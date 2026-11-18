"""Session-aware OHLCV resampling for NSE bars.

Bars are labelled by their start time (tz-aware Asia/Kolkata). Higher timeframe
bars are built inside each session, anchored at the session open (09:15), so no
bar ever spans two sessions. The last bar of a session may be short (the 1h bar
15:15-15:30); it is a complete session bar and is kept.

Completeness rule: a bar is kept only if it has every constituent base bar its
span requires. That drops an incomplete leading or trailing bar at either end of
the data, and a bar with an internal hole. Daily bars ("D") are kept when the
session has its first and its last base bar (so open and close are the true
session open and close); internal holes inside a session do not drop a daily bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TF_MINUTES, Config, hhmm_to_minutes

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Minutes after midnight (IST wall clock) for each timestamp."""
    return (index.hour * 60 + index.minute).to_numpy()


def session_key(index: pd.DatetimeIndex) -> np.ndarray:
    """An integer per session (the calendar date as a day ordinal)."""
    return (index.normalize().tz_localize(None).asi8 // (86400 * 10**9)).astype(np.int64)


def infer_bar_minutes(df: pd.DataFrame) -> int | None:
    """Most common spacing in minutes between consecutive bars of the same session."""
    if len(df) < 2:
        return None
    idx = df.index
    same = session_key(idx)[1:] == session_key(idx)[:-1]
    diffs = (np.diff(idx.asi8) // (60 * 10**9))[same]
    if len(diffs) == 0:
        return None
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


def bar_end_minutes(index: pd.DatetimeIndex, tf: str, cfg: Config | None = None) -> np.ndarray:
    """End time of each bar in minutes after midnight, capped at the session close."""
    cfg = cfg or Config()
    close = hhmm_to_minutes(cfg.session_close)
    return np.minimum(minute_of_day(index) + TF_MINUTES[tf], close)


def resample_ohlcv(df: pd.DataFrame, tf: str, cfg: Config | None = None) -> pd.DataFrame:
    """Resample intraday bars to `tf` ("5m", "15m", "1h" or "D") within each session.

    Args:
        df: OHLCV bars indexed by tz-aware start time, all inside the session.
        tf: Target timeframe.
        cfg: Session settings (defaults to Config()).

    Returns:
        OHLCV DataFrame labelled by bar start (label="left", closed="left"),
        with incomplete bars dropped as described in the module docstring.
    """
    cfg = cfg or Config()
    cols = ["open", "high", "low", "close", "volume"]
    if df.empty:
        return df[cols].copy()
    base = infer_bar_minutes(df)
    if base is None:
        return df[cols].iloc[0:0].copy()
    open_min = hhmm_to_minutes(cfg.session_open)
    close_min = hhmm_to_minutes(cfg.session_close)
    idx = df.index
    mod = minute_of_day(idx)
    day = idx.normalize()

    if tf == "D":
        keys = day
        grouped = df[cols].groupby(keys)
        out = grouped.agg(AGG)
        mod_s = pd.Series(mod, index=idx).groupby(keys)
        first = mod_s.min()
        last = mod_s.max()
        keep = (first == open_min) & (last == close_min - base)
        out = out[keep.to_numpy()]
        out.index = out.index + pd.Timedelta(minutes=open_min)
        out.index.name = df.index.name
        return out.astype(float)

    minutes = TF_MINUTES[tf]
    if minutes % base != 0:
        raise ValueError(f"Cannot build {tf} bars from {base}-minute bars")
    bucket = (mod - open_min) // minutes
    start_min = open_min + bucket * minutes
    keys = day + pd.to_timedelta(start_min, unit="min")
    grouped = df[cols].groupby(keys)
    out = grouped.agg(AGG)
    counts = grouped.size()
    label_min = minute_of_day(out.index)
    span = np.minimum(label_min + minutes, close_min) - label_min
    expected = span // base
    out = out[counts.to_numpy() == expected]
    out.index.name = df.index.name
    return out.astype(float)
