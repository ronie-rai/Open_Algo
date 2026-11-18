"""Deterministic synthetic NSE-shaped bars for tests. Not market data."""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"


def session_index(
    days: list[str] | pd.DatetimeIndex, minutes: int = 5, start="09:15", end="15:30"
) -> pd.DatetimeIndex:
    """Bar start times for whole sessions on the given days."""
    stamps = []
    for d in pd.DatetimeIndex(days):
        day = pd.Timestamp(d.date())
        t = day + pd.Timedelta(start + ":00")
        stop = day + pd.Timedelta(end + ":00")
        while t < stop:
            stamps.append(t)
            t += pd.Timedelta(minutes=minutes)
    return pd.DatetimeIndex(stamps).tz_localize(TZ)


def frame(index: pd.DatetimeIndex, o, h, lo, c, v=None) -> pd.DataFrame:
    """OHLCV frame from arrays (volume defaults to 1000)."""
    n = len(index)
    v = np.full(n, 1000.0) if v is None else v
    return pd.DataFrame(
        {
            "open": np.asarray(o, float),
            "high": np.asarray(h, float),
            "low": np.asarray(lo, float),
            "close": np.asarray(c, float),
            "volume": np.asarray(v, float),
        },
        index=index,
    ).rename_axis("timestamp")


def flat_bars(index: pd.DatetimeIndex, price: float = 100.0, spread: float = 0.02) -> pd.DataFrame:
    """Bars that never move: open = close = price, high/low +- spread."""
    n = len(index)
    p = np.full(n, price)
    return frame(index, p, p + spread, p - spread, p)


def random_sessions(sessions: int = 60, seed: int = 7, start: str = "2025-11-03") -> pd.DataFrame:
    """Mean-reverting 5m bars with a slow wave in the mean, so levels and both biases occur."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=sessions)
    idx = session_index(days)
    n = len(idx)
    k = np.arange(n)
    mean = 1000 * (1 + 0.04 * np.sin(2 * np.pi * k / (75 * 20)))
    price = np.empty(n)
    opens = np.empty(n)
    p = 1000.0
    new_session = np.r_[True, idx.normalize()[1:] != idx.normalize()[:-1]]
    for i in range(n):
        if new_session[i] and i:
            p = p * (1 + rng.normal(0, 0.003))
        opens[i] = p
        p = p + 0.08 * (mean[i] - p) + rng.normal(0, 1.2)
        price[i] = p
    highs = np.maximum(opens, price) + np.abs(rng.normal(0, 0.8, n))
    lows = np.minimum(opens, price) - np.abs(rng.normal(0, 0.8, n))
    return frame(idx, opens.round(2), highs.round(2), lows.round(2), price.round(2))
