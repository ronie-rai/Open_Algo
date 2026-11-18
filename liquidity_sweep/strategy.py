"""Building blocks of the liquidity sweep strategy: pivots, levels, ATR and bias.

Nothing here looks ahead. A pivot at bar i is only confirmed at bar i+n and the
engine adds it to the level book from bar i+n+1. ATR is a causal recursion. A
bias bar is usable only from the time it has fully closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import TF_MINUTES, Config, hhmm_to_minutes
from resample import minute_of_day, resample_ohlcv


def find_pivots(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pivot highs and lows.

    A pivot high at bar i is where high[i] is the strict maximum of the highs in
    [i-n, i+n] (no other bar in the window equals it). Pivot lows mirror this.
    Bars closer than n to either end can never be pivots.

    Returns:
        Two boolean arrays (pivot_high, pivot_low), indexed by the pivot bar i.
    """
    size = len(high)
    ph = np.zeros(size, dtype=bool)
    pl = np.zeros(size, dtype=bool)
    if size < 2 * n + 1:
        return ph, pl
    win_h = np.lib.stride_tricks.sliding_window_view(high, 2 * n + 1)
    win_l = np.lib.stride_tricks.sliding_window_view(low, 2 * n + 1)
    centre_h = high[n : size - n]
    centre_l = low[n : size - n]
    ph[n : size - n] = (win_h.max(axis=1) == centre_h) & (
        (win_h == centre_h[:, None]).sum(axis=1) == 1
    )
    pl[n : size - n] = (win_l.min(axis=1) == centre_l) & (
        (win_l == centre_l[:, None]).sum(axis=1) == 1
    )
    return ph, pl


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR, seeded with the simple mean of the first `period` true ranges.

    The first true range is high-low (there is no previous close). ATR is NaN
    before `period` true ranges exist. Each value uses only bars up to and
    including its own bar.
    """
    size = len(high)
    atr = np.full(size, np.nan)
    if size < period:
        return atr
    prev_close = np.empty(size)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    atr[period - 1] = tr[:period].mean()
    for i in range(period, size):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def seeded_ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA with alpha 2/(span+1), seeded with the simple mean of the first `span` values.

    NaN until `span` values exist, so "the EMA exists" means span closes are in.
    """
    out = np.full(len(values), np.nan)
    if len(values) < span:
        return out
    alpha = 2.0 / (span + 1.0)
    out[span - 1] = values[:span].mean()
    for i in range(span, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def build_bias_bars(base: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Bias bars (1h or daily) from intraday bars, with EMA and availability time.

    Columns: close, ema, available_at (tz-aware time from which the bar may be used).
    A 1h bar is available from its end (start + 1h, capped at the session close,
    so the short 15:15 bar is available from 15:30). A daily bar is available
    from that session's close, so in practice from the next session.
    """
    bars = resample_ohlcv(base, cfg.bias_tf, cfg)
    out = pd.DataFrame(index=bars.index)
    out["close"] = bars["close"].to_numpy()
    out["ema"] = seeded_ema(bars["close"].to_numpy(dtype=float), cfg.bias_ema)
    close_min = hhmm_to_minutes(cfg.session_close)
    if cfg.bias_tf == "D":
        day = bars.index.normalize()
        out["available_at"] = day + pd.Timedelta(minutes=close_min)
    else:
        start = minute_of_day(bars.index)
        end = np.minimum(start + TF_MINUTES[cfg.bias_tf], close_min)
        out["available_at"] = bars.index + pd.to_timedelta(end - start, unit="min")
    return out


def bias_direction(bar_index: pd.DatetimeIndex, bias: pd.DataFrame) -> np.ndarray:
    """Bias per trading bar: +1 longs allowed, -1 shorts allowed, 0 none.

    Uses the last bias bar whose available_at is at or before the trading bar's
    start, so the bias bar still in progress is never used. 0 until the EMA exists.
    """
    out = np.zeros(len(bar_index), dtype=np.int8)
    if bias.empty:
        return out
    avail = pd.DatetimeIndex(bias["available_at"]).as_unit("ns").asi8
    order = np.argsort(avail, kind="stable")
    avail = avail[order]
    close = bias["close"].to_numpy()[order]
    ema = bias["ema"].to_numpy()[order]
    pos = np.searchsorted(avail, bar_index.asi8, side="right") - 1
    ok = pos >= 0
    c = np.where(ok, close[np.clip(pos, 0, None)], np.nan)
    e = np.where(ok, ema[np.clip(pos, 0, None)], np.nan)
    valid = ok & ~np.isnan(e)
    out[valid & (c > e)] = 1
    out[valid & (c < e)] = -1
    return out


@dataclass
class Level:
    """A cluster of pivots on one side. kind "R" resistance, "S" support."""

    kind: str
    price: float
    touches: int
    members: tuple[int, ...]  # pivot bar indices


class LevelBook:
    """Live pivots per side, clustered into levels on demand.

    A pivot is identified by its bar index (a bar is at most one pivot high and
    one pivot low). Killed pivots never come back; new pivots can form a new level.
    """

    def __init__(self, cfg: Config):
        self.tol = cfg.level_tol
        self.lookback = cfg.level_lookback
        self.live: dict[str, dict[int, float]] = {"R": {}, "S": {}}

    def add(self, kind: str, idx: int, price: float) -> None:
        self.live[kind][idx] = float(price)

    def prune(self, t: int) -> None:
        """Keep only pivots from the last `lookback` bars before bar t (t - i <= lookback)."""
        for side in self.live.values():
            for idx in [i for i in side if t - i > self.lookback]:
                del side[idx]

    def kill(self, kind: str, members: tuple[int, ...]) -> None:
        side = self.live[kind]
        for idx in members:
            side.pop(idx, None)

    def levels(self, kind: str) -> list[Level]:
        """Cluster live pivots of one side.

        Pivots are sorted by price and grouped greedily: a pivot joins the current
        cluster while it is within level_tol of the cluster's lowest member, so
        every pair in a cluster is within level_tol. Level price is the mean.
        """
        items = sorted(self.live[kind].items(), key=lambda kv: (kv[1], kv[0]))
        out: list[Level] = []
        cluster: list[tuple[int, float]] = []
        for idx, price in items:
            if cluster and price > cluster[0][1] * (1 + self.tol):
                out.append(self._make(kind, cluster))
                cluster = []
            cluster.append((idx, price))
        if cluster:
            out.append(self._make(kind, cluster))
        return out

    @staticmethod
    def _make(kind: str, cluster: list[tuple[int, float]]) -> Level:
        prices = [p for _, p in cluster]
        return Level(
            kind=kind,
            price=float(np.mean(prices)),
            touches=len(cluster),
            members=tuple(sorted(i for i, _ in cluster)),
        )


def pick_signal(
    high: float,
    low: float,
    close: float,
    resistance: list[Level],
    support: list[Level],
    min_touches: int,
) -> tuple[int, Level | None]:
    """Sweep signal on one bar, before the bias filter.

    Short: a resistance L with touches >= min_touches, high > L and close < L.
    Long: a support L with touches >= min_touches, low < L and close > L.
    Several levels: most touches, then the most extreme price (highest
    resistance, lowest support). A bar that sweeps both sides gives no signal.

    Returns:
        (side, level): side -1 short, +1 long, 0 none.
    """
    shorts = [
        lv
        for lv in resistance
        if lv.touches >= min_touches and high > lv.price and close < lv.price
    ]
    longs = [
        lv for lv in support if lv.touches >= min_touches and low < lv.price and close > lv.price
    ]
    if shorts and longs:
        return 0, None
    if shorts:
        return -1, max(shorts, key=lambda lv: (lv.touches, lv.price))
    if longs:
        return 1, max(longs, key=lambda lv: (lv.touches, -lv.price))
    return 0, None
