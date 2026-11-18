"""Bar-by-bar engine for the liquidity sweep strategy on one symbol.

Order of work on bar t:
  1. Add pivots confirmed at bar t-1 (pivot bar t-1-n) and drop pivots older
     than the lookback.
  2. Fill a pending entry at this bar's open (skipped if the open is already
     beyond the stop). An entered level is killed (its pivots removed).
  3. Manage an open trade: stop first, then target, then the time exit after
     max_bars bars, then the session square-off at the bar ending at 15:15.
  4. If flat (including flat because of an exit on this bar), look for a sweep
     signal; it becomes a pending entry for the next bar only if that bar is in
     the same session and opens before 15:00.
  5. Apply level invalidations: a level dies once this bar closes beyond it by
     more than level_tol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import TF_MINUTES, Config, hhmm_to_minutes
from resample import bar_end_minutes, infer_bar_minutes, minute_of_day, session_key
from strategy import (
    LevelBook,
    bias_direction,
    build_bias_bars,
    find_pivots,
    pick_signal,
    wilder_atr,
)

TRADE_COLUMNS = [
    "symbol",
    "side",
    "entry_time",
    "entry_price",
    "stop",
    "target",
    "exit_time",
    "exit_price",
    "exit_reason",
    "r_gross",
    "r_net",
    "level",
    "touches",
]


@dataclass
class _Pending:
    side: int
    stop: float
    level_kind: str
    level_price: float
    touches: int
    members: tuple[int, ...]


@dataclass
class _Position:
    side: int
    entry_idx: int
    entry: float
    stop: float
    target: float
    session: int
    level_price: float
    touches: int


def trade_r(
    side: int, entry: float, exit_price: float, stop: float, fee: float, slippage: float
) -> tuple[float, float]:
    """(r_gross, r_net) for one trade under the given per-side fee and slippage.

    R_gross = (exit - entry) / risk for longs, sign flipped for shorts.
    cost_R = (fee + slippage) * (entry + exit) / risk.
    """
    risk = abs(entry - stop)
    r_gross = side * (exit_price - entry) / risk
    cost_r = (fee + slippage) * (entry + exit_price) / risk
    return r_gross, r_gross - cost_r


def check_timeframe(bars: pd.DataFrame, cfg: Config) -> None:
    """Raise if the bars' spacing does not match cfg.timeframe."""
    found = infer_bar_minutes(bars)
    want = TF_MINUTES[cfg.timeframe]
    if found is not None and found != want:
        raise ValueError(f"Bars are {found}-minute but config.timeframe is {cfg.timeframe}")


def run_backtest(
    bars: pd.DataFrame,
    cfg: Config,
    symbol: str = "",
    bias_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the strategy on one symbol.

    Args:
        bars: OHLCV bars of cfg.timeframe, tz-aware start-time index, sorted.
        cfg: Parameters.
        symbol: Label written to the trades.
        bias_source: Intraday bars to build the bias timeframe from (normally the
            5-minute data). Defaults to `bars`.

    Returns:
        One row per trade with TRADE_COLUMNS. Times are bar start labels.
    """
    check_timeframe(bars, cfg)
    if bars.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    idx = bars.index
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    size = len(bars)
    n = cfg.pivot_n

    sess = session_key(idx)
    start_min = minute_of_day(idx)
    end_min = bar_end_minutes(idx, cfg.timeframe, cfg)
    last_entry = hhmm_to_minutes(cfg.last_entry)
    square_off = hhmm_to_minutes(cfg.square_off)

    ph, pl = find_pivots(h, lo, n)
    atr = wilder_atr(h, lo, c, cfg.atr_period)
    if cfg.use_bias:
        bias = bias_direction(
            idx, build_bias_bars(bars if bias_source is None else bias_source, cfg)
        )
    else:
        bias = np.zeros(size, dtype=np.int8)

    book = LevelBook(cfg)
    pending: _Pending | None = None
    pos: _Position | None = None
    trades: list[dict] = []

    def close_trade(p: _Position, exit_i: int, price: float, reason: str) -> None:
        r_gross, r_net = trade_r(p.side, p.entry, price, p.stop, cfg.fee, cfg.slippage)
        trades.append(
            {
                "symbol": symbol,
                "side": "long" if p.side > 0 else "short",
                "entry_time": idx[p.entry_idx],
                "entry_price": p.entry,
                "stop": p.stop,
                "target": p.target,
                "exit_time": idx[exit_i],
                "exit_price": price,
                "exit_reason": reason,
                "r_gross": r_gross,
                "r_net": r_net,
                "level": p.level_price,
                "touches": p.touches,
            }
        )

    for t in range(size):
        # 1. Pivots confirmed at bar t-1 become usable now.
        j = t - 1 - n
        if j >= 0:
            if ph[j]:
                book.add("R", j, h[j])
            if pl[j]:
                book.add("S", j, lo[j])
        book.prune(t)

        # 2. Pending entry at this bar's open.
        if pending is not None:
            p = pending
            pending = None
            beyond = (p.side > 0 and o[t] <= p.stop) or (p.side < 0 and o[t] >= p.stop)
            if not beyond:
                risk = abs(o[t] - p.stop)
                pos = _Position(
                    side=p.side,
                    entry_idx=t,
                    entry=o[t],
                    stop=p.stop,
                    target=o[t] + p.side * cfg.rr * risk,
                    session=sess[t],
                    level_price=p.level_price,
                    touches=p.touches,
                )
                book.kill(p.level_kind, p.members)

        # 3. Manage the open trade.
        if pos is not None:
            if sess[t] != pos.session:
                # The square-off bar was missing from the data: flatten at the
                # close of the session's last bar.
                close_trade(pos, t - 1, c[t - 1], "session_end")
                pos = None
            elif start_min[t] >= square_off:
                # Bar ending at 15:15 was missing: flatten at this bar's open.
                close_trade(pos, t, o[t], "session_end")
                pos = None
        if pos is not None:
            s = pos.side
            exit_price = None
            reason = ""
            gap_stop = (s > 0 and o[t] <= pos.stop) or (s < 0 and o[t] >= pos.stop)
            hit_stop = (s > 0 and lo[t] <= pos.stop) or (s < 0 and h[t] >= pos.stop)
            hit_target = (s > 0 and h[t] >= pos.target) or (s < 0 and lo[t] <= pos.target)
            if gap_stop and t > pos.entry_idx:
                exit_price, reason = o[t], "stop"
            elif hit_stop:
                exit_price, reason = pos.stop, "stop"
            elif hit_target:
                exit_price, reason = pos.target, "target"
            elif t - pos.entry_idx + 1 >= cfg.max_bars:
                exit_price, reason = c[t], "time"
            elif end_min[t] >= square_off:
                exit_price, reason = c[t], "session_end"
            if exit_price is not None:
                close_trade(pos, t, exit_price, reason)
                pos = None

        resistance = book.levels("R")
        support = book.levels("S")

        # 4. Signal on a flat bar.
        if pos is None:
            side, level = pick_signal(h[t], lo[t], c[t], resistance, support, cfg.min_touches)
            if side != 0 and cfg.use_bias and bias[t] != side:
                side = 0
            if side != 0 and cfg.atr_mult > 0 and np.isnan(atr[t]):
                side = 0
            tradable_next = (
                t + 1 < size and sess[t + 1] == sess[t] and start_min[t + 1] < last_entry
            )
            if side != 0 and tradable_next:
                extra = cfg.atr_mult * atr[t] if cfg.atr_mult > 0 else 0.0
                if side < 0:
                    stop = max(h[t] * (1 + cfg.stop_buffer), h[t] + extra)
                else:
                    stop = min(lo[t] * (1 - cfg.stop_buffer), lo[t] - extra)
                pending = _Pending(
                    side, stop, level.kind, level.price, level.touches, level.members
                )

        # 5. Invalidate levels this bar closed beyond.
        for lv in resistance:
            if c[t] > lv.price * (1 + cfg.level_tol):
                book.kill("R", lv.members)
        for lv in support:
            if c[t] < lv.price * (1 - cfg.level_tol):
                book.kill("S", lv.members)

    if pos is not None:
        close_trade(pos, size - 1, c[size - 1], "end_of_data")
    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


def net_r_at_costs(
    trades: pd.DataFrame, fee: float | None = None, slippage: float | None = None
) -> pd.Series:
    """Re-price the same trades' net R under another fee and slippage per side.

    Defaults to the low-cost scenario (Config.low_fee, Config.low_slippage).
    """
    fee = Config().low_fee if fee is None else fee
    slippage = Config().low_slippage if slippage is None else slippage
    if trades.empty:
        return pd.Series(dtype=float)
    risk = (trades["entry_price"] - trades["stop"]).abs()
    cost_r = (fee + slippage) * (trades["entry_price"] + trades["exit_price"]) / risk
    return trades["r_gross"] - cost_r
