#!/usr/bin/env python
"""
Supertrend entries with fair value gap stops and targets, across many NSE stocks

How it trades
  1. Every closed 5 minute bar, for every symbol in SYMBOLS, the TradingView
     "Supertrend" study (Pine v4: Periods 10, source hl2, Multiplier 3,
     changeATR on) is computed with a line-for-line port. The bar still
     forming is never used.
  2. A Buy signal on the last closed bar opens a long; a Sell signal opens a
     short. Intraday (MIS) only.
  3. Only ONE trade is open across all symbols at any time. While a trade is
     open or an entry order is pending, signals on every other symbol are
     ignored (each one is logged and journalled once).
  4. The stop and the target come from fair value gaps (FVG) and order blocks
     (OB) formed in the last LEVEL_LOOKBACK_DAYS trading days:

       BUY   stop   = the LOW of the nearest (highest) unfilled bullish FVG
                      or unmitigated bullish OB lying entirely below entry
             target = the MIDPOINT of the nearest (lowest) bearish FVG above
                      entry whose midpoint has not been reached since it formed
       SELL  stop   = the HIGH of the nearest (lowest) unfilled bearish FVG
                      or unmitigated bearish OB lying entirely above entry
             target = the MIDPOINT of the nearest (highest) bullish FVG below
                      entry whose midpoint has not been reached since it formed

     The nearest stop at least MIN_STOP_ATR x ATR(14) away is used, then the
     nearest target paying at least MIN_RR times that risk. If there is no
     valid stop or no valid target, the signal is skipped and the reason is
     logged and journalled.
  5. Quantity: min(floor(MAX_LOSS_PER_TRADE / stop distance),
     floor(CAPITAL / entry)), rounded down to the lot size, with no leverage.
     Both prices are padded by SIZING_BUFFER_PCT first so a little slippage
     cannot push the loss at the stop or the position value over the caps.
     A quantity below one lot skips the trade.
  6. Exits: a stop-loss order at the broker (SL-M, or SL when the broker
     refuses SL-M), the target watched by this script on the LTP and exited at
     market, an opposite Supertrend signal on the traded symbol (EXIT_ON_FLIP),
     and square-off at SQUARE_OFF. An exit always cancels the broker stop first
     and closes with placesmartorder to a position of zero, so it never
     reverses a position.
  7. After an exit caused by a flip, the flip is itself a fresh signal. With
     ENTER_ON_FLIP on, the opposite trade is considered only once the broker
     reports the position flat, and it goes through every normal check (entry
     window, daily caps, a valid stop and target) as a separate new order.

Definitions (all evaluated with bars up to the last closed bar only)
  Fair value gap, FVG_MODE = "3candle" (default), candles i-2, i-1, i:
      bullish  high[i-2] < low[i]   zone [high[i-2], low[i]]
      bearish  low[i-2]  > high[i]  zone [high[i],   low[i-2]]
  FVG_MODE = "2candle" uses two consecutive candles i-1, i instead
  (high[i-1] < low[i] bullish, low[i-1] > high[i] bearish). On 5 minute bars a
  true two-candle gap is rare, which is why the three-candle gap is the default.
  Gaps narrower than FVG_MIN_ATR x ATR(14) are ignored as tick noise.

  Unfilled (for a stop): no bar after the gap formed traded beyond its far
  edge, that is below the low of a bullish gap or above the high of a bearish
  one. Price may have entered the gap.
  Midpoint not reached (for a target): no bar after the gap formed traded at
  or through its midpoint (low <= mid for a bullish gap, high >= mid for a
  bearish one). A target gap must have its midpoint beyond entry.

  Order block: the last opposite-colour candle in the OB_SWING_LEN bars
  before a displacement candle whose body is at least OB_IMPULSE_ATR x ATR(14)
  and whose close breaks the highest high (bullish) or lowest low (bearish) of
  those bars, confirmed on that candle's close. Zone = the block candle's
  low to high. It counts as unmitigated until a later bar trades back into
  the zone. Only blocks lying entirely on the protective side of entry are
  used, and only for the stop.

Setup
  - Edit the CONFIG block below, or override any value with an STFVG_* entry
    in .env, for example STFVG_SYMBOLS = 'SBIN,TCS,LT'.
  - Upload at /python, exchange NSE, schedule 09:15 - 15:20, Mon-Fri.
  - Backtest it first with the Backtest button on its card.
  - Then run it in analyzer (sandbox) mode before going live. The log says
    which mode is active.
  - Telegram alerts are optional: set STFVG_TELEGRAM_USER to your OpenAlgo
    login name.

Every signal, skip, entry and exit is written to supertrend_fvg_data/journal.jsonl
beside this file, and the open trade is kept in state.json so a restart picks
it up again. On a restart each symbol's current trend is only recorded: a
signal from a bar before the restart is never traded.
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openalgo import api


def _env(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    value = value.strip()
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value


def _env_list(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [s.strip().upper() for s in value.split(",") if s.strip()]


# =============================================================================
# CONFIG
# =============================================================================
STRATEGY_NAME = _env("STFVG_STRATEGY_NAME", "Supertrend FVG")
SYMBOLS = _env_list(
    "STFVG_SYMBOLS",
    [
        "ICICIBANK",
        "HDFCBANK",
        "BSE",
        "BAJFINANCE",
        "M&M",
        "RELIANCE",
        "SBIN",
        "CUMMINSIND",
        "BHARTIARTL",
        "PIIND",
        "LENSKART",
        "GROWW",
        "TCS",
        "ASHOKLEY",
        "LT",
        "TONBO",
        "MARUTI",
    ],
)
EXCHANGE = os.getenv("OPENALGO_STRATEGY_EXCHANGE") or _env("STFVG_EXCHANGE", "NSE")
PRODUCT = _env("STFVG_PRODUCT", "MIS")
INTERVAL = "5m"
BAR_MINUTES = 5

# Supertrend (TradingView study inputs)
ST_PERIOD = _env("STFVG_ST_PERIOD", 10)  # "ATR Period"
ST_MULTIPLIER = _env("STFVG_ST_MULTIPLIER", 3.0)  # "ATR Multiplier"
ST_SOURCE = _env("STFVG_ST_SOURCE", "hl2")  # hl2, close, open, high, low, hlc3, ohlc4
ST_CHANGE_ATR = _env("STFVG_ST_CHANGE_ATR", True)  # True: ta.atr (RMA); False: SMA of tr

# Stop and target levels
RISK_ATR_PERIOD = 14  # ATR(14), Wilder, for MIN_STOP_ATR, FVG_MIN_ATR and OB_IMPULSE_ATR
FVG_MODE = _env("STFVG_FVG_MODE", "3candle")  # "3candle" or "2candle"
FVG_MIN_ATR = _env("STFVG_FVG_MIN_ATR", 0.05)  # ignore gaps narrower than this x ATR
USE_ORDER_BLOCKS = _env("STFVG_USE_ORDER_BLOCKS", True)  # order blocks as stop candidates
OB_SWING_LEN = _env("STFVG_OB_SWING_LEN", 10)
OB_IMPULSE_ATR = _env("STFVG_OB_IMPULSE_ATR", 1.0)
LEVEL_LOOKBACK_DAYS = _env("STFVG_LOOKBACK_DAYS", 3)  # trading days of gaps and blocks
MIN_STOP_ATR = _env("STFVG_MIN_STOP_ATR", 0.3)
MIN_RR = _env("STFVG_MIN_RR", 1.0)

# Sizing (no leverage)
CAPITAL = _env("STFVG_CAPITAL", 25000.0)  # most a position may be worth
MAX_LOSS_PER_TRADE = _env("STFVG_MAX_LOSS_PER_TRADE", 2000.0)  # loss if the stop fills
SIZING_BUFFER_PCT = _env("STFVG_SIZING_BUFFER_PCT", 0.1)  # slippage allowance, % of price

# Daily safety caps
MAX_TRADES_PER_DAY = _env("STFVG_MAX_TRADES_PER_DAY", 5)
MAX_DAILY_LOSS = _env("STFVG_MAX_DAILY_LOSS", 4000.0)  # no new trades after this loss

# Session (IST)
ENTRY_START = _env("STFVG_ENTRY_START", "09:30")
LAST_ENTRY = _env("STFVG_LAST_ENTRY", "14:45")
SQUARE_OFF = _env("STFVG_SQUARE_OFF", "15:10")

# Exits
EXIT_ON_FLIP = _env("STFVG_EXIT_ON_FLIP", True)
ENTER_ON_FLIP = _env("STFVG_ENTER_ON_FLIP", True)  # opposite trade after a confirmed flat

# Plumbing
HISTORY_DAYS = 12  # calendar days of 5m bars, so ATR and Supertrend are warmed up
HISTORY_PAUSE_SEC = 0.3  # between history calls, for the broker's rate limit
POLL_SECONDS = 10
ENTRY_TIMEOUT_SEC = 60  # an entry order not final by then is cancelled
TELEGRAM_USER = _env("STFVG_TELEGRAM_USER", "")  # your OpenAlgo login name, optional

API_KEY = os.getenv("OPENALGO_API_KEY")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL") or (
    f"ws://{os.getenv('WEBSOCKET_HOST', '127.0.0.1')}:{os.getenv('WEBSOCKET_PORT', '8765')}"
)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path(__file__).resolve().parent / "supertrend_fvg_data"
FVG_MODES = ("3candle", "2candle")


def now_ist():
    return datetime.now(IST).replace(tzinfo=None)


def hhmm(text):
    h, m = text.split(":")
    return int(h) * 60 + int(m)


def minutes_of(dt):
    return dt.hour * 60 + dt.minute


def log(msg):
    print(f"[{now_ist():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def round_tick(price, tick):
    return round(round(price / tick) * tick, 4)


# =============================================================================
# SUPERTREND (line-for-line port of the TradingView Pine v4 study)
# =============================================================================
def true_range(high, low, close, handle_na):
    """Pine tr(handle_na). On bar 0, close[1] is na: high - low if handle_na, else na."""
    n = len(close)
    tr = np.empty(n)
    if n:
        tr[0] = high[0] - low[0] if handle_na else np.nan
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return tr


def pine_sma(src, length):
    """Pine sma(): na until `length` values exist, and na while any value in the window is na."""
    n = len(src)
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        window = src[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = window.mean()
    return out


def pine_rma(src, length):
    """Pine rma(): seeded with sma(src, length), then alpha = 1 / length."""
    seed = pine_sma(src, length)
    out = np.full(len(src), np.nan)
    for i in range(len(src)):
        if i == 0 or np.isnan(out[i - 1]):
            out[i] = seed[i]
        else:
            out[i] = (src[i] + (length - 1) * out[i - 1]) / length
    return out


def atr_rma(high, low, close, length):
    """ta.atr(length): rma(tr(true), length)."""
    return pine_rma(true_range(high, low, close, True), length)


def source_series(o, h, lo, c, name):
    name = str(name).lower()
    if name == "hl2":
        return (h + lo) / 2
    if name == "hlc3":
        return (h + lo + c) / 3
    if name == "ohlc4":
        return (o + h + lo + c) / 4
    series = {"open": o, "high": h, "low": lo, "close": c}.get(name)
    if series is None:
        raise ValueError(f"unknown Supertrend source {name!r}")
    return np.asarray(series, dtype=float)


def _nz(x, replacement):
    return replacement if np.isnan(x) else x


def supertrend(o, h, lo, c, period=None, multiplier=None, source=None, change_atr=None):
    """Bar-by-bar port of:

        atr2 = sma(tr, Periods)
        atr = changeATR ? atr(Periods) : atr2
        up = src - (Multiplier * atr)
        up1 = nz(up[1], up)
        up := close[1] > up1 ? max(up, up1) : up
        dn = src + (Multiplier * atr)
        dn1 = nz(dn[1], dn)
        dn := close[1] < dn1 ? min(dn, dn1) : dn
        trend = 1
        trend := nz(trend[1], trend)
        trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        buySignal = trend == 1 and trend[1] == -1
        sellSignal = trend == -1 and trend[1] == 1

    `tr` in sma(tr, ...) is Pine's tr variable, tr(false), which is na on bar 0;
    atr() uses tr(true). Comparisons with na are false, as in Pine.
    Returns a dict of numpy arrays: trend, up, dn, atr, buy, sell.
    """
    period = ST_PERIOD if period is None else period
    multiplier = ST_MULTIPLIER if multiplier is None else multiplier
    source = ST_SOURCE if source is None else source
    change_atr = ST_CHANGE_ATR if change_atr is None else change_atr

    o, h, lo, c = (np.asarray(x, dtype=float) for x in (o, h, lo, c))
    src = source_series(o, h, lo, c, source)
    if change_atr:
        atr = atr_rma(h, lo, c, period)
    else:
        atr = pine_sma(true_range(h, lo, c, False), period)

    n = len(c)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)
    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)
    for i in range(n):
        prev_close = c[i - 1] if i > 0 else np.nan
        u = src[i] - multiplier * atr[i]
        u1 = _nz(up[i - 1] if i > 0 else np.nan, u)
        if prev_close > u1:
            u = max(u, u1)
        d = src[i] + multiplier * atr[i]
        d1 = _nz(dn[i - 1] if i > 0 else np.nan, d)
        if prev_close < d1:
            d = min(d, d1)
        up[i], dn[i] = u, d

        prev_trend = trend[i - 1] if i > 0 else None
        t = 1 if prev_trend is None else int(prev_trend)  # nz(trend[1], trend)
        if t == -1 and c[i] > d1:
            t = 1
        elif t == 1 and c[i] < u1:
            t = -1
        trend[i] = t
        buy[i] = prev_trend is not None and t == 1 and prev_trend == -1
        sell[i] = prev_trend is not None and t == -1 and prev_trend == 1
    return {"trend": trend, "up": up, "dn": dn, "atr": atr, "buy": buy, "sell": sell}


# =============================================================================
# FAIR VALUE GAPS AND ORDER BLOCKS
# =============================================================================
@dataclass
class Gap:
    kind: str  # "bull" or "bear"
    top: float
    bottom: float
    created_idx: int  # the bar that completes the gap
    filled_idx: int = -1  # first later bar beyond the far edge, -1 while unfilled
    mid_hit_idx: int = -1  # first later bar at or through the midpoint, -1 while not reached

    @property
    def mid(self):
        return (self.top + self.bottom) / 2


def detect_fvgs(high, low, atr, mode=None, min_atr=None, end=None):
    """Fair value gaps formed up to bar `end` (default: the last bar), with their
    fill state as of `end`. Bars after `end` are never read.

    "3candle": bullish when high[i-2] < low[i], bearish when low[i-2] > high[i].
    "2candle": the same test between consecutive candles i-1 and i.
    """
    mode = FVG_MODE if mode is None else mode
    min_atr = FVG_MIN_ATR if min_atr is None else min_atr
    if mode not in FVG_MODES:
        raise ValueError(f"FVG_MODE must be one of {FVG_MODES}, not {mode!r}")
    span = 2 if mode == "3candle" else 1
    end = len(high) - 1 if end is None else int(end)

    gaps = []
    for i in range(span, end + 1):
        a = atr[i] if i < len(atr) else np.nan
        min_size = min_atr * a if np.isfinite(a) else 0.0
        if high[i - span] < low[i] and low[i] - high[i - span] >= min_size:
            gaps.append(Gap("bull", float(low[i]), float(high[i - span]), i))
        if low[i - span] > high[i] and low[i - span] - high[i] >= min_size:
            gaps.append(Gap("bear", float(low[i - span]), float(high[i]), i))

    for g in gaps:
        mid = g.mid
        for k in range(g.created_idx + 1, end + 1):
            if g.kind == "bull":
                if g.mid_hit_idx < 0 and low[k] <= mid:
                    g.mid_hit_idx = k
                if low[k] < g.bottom:
                    g.filled_idx = k
                    break
            else:
                if g.mid_hit_idx < 0 and high[k] >= mid:
                    g.mid_hit_idx = k
                if high[k] > g.top:
                    g.filled_idx = k
                    break
    return gaps


@dataclass
class Block:
    kind: str  # "bull" (demand) or "bear" (supply)
    top: float
    bottom: float
    ob_idx: int  # the order-block candle
    created_idx: int  # the displacement candle that confirmed it
    mitigated_idx: int = -1  # first later bar that traded back into it, -1 while fresh


def detect_order_blocks(o, h, lo, c, atr, swing_len=None, impulse_atr=None, end=None):
    """Order blocks confirmed up to bar `end` (default: the last bar), with their
    mitigation state as of `end`. Bars after `end` are never read.

    Bullish: bar t closes above the highest high of the `swing_len` bars before
    it with a bullish body of at least `impulse_atr` x ATR; the block is the
    last bearish candle among those bars. Bearish is the mirror.
    """
    swing_len = OB_SWING_LEN if swing_len is None else swing_len
    impulse_atr = OB_IMPULSE_ATR if impulse_atr is None else impulse_atr
    end = len(c) - 1 if end is None else int(end)
    blocks, used = [], set()
    for t in range(swing_len, end + 1):
        a = atr[t]
        if not np.isfinite(a):
            continue
        body = c[t] - o[t]
        window = range(t - 1, t - swing_len - 1, -1)  # t-1 back to t-swing_len
        if body >= impulse_atr * a and c[t] > max(h[k] for k in window):
            for k in window:
                if c[k] < o[k]:
                    if ("bull", k) not in used:
                        used.add(("bull", k))
                        blocks.append(Block("bull", float(h[k]), float(lo[k]), k, t))
                    break
        if -body >= impulse_atr * a and c[t] < min(lo[k] for k in window):
            for k in window:
                if c[k] > o[k]:
                    if ("bear", k) not in used:
                        used.add(("bear", k))
                        blocks.append(Block("bear", float(h[k]), float(lo[k]), k, t))
                    break

    for b in blocks:
        for k in range(b.created_idx + 1, end + 1):
            if h[k] >= b.bottom and lo[k] <= b.top:
                b.mitigated_idx = k
                break
    return blocks


def lookback_start(index, days):
    """Index of the first bar of the last `days` trading days in `index`."""
    dates = [ts.date() for ts in index]
    unique = sorted(set(dates))
    if len(unique) <= days:
        return 0
    cutoff = unique[-days]
    return next(i for i, d in enumerate(dates) if d >= cutoff)


# =============================================================================
# STOP, TARGET AND SIZE
# =============================================================================
def select_levels(side, entry, atr, gaps, blocks, tick=0.05, min_stop_atr=None, min_rr=None):
    """Return (levels dict, "") or (None, why the trade is skipped).

    `gaps` and `blocks` must already be limited to the lookback; their fill and
    mitigation state is used as given.
    """
    min_stop_atr = MIN_STOP_ATR if min_stop_atr is None else min_stop_atr
    min_rr = MIN_RR if min_rr is None else min_rr
    if not (np.isfinite(atr) and atr > 0):
        return None, "ATR is not available yet"
    min_risk = min_stop_atr * atr
    long = side == "BUY"
    protect, opposite = ("bull", "bear") if long else ("bear", "bull")

    stops = []
    for g in gaps:
        if g.kind == protect and g.filled_idx < 0:
            if long and g.top < entry:
                stops.append((g.bottom, f"bullish FVG {g.bottom:.2f}-{g.top:.2f}"))
            elif not long and g.bottom > entry:
                stops.append((g.top, f"bearish FVG {g.bottom:.2f}-{g.top:.2f}"))
    for b in blocks:
        if b.kind == protect and b.mitigated_idx < 0:
            if long and b.top < entry:
                stops.append((b.bottom, f"bullish OB {b.bottom:.2f}-{b.top:.2f}"))
            elif not long and b.bottom > entry:
                stops.append((b.top, f"bearish OB {b.bottom:.2f}-{b.top:.2f}"))
    if not stops:
        kind = "bullish" if long else "bearish"
        return None, f"no unfilled {kind} FVG or order block on the stop side of {entry:.2f}"
    stops.sort(key=lambda s: -s[0] if long else s[0])  # nearest first

    stop = stop_src = None
    for level, src in stops:
        level = round_tick(level, tick)
        if abs(entry - level) >= min_risk and (level < entry if long else level > entry):
            stop, stop_src = level, src
            break
    if stop is None:
        return None, (
            f"every stop candidate is closer than {min_stop_atr} ATR ({min_risk:.2f}) to {entry:.2f}"
        )
    risk = abs(entry - stop)

    targets = []
    for g in gaps:
        if g.kind == opposite and g.mid_hit_idx < 0:
            if long and g.mid > entry:
                targets.append((g.mid, f"bearish FVG {g.bottom:.2f}-{g.top:.2f}"))
            elif not long and g.mid < entry:
                targets.append((g.mid, f"bullish FVG {g.bottom:.2f}-{g.top:.2f}"))
    if not targets:
        kind = "bearish" if long else "bullish"
        side_word = "above" if long else "below"
        return None, f"no {kind} FVG with an unreached midpoint {side_word} {entry:.2f}"
    targets.sort(key=lambda s: s[0] if long else -s[0])  # nearest first

    for level, src in targets:
        level = round_tick(level, tick)
        reward = (level - entry) if long else (entry - level)
        if reward > 0 and reward >= min_rr * risk:
            return {
                "stop": stop,
                "target": level,
                "risk": round(risk, 4),
                "rr": round(reward / risk, 2),
                "stop_source": stop_src,
                "target_source": src,
            }, ""
    return None, f"no FVG target pays {min_rr}R on a stop {risk:.2f} away (stop at {stop})"


def size_position(entry, stop, lot=1, capital=None, max_loss=None, buffer_pct=None):
    """Return (qty, which cap bound it). qty 0 means the trade is skipped."""
    capital = CAPITAL if capital is None else capital
    max_loss = MAX_LOSS_PER_TRADE if max_loss is None else max_loss
    buffer_pct = SIZING_BUFFER_PCT if buffer_pct is None else buffer_pct
    risk = abs(entry - stop)
    if not (entry > 0 and risk > 0):
        return 0, "invalid prices"
    pad = entry * buffer_pct / 100.0
    by_risk = math.floor(max_loss / (risk + pad))
    by_capital = math.floor(capital / (entry + pad))
    qty = min(by_risk, by_capital)
    lot = max(1, int(lot))
    qty = (qty // lot) * lot
    return max(qty, 0), ("risk" if by_risk <= by_capital else "capital")


@dataclass
class Plan:
    symbol: str
    side: str
    entry: float
    stop: float
    target: float
    qty: int
    risk: float
    rr: float
    atr: float
    stop_source: str
    target_source: str
    sized_by: str
    signal_bar: str


def plan_trade(symbol, side, bars, entry, tick=0.05, lot=1):
    """Plan a trade from closed bars (the last row is the signal bar).

    Returns (Plan, "") or (None, why it is skipped).
    """
    o, h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = atr_rma(h, lo, c, RISK_ATR_PERIOD)
    last = len(c) - 1
    a = float(atr[last]) if last >= 0 else float("nan")
    if not np.isfinite(a):
        return None, "not enough history for ATR"
    start = lookback_start(bars.index, LEVEL_LOOKBACK_DAYS)
    gaps = [g for g in detect_fvgs(h, lo, atr) if g.created_idx >= start]
    blocks = []
    if USE_ORDER_BLOCKS:
        blocks = [b for b in detect_order_blocks(o, h, lo, c, atr) if b.ob_idx >= start]
    levels, why = select_levels(side, entry, a, gaps, blocks, tick)
    if levels is None:
        return None, why
    qty, sized_by = size_position(entry, levels["stop"], lot)
    if qty < 1:
        return None, (
            f"quantity rounds to 0: stop {levels['risk']:.2f} away at {entry:.2f} "
            f"with Rs {MAX_LOSS_PER_TRADE:.0f} risk, Rs {CAPITAL:.0f} capital, lot {lot}"
        )
    return Plan(
        symbol=symbol,
        side=side,
        entry=round(entry, 4),
        stop=levels["stop"],
        target=levels["target"],
        qty=qty,
        risk=levels["risk"],
        rr=levels["rr"],
        atr=round(a, 4),
        stop_source=levels["stop_source"],
        target_source=levels["target_source"],
        sized_by=sized_by,
        signal_bar=f"{bars.index[-1]:%Y-%m-%d %H:%M}",
    ), ""


# =============================================================================
# STRATEGY
# =============================================================================
class SupertrendFvgStrategy:
    def __init__(self, client, symbols=None, clock=now_ist, data_dir=DATA_DIR, pause=None):
        self.client = client
        self.symbols = list(symbols if symbols is not None else SYMBOLS)
        self.clock = clock
        self.pause = HISTORY_PAUSE_SEC if pause is None else pause
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = data_dir / "state.json"
        self.journal_path = data_dir / "journal.jsonl"
        self.meta = {}  # symbol -> (tick, lot)
        self.last_bar_done = None
        self.bar_done = {}  # symbol -> closed bar processed for it this cycle
        self.last_seen_bar = {}  # symbol -> timestamp of the last closed bar evaluated
        self.trend = {}  # symbol -> current Supertrend direction
        self.untracked = set()  # symbols with a position this strategy does not manage
        self.state = self._load_state()

    # ---------- persistence ----------
    def _fresh_day(self, day, previous=None):
        previous = previous or {}
        return {
            "date": day,
            "trades": 0,
            "pnl": 0.0,
            "open": previous.get("open"),  # verified against the broker in reconcile()
            "pending": previous.get("pending"),
        }

    def _load_state(self):
        day = self.clock().date().isoformat()
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return self._fresh_day(day)
        if state.get("date") != day:
            return self._fresh_day(day, state)
        for key, value in self._fresh_day(day).items():
            state.setdefault(key, value)
        return state

    def _save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)

    def journal(self, event, **data):
        row = {"ts": self.clock().isoformat(timespec="seconds"), "event": event, **data}
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def notify(self, text):
        log(text)
        if TELEGRAM_USER:
            self.call("telegram", username=TELEGRAM_USER, message=f"{STRATEGY_NAME}: {text}")

    # ---------- broker helpers ----------
    def call(self, method, quiet=False, **kwargs):
        try:
            resp = getattr(self.client, method)(**kwargs)
        except Exception as e:
            if not quiet:
                log(f"{method} failed: {e}")
            return None
        if isinstance(resp, dict) and resp.get("status") == "error":
            if not quiet:
                log(f"{method} error: {resp.get('message')}")
            return None
        return resp

    def tick_lot(self, symbol):
        return self.meta.get(symbol, (0.05, 1))

    def ltp(self, symbol):
        resp = self.call("quotes", symbol=symbol, exchange=EXCHANGE)
        try:
            return float(resp["data"]["ltp"])
        except (TypeError, KeyError, ValueError):
            return None

    def position_qty(self, symbol):
        resp = self.call(
            "openposition",
            strategy=STRATEGY_NAME,
            symbol=symbol,
            exchange=EXCHANGE,
            product=PRODUCT,
        )
        try:
            return int(float(resp["quantity"]))
        except (TypeError, KeyError, ValueError):
            return None

    def order_status(self, order_id):
        resp = self.call("orderstatus", order_id=order_id, strategy=STRATEGY_NAME)
        try:
            data = resp["data"]
            return str(data.get("order_status", "")).lower(), float(data.get("average_price") or 0)
        except (TypeError, KeyError, ValueError, AttributeError):
            return None, 0.0

    def place(self, symbol, action, qty, price_type="MARKET", price=0, trigger_price=0):
        resp = self.call(
            "placeorder",
            strategy=STRATEGY_NAME,
            symbol=symbol,
            action=action,
            exchange=EXCHANGE,
            price_type=price_type,
            product=PRODUCT,
            quantity=qty,
            price=price,
            trigger_price=trigger_price,
        )
        return (resp or {}).get("orderid")

    def wait_final(self, order_id, timeout=30):
        deadline = time.monotonic() + timeout
        status, avg = None, 0.0
        while time.monotonic() < deadline:
            status, avg = self.order_status(order_id)
            if status in ("complete", "rejected", "cancelled"):
                break
            time.sleep(2)
        return status, avg

    # ---------- startup ----------
    def setup(self):
        valid = []
        for symbol in self.symbols:
            try:
                info = self.client.symbol(symbol=symbol, exchange=EXCHANGE)
            except Exception:
                info = None
            data = info.get("data") if isinstance(info, dict) else None
            if not (isinstance(info, dict) and info.get("status") == "success" and data):
                log(f"{symbol} is not in the {EXCHANGE} symbol list; it is skipped")
                continue
            try:
                tick = float(data.get("tick_size") or 0.05)
                lot = int(float(data.get("lotsize") or 1))
            except (TypeError, ValueError):
                tick, lot = 0.05, 1
            self.meta[symbol] = (tick, max(lot, 1))
            valid.append(symbol)
        self.symbols = valid
        mode = self.call("analyzerstatus")
        mode_name = ((mode or {}).get("data") or {}).get("mode", "unknown")
        log(f"{STRATEGY_NAME}: {len(valid)} symbols on {EXCHANGE} {PRODUCT}: {', '.join(valid)}")
        log(
            f"Supertrend {ST_PERIOD}/{ST_MULTIPLIER} {ST_SOURCE} changeATR={ST_CHANGE_ATR}, "
            f"FVG {FVG_MODE}, capital Rs {CAPITAL:.0f}, max loss Rs {MAX_LOSS_PER_TRADE:.0f}/trade"
        )
        log(f"OpenAlgo mode: {mode_name}  (analyze = sandbox orders, live = real orders)")
        self.reconcile()

    def reconcile(self):
        """Make the saved state agree with the broker's positions."""
        trade = self.state.get("open")
        pending = self.state.get("pending")
        if pending:
            self.manage_pending(self.clock(), force=True)
            trade = self.state.get("open")
        if trade:
            qty = self.position_qty(trade["symbol"])
            if qty == 0:
                self.notify(
                    f"saved {trade['symbol']} trade is no longer open at the broker; "
                    "clearing it (P&L not recorded)"
                )
                self.state["open"] = None
            elif qty is not None:
                trade["qty"] = abs(qty)
                log(
                    f"resuming {trade['side']} {abs(qty)} {trade['symbol']} @ {trade['entry_price']}"
                )
        for symbol in self.symbols:
            if trade and symbol == trade["symbol"]:
                continue
            qty = self.position_qty(symbol)
            if qty:
                self.untracked.add(symbol)
                log(
                    f"{symbol}: a position of {qty} exists that this strategy is not managing; "
                    "no new entries until it is flat"
                )
        self._save_state()

    # ---------- market data ----------
    def load_bars(self, symbol, now):
        start = (now - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
        bars = self.call(
            "history",
            quiet=True,
            symbol=symbol,
            exchange=EXCHANGE,
            interval=INTERVAL,
            start_date=start,
            end_date=now.strftime("%Y-%m-%d"),
        )
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            return None
        bars = self._naive(bars)
        forming_start = now.replace(second=0, microsecond=0) - timedelta(
            minutes=now.minute % BAR_MINUTES
        )
        return bars[bars.index < forming_start]

    @staticmethod
    def _naive(df):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        return df.sort_index()[["open", "high", "low", "close"]].astype(float)

    # ---------- main cycle ----------
    def step(self):
        now = self.clock()
        if self.state["date"] != now.date().isoformat():
            self.state = self._fresh_day(now.date().isoformat(), self.state)
            self._save_state()
        if self.state.get("pending"):
            self.manage_pending(now)
        if self.state.get("open"):
            self.manage_open(now)

        closed_bar = now.replace(second=0, microsecond=0) - timedelta(
            minutes=now.minute % BAR_MINUTES + BAR_MINUTES
        )
        if closed_bar != self.last_bar_done and now >= closed_bar + timedelta(
            minutes=BAR_MINUTES, seconds=5
        ):
            if self.on_bar(now, closed_bar):
                self.last_bar_done = closed_bar

    def can_enter(self, now):
        s = self.state
        if s.get("open"):
            return False, f"a {s['open']['symbol']} trade is already open"
        if s.get("pending"):
            return False, f"a {s['pending']['plan']['symbol']} entry order is pending"
        secs = now.hour * 3600 + now.minute * 60 + now.second
        if not hhmm(ENTRY_START) * 60 <= secs <= hhmm(LAST_ENTRY) * 60:
            return False, f"outside the entry window {ENTRY_START}-{LAST_ENTRY}"
        if s["trades"] >= MAX_TRADES_PER_DAY:
            return False, f"{MAX_TRADES_PER_DAY} trades already taken today"
        if s["pnl"] <= -MAX_DAILY_LOSS:
            return False, f"daily loss limit Rs {MAX_DAILY_LOSS:.0f} reached"
        if self.untracked:
            return False, f"untracked positions in {', '.join(sorted(self.untracked))}"
        return True, ""

    def on_bar(self, now, closed_bar):
        """Process one closed bar for every symbol. Returns False to retry the
        symbols whose bar has not arrived yet on the next poll."""
        grace_over = now >= closed_bar + timedelta(minutes=BAR_MINUTES, seconds=60)
        signals, missing, waiting = [], [], False
        for n, symbol in enumerate(self.symbols):
            if self.bar_done.get(symbol) == closed_bar:
                continue
            if n:
                time.sleep(self.pause)
            bars = self.load_bars(symbol, now)
            if bars is None or len(bars) < max(ST_PERIOD, RISK_ATR_PERIOD) + 5:
                if grace_over:
                    missing.append(symbol)
                    self.bar_done[symbol] = closed_bar
                else:
                    waiting = True
                continue
            if bars.index[-1] < closed_bar and not grace_over:
                waiting = True  # the broker has not published the bar yet
                continue
            self.bar_done[symbol] = closed_bar
            signal = self.evaluate_symbol(symbol, bars)
            if signal:
                signals.append((symbol, signal, bars))

        self.untracked = {s for s in self.untracked if self.position_qty(s)}
        if signals:
            self.act_on_signals(now, signals)
        if waiting and not grace_over:
            return False
        ups = sum(1 for s in self.symbols if self.trend.get(s) == 1)
        downs = sum(1 for s in self.symbols if self.trend.get(s) == -1)
        line = f"bar {closed_bar:%H:%M} done: trend up {ups} / down {downs}"
        if missing:
            line += f" | no data for {', '.join(missing)}"
        open_trade = self.state.get("open")
        if open_trade:
            line += f" | open {open_trade['side']} {open_trade['symbol']}"
        log(line)
        return True

    def evaluate_symbol(self, symbol, bars):
        """Supertrend on closed bars. Returns "BUY", "SELL" or None.

        The first evaluation after a start only records the trend, and a bar
        already evaluated never signals again.
        """
        last_ts = bars.index[-1]
        seen = self.last_seen_bar.get(symbol)
        st = supertrend(*(bars[k].to_numpy() for k in ("open", "high", "low", "close")))
        self.trend[symbol] = int(st["trend"][-1])
        self.last_seen_bar[symbol] = last_ts
        if seen is None or last_ts <= seen:
            return None
        if st["buy"][-1]:
            return "BUY"
        if st["sell"][-1]:
            return "SELL"
        return None

    def act_on_signals(self, now, signals):
        trade = self.state.get("open")
        ordered = list(signals)
        if trade:
            for item in signals:
                symbol, side, _bars = item
                if symbol != trade["symbol"] or side == trade["side"] or not EXIT_ON_FLIP:
                    continue
                ordered.remove(item)
                self.journal("flip", symbol=symbol, side=side, trade=trade)
                self.notify(f"{symbol}: Supertrend flipped to {side}; exiting the {trade['side']}")
                closed = self.exit_trade("supertrend flip")
                if closed and ENTER_ON_FLIP and self.position_qty(symbol) == 0:
                    ordered.insert(0, item)  # a fresh signal, now that the book is flat
                break

        for symbol, side, bars in ordered:
            ok, why = self.can_enter(now)
            if ok and self.position_qty(symbol):
                ok, why = False, f"a position in {symbol} is already open at the broker"
            if not ok:
                log(f"{symbol} {side} signal ignored: {why}")
                self.journal(
                    "signal_ignored", symbol=symbol, side=side, bar=f"{bars.index[-1]}", why=why
                )
                continue
            tick, lot = self.tick_lot(symbol)
            entry = self.ltp(symbol) or float(bars["close"].iloc[-1])
            plan, why = plan_trade(symbol, side, bars, entry, tick, lot)
            if plan is None:
                log(f"{symbol} {side} signal skipped: {why}")
                self.journal(
                    "skip", symbol=symbol, side=side, bar=f"{bars.index[-1]}", entry=entry, why=why
                )
                continue
            self.journal("signal", plan=asdict(plan))
            log(
                f"{symbol} {side} {plan.qty} @ ~{plan.entry} SL {plan.stop} ({plan.stop_source}) "
                f"TG {plan.target} ({plan.target_source}) RR {plan.rr}, sized by {plan.sized_by}"
            )
            self.enter(plan, now)

    # ---------- execution ----------
    def enter(self, plan, now):
        oid = self.place(plan.symbol, plan.side, plan.qty, "MARKET")
        if not oid:
            self.notify(f"{plan.symbol} {plan.side} entry order was refused; no trade")
            return
        self.state["trades"] += 1
        self.state["pending"] = {
            "order_id": oid,
            "plan": asdict(plan),
            "placed": now.isoformat(timespec="seconds"),
        }
        self._save_state()
        status, avg = self.wait_final(oid)
        if status in ("complete", "rejected", "cancelled"):
            self.state["pending"] = None
            qty = self.position_qty(plan.symbol)
            if qty:
                self.on_filled(plan, abs(qty), avg or self.ltp(plan.symbol) or plan.entry)
            else:
                self._save_state()
                self.notify(f"{plan.symbol} entry order ended '{status}' with no position")
        else:
            log(f"{plan.symbol} entry order is still '{status}'; checking again next cycle")

    def manage_pending(self, now, force=False):
        p = self.state["pending"]
        plan = Plan(**p["plan"])
        status, avg = self.order_status(p["order_id"])
        if status not in ("complete", "rejected", "cancelled"):
            age = (now - datetime.fromisoformat(p["placed"])).total_seconds()
            if not force and age < ENTRY_TIMEOUT_SEC:
                return
            self.call("cancelorder", order_id=p["order_id"], strategy=STRATEGY_NAME)
            status, avg = self.wait_final(p["order_id"], timeout=10)
            if status not in ("complete", "rejected", "cancelled"):
                log(f"{plan.symbol} entry order could not be cancelled ({status}); retrying")
                return
        self.state["pending"] = None
        qty = self.position_qty(plan.symbol)  # covers a fill that raced the cancel
        if qty:
            self.on_filled(plan, abs(qty), avg or self.ltp(plan.symbol) or plan.entry)
        else:
            self._save_state()
            self.notify(f"{plan.symbol} entry order ended '{status}' with no position")

    def stop_order_prices(self, symbol, exit_action, stop, price_type):
        """(price_type, limit price) for a stop order at `stop`."""
        if price_type == "SL-M":
            return "SL-M", 0
        tick, _lot = self.tick_lot(symbol)
        buffer = max(tick, round_tick(stop * 0.003, tick))
        limit = stop - buffer if exit_action == "SELL" else stop + buffer
        return "SL", round_tick(limit, tick)

    def on_filled(self, plan, qty, fill):
        exit_action = "SELL" if plan.side == "BUY" else "BUY"
        trade = {
            "symbol": plan.symbol,
            "side": plan.side,
            "qty": qty,
            "entry_price": fill,
            "stop": plan.stop,
            "target": plan.target,
            "stop_source": plan.stop_source,
            "target_source": plan.target_source,
            "stop_order_id": None,
            "stop_type": "SL-M",
            "exit_retry": None,
            "opened": self.clock().isoformat(timespec="seconds"),
        }
        self.state["open"] = trade
        self._save_state()
        self.journal(
            "entry",
            trade=trade,
            plan=asdict(plan),
            planned_loss=round(qty * abs(fill - plan.stop), 2),
            position_value=round(qty * fill, 2),
        )
        self.notify(
            f"{plan.side} {qty} {plan.symbol} filled @ {fill:.2f}, SL {plan.stop}, TG {plan.target}"
        )

        gapped = fill <= plan.stop if plan.side == "BUY" else fill >= plan.stop
        if gapped:
            self.exit_trade("filled beyond the stop")
            return

        oid = self.place(plan.symbol, exit_action, qty, "SL-M", trigger_price=plan.stop)
        if not oid:
            _, limit = self.stop_order_prices(plan.symbol, exit_action, plan.stop, "SL")
            oid = self.place(
                plan.symbol, exit_action, qty, "SL", price=limit, trigger_price=plan.stop
            )
            trade["stop_type"] = "SL"
        if oid:
            trade["stop_order_id"] = oid
        else:
            self.notify(
                f"{plan.symbol}: the broker refused the stop-loss order; "
                "the stop is now watched by this script only"
            )
        self._save_state()

    def manage_open(self, now):
        trade = self.state["open"]
        if trade.get("stop_order_id"):
            status, avg = self.order_status(trade["stop_order_id"])
            if status == "complete":
                self.close_book(avg or trade["stop"], "stop loss")
                return
            if status in ("rejected", "cancelled"):
                trade["stop_order_id"] = None
                self._save_state()
                self.notify(
                    f"{trade['symbol']}: stop-loss order {status} by the broker; "
                    "the stop is now watched by this script only"
                )

        if trade.get("exit_retry"):
            self.exit_trade(trade["exit_retry"])
            return
        if minutes_of(now) >= hhmm(SQUARE_OFF):
            self.exit_trade("square-off time")
            return
        ltp = self.ltp(trade["symbol"])
        if ltp is None:
            return
        long = trade["side"] == "BUY"
        if (ltp >= trade["target"]) if long else (ltp <= trade["target"]):
            self.exit_trade("target")
        elif not trade.get("stop_order_id") and (
            (ltp <= trade["stop"]) if long else (ltp >= trade["stop"])
        ):
            self.exit_trade("stop loss (script)")

    def exit_trade(self, reason):
        """Close the open trade. Returns True once it is booked as closed."""
        trade = self.state["open"]
        symbol = trade["symbol"]
        if trade.get("stop_order_id"):
            self.call("cancelorder", order_id=trade["stop_order_id"], strategy=STRATEGY_NAME)
            status, avg = self.wait_final(trade["stop_order_id"], timeout=10)
            if status == "complete":  # the stop filled first; nothing left to close
                self.close_book(avg or trade["stop"], "stop loss")
                return True
            if status != "cancelled":
                # Still live at the broker: exiting now as well could reverse the position
                trade["exit_retry"] = reason
                self._save_state()
                log(
                    f"{symbol}: stop-loss order could not be cancelled ({status}); "
                    f"retrying the {reason} exit next cycle"
                )
                return False
            trade["stop_order_id"] = None
            self._save_state()
        qty = self.position_qty(symbol)
        if qty is None:
            trade["exit_retry"] = reason
            self._save_state()
            log(f"{symbol}: could not read the position; retrying the {reason} exit next cycle")
            return False
        if qty == 0:
            self.close_book(self.ltp(symbol) or trade["entry_price"], reason + " (already flat)")
            return True
        exit_action = "SELL" if qty > 0 else "BUY"
        resp = self.call(
            "placesmartorder",
            strategy=STRATEGY_NAME,
            symbol=symbol,
            action=exit_action,
            exchange=EXCHANGE,
            price_type="MARKET",
            product=PRODUCT,
            quantity=abs(qty),
            position_size=0,
        )
        oid = (resp or {}).get("orderid")
        if not oid:
            trade["exit_retry"] = reason
            self._save_state()
            self.notify(f"{symbol}: EXIT ORDER REFUSED ({reason}); position still open, will retry")
            return False
        _status, avg = self.wait_final(oid)
        left = self.position_qty(symbol)
        if left:
            trade["exit_retry"] = reason
            trade["qty"] = abs(left)
            self._save_state()
            log(f"{symbol}: {abs(left)} still open after the exit order; retrying next cycle")
            return False
        self.close_book(avg or self.ltp(symbol) or trade["entry_price"], reason)
        return True

    def close_book(self, exit_price, reason):
        trade = self.state["open"]
        sign = 1 if trade["side"] == "BUY" else -1
        pnl = sign * (exit_price - trade["entry_price"]) * trade["qty"]
        self.state["pnl"] += pnl
        self.state["open"] = None
        self._save_state()
        self.journal("exit", trade=trade, exit_price=exit_price, reason=reason, pnl=round(pnl, 2))
        self.notify(
            f"closed {trade['side']} {trade['qty']} {trade['symbol']} @ {exit_price:.2f} "
            f"({reason}), P&L {pnl:.2f}, day {self.state['pnl']:.2f}"
        )


def main():
    if not API_KEY:
        log("OPENALGO_API_KEY is not set. Start this strategy from the /python page.")
        raise SystemExit(1)
    if FVG_MODE not in FVG_MODES:
        log(f"STFVG_FVG_MODE must be one of {', '.join(FVG_MODES)}; it is {FVG_MODE!r}")
        raise SystemExit(1)
    strategy = SupertrendFvgStrategy(api(api_key=API_KEY, host=HOST, ws_url=WS_URL))
    strategy.setup()
    if not strategy.symbols:
        log("none of the configured symbols could be found; nothing to trade")
        raise SystemExit(1)
    while True:
        try:
            strategy.step()
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"error in cycle: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
