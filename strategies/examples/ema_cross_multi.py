#!/usr/bin/env python
"""
EMA crossover longs with an ATR stop, across many NSE stocks

Credit: the entry rule is as in LuxAlgo's "Universal Signal Backtester" on
TradingView, default 9/21 EMA cross. That script is published under
CC BY-NC-SA 4.0, a non-commercial licence. No code or text from it is used
here: this file is an independent implementation of a standard EMA crossover
with an ATR stop, written for the OpenAlgo /python host.

How it trades
  1. Every closed 5 minute bar, for every symbol in SYMBOLS, EMA(close, 9) and
     EMA(close, 21) are computed the way TradingView's ta.ema computes them
     (see "Indicators" below). The bar still forming is never used.
  2. LONG ONLY, intraday (MIS). A buy signal is a crossover on the last closed
     bar: EMA9 above EMA21 on that bar, and EMA9 at or below EMA21 on the bar
     before. A crossunder never opens a short.
     Optional filter, off by default (ATR_FILTER): the signal is taken only
     when ATR(14) is above the 50 bar simple average of ATR(14) on the signal
     bar, so quiet, choppy stretches are skipped.
  3. Only ONE trade is open across all symbols at any time. While a trade is
     open or an entry order is being confirmed, signals on every other symbol
     are ignored (each one is logged and journalled as signal_ignored).
  4. Stop loss = signal bar close - 1.5 x ATR(14), rounded down to the tick.
     It is placed at the broker as one SL-M order (SL only when the broker
     clearly refuses SL-M). The signal is skipped when the price at entry is
     already at or below that stop, or when entry - stop is less than
     MIN_STOP_ATR x ATR(14) (only possible if the multiplier is lowered). A
     fill at or below the stop is closed at once.
  5. Take profit: no price target. The trade is closed at market when, on a
     closed bar AFTER the entry bar, EMA13 crosses under EMA21 (EMA13 below
     EMA21 on that bar, at or above it on the bar before). The entry bar is
     the signal bar, the bar whose close opened the trade, so a crossunder on
     the bar the entry filled in, or any later one, counts; a crossunder on or
     before the signal bar never does. The opposite 9/21 cross does NOT exit.
     A crossunder after the entry bar that happened while the script was
     stopped closes the trade on restart: that only ever reduces risk.
  6. Square-off at SQUARE_OFF (15:10). An exit cancels the broker stop and
     confirms it cancelled before it closes the position with placesmartorder
     to a position of zero.
  7. Quantity: min(floor(MAX_LOSS_PER_TRADE / (entry - stop)),
     floor(CAPITAL / entry)), rounded down to the lot size, with no leverage.
     Both are padded by SIZING_BUFFER_PCT of the price first, so a little
     slippage cannot push the loss at the stop or the position value over the
     caps. A quantity below one lot skips the trade, with the reason.
  8. Entry window: a signal is judged by the close time of its bar. A bar
     closing between ENTRY_START and LAST_ENTRY (09:30 - 14:45) may trade, so
     the 14:40 - 14:45 bar still trades when it is processed a few seconds
     after 14:45. No entry is ever placed more than ENTRY_GRACE_SEC (2 minutes)
     after LAST_ENTRY, nor before ENTRY_START.
  9. Daily caps: MAX_TRADES_PER_DAY entries (5), and no new entry once the
     day's realised loss reaches MAX_DAILY_LOSS (Rs 4000). The loss counted is
     from fill prices: the entry fill, and the exit, stop or (when the
     position was already flat) the latest closing fill in the trade book,
     falling back to the LTP. Brokerage, taxes and positions not opened by
     this script are not counted.
 10. Restart: the first evaluation of each symbol after a start only records
     where its EMAs are. A crossover from a bar before the restart is never
     traded.

Indicators (TradingView Pine v6 semantics, closed bars only)
  ta.ema(close, n): alpha = 2 / (n + 1). It is na for the first n - 1 bars,
      starts on bar n - 1 with the simple average of the first n closes, then
      ema = alpha x close + (1 - alpha) x ema[1].
  ta.atr(14): true range, with high - low on bar 0 (no previous close), then
      Wilder's average (alpha = 1 / 14) seeded with the simple average of the
      first 14 true ranges.
  ta.crossover(a, b): a > b and a[1] <= b[1]. ta.crossunder(a, b): a < b and
      a[1] >= b[1]. Any comparison with na is false, so no cross is reported
      until both lines and their previous values exist.

Order safety
  Every order call ends in one of three ways: sent (an order id came back),
  refused (a clear error), or unknown (the order may exist). The openalgo SDK
  does not raise on a failed call; it returns an error, read like this:
    unknown   timeout_error; http_error with a status of 500 or more, or with
              no status (the request was cut off); unknown_error; json_error;
              any message saying the order "may have been placed"; an
              exception or an empty reply
    refused   connection_error (nothing was sent); http_error with a 4xx
              status; api_error (the app or the broker said no)
  An unknown order may exist at the broker and may not show in the order book
  straight away, so after one nothing replaces it at once.

  What is guaranteed, as long as the broker answers truthfully:
  - A position that cannot be read is unknown, never flat. Entries on every
    symbol are blocked while an entry is being confirmed, while a trade is
    open, while a symbol's position is unreadable or not managed here, and for
    SWEEP_SEC (300 s) after a trade closed with an order answer still unknown.
  - Entry: a refused entry is dropped. An unknown one blocks every entry while
    the position and the order book are read for ENTRY_CONFIRM_SEC: a fill is
    taken over and given its stop; a working order that matches the planned
    one (BUY, MARKET, the planned quantity, this strategy's tag when the book
    shows one, placed after the attempt) is cancelled. Any other working
    order in that symbol, such as one you placed by hand, is left alone. An
    entry the broker calls complete while the position reads flat keeps
    blocking entries, with an alert, until the position shows it.
  - Stop: SL is sent only after SL-M was clearly refused. After an unknown
    stop answer the order book is read every cycle; a stop that shows up is
    adopted, and a new one is placed only if none has shown after
    STOP_CONFIRM_SEC. Meanwhile the script watches the stop on the LTP. If
    the price reaches it inside that grace period, the position is closed at
    market at once, but only when the order book reads, shows no working
    stop (a stop that shows is adopted and left to work), and the position
    reads open. Any other working sell in that symbol is cancelled and
    confirmed final first. If the book or the position cannot be read, the
    script waits and tries again every cycle. After such a close the sweep
    below runs.
  - A filled stop is booked as closed only after other closing orders in that
    symbol are cancelled and the position reads flat. If it is not flat (the
    stop was smaller than the position, or a second stop filled), what is
    left is closed at market and entries stay blocked until it is. Once a
    bar the managed quantity is compared with the broker position and the
    stop is resized.
  - Exit: one closing order at a time. A working exit is left to fill until
    its backoff (EXIT_RETRY_SEC doubling to EXIT_RETRY_MAX_SEC) ends, then
    cancelled and confirmed final before another is sent; after
    EXIT_ALERT_AFTER attempts you get an alert (log and Telegram). After an
    unknown exit answer, a new closing order is sent only when the order book
    shows no working closing order in that symbol and the position is still
    open after EXIT_CONFIRM_SEC. Any working closing order that shows up in
    the book is cancelled first, and that includes one you placed by hand in
    the same symbol, because the order book is account-wide.
  - Sweep: for SWEEP_SEC after a close with an order answer still unknown,
    every cycle, any sell order that appears in that symbol is cancelled
    (with an alert), and a short position that appears there (a late stop or
    exit filling after the close) is taken over and closed through the same
    exit path, with an alert. For the rest of the day (or until a new trade
    opens in it) that symbol stays watched: a short found there by the
    once-a-bar position scan is taken over and closed the same way. Only one
    position is managed at a time, so a short found while another trade is
    open is reported and blocks entries, and is taken over once that trade
    has closed.
  - Once a bar the position book is read. Any other open position in a
    configured symbol that this strategy is not managing is reported once
    (log and Telegram) and blocks new entries until it is flat. It is never
    traded.

  What is not guaranteed (two faults at once, or a broker that misreports):
  - An unknown order that stays out of the order book for longer than its
    grace period is taken as never placed. If it then appears, two closing
    orders can exist at once. Inside SWEEP_SEC a late sell order is cancelled
    when seen; after it, it is not. A short it opens is still found and
    closed (above), but up to one bar later, at market, and with a second
    round of slippage. The once-a-bar scan only knows the symbols closed
    today with an unknown answer; a short in any other configured symbol is
    reported and blocks entries, but is not closed.
  - The market close inside the stop's grace period can race the lost stop:
    if both fill, the account is short until the sweep sees it, usually one
    cycle (POLL_SECONDS), and closing it costs a second fill.
  - A fast move through the stop costs more than the planned loss whenever
    the stop is watched by this script rather than held at the broker: after
    an unknown answer, after a refusal, or while the book or the position
    cannot be read. An SL-M order also fills at whatever price the market
    gives once triggered.
  - Positions opened by hand in a configured symbol are not closed (outside
    the lost state file case below); a hand-placed sell in a symbol this
    script is closing can be cancelled.

  Lost state file: if state.json cannot be read it is kept aside with a
  timestamp, and every open position at the broker in a configured symbol is
  taken over and closed at market, one after another, after any closing
  order still working in that symbol is cancelled. This includes a position
  you opened by hand in one of those symbols: it cannot be told apart. Their
  P&L is not counted. If state.json reads fine, a position this script did
  not open is left alone and only blocks new entries.

  Symbols: one the app says does not exist is skipped. One that could not be
  checked (OpenAlgo unreachable, broker not logged in) is checked again every
  bar and traded only once it is confirmed.

Setup
  - Edit the CONFIG block below, or override any value with an EMAX_* entry
    in .env, for example EMAX_SYMBOLS = 'SBIN,TCS,LT' or EMAX_CAPITAL = '50000'.
  - Upload at /python, exchange NSE, schedule 09:15 - 15:20, Mon-Fri.
  - Backtest it first with the Backtest button on its card.
  - Then run it in analyzer mode (sandbox orders) before going live. The log
    says which mode is active.
  - Telegram alerts are optional: set EMAX_TELEGRAM_USER to your OpenAlgo
    login name.

Every signal, skip, entry and exit is written to ema_cross_data/journal.jsonl
beside this file, and the open trade is kept in state.json so a restart picks
it up again.
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
STRATEGY_NAME = _env("EMAX_STRATEGY_NAME", "EMA Cross 9-21")
SYMBOLS = _env_list(
    "EMAX_SYMBOLS",
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
EXCHANGE = os.getenv("OPENALGO_STRATEGY_EXCHANGE") or _env("EMAX_EXCHANGE", "NSE")
PRODUCT = _env("EMAX_PRODUCT", "MIS")
INTERVAL = "5m"
BAR_MINUTES = 5

# Entry: EMA fast crosses over EMA slow
FAST_LEN = _env("EMAX_FAST_LEN", 9)
SLOW_LEN = _env("EMAX_SLOW_LEN", 21)
# Take profit: EMA exit-fast crosses under EMA exit-slow
EXIT_FAST_LEN = _env("EMAX_EXIT_FAST_LEN", 13)
EXIT_SLOW_LEN = _env("EMAX_EXIT_SLOW_LEN", 21)

# Stop and optional filter
ATR_PERIOD = _env("EMAX_ATR_PERIOD", 14)
STOP_ATR_MULT = _env("EMAX_STOP_ATR_MULT", 1.5)  # stop = signal close - this x ATR
MIN_STOP_ATR = _env("EMAX_MIN_STOP_ATR", 0.3)  # skip if entry - stop is less than this x ATR
ATR_FILTER = _env("EMAX_ATR_FILTER", False)  # only when ATR > SMA(ATR, ATR_FILTER_LEN)
ATR_FILTER_LEN = _env("EMAX_ATR_FILTER_LEN", 50)

# Sizing (no leverage)
CAPITAL = _env("EMAX_CAPITAL", 30000.0)  # most a position may be worth
MAX_LOSS_PER_TRADE = _env("EMAX_MAX_LOSS_PER_TRADE", 2000.0)  # loss if the stop fills
SIZING_BUFFER_PCT = _env("EMAX_SIZING_BUFFER_PCT", 0.1)  # slippage allowance, % of price

# Daily safety caps
MAX_TRADES_PER_DAY = _env("EMAX_MAX_TRADES_PER_DAY", 5)
MAX_DAILY_LOSS = _env("EMAX_MAX_DAILY_LOSS", 4000.0)  # no new trades after this loss

# Session (IST)
ENTRY_START = _env("EMAX_ENTRY_START", "09:30")
LAST_ENTRY = _env("EMAX_LAST_ENTRY", "14:45")
SQUARE_OFF = _env("EMAX_SQUARE_OFF", "15:10")

# Plumbing
HISTORY_DAYS = 12  # calendar days of 5m bars, so the EMAs and ATR are warmed up
HISTORY_PAUSE_SEC = 0.3  # between history calls, for the broker's rate limit
POLL_SECONDS = 10
ENTRY_TIMEOUT_SEC = 60  # an entry order not final by then is cancelled
ENTRY_GRACE_SEC = 120  # a signal from the LAST_ENTRY bar may still be entered this long after

# Order answers that are not clear (no reply, or "may have been placed"): how long
# the order book and position are watched before anything new is sent.
ENTRY_CONFIRM_SEC = _env("EMAX_ENTRY_CONFIRM_SEC", 60)  # a lost entry answer
ENTRY_CONFIRM_READS = 3  # position reads after a fill the position does not show yet
STOP_CONFIRM_SEC = _env("EMAX_STOP_CONFIRM_SEC", 60)  # a lost stop-loss answer
EXIT_CONFIRM_SEC = _env("EMAX_EXIT_CONFIRM_SEC", 45)  # a lost exit answer
EXIT_RETRY_SEC = 10  # an exit that does not fill is replaced after 10, 20, 40 ... s
EXIT_RETRY_MAX_SEC = 300
EXIT_ALERT_AFTER = 3  # alert after this many exit attempts
SWEEP_SEC = _env("EMAX_SWEEP_SEC", 300)  # after such a close: late orders are cancelled
TELEGRAM_USER = _env("EMAX_TELEGRAM_USER", "")  # your OpenAlgo login name, optional

API_KEY = os.getenv("OPENALGO_API_KEY")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL") or (
    f"ws://{os.getenv('WEBSOCKET_HOST', '127.0.0.1')}:{os.getenv('WEBSOCKET_PORT', '8765')}"
)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path(__file__).resolve().parent / "ema_cross_data"


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


def floor_tick(price, tick):
    """Round down to the tick (a long's stop never moves closer to entry)."""
    return round(math.floor(price / tick + 1e-9) * tick, 4)


def min_bars():
    """Closed bars needed before a symbol is evaluated at all."""
    need = max(SLOW_LEN, FAST_LEN, EXIT_FAST_LEN, EXIT_SLOW_LEN, ATR_PERIOD)
    if ATR_FILTER:
        need = max(need, ATR_PERIOD + ATR_FILTER_LEN)
    return need + 5


# =============================================================================
# INDICATORS (TradingView Pine v6 semantics)
# =============================================================================
def pine_sma(src, length):
    """ta.sma: na until `length` values exist, and na while any value in the window is na."""
    s = pd.Series(np.asarray(src, dtype=float))
    return s.rolling(int(length), min_periods=int(length)).mean().to_numpy()


def _seeded(src, length, alpha):
    """x = na(x[1]) ? sma(src, length) : alpha * src + (1 - alpha) * x[1]"""
    src = np.asarray(src, dtype=float)
    seed = pine_sma(src, length)
    out = np.full(len(src), np.nan)
    prev = np.nan
    for i in range(len(src)):
        prev = seed[i] if np.isnan(prev) else alpha * src[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def pine_ema(src, length):
    """ta.ema(src, length): alpha 2 / (length + 1), seeded with sma(src, length)."""
    return _seeded(src, length, 2.0 / (length + 1))


def pine_rma(src, length):
    """ta.rma(src, length): alpha 1 / length, seeded with sma(src, length)."""
    return _seeded(src, length, 1.0 / length)


def true_range(high, low, close):
    """ta.tr(true): high - low on bar 0, where close[1] is na."""
    high, low, close = (np.asarray(x, dtype=float) for x in (high, low, close))
    n = len(close)
    tr = np.empty(n)
    if n:
        tr[0] = high[0] - low[0]
    if n > 1:
        prev = close[:-1]
        tr[1:] = np.maximum(
            high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev))
        )
    return tr


def pine_atr(high, low, close, length):
    """ta.atr(length) = ta.rma(ta.tr(true), length)."""
    return pine_rma(true_range(high, low, close), length)


def crossover(a, b):
    """ta.crossover: a > b and a[1] <= b[1]; false wherever a value is na."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.zeros(len(a), dtype=bool)
    if len(a) > 1:
        with np.errstate(invalid="ignore"):
            out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def crossunder(a, b):
    """ta.crossunder: a < b and a[1] >= b[1]; false wherever a value is na."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.zeros(len(a), dtype=bool)
    if len(a) > 1:
        with np.errstate(invalid="ignore"):
            out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


def compute_signals(high, low, close, fast=None, slow=None, exit_fast=None, exit_slow=None):
    """Entry and exit crosses on a series of closed bars. Bar i uses bars 0..i only.

    Returns a dict of numpy arrays: fast, slow, atr, entry (crossover, with the
    ATR filter applied when it is on), exit (crossunder of the exit EMAs).
    """
    fast = FAST_LEN if fast is None else fast
    slow = SLOW_LEN if slow is None else slow
    exit_fast = EXIT_FAST_LEN if exit_fast is None else exit_fast
    exit_slow = EXIT_SLOW_LEN if exit_slow is None else exit_slow
    close = np.asarray(close, dtype=float)
    cache = {}

    def ema(n):
        if n not in cache:
            cache[n] = pine_ema(close, n)
        return cache[n]

    atr = pine_atr(high, low, close, ATR_PERIOD)
    entry = crossover(ema(fast), ema(slow))
    if ATR_FILTER:
        with np.errstate(invalid="ignore"):
            entry &= atr > pine_sma(atr, ATR_FILTER_LEN)
    return {
        "fast": ema(fast),
        "slow": ema(slow),
        "atr": atr,
        "entry": entry,
        "exit": crossunder(ema(exit_fast), ema(exit_slow)),
    }


# =============================================================================
# STOP AND SIZE
# =============================================================================
def size_position(entry, stop, lot=1, capital=None, max_loss=None, buffer_pct=None):
    """Return (qty, which cap bound it). qty 0 means the trade is skipped."""
    capital = CAPITAL if capital is None else capital
    max_loss = MAX_LOSS_PER_TRADE if max_loss is None else max_loss
    buffer_pct = SIZING_BUFFER_PCT if buffer_pct is None else buffer_pct
    risk = entry - stop
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
    qty: int
    risk: float
    atr: float
    signal_close: float
    sized_by: str
    signal_bar: str


def plan_trade(symbol, bars, entry, tick=0.05, lot=1):
    """Plan a long from closed bars (the last row is the signal bar).

    Returns (Plan, "") or (None, why it is skipped).
    """
    h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    if not len(c):
        return None, "no bars"
    a = float(pine_atr(h, lo, c, ATR_PERIOD)[-1])
    if not (np.isfinite(a) and a > 0):
        return None, "not enough history for ATR"
    close = float(c[-1])
    stop = floor_tick(close - STOP_ATR_MULT * a, tick)
    if not entry > stop:
        return None, f"the price {entry:.2f} is already at or below the stop {stop:.2f}"
    risk = entry - stop
    if risk < MIN_STOP_ATR * a:
        return None, (
            f"the stop {stop:.2f} is closer than {MIN_STOP_ATR} ATR ({MIN_STOP_ATR * a:.2f}) "
            f"to {entry:.2f}"
        )
    qty, sized_by = size_position(entry, stop, lot)
    if qty < 1:
        return None, (
            f"quantity rounds to 0: stop {risk:.2f} away at {entry:.2f} "
            f"with Rs {MAX_LOSS_PER_TRADE:.0f} risk, Rs {CAPITAL:.0f} capital, lot {lot}"
        )
    return Plan(
        symbol=symbol,
        side="BUY",
        entry=round(entry, 4),
        stop=stop,
        qty=qty,
        risk=round(risk, 4),
        atr=round(a, 4),
        signal_close=round(close, 4),
        sized_by=sized_by,
        signal_bar=f"{bars.index[-1]:%Y-%m-%d %H:%M}",
    ), ""


# =============================================================================
# STRATEGY
# =============================================================================
FINAL = ("complete", "rejected", "cancelled")
# What OpenAlgo answers when an order was sent but the broker's reply never came
# back (utils/httpx_client.py UNANSWERED_ORDER_MESSAGE). The order may exist.
UNANSWERED_HINT = "may have been placed"
SENT, REFUSED, UNKNOWN, NOTHING = "sent", "refused", "unknown", "nothing to do"


def error_is_unknown(resp):
    """Whether an SDK error dict means the order may still have been placed.

    From openalgo/orders.py (SDK 2.0.5), _make_request and _handle_response:
      timeout_error              sent, no reply in time            -> unknown
      http_error, code >= 500    a 502 with OpenAlgo's "may have been
                                 placed", or a 504 from a gateway   -> unknown
      http_error, no code        dropped mid-request, for example
                                 "Server disconnected"              -> unknown
      unknown_error, json_error  a reply that cannot be trusted     -> unknown
      connection_error           could not connect: nothing sent    -> refused
      http_error, code 4xx       the app refused it                 -> refused
      api_error, or no type      the app or broker refused it       -> refused
    A message saying the order "may have been placed" is unknown whatever
    its type.
    """
    if UNANSWERED_HINT in str(resp.get("message") or "").lower():
        return True
    kind = str(resp.get("error_type") or "").lower()
    if kind in ("timeout_error", "unknown_error", "json_error"):
        return True
    if kind == "http_error":
        try:
            return int(resp.get("code")) >= 500
        except (TypeError, ValueError):
            return True  # an HTTP failure with no status: the request may have arrived
    return False


def _parse_ts(text):
    for fmt in ("%d-%b-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(text).strip()[:20], fmt)
        except (TypeError, ValueError):
            continue
    return None


class EmaCrossStrategy:
    def __init__(self, client, symbols=None, clock=now_ist, data_dir=DATA_DIR, pause=None):
        self.client = client
        self.symbols = list(symbols if symbols is not None else SYMBOLS)
        self.unchecked = []  # symbols whose existence could not be checked yet
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
        self.above = {}  # symbol -> EMA fast above EMA slow on the last closed bar
        self.untracked = set()  # symbols with a position not managed here, or unreadable
        self.state_lost = False  # the state file existed but could not be read
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
            "sweep": previous.get("sweep"),
            "watch": {},  # symbol -> closing action, after a close with an unknown answer
        }

    def _load_state(self):
        day = self.clock().date().isoformat()
        try:
            state = json.loads(self.state_path.read_text())
            if not isinstance(state, dict):
                raise ValueError("state is not an object")
        except FileNotFoundError:
            return self._fresh_day(day)
        except (OSError, ValueError):
            self.state_lost = True
            aside = self.state_path.with_name(f"state.unreadable-{self.clock():%Y%m%d-%H%M%S}.json")
            try:
                self.state_path.replace(aside)
                where = f"kept aside as {aside.name}"
            except OSError:
                where = "it could not be moved aside"
            log(
                f"the saved state could not be read ({where}); every position found at the "
                "broker in the configured symbols is closed at market, one at a time"
            )
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

    def _since(self, iso):
        """Seconds from an ISO time to now, on the strategy's clock."""
        try:
            return (self.clock() - datetime.fromisoformat(iso)).total_seconds()
        except (TypeError, ValueError):
            return float("inf")

    # ---------- broker helpers ----------
    def call(self, method, quiet=False, **kwargs):
        """The response, or None when the call failed or answered with an error.

        Read calls only: for orders use send(), which tells a refusal from a
        lost answer.
        """
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

    def send(self, method, **kwargs):
        """Send an order. Returns (outcome, order id, message), outcome one of:

        SENT     the broker took it and gave an order id
        REFUSED  a clear refusal: the order does not exist
        UNKNOWN  no clear answer: the order may exist
        NOTHING  accepted with nothing to trade (a smart order already matched)

        The openalgo SDK (orders.py _make_request / _handle_response) never
        raises; it returns {"status": "error", "error_type": ..., "code": ...}.
        How each is read is in error_is_unknown(). An exception from a client
        is UNKNOWN too.
        """
        try:
            resp = getattr(self.client, method)(**kwargs)
        except Exception as e:
            log(f"{method} got no answer: {e}")
            return UNKNOWN, None, str(e)
        if not isinstance(resp, dict):
            return UNKNOWN, None, "no answer"
        message = str(resp.get("message") or "")
        if resp.get("status") == "error":
            if error_is_unknown(resp):
                log(f"{method} got no clear answer: {message[:200]}")
                return UNKNOWN, None, message
            log(f"{method} refused: {message[:200]}")
            return REFUSED, None, message
        oid = resp.get("orderid")
        if oid:
            return SENT, str(oid), message
        if resp.get("status") == "success":
            return NOTHING, None, message
        return UNKNOWN, None, message

    def tick_lot(self, symbol):
        return self.meta.get(symbol, (0.05, 1))

    def ltp(self, symbol):
        resp = self.call("quotes", symbol=symbol, exchange=EXCHANGE)
        try:
            return float(resp["data"]["ltp"])
        except (TypeError, KeyError, ValueError):
            return None

    def position_qty(self, symbol):
        """Signed net quantity, or None when it could not be read (never "flat")."""
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

    def order_info(self, order_id):
        """(status, average price, quantity) of an order; status None when unreadable."""
        resp = self.call("orderstatus", order_id=order_id, strategy=STRATEGY_NAME)
        try:
            data = resp["data"]
            status = str(data.get("order_status", "")).lower() or None
            avg = float(data.get("average_price") or 0)
            qty = int(float(data.get("quantity") or 0))
            return status, avg, qty
        except (TypeError, KeyError, ValueError, AttributeError):
            return None, 0.0, 0

    def order_status(self, order_id):
        status, avg, _qty = self.order_info(order_id)
        return status, avg

    def open_orders(self, symbol):
        """Orders still working at the broker in `symbol`, or None if the order book
        could not be read. The order book is account-wide."""
        resp = self.call("orderbook")
        if not isinstance(resp, dict):
            return None
        data = resp.get("data")
        orders = data.get("orders") if isinstance(data, dict) else data
        if orders is None:
            orders = []
        if not isinstance(orders, list):
            return None
        working = []
        for o in orders:
            if not isinstance(o, dict):
                continue
            if str(o.get("symbol", "")).upper() != symbol.upper():
                continue
            if str(o.get("exchange") or EXCHANGE).upper() != EXCHANGE.upper():
                continue
            if str(o.get("product") or PRODUCT).upper() != PRODUCT.upper():
                continue
            if str(o.get("order_status", "")).lower() in FINAL:
                continue
            working.append(o)
        return working

    @staticmethod
    def _action(o):
        return str(o.get("action", "")).upper()

    @staticmethod
    def _ptype(o):
        return str(o.get("pricetype") or o.get("price_type") or "").upper()

    def cancel_orders(self, orders):
        """Cancel each order and wait until it is final. True when all of them are."""
        for o in orders:
            oid = o.get("orderid")
            if not oid:
                return False
            self.call("cancelorder", order_id=oid, strategy=STRATEGY_NAME)
            status, _avg = self.wait_final(oid, timeout=10)
            if status not in FINAL:
                log(f"order {oid} could not be cancelled ({status}); trying again next cycle")
                return False
        return True

    def book_fill_price(self, symbol, action):
        """Average price of the latest fill in `symbol` on `action`, from the trade book."""
        resp = self.call("tradebook", quiet=True)
        data = (resp or {}).get("data") if isinstance(resp, dict) else None
        if not isinstance(data, list):
            return None
        for row in reversed(data):
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol", "")).upper() != symbol.upper():
                continue
            if self._action(row) != action:
                continue
            try:
                price = float(row.get("average_price") or 0)
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
        return None

    def place(self, symbol, action, qty, price_type="MARKET", price=0, trigger_price=0):
        """placeorder through send(): (outcome, order id)."""
        outcome, oid, _msg = self.send(
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
        return outcome, oid

    def wait_final(self, order_id, timeout=30):
        deadline = time.monotonic() + timeout
        status, avg = None, 0.0
        while time.monotonic() < deadline:
            status, avg = self.order_status(order_id)
            if status in FINAL:
                break
            time.sleep(2)
        return status, avg

    # ---------- startup ----------
    def setup(self):
        self.unchecked, self.symbols = list(self.symbols), []
        self.check_symbols()
        mode = self.call("analyzerstatus")
        mode_name = ((mode or {}).get("data") or {}).get("mode", "unknown")
        log(
            f"{STRATEGY_NAME}: {len(self.symbols)} symbols on {EXCHANGE} {PRODUCT}: "
            f"{', '.join(self.symbols)}"
        )
        log(
            f"long only: EMA {FAST_LEN}/{SLOW_LEN} cross over to enter, EMA "
            f"{EXIT_FAST_LEN}/{EXIT_SLOW_LEN} cross under to exit, stop {STOP_ATR_MULT} x "
            f"ATR({ATR_PERIOD}), ATR filter {'on' if ATR_FILTER else 'off'}, capital Rs "
            f"{CAPITAL:.0f}, max loss Rs {MAX_LOSS_PER_TRADE:.0f}/trade"
        )
        log(f"OpenAlgo mode: {mode_name}  (analyze = sandbox orders, live = real orders)")
        self.reconcile()

    def check_symbols(self):
        """Confirm symbols against the master contract. A symbol the app says does
        not exist is dropped; one that could not be checked is tried again later."""
        still, added = [], []
        for symbol in self.unchecked:
            try:
                info = self.client.symbol(symbol=symbol, exchange=EXCHANGE)
            except Exception:
                info = None
            data = info.get("data") if isinstance(info, dict) else None
            if (
                isinstance(info, dict)
                and info.get("status") == "success"
                and isinstance(data, dict)
            ):
                try:
                    tick = float(data.get("tick_size") or 0.05)
                    lot = int(float(data.get("lotsize") or 1))
                except (TypeError, ValueError):
                    tick, lot = 0.05, 1
                self.meta[symbol] = (tick, max(lot, 1))
                self.symbols.append(symbol)
                added.append(symbol)
            elif isinstance(info, dict) and "not found" in str(info.get("message", "")).lower():
                log(f"{symbol} is not in the {EXCHANGE} symbol list; it is skipped")
            else:
                still.append(symbol)
        if still:
            log(
                f"could not check {', '.join(still)}: OpenAlgo could not be reached, or the "
                "broker is not logged in. They are checked again every bar and are not "
                "traded until then"
            )
        self.unchecked = still
        return added

    def reconcile(self):
        """Make the saved state agree with the broker's positions."""
        if self.state.get("pending"):
            self.manage_pending(self.clock(), force=True)
        trade = self.state.get("open")
        if trade:
            qty = self.position_qty(trade["symbol"])
            if qty == 0 and not (trade.get("exit_unanswered") or trade.get("stop_unknown_since")):
                self.notify(
                    f"saved {trade['symbol']} trade is no longer open at the broker; "
                    "clearing it (P&L not recorded)"
                )
                self.state["open"] = None
            elif qty is None:
                log(
                    f"could not read the {trade['symbol']} position; still managing the saved trade"
                )
            elif qty:
                trade["qty"] = abs(qty)
                log(
                    f"resuming {trade['side']} {abs(qty)} {trade['symbol']} @ {trade['entry_price']}"
                )
        self.check_positions(self.symbols)
        self._save_state()

    def _busy_symbols(self):
        busy = {(self.state.get("open") or {}).get("symbol")}
        busy.add(((self.state.get("pending") or {}).get("plan") or {}).get("symbol"))
        return busy

    def check_positions(self, symbols):
        """Note positions this strategy is not managing. After a lost state file,
        each one is taken over and closed at market in turn."""
        busy = self._busy_symbols()
        for symbol in symbols:
            if symbol in busy:
                continue
            qty = self.position_qty(symbol)
            if qty == 0:
                continue
            if qty is None:
                self.untracked.add(symbol)
                log(f"{symbol}: could not read the position; no new entries until it can be read")
            elif self.state_lost and not self.state.get("open"):
                self.recover(symbol, qty)
                busy.add(symbol)
            else:
                self.untracked.add(symbol)
                self.notify(
                    f"{symbol}: a position of {qty} exists that this strategy is not managing; "
                    "no new entries until it is flat"
                )

    def scan_positions(self):
        """Once a bar: any open position in a configured symbol that this strategy
        is not managing is reported (log and Telegram) and blocks new entries. A
        short in a symbol closed today with an order answer unknown is taken over
        and closed (handle_unmanaged). One position book read."""
        resp = self.call("positionbook", quiet=True)
        rows = (resp or {}).get("data") if isinstance(resp, dict) else None
        if not isinstance(rows, list):
            return
        busy, mine = self._busy_symbols(), set(self.symbols)
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper()
            if symbol not in mine or symbol in busy:
                continue
            if str(row.get("exchange") or EXCHANGE).upper() != EXCHANGE.upper():
                continue
            if str(row.get("product") or PRODUCT).upper() != PRODUCT.upper():
                continue
            try:
                qty = int(float(row.get("quantity") or 0))
            except (TypeError, ValueError):
                continue
            if qty and (symbol not in self.untracked or self.is_reversal(symbol, qty)):
                self.handle_unmanaged(symbol, qty)
                busy = self._busy_symbols()

    def recover(
        self,
        symbol,
        qty,
        reason="recovered after the saved state was lost",
        message=None,
    ):
        """Take over a position and close it: one left behind by a lost state file,
        or one that a late order opened after a close.

        Its stop cannot be rebuilt, so it is closed: any working order
        in the closing direction (a stop left at the broker) is cancelled first,
        then the position is closed to zero. A position opened by hand in a
        configured symbol cannot be told apart and is closed too.
        """
        self.untracked.discard(symbol)
        self.state["open"] = {
            "symbol": symbol,
            "side": "BUY" if qty > 0 else "SELL",
            "qty": abs(qty),
            "entry_price": self.ltp(symbol) or 0.0,
            "stop": None,
            "target": None,
            "stop_order_id": None,
            "stop_type": None,
            "stop_check": True,  # look for and cancel any stop left at the broker
            "exit_order_id": None,
            "exit_unanswered": False,
            "exit_retry": reason,
            "recovered": True,
            "opened": self.clock().isoformat(timespec="seconds"),
        }
        self._save_state()
        self.journal("recovered", symbol=symbol, qty=qty, reason=reason)
        self.notify(
            message
            or f"{symbol}: a position of {qty} was found after the saved state was lost; "
            "closing it at market (its stop is unknown)"
        )

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
        if self.state.get("sweep"):
            self.run_sweep()

        closed_bar = now.replace(second=0, microsecond=0) - timedelta(
            minutes=now.minute % BAR_MINUTES + BAR_MINUTES
        )
        if closed_bar != self.last_bar_done and now >= closed_bar + timedelta(
            minutes=BAR_MINUTES, seconds=5
        ):
            if self.on_bar(now, closed_bar):
                self.last_bar_done = closed_bar

    def can_enter(self, now, bar_close=None):
        """Whether a new entry may be placed now for a signal on the bar closing at
        `bar_close` (default: judge the signal by `now`)."""
        s = self.state
        if s.get("open"):
            return False, f"a {s['open']['symbol']} trade is already open"
        if s.get("pending"):
            return False, f"a {s['pending']['plan']['symbol']} entry is being confirmed"
        if s.get("sweep"):
            return False, f"an order answer in {s['sweep']['symbol']} is still being checked"
        signal_at = bar_close or now
        signal_secs = signal_at.hour * 3600 + signal_at.minute * 60 + signal_at.second
        now_secs = now.hour * 3600 + now.minute * 60 + now.second
        start, last = hhmm(ENTRY_START) * 60, hhmm(LAST_ENTRY) * 60
        if not (start <= signal_secs <= last and start <= now_secs <= last + ENTRY_GRACE_SEC):
            return False, f"outside the entry window {ENTRY_START}-{LAST_ENTRY}"
        if s["trades"] >= MAX_TRADES_PER_DAY:
            return False, f"{MAX_TRADES_PER_DAY} trades already taken today"
        if s["pnl"] <= -MAX_DAILY_LOSS:
            return False, f"daily loss limit Rs {MAX_DAILY_LOSS:.0f} reached"
        if self.untracked:
            return False, f"positions not managed here in {', '.join(sorted(self.untracked))}"
        return True, ""

    def on_bar(self, now, closed_bar):
        """Process one closed bar for every symbol. Returns False to retry the
        symbols whose bar has not arrived yet on the next poll."""
        if self.unchecked:
            added = self.check_symbols()
            if added:
                log(f"now trading {', '.join(added)} as well")
                self.check_positions(added)
                self._save_state()
        grace_over = now >= closed_bar + timedelta(minutes=BAR_MINUTES, seconds=60)
        signals, missing, waiting, take_profit = [], [], False, False
        for n, symbol in enumerate(self.symbols):
            if self.bar_done.get(symbol) == closed_bar:
                continue
            if n:
                time.sleep(self.pause)
            try:
                bars = self.load_bars(symbol, now)
                if bars is None or len(bars) < min_bars():
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
                trade = self.state.get("open")
                if trade and trade["symbol"] == symbol and self.take_profit_due(trade, bars):
                    take_profit = True
            except Exception as e:
                self.bar_done[symbol] = closed_bar
                log(f"{symbol}: its 5 minute bars could not be read this bar ({e}); skipped")
                continue
            if signal:
                signals.append((symbol, signal, bars))

        self.untracked = {s for s in self.untracked if self.position_qty(s) != 0}
        self.scan_positions()
        if take_profit:
            self.take_profit()
        if signals:
            self.act_on_signals(now, signals)
        if self.state.get("open"):
            self.sync_open_qty()
        if waiting and not grace_over:
            return False
        ups = sum(1 for s in self.symbols if self.above.get(s) is True)
        downs = sum(1 for s in self.symbols if self.above.get(s) is False)
        line = (
            f"bar {closed_bar:%H:%M} done: EMA {FAST_LEN} above EMA {SLOW_LEN} in {ups}, "
            f"below in {downs}"
        )
        if missing:
            line += f" | no data for {', '.join(missing)}"
        open_trade = self.state.get("open")
        if open_trade:
            line += f" | open {open_trade['side']} {open_trade['symbol']}"
        log(line)
        return True

    def evaluate_symbol(self, symbol, bars):
        """EMA crossover on closed bars. Returns "BUY" or None (long only).

        The first evaluation after a start only records where the EMAs are, and
        a bar already evaluated never signals again.
        """
        last_ts = bars.index[-1]
        seen = self.last_seen_bar.get(symbol)
        h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
        sig = compute_signals(h, lo, c)
        if np.isfinite(sig["fast"][-1]) and np.isfinite(sig["slow"][-1]):
            self.above[symbol] = bool(sig["fast"][-1] > sig["slow"][-1])
        self.last_seen_bar[symbol] = last_ts
        if seen is None or last_ts <= seen:
            return None
        return "BUY" if sig["entry"][-1] else None

    def take_profit_due(self, trade, bars):
        """Whether EMA exit-fast has crossed under EMA exit-slow on a closed bar
        after the trade's entry bar (its signal bar). A cross on or before that
        bar never counts."""
        entry_bar = trade.get("entry_bar")
        if not entry_bar or trade.get("exit_retry"):
            return False
        h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
        crosses = compute_signals(h, lo, c)["exit"]
        after = np.asarray(bars.index > pd.Timestamp(entry_bar))
        return bool((crosses & after).any())

    def take_profit(self):
        trade = self.state.get("open")
        if not trade or trade.get("exit_retry"):
            return
        reason = f"take profit: EMA {EXIT_FAST_LEN} crossed under EMA {EXIT_SLOW_LEN}"
        self.journal("take_profit", symbol=trade["symbol"], trade=dict(trade))
        log(f"{trade['symbol']}: {reason}; exiting the long")
        self.exit_trade(reason)

    def act_on_signals(self, now, signals):
        for symbol, side, bars in signals:
            bar_close = bars.index[-1].to_pydatetime() + timedelta(minutes=BAR_MINUTES)
            ok, why = self.can_enter(now, bar_close)
            if ok:
                qty = self.position_qty(symbol)
                if qty is None:
                    ok, why = False, f"the {symbol} position could not be read"
                elif qty:
                    ok, why = False, f"a position in {symbol} is already open at the broker"
            if not ok:
                log(f"{symbol} {side} signal ignored: {why}")
                self.journal(
                    "signal_ignored", symbol=symbol, side=side, bar=f"{bars.index[-1]}", why=why
                )
                continue
            tick, lot = self.tick_lot(symbol)
            entry = self.ltp(symbol) or float(bars["close"].iloc[-1])
            plan, why = plan_trade(symbol, bars, entry, tick, lot)
            if plan is None:
                log(f"{symbol} {side} signal skipped: {why}")
                self.journal(
                    "skip", symbol=symbol, side=side, bar=f"{bars.index[-1]}", entry=entry, why=why
                )
                continue
            self.journal("signal", plan=asdict(plan))
            log(
                f"{symbol} BUY {plan.qty} @ ~{plan.entry} SL {plan.stop} "
                f"(close {plan.signal_close} - {STOP_ATR_MULT} x ATR {plan.atr}), "
                f"sized by {plan.sized_by}"
            )
            self.enter(plan, now)

    # ---------- entry ----------
    def enter(self, plan, now):
        """Place the entry and keep it as `pending` until the broker's answer is
        confirmed. Every entry is blocked on every symbol while it is pending."""
        self.state["pending"] = {
            "order_id": None,
            "plan": asdict(plan),
            "placed": now.isoformat(timespec="seconds"),
        }
        self._save_state()  # saved before sending, so a crash mid-send is resolved on restart
        outcome, oid = self.place(plan.symbol, plan.side, plan.qty, "MARKET")
        if outcome == REFUSED:
            self.state["pending"] = None
            self._save_state()
            self.notify(f"{plan.symbol} {plan.side} entry order was refused; no trade")
            return
        if outcome == SENT:
            self.state["pending"]["order_id"] = oid
            self._save_state()
            self.wait_final(oid)
        else:
            # The order may have reached the broker even though no answer came back.
            log(
                f"{plan.symbol} entry order got no clear answer; checking the position "
                "and the order book before anything else"
            )
            time.sleep(3)
        self.manage_pending(now)

    def _ours(self, o, plan, placed):
        """Whether a working order looks like this strategy's planned entry."""
        if self._action(o) != plan.side or self._ptype(o) not in ("MARKET", ""):
            return False
        try:
            if int(float(o.get("quantity") or 0)) != int(plan.qty):
                return False
        except (TypeError, ValueError):
            return False
        tag = o.get("strategy")
        if tag and str(tag) != STRATEGY_NAME:
            return False
        ts = _parse_ts(o.get("timestamp") or o.get("order_timestamp") or "")
        if ts is not None and ts < placed - timedelta(seconds=5):
            return False
        return True

    def manage_pending(self, now, force=False):
        p = self.state["pending"]
        plan = Plan(**p["plan"])
        oid = p.get("order_id")
        placed = datetime.fromisoformat(p["placed"])
        status, avg, filled = None, 0.0, 0
        if oid:
            status, avg, filled = self.order_info(oid)
            if status is None:
                log(f"{plan.symbol} entry order status could not be read; checking again")
                return
            if status not in FINAL:
                if not force and (now - placed).total_seconds() < ENTRY_TIMEOUT_SEC:
                    return
                self.call("cancelorder", order_id=oid, strategy=STRATEGY_NAME)
                status, avg = self.wait_final(oid, timeout=10)
                if status not in FINAL:
                    log(f"{plan.symbol} entry order could not be cancelled ({status}); retrying")
                    return
        qty = self.position_qty(plan.symbol)
        if qty is None:
            if status == "complete":
                # Filled in full by the broker's word; the position check follows each bar.
                done = filled or plan.qty
                self.adopt_entry(plan, done if plan.side == "BUY" else -done, avg)
            else:
                log(f"{plan.symbol} position could not be read; the entry is checked again")
            return
        if qty:
            self.adopt_entry(plan, qty, avg)
            return

        # The position reads flat.
        if status == "complete":
            # Inconsistent: the broker says it filled. Read again before anything.
            for _ in range(ENTRY_CONFIRM_READS - 1):
                time.sleep(2)
                qty = self.position_qty(plan.symbol)
                if qty:
                    self.adopt_entry(plan, qty, avg)
                    return
            if placed.date() < now.date():
                self.state["pending"] = None
                self._save_state()
                log(f"{plan.symbol}: yesterday's unconfirmed entry is cleared (MIS is squared off)")
                return
            if not p.get("alerted"):
                p["alerted"] = True
                self._save_state()
                self.notify(
                    f"{plan.symbol}: the broker reports the entry order filled but the position "
                    "reads flat. No new entries until this resolves; check the position book"
                )
            return
        if not oid:
            working = self.open_orders(plan.symbol)
            if working is None:
                log(f"{plan.symbol}: the order book could not be read; the entry is checked again")
                return
            ours = [o for o in working if self._ours(o, plan, placed)]
            if ours:
                self.cancel_orders(ours)
                return  # checked again next cycle, a fill that raced the cancel included
            if (now - placed).total_seconds() < ENTRY_CONFIRM_SEC:
                if not p.get("waiting"):
                    p["waiting"] = True
                    self._save_state()
                    others = [o for o in working if self._action(o) == plan.side]
                    note = " (a working order there that this strategy did not place is left alone)"
                    log(
                        f"{plan.symbol}: no sign of the entry yet; checking again for "
                        f"{ENTRY_CONFIRM_SEC}s before treating it as not placed"
                        + (note if others else "")
                    )
                return
        self.state["pending"] = None
        self._save_state()
        self.notify(f"{plan.symbol} {plan.side} entry was not filled; no trade")
        self.journal("entry_failed", symbol=plan.symbol, side=plan.side, status=status)

    def adopt_entry(self, plan, qty, avg):
        self.state["pending"] = None
        if (qty > 0) != (plan.side == "BUY"):
            self.untracked.add(plan.symbol)
            self._save_state()
            self.notify(
                f"{plan.symbol}: the position is {qty}, the opposite of the {plan.side} entry; "
                "it is not managed here and no new entries are made until it is flat"
            )
            return
        self.on_filled(plan, abs(qty), avg or self.ltp(plan.symbol) or plan.entry)

    def on_filled(self, plan, qty, fill):
        self.state["trades"] += 1
        trade = {
            "symbol": plan.symbol,
            "side": plan.side,
            "qty": qty,
            "entry_price": fill,
            "stop": plan.stop,
            "target": None,  # no price target: the exit-EMA crossunder takes profit
            "entry_bar": plan.signal_bar,
            "stop_order_id": None,
            "stop_type": None,
            "stop_check": False,
            "exit_order_id": None,
            "exit_unanswered": False,
            "exit_retry": None,
            "opened": self.clock().isoformat(timespec="seconds"),
        }
        self.state["open"] = trade
        (self.state.get("watch") or {}).pop(plan.symbol, None)  # a new trade ends the watch
        self._save_state()
        self.journal(
            "entry",
            trade=trade,
            plan=asdict(plan),
            planned_loss=round(qty * abs(fill - plan.stop), 2),
            position_value=round(qty * fill, 2),
        )
        self.notify(f"{plan.side} {qty} {plan.symbol} filled @ {fill:.2f}, SL {plan.stop}")
        gapped = fill <= plan.stop if plan.side == "BUY" else fill >= plan.stop
        if gapped:
            self.exit_trade("filled beyond the stop")
            return
        self.place_stop(trade)

    # ---------- the stop ----------
    def stop_order_prices(self, symbol, exit_action, stop, price_type):
        """(price_type, limit price) for a stop order at `stop`."""
        if price_type == "SL-M":
            return "SL-M", 0
        tick, _lot = self.tick_lot(symbol)
        buffer = max(tick, round_tick(stop * 0.003, tick))
        limit = stop - buffer if exit_action == "SELL" else stop + buffer
        return "SL", round_tick(limit, tick)

    def open_stops(self, trade):
        working = self.open_orders(trade["symbol"])
        if working is None:
            return None
        exit_action = "SELL" if trade["side"] == "BUY" else "BUY"
        return [
            o
            for o in working
            if self._action(o) == exit_action and self._ptype(o) in ("SL", "SL-M")
        ]

    def adopt_stop(self, trade, found):
        """Keep the first working stop found at the broker and cancel any other."""
        keep = found[0]
        if len(found) > 1:
            self.cancel_orders(found[1:])
        trade["stop_order_id"] = keep.get("orderid")
        trade["stop_type"] = self._ptype(keep) or "SL-M"
        trade["stop_check"] = False
        trade.pop("stop_unknown_since", None)
        self._save_state()
        log(f"{trade['symbol']}: the stop-loss order {trade['stop_order_id']} is working")

    def place_stop(self, trade):
        """Place the broker stop: SL-M, and SL only after a clear refusal of SL-M.

        After an answer that is not clear, nothing more is sent: the script
        watches the stop, and the order book is read every cycle for
        STOP_CONFIRM_SEC before a new stop may be placed.
        """
        exit_action = "SELL" if trade["side"] == "BUY" else "BUY"
        for price_type in ("SL-M", "SL"):
            _, limit = self.stop_order_prices(
                trade["symbol"], exit_action, trade["stop"], price_type
            )
            outcome, oid = self.place(
                trade["symbol"],
                exit_action,
                trade["qty"],
                price_type,
                price=limit,
                trigger_price=trade["stop"],
            )
            if outcome == SENT:
                trade.update(stop_order_id=oid, stop_type=price_type, stop_check=False)
                trade.pop("stop_unknown_since", None)
                self._save_state()
                return
            if outcome != REFUSED:
                trade["stop_check"] = True
                trade["stop_unknown_since"] = self.clock().isoformat(timespec="seconds")
                self._save_state()
                self.notify(
                    f"{trade['symbol']}: no clear answer to the stop-loss order. The stop is "
                    "watched by this script and the order book is checked before anything "
                    "else is sent"
                )
                return
        trade["stop_check"] = False
        self._save_state()
        self.notify(
            f"{trade['symbol']}: the broker refused the stop-loss order; "
            "the stop is now watched by this script only"
        )

    def on_stop_filled(self, avg, filled):
        """The broker stop filled. Book it closed only once the position reads flat;
        anything left over is flattened as a managed residual."""
        trade = self.state["open"]
        symbol = trade["symbol"]
        closing = "SELL" if trade["side"] == "BUY" else "BUY"
        price = avg or trade["stop"]
        uncertain = trade.get("stop_unknown_since") or trade.get("exit_unanswered")
        working = self.open_orders(symbol)
        if working is None and uncertain:
            log(f"{symbol}: the stop filled; reading the order book before booking it")
            return
        others = [
            o
            for o in (working or [])
            if self._action(o) == closing and o.get("orderid") != trade["stop_order_id"]
        ]
        if others and not self.cancel_orders(others):
            return
        qty = self.position_qty(symbol)
        if qty is None:
            self.close_book(price, "stop loss")
            self.untracked.add(symbol)  # blocks entries until it reads flat
            log(f"{symbol}: the position could not be confirmed flat; no new entries until it is")
            return
        if qty == 0:
            self.close_book(price, "stop loss")
            return
        # Something is left: a stop smaller than the position, or a second stop.
        done = filled or trade["qty"]
        sign = 1 if trade["side"] == "BUY" else -1
        pnl = sign * (price - trade["entry_price"]) * done
        self.state["pnl"] += pnl
        self.journal("partial_exit", trade=dict(trade), qty=done, price=price, pnl=round(pnl, 2))
        side = "BUY" if qty > 0 else "SELL"
        if side != trade["side"]:
            trade["entry_price"] = price
        trade.update(
            side=side,
            qty=abs(qty),
            stop=None,
            target=None,
            stop_order_id=None,
            stop_type=None,
            stop_check=True,
            exit_order_id=None,
            exit_unanswered=False,
            exit_retry="a position was left after the stop filled",
            residual=True,
        )
        trade.pop("stop_unknown_since", None)
        self._save_state()
        self.notify(
            f"{symbol}: the stop filled but {qty} is still open; closing it at market. "
            "No new entries until it is flat"
        )

    def sync_open_qty(self):
        """Once a bar: the managed quantity follows the broker position, and the
        stop is resized to cover all of it."""
        trade = self.state["open"]
        if trade.get("exit_retry") or trade.get("residual") or trade.get("stop") is None:
            return
        qty = self.position_qty(trade["symbol"])
        if not qty:
            return  # unreadable, or flat: the stop and exit handling deal with that
        if (qty > 0) != (trade["side"] == "BUY"):
            trade.update(side="BUY" if qty > 0 else "SELL", qty=abs(qty), residual=True)
            trade.update(stop_check=True, exit_retry="the position is the opposite of the trade")
            self._save_state()
            self.notify(f"{trade['symbol']}: the position is {qty}, against the trade; closing it")
            return
        if abs(qty) == trade["qty"] or trade.get("stop_check"):
            return
        log(f"{trade['symbol']}: the position is {abs(qty)}, not {trade['qty']}; resizing the stop")
        if trade.get("stop_order_id"):
            self.call("cancelorder", order_id=trade["stop_order_id"], strategy=STRATEGY_NAME)
            status, _avg = self.wait_final(trade["stop_order_id"], timeout=10)
            if status != "cancelled":
                return  # filled or still working: handled next cycle
            trade["stop_order_id"] = None
        trade["qty"] = abs(qty)
        self._save_state()
        self.place_stop(trade)

    # ---------- the open trade ----------
    def manage_open(self, now):
        trade = self.state["open"]
        if trade.get("stop_order_id"):
            status, avg, filled = self.order_info(trade["stop_order_id"])
            if status == "complete":
                self.on_stop_filled(avg, filled)
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
        if trade.get("stop_check"):
            found = self.open_stops(trade)
            if found:
                self.adopt_stop(trade, found)
            elif (
                found is not None
                and trade.get("stop") is not None
                and self._since(trade.get("stop_unknown_since")) >= STOP_CONFIRM_SEC
            ):
                self.place_stop(trade)  # the earlier one never appeared
        if minutes_of(now) >= hhmm(SQUARE_OFF):
            self.exit_trade("square-off time")
            return
        ltp = self.ltp(trade["symbol"])
        if ltp is None:
            return
        long = trade["side"] == "BUY"
        if trade.get("target") is not None and (
            (ltp >= trade["target"]) if long else (ltp <= trade["target"])
        ):
            self.exit_trade("target")
        elif (
            trade.get("stop") is not None
            and not trade.get("stop_order_id")
            and ((ltp <= trade["stop"]) if long else (ltp >= trade["stop"]))
        ):
            self.exit_trade("stop loss (script)")

    def exit_trade(self, reason):
        """Close the open trade. Returns True once it is booked as closed.

        Never two closing orders at once. An earlier exit order is left to work
        until its backoff ends, then cancelled and confirmed final. After an
        order answer that was not clear, the order book and the position are
        read every cycle and nothing new is sent until the grace period ends
        with the position still open and no closing order working. The broker
        stop is cancelled and confirmed before the position is closed to zero.
        """
        trade = self.state["open"]
        symbol = trade["symbol"]
        exit_action = "SELL" if trade["side"] == "BUY" else "BUY"
        trade["exit_retry"] = reason
        self._save_state()

        # 1. an earlier exit order of this trade
        if trade.get("exit_order_id"):
            oid = trade["exit_order_id"]
            status, avg = self.order_status(oid)
            if status is None:
                log(f"{symbol}: the earlier exit order could not be read; retrying next cycle")
                return False
            if status not in FINAL:
                if self._since(trade.get("next_exit_at")) < 0:
                    return False  # let it work until its backoff ends
                self.call("cancelorder", order_id=oid, strategy=STRATEGY_NAME)
                status, avg = self.wait_final(oid, timeout=10)
                if status not in FINAL:
                    log(f"{symbol}: the earlier exit order is still working; retrying next cycle")
                    return False
            trade["exit_order_id"] = None
            if status == "complete" and avg:
                trade["last_exit_avg"] = avg
            self._save_state()

        # 2. orders whose answer was not clear: find them in the order book
        if trade.get("stop_check") or trade.get("exit_unanswered"):
            working = self.open_orders(symbol)
            if working is None:
                log(
                    f"{symbol}: the order book could not be read, so the {reason} exit waits "
                    "(an order sent earlier may still be working); retrying next cycle"
                )
                return False
            closing = [o for o in working if self._action(o) == exit_action]
            if trade.get("stop_order_id"):
                closing = [o for o in closing if o.get("orderid") != trade["stop_order_id"]]
            if closing:
                if not self.cancel_orders(closing):
                    return False
            else:
                qty = self.position_qty(symbol)
                if qty is None:
                    log(f"{symbol}: could not read the position; retrying the {reason} exit")
                    return False
                if qty == 0:
                    return self._book_flat(trade, exit_action, reason)
                # An unknown STOP answer does not hold the exit back: the book shows
                # no working closing order and the position reads open, so it is
                # closed now, and the sweep after the close cancels a stop that
                # shows up late. An unknown EXIT answer is waited out, because that
                # order is itself the closing order and may simply not show yet.
                wait = EXIT_CONFIRM_SEC - self._since(trade.get("exit_unknown_since"))
                if trade.get("exit_unanswered") and wait > 0:
                    log(
                        f"{symbol}: the earlier exit order's answer was not clear; waiting up "
                        f"to {wait:.0f}s for it to show before sending another"
                    )
                    return False
            trade["stop_check"] = trade["exit_unanswered"] = False
            self._save_state()

        # 3. the broker stop
        if trade.get("stop_order_id"):
            self.call("cancelorder", order_id=trade["stop_order_id"], strategy=STRATEGY_NAME)
            status, avg = self.wait_final(trade["stop_order_id"], timeout=10)
            if status == "complete":  # the stop filled first
                _s, _a, filled = self.order_info(trade["stop_order_id"])
                self.on_stop_filled(avg, filled)
                return self.state["open"] is None
            if status != "cancelled":
                # Still live at the broker: exiting now as well could reverse the position
                log(
                    f"{symbol}: stop-loss order could not be cancelled ({status}); "
                    f"retrying the {reason} exit next cycle"
                )
                return False
            trade["stop_order_id"] = None
            self._save_state()

        # 4. the position
        qty = self.position_qty(symbol)
        if qty is None:
            log(f"{symbol}: could not read the position; retrying the {reason} exit next cycle")
            return False
        if qty == 0:
            return self._book_flat(trade, exit_action, reason)
        if self._since(trade.get("retry_at")) < 0:
            return False  # backing off after a refused exit order
        outcome, oid, message = self.send(
            "placesmartorder",
            strategy=STRATEGY_NAME,
            symbol=symbol,
            action="SELL" if qty > 0 else "BUY",
            exchange=EXCHANGE,
            price_type="MARKET",
            product=PRODUCT,
            quantity=abs(qty),
            position_size=0,
        )
        attempts = int(trade.get("exit_attempts") or 0) + 1
        trade["exit_attempts"] = attempts
        backoff = min(EXIT_RETRY_SEC * 2 ** (attempts - 1), EXIT_RETRY_MAX_SEC)
        until = (self.clock() + timedelta(seconds=backoff)).isoformat(timespec="seconds")
        trade["next_exit_at"] = until  # a working exit order is left alone until then
        if attempts == EXIT_ALERT_AFTER or (attempts > EXIT_ALERT_AFTER and attempts % 5 == 0):
            self.notify(
                f"{symbol}: the {reason} exit has not filled after {attempts} attempts; the "
                "stock may be at its circuit limit. Check the order book and close it by "
                "hand if needed"
            )
        if outcome == UNKNOWN:
            trade["exit_unanswered"] = True
            trade["exit_unknown_since"] = self.clock().isoformat(timespec="seconds")
            self._save_state()
            self.notify(
                f"{symbol}: no clear answer to the exit order ({reason}); the position is still "
                "managed and nothing else is sent until the order book and position show it"
            )
            return False
        if outcome != SENT:
            trade["retry_at"] = until
            self._save_state()
            log(f"{symbol}: the exit order was not placed ({message}); retrying in {backoff}s")
            return False
        trade["exit_order_id"] = oid
        self._save_state()
        _status, avg = self.wait_final(oid)
        left = self.position_qty(symbol)
        if left is None or left != 0:
            what = "could not be read" if left is None else f"is still {left}"
            log(f"{symbol}: after the exit order the position {what}; checking again next cycle")
            return False
        trade["exit_order_id"] = None
        self.close_book(avg or self.ltp(symbol) or trade["entry_price"], reason)
        return True

    def _book_flat(self, trade, exit_action, reason):
        price = (
            trade.pop("last_exit_avg", None)
            or self.book_fill_price(trade["symbol"], exit_action)
            or self.ltp(trade["symbol"])
            or trade["entry_price"]
        )
        self.close_book(price, reason + " (already flat)")
        return True

    def close_book(self, exit_price, reason):
        trade = self.state["open"]
        sign = 1 if trade["side"] == "BUY" else -1
        if trade.get("recovered"):
            pnl = 0.0  # its entry price is not known
        else:
            pnl = sign * (exit_price - trade["entry_price"]) * trade["qty"]
        self.state["pnl"] += pnl
        self.state["open"] = None
        if trade.get("stop_unknown_since") or trade.get("exit_unknown_since"):
            # An order whose answer was not clear may still appear: keep cancelling
            # closing orders in this symbol for a while, and enter nothing meanwhile.
            self.state["sweep"] = {
                "symbol": trade["symbol"],
                "action": "SELL" if trade["side"] == "BUY" else "BUY",
                "until": (self.clock() + timedelta(seconds=SWEEP_SEC)).isoformat(
                    timespec="seconds"
                ),
            }
            watch = self.state.get("watch")
            if not isinstance(watch, dict):
                watch = self.state["watch"] = {}
            watch[trade["symbol"]] = self.state["sweep"]["action"]
        self._save_state()
        self.journal("exit", trade=trade, exit_price=exit_price, reason=reason, pnl=round(pnl, 2))
        note = (
            " (taken over after a lost state file; P&L not counted)"
            if trade.get("recovered")
            else ""
        )
        self.notify(
            f"closed {trade['side']} {trade['qty']} {trade['symbol']} @ {exit_price:.2f} "
            f"({reason}), P&L {pnl:.2f}, day {self.state['pnl']:.2f}{note}"
        )
        if trade.get("recovered") and self.state_lost:
            self.check_positions(sorted(self.untracked))  # the next one, if any

    def run_sweep(self):
        """For SWEEP_SEC after a trade closed with an order answer still unknown:
        cancel closing orders that show up in that symbol, and take over a
        position that a late order opened. Entries are blocked meanwhile."""
        sweep = self.state["sweep"]
        symbol = sweep["symbol"]
        working = self.open_orders(symbol)
        if working is not None:
            stray = [o for o in working if self._action(o) == sweep["action"]]
            if stray:
                self.notify(f"{symbol}: cancelling a closing order that appeared after the close")
                self.cancel_orders(stray)
        qty = self.position_qty(symbol)
        if qty and not self.state.get("open") and not self.state.get("pending"):
            self.handle_unmanaged(symbol, qty)
        if self._since(sweep["until"]) >= 0:
            self.state["sweep"] = None
            self._save_state()

    def is_reversal(self, symbol, qty):
        """A position on the side a late closing order would leave, in a symbol this
        strategy closed today while an order answer was unknown."""
        action = (self.state.get("watch") or {}).get(symbol)
        return bool(action) and (qty < 0 if action == "SELL" else qty > 0)

    def handle_unmanaged(self, symbol, qty):
        """A position this strategy is not managing. One that a late closing order
        reversed is taken over and closed; any other only blocks entries."""
        if self.is_reversal(symbol, qty) and not self.state.get("open"):
            self.recover(
                symbol,
                qty,
                "a late order reversed the position after the close",
                f"{symbol}: a position of {qty} appeared after the trade was closed, most "
                "likely from an order whose answer was lost; closing it at market",
            )
            return
        if symbol not in self.untracked:
            self.untracked.add(symbol)
            self._save_state()
            self.notify(
                f"{symbol}: a position of {qty} is open that this strategy is not managing; "
                "no new entries until it is flat"
            )


def main():
    if not API_KEY:
        log("OPENALGO_API_KEY is not set. Start this strategy from the /python page.")
        raise SystemExit(1)
    if min(FAST_LEN, SLOW_LEN, EXIT_FAST_LEN, EXIT_SLOW_LEN, ATR_PERIOD) < 1:
        log("every EMA length and the ATR period must be 1 or more; check the EMAX_ settings")
        raise SystemExit(1)
    if not STOP_ATR_MULT > 0:
        log("EMAX_STOP_ATR_MULT must be above 0")
        raise SystemExit(1)
    strategy = EmaCrossStrategy(api(api_key=API_KEY, host=HOST, ws_url=WS_URL))
    strategy.setup()
    if not strategy.symbols and not strategy.unchecked:
        log("none of the configured symbols exist on the exchange; nothing to trade")
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
