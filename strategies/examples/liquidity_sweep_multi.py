#!/usr/bin/env python
"""
Liquidity Sweep: stop-hunt reversals at well-touched levels, intraday, many NSE stocks

Research use first. Backtest it with the Backtest button on /python, then run it
in analyzer mode (sandbox orders), and only then live.

The idea: stops collect just beyond a support or resistance level that price has
touched three or more times. Price wicks through the level (the sweep), takes
those stops, closes back inside and reverses. This strategy trades that
reversal, only in the direction of the higher-timeframe trend, intraday (MIS),
long and short, and nothing is held overnight.

The signal rules are the research backtester's (the liquidity_sweep folder in
this repository), copied into this file unchanged in meaning and under the same
function names (find_pivots, LevelBook, pick_signal, wilder_atr, seeded_ema,
resample_ohlcv, build_bias_bars, bias_direction), because a /python strategy runs
as one file in its own process and cannot import that folder. A parity test
(test/test_liquidity_sweep_strategy.py) runs both on the same bars and checks
that they give the same signals.

Timeframe
  LSW_TIMEFRAME is 5m (the default), 15m or 1h. Bars are anchored at the 09:15
  open: 5m and 15m bars fill the session exactly; the 1h bars are 09:15, 10:15
  ... 14:15 plus a short 15:15-15:30 bar. A bar counts as completed only once
  its end (start + the timeframe, 15:30 for the short bar) is at or before the
  current time; the bar still forming is always dropped. The history call stamps
  each intraday bar with its START time.
  Per timeframe, as in the research backtester:
    5m    level lookback 225 bars (3 sessions), bias on 1h bars
    15m   level lookback 75 bars, bias on 1h bars
    1h    level lookback 21 bars, bias on daily bars

The rules (every value is an LSW_ setting, defaults in brackets)
  1. Pivots. A pivot high at bar i is where high[i] is the strict maximum of the
     highs in [i-n, i+n] (no other bar there equals it), n = PIVOT_N (5). A pivot
     low mirrors it. A pivot is confirmed at bar i+n and used only from bar i+n+1.
  2. Levels. Live pivot highs (resistance) and pivot lows (support) are
     clustered: sorted by price, a pivot joins the current cluster while it is
     within LEVEL_TOL (0.0015, 0.15%) of the cluster's lowest member. A level's
     price is the mean of its pivots and its touches are the number of pivots.
     Only pivots from the last lookback bars count (t - i <= lookback). A level
     dies when a bar closes beyond it by more than LEVEL_TOL, or once it has been
     traded (its entry filled); its pivots are removed, so new pivots can later
     form a fresh level.
  3. Signal on bar t. Short when a resistance L with at least MIN_TOUCHES (3)
     touches has high[t] > L and close[t] < L. Long is the mirror at support.
     Several levels: most touches, then the most extreme price. A bar that sweeps
     both sides is skipped. The signal is checked first, then the invalidations
     of bar t are applied.
  4. Bias (USE_BIAS, on). EMA(BIAS_EMA = 50) of the bias bars' closes, seeded
     with the simple mean of the first 50 closes. Shorts only when the last
     fully closed bias bar closed below its EMA, longs only above. A 1h bias bar
     counts from its end; a daily bar from that session's close, so from the
     next session. No trade until the EMA exists.
  5. Stop. Short: max(sweep high x (1 + STOP_BUFFER), sweep high + ATR_MULT x
     ATR); long: min(sweep low x (1 - STOP_BUFFER), sweep low - ATR_MULT x ATR).
     STOP_BUFFER 0.0005, ATR_MULT 0 (off), ATR = Wilder's ATR(ATR_PERIOD = 14)
     seeded with the mean of the first 14 true ranges, up to and including the
     sweep bar. With ATR_MULT above 0, no signal while ATR does not exist yet.
     The stop is rounded to the tick AWAY from the entry (a long's down, a
     short's up), so it is never tighter than the rule.
  6. Target = RR (2.0) x risk, the risk measured from the ACTUAL fill price to
     the stop, rounded to the tick toward the entry. Watched on the last traded
     price.
  7. Time exit: closed at market once MAX_BARS (48) bars of the timeframe have
     passed since entry; the entry bar counts as bar 1, so the exit is due at
     the entry bar's open + 48 x the timeframe. With the 15:15 square-off this
     can bind only on 5m (4 hours).
  8. Entry is a MARKET order at the next bar's open, only if that bar is in the
     same session as the signal bar and opens before LAST_ENTRY (15:00). The
     square-off is at SQUARE_OFF (15:15). So a signal on a session's last bar,
     or on a bar ending at or after 15:00, is never traded.

How the live strategy runs them
  - After each completed bar, the entry window opens ENTRY_OFFSET_SEC (10)
    seconds after the close, so the broker has published the bar, and lasts
    ENTRY_WINDOW_MIN minutes: 2 on 5m, 5 on 15m and 10 on 1h by default. It
    always stays inside the first half of the next bar, so the entry is taken
    close to that bar's open. A window whose entry bar would open at or after
    LAST_ENTRY does not open at all, and none runs past SQUARE_OFF.
  - At the start of each window the bars of every symbol are read once, with
    the pause between calls the broker's rate limit needs. A read that fails, or
    that does not yet show the bar that has just closed, is tried again every
    SCAN_RETRY_SEC (15) until the window ends; a symbol still without that bar
    is journalled as skipped for that bar. Entries wait up to SCAN_GRACE_SEC (30)
    for every symbol, so the ranking is whole.
  - History depth. Enough calendar days of bars for the pivots, the level
    lookback, the ATR and twice the bias EMA's warm-up, so the EMA has settled:
    28 days of 5m or 15m bars (about 16 sessions, about 110 hourly bias bars),
    and on 1h, 17 days of hourly bars plus 147 days of daily bars (about 100
    sessions) read once a day. LSW_HISTORY_DAYS and LSW_DAILY_HISTORY_DAYS
    override them. On 5m and 15m the hourly bias bars are built from the same
    bars (session-anchored, a bar kept only when every bar of its span is
    there), exactly as the research backtester builds them from 5m bars.
  - The level book is rebuilt from that history on every scan, bar by bar, with
    the same code and in the same order as the research engine, so a restart
    gives the same levels. To keep a backtest fast, the book reached on one scan
    is kept in memory and continued on the next one, but only while the bars it
    was built from come back unchanged and no traded level was added inside
    them; anything else rebuilds it from the first bar. Either way the result is
    the same.
  - "Killed once traded" follows THIS strategy's own entries. Every entry that
    fills is kept in state.json as (symbol, side, signal bar, level price,
    touches) and is NOT cleared at the day roll; entries older than the history
    depth plus a week are pruned. During the replay, at that signal bar, the
    level the replay picks there (or, if the bars were corrected since, the
    level on the same side closest to the saved price within LEVEL_TOL) has its
    pivots removed from the next bar on, exactly where the research engine
    removes a traded level. A signal that was not taken (another trade open,
    the price already beyond the stop, a refused order) does not kill its level,
    as in the research engine, where a level dies only when its entry fills.
  - Caveat on the comparison with the research tables. The research backtester
    trades every symbol independently, one position per stock. This strategy
    holds ONE trade across the whole basket (below). So it, and its OpenAlgo
    backtest, take fewer trades than the research tables show, and which trades
    it takes depends on the ranking. Its levels also come from a rolling window
    of history rather than from a year of data, so a level whose life depends on
    bars older than the window, and an EMA seeded inside the window, can differ
    slightly from a research run started months earlier. On the same bars, with
    the same traded levels, the signals are the same.

One trade at a time, and which one
  - Only ONE trade is open across the whole basket. While a trade is open, the
    bars are still read after each close when its stop is confirmed at the
    broker, and each signal found is journalled. Between the symbol reads the
    open trade is checked again whenever POLL_SECONDS have passed, exactly as
    the main cycle checks it (target, time exit, square-off, its stop, an exit
    under way), so reading the basket never leaves it unwatched for longer than
    about one poll plus one symbol's read. The read waits, or stops part-way,
    while an entry is being confirmed, the stop is not confirmed at the broker
    or an exit is under way; the symbols left unread are read once that is
    settled inside the window, or journalled as not scanned and why. A signal
    is dropped when its window ends; if the open trade closes inside the
    window, the best signal still valid is taken, from the whole basket's
    ranking and with the clock read again just before the order is sent. A
    signal is never carried to a later bar.
  - When several stocks signal on the same bar they are ranked by touches (most
    first), then by reward:risk at the reference price (the signal bar's
    close), then by the order of SYMBOLS. With the target at RR x risk the
    reward:risk is RR for every signal, so in practice equal touches fall to the
    SYMBOLS order. The best one still valid on the live price is taken.
  - Live price at entry time: a long is dropped when the price is at or below
    the stop, a short at or above it. A fill at or beyond the stop is closed at
    once.
  - No limit on trades a day (MAX_TRADES_PER_DAY 0) and no daily loss limit
    (MAX_DAILY_LOSS 0); both are available as settings. A symbol is entered at
    most once on one signal bar, even after a restart.

Sizing, and leverage by mode
  quantity = floor(CAPITAL x leverage / entry price), rounded down to the lot
  size, the price padded by SIZING_BUFFER_PCT first.
  The leverage is read with analyzerstatus just before every entry is sized,
  and the mode and leverage are logged and journalled with the trade:
    Live mode (analyze mode off)      LEVERAGE (2.0): CAPITAL 30000 x 2 = Rs 60000
    Backtest                          LEVERAGE (2.0): Rs 60000
    Analyzer mode (sandbox orders)    1x: Rs 30000
    Mode could not be read            1x, the safe side, and logged
  A backtest is recognised by the OPENALGO_BACKTEST=1 the backtest runner sets,
  or by the backtester's own analyzerstatus answer (mode "backtest"). OpenAlgo
  has no MTF product: the 2x is MIS intraday leverage, available only if your
  broker gives at least 2x intraday margin on the stock.
  Loss per trade, Rs 4000 at most (MAX_LOSS_PER_TRADE). The stop PRICE is the
  rule's and is never moved to fit the cap; the QUANTITY is cut instead:
    quantity = floor(4000 / (risk + pad)), risk = |live price - stop|,
    pad = live price x SIZING_BUFFER_PCT / 100,
  rounded down to the lot size, and the smaller of this and the position value
  quantity is used. A signal whose quantity comes to 0 is skipped and
  journalled. What the cap cannot hold: a fill worse than the live price by
  more than the pad, an SL-M stop that fills beyond its trigger, a gap through
  the stop, and a stop watched by this script rather than held at the broker.

Restart
  The signals depend only on completed bars and on the traded levels kept in
  state.json, so after a restart the latest bar's signals come out the same; if
  its window has ended, nothing is entered until the next close. The entries
  sent today (symbol and signal bar), the signals dropped, the entries with no
  clear answer, the trade count, the open trade and the traded levels are kept
  in state.json. The order steps are saved BEFORE they are sent, exactly as
  below.
  If state.json or the journal cannot be written you get one alert, the open
  trade is still managed and its stop still placed, and no new entry is made
  until the state file can be written again.

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

  What is guaranteed, as long as the broker answers truthfully:
  - A position that cannot be read is unknown, never flat. Entries on every
    symbol are blocked while an entry is being confirmed, while a trade is
    open, while a symbol's position is unreadable or not managed here, and for
    SWEEP_SEC (300 s) after a trade closed with an order answer still unknown.
  - Entry: a refused entry is dropped. An unknown one blocks every entry while
    the position and the order book are read for ENTRY_CONFIRM_SEC: a fill is
    taken over and given its stop; a working order that matches the planned
    one (same side, MARKET, the planned quantity, this strategy's tag when the
    book shows one, placed after the attempt) is cancelled. Any other working
    order in that symbol is left alone. An entry the broker calls complete
    while the position reads flat keeps blocking entries, with an alert, until
    the position shows it. Whatever the outcome, that symbol is not entered
    again on that signal bar. An unknown entry that shows neither an order nor
    a position within ENTRY_CONFIRM_SEC is taken as not placed, but that symbol
    is not entered again that day and is read every cycle: an entry order that
    shows up still working is cancelled, and a position that shows up on the
    planned side is taken over and given its stop, with an alert.
  - Stop: one SL-M order at the broker at the stop price, sent straight after
    the fill, before the journal and the alert. SL is sent only after SL-M was
    clearly refused. After an unknown stop answer (or a restart before the
    answer) the order book is read every cycle; a stop that shows up is
    adopted, and a new one is placed only if none has shown after
    STOP_CONFIRM_SEC. Meanwhile the script watches the stop on the LTP and, if
    the price reaches it, closes the position at market once the order book
    shows no working stop and the position reads open.
  - A filled stop is booked as closed only after other closing orders in that
    symbol are cancelled and the position reads flat; anything left is closed
    at market. Every CHECK_MINUTES the managed quantity is compared with the
    broker position and the stop is resized (its price is never moved).
  - Exit: one closing order at a time, a smart order to a position of zero with
    quantity 0, so it cannot open the opposite position. A working exit is left
    to fill until its backoff (EXIT_RETRY_SEC doubling to EXIT_RETRY_MAX_SEC)
    ends, then cancelled and confirmed final before another is sent; after
    EXIT_ALERT_AFTER attempts you get an alert. After an unknown exit answer a
    new closing order is sent only when the order book shows no working
    closing order in that symbol and the position is still open after
    EXIT_CONFIRM_SEC.
  - Sweep: for SWEEP_SEC after a close with an order answer still unknown, any
    closing-side order that appears in that symbol is cancelled, and an
    opposite position that appears there is taken over and closed, with an
    alert. For the rest of the day that symbol stays watched.
  - Every CHECK_MINUTES the position book is read. Any other open position in a
    configured symbol that this strategy is not managing is reported once and
    blocks new entries until it is flat. It is never traded.

  What is not guaranteed (two faults at once, or a broker that misreports):
  - An unknown order that stays out of the order book for longer than its
    grace period is taken as never placed; if it then appears, two closing
    orders can exist at once. An opposite position it opens is still found and
    closed on the next cycle, at market.
  - A fast move through the stop costs more than the planned loss whenever the
    stop is watched by this script rather than held at the broker, and an SL-M
    order fills at whatever price the market gives once triggered.
  - The target is watched on the LTP every POLL_SECONDS and closed at market,
    so it can fill short of the target on a fast reversal. While the bars of
    the basket are being read after a close, the open trade is checked between
    symbol reads once a poll has passed, so a check can come up to one symbol's
    read (a second or two live) later than POLL_SECONDS; the stop is at the
    broker throughout.
  - Bars are expected on the NSE grid that starts at 09:15. A broker whose bars
    start elsewhere never gives the expected signal bar, so every symbol is
    skipped every bar and nothing trades: the journal says so.
  - If state.json is lost, the traded levels go with it, so a level traded in
    the last few sessions can be traded once more.

  Lost state file: if state.json cannot be read it is kept aside with a
  timestamp, every open position at the broker in a configured symbol is taken
  over and closed at market (one you opened by hand in one of those symbols
  included: it cannot be told apart), and NO new entry is made for the rest of
  that day. If state.json reads fine, a position this script did not open is
  left alone and only blocks new entries.

  Symbols: one the app says does not exist is skipped. One that could not be
  checked is checked again every CHECK_MINUTES and traded only once confirmed.

Setup
  - Edit the CONFIG block below, or override any value with an LSW_* entry in
    .env, for example LSW_TIMEFRAME = '15m' or LSW_SYMBOLS = 'SBIN,TCS,LT'.
    Times are written HH:MM. A value that cannot be read is named in the log
    and the strategy does not start.
  - Upload at /python, exchange NSE, schedule 09:15 - 15:20, Mon-Fri.
  - Backtest it first: on the /python page press Backtest on this strategy's
    card, choose the dates, the capital and the costs, and start it. Set
    LSW_TIMEFRAME in .env (5m, 15m or 1h) to choose what is tested; the LSW_*
    values apply to the backtest too. The backtester replays one-minute bars,
    builds the 5m, 15m and 1h bars from them anchored at 09:15, and sizes at 2x.
    It loads 15 days of one-minute bars before the first test day, about 10
    sessions, so on 5m and 15m the hourly EMA(50) exists from the first day but
    has had less time to settle in the first week; on 1h the daily EMA comes
    from about 80 daily bars it loads before the start.
  - Then run it in analyzer mode (sandbox orders, sized at 1x) before going
    live. The log says which mode is active and the leverage each entry used.
  - Telegram alerts are optional: set LSW_TELEGRAM_USER to your OpenAlgo login
    name.

Every scan, candidate, skip, entry and exit is written to
liquidity_sweep_data/journal.jsonl beside this file, and the open trade, the
day's record and the traded levels are kept in state.json there.

Research only, not financial advice.
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openalgo import api

# Settings that could not be read: (name, value, what was expected). Nothing is
# raised while this file is imported; main() reports each one and stops.
BAD_SETTINGS = []


def _env(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    value = value.strip()
    if isinstance(default, bool):
        if value.lower() in ("1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        BAD_SETTINGS.append((name, value, "true or false"))
        return default
    if isinstance(default, (int, float)):
        whole = isinstance(default, int)
        try:
            number = float(value)
            if not math.isfinite(number) or (whole and number != int(number)):
                raise ValueError(value)
        except ValueError:
            BAD_SETTINGS.append((name, value, "a whole number" if whole else "a number"))
            return default
        return int(number) if whole else number
    return value


def _env_time(name, default):
    """A time of day as HH:MM (24 hour), for example 14:30."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    value = value.strip()
    hour, _, minute = value.partition(":")
    if (
        hour.isdigit()
        and minute.isdigit()
        and len(hour) <= 2
        and len(minute) == 2
        and int(hour) < 24
        and int(minute) < 60
    ):
        return f"{int(hour):02d}:{minute}"
    BAD_SETTINGS.append((name, value, "a time written as HH:MM, for example 14:30"))
    return default


def _env_list(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _env_choice(name, default, choices):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    value = value.strip().lower()
    if value in choices:
        return value
    BAD_SETTINGS.append((name, value, f"one of {', '.join(choices)}"))
    return default


# Minutes per bar of each timeframe, and the defaults that depend on it.
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
TIMEFRAMES = ("5m", "15m", "1h")
DEFAULT_WINDOW_MIN = {"5m": 2, "15m": 5, "1h": 10}

# =============================================================================
# CONFIG
# =============================================================================
STRATEGY_NAME = _env("LSW_STRATEGY_NAME", "Liquidity Sweep")  # also the order tag
SYMBOLS = _env_list(
    "LSW_SYMBOLS",
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
EXCHANGE = os.getenv("OPENALGO_STRATEGY_EXCHANGE") or _env("LSW_EXCHANGE", "NSE")
PRODUCT = "MIS"  # intraday on both sides; nothing is held overnight
TIMEFRAME = _env_choice("LSW_TIMEFRAME", "5m", TIMEFRAMES)  # 5m, 15m or 1h
INTERVAL = TIMEFRAME  # the history interval of the signal bars
BAR_MINUTES = TF_MINUTES[TIMEFRAME]
SESSION_OPEN = "09:15"  # bars are anchored here
SESSION_CLOSE = "15:30"  # the last 1h bar, 15:15, is a short one and closes here

# The rules (the research backtester's defaults)
PIVOT_N = _env("LSW_PIVOT_N", 5)
LEVEL_TOL = _env("LSW_LEVEL_TOL", 0.0015)
MIN_TOUCHES = _env("LSW_MIN_TOUCHES", 3)
USE_BIAS = _env("LSW_USE_BIAS", True)
BIAS_EMA = _env("LSW_BIAS_EMA", 50)
STOP_BUFFER = _env("LSW_STOP_BUFFER", 0.0005)
ATR_MULT = _env("LSW_ATR_MULT", 0.0)
ATR_PERIOD = _env("LSW_ATR_PERIOD", 14)
RR = _env("LSW_RR", 2.0)
MAX_BARS = _env("LSW_MAX_BARS", 48)

# Sizing
CAPITAL = _env("LSW_CAPITAL", 30000.0)
# MIS intraday leverage: position value <= CAPITAL x this. Used in Live mode and in
# a backtest only; in analyzer mode, or when the mode cannot be read, it is 1.
LEVERAGE = _env("LSW_LEVERAGE", 2.0)
# Rs at the stop per trade: the quantity is cut so the loss at the stop stays within it.
# 0 = off (then only the position value limits the quantity).
MAX_LOSS_PER_TRADE = _env("LSW_MAX_LOSS_PER_TRADE", 4000.0)
SIZING_BUFFER_PCT = _env("LSW_SIZING_BUFFER_PCT", 0.1)  # slippage allowance, % of price

# Daily safety caps
MAX_TRADES_PER_DAY = _env("LSW_MAX_TRADES_PER_DAY", 0)  # 0 = no limit; one trade open at a time
MAX_DAILY_LOSS = _env("LSW_MAX_DAILY_LOSS", 0.0)  # no new trades after this loss; 0 = off

# Session (IST). After each bar closes, entries are allowed from ENTRY_OFFSET_SEC
# seconds after the close, for ENTRY_WINDOW_MIN minutes. No entry on a bar opening
# at or after LAST_ENTRY; square-off at SQUARE_OFF.
ENTRY_OFFSET_SEC = _env("LSW_ENTRY_OFFSET_SEC", 10)  # time for the broker to publish the bar
ENTRY_WINDOW_MIN = _env("LSW_ENTRY_WINDOW_MIN", 0) or DEFAULT_WINDOW_MIN[TIMEFRAME]
LAST_ENTRY = _env_time("LSW_LAST_ENTRY", "15:00")
SQUARE_OFF = _env_time("LSW_SQUARE_OFF", "15:15")

# Plumbing
HISTORY_DAYS = _env("LSW_HISTORY_DAYS", 0)  # calendar days of signal bars; 0 = worked out
DAILY_HISTORY_DAYS = _env("LSW_DAILY_HISTORY_DAYS", 0)  # daily bars on 1h; 0 = worked out
HISTORY_PAUSE_SEC = 0.3  # between history calls, for the broker's rate limit
POLL_SECONDS = 10
CHECK_MINUTES = 5  # how often positions are checked against the broker
SCAN_RETRY_SEC = 15  # a symbol whose bars could not be read, or are not out yet
SCAN_GRACE_SEC = 30  # entries wait this long for every symbol's bars, so the ranking is whole
PRICE_READS = 3  # a candidate whose price cannot be read this many times is dropped
ENTRY_TIMEOUT_SEC = 60  # an entry order not final by then is cancelled
TRADED_KEEP_EXTRA_DAYS = 7  # traded levels are kept for the history depth plus this

# Order answers that are not clear (no reply, or "may have been placed"): how long
# the order book and position are watched before anything new is sent.
ENTRY_CONFIRM_SEC = _env("LSW_ENTRY_CONFIRM_SEC", 60)  # a lost entry answer
ENTRY_CONFIRM_READS = 3  # position reads after a fill the position does not show yet
STOP_CONFIRM_SEC = _env("LSW_STOP_CONFIRM_SEC", 60)  # a lost stop-loss answer
EXIT_CONFIRM_SEC = _env("LSW_EXIT_CONFIRM_SEC", 45)  # a lost exit answer
EXIT_RETRY_SEC = 10  # an exit that does not fill is replaced after 10, 20, 40 ... s
EXIT_RETRY_MAX_SEC = 300
EXIT_ALERT_AFTER = 3  # alert after this many exit attempts
SWEEP_SEC = _env("LSW_SWEEP_SEC", 300)  # after such a close: late orders are cancelled
TELEGRAM_USER = _env("LSW_TELEGRAM_USER", "")  # your OpenAlgo login name, optional

API_KEY = os.getenv("OPENALGO_API_KEY")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL") or (
    f"ws://{os.getenv('WEBSOCKET_HOST', '127.0.0.1')}:{os.getenv('WEBSOCKET_PORT', '8765')}"
)

IST = timezone(timedelta(hours=5, minutes=30))
TZ = "Asia/Kolkata"
DATA_DIR = Path(__file__).resolve().parent / "liquidity_sweep_data"


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


def ceil_tick(price, tick):
    """Round up to the tick (a short's stop never moves closer to entry)."""
    return round(math.ceil(price / tick - 1e-9) * tick, 4)


# =============================================================================
# THE SIGNAL, copied from the research backtester (liquidity_sweep/config.py,
# resample.py and strategy.py). Same names, same semantics; the parity test
# compares the two on the same bars. Do not change one without the other.
# =============================================================================
LOOKBACK_SESSIONS = 3  # the level lookback, in sessions


@dataclass(frozen=True)
class Config:
    """The signal parameters, with the research backtester's field names."""

    timeframe: str = "5m"
    pivot_n: int = 5
    level_tol: float = 0.0015
    level_lookback: int = 225
    min_touches: int = 3
    use_bias: bool = True
    bias_tf: str = "1h"
    bias_ema: int = 50
    stop_buffer: float = 0.0005
    atr_mult: float = 0.0
    atr_period: int = 14
    rr: float = 2.0
    max_bars: int = 48
    tz: str = TZ
    session_open: str = "09:15"
    session_close: str = "15:30"
    last_entry: str = "15:00"  # no entry on a bar opening at or after this time
    square_off: str = "15:15"


def hhmm_to_minutes(value: str) -> int:
    """Convert "HH:MM" to minutes after midnight."""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def bars_per_session(tf, cfg=None):
    """Number of session-anchored bars of `tf` in one full session (short last bar counts)."""
    cfg = cfg or Config()
    span = hhmm_to_minutes(cfg.session_close) - hhmm_to_minutes(cfg.session_open)
    minutes = TF_MINUTES[tf]
    return -(-span // minutes)


def config_for_timeframe(tf, **overrides):
    """Config for a timeframe: 3-session lookback and the matching bias timeframe.

    5m -> lookback 225, bias 1h; 15m -> 75, bias 1h; 1h -> 21, bias daily ("D").
    """
    if tf not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe {tf!r}; use 5m, 15m or 1h")
    base = Config()
    cfg = replace(
        base,
        timeframe=tf,
        level_lookback=LOOKBACK_SESSIONS * bars_per_session(tf, base),
        bias_tf="D" if tf == "1h" else "1h",
    )
    return replace(cfg, **overrides) if overrides else cfg


def signal_config():
    """The Config this strategy trades with, from the LSW_ settings."""
    return config_for_timeframe(
        TIMEFRAME,
        pivot_n=PIVOT_N,
        level_tol=LEVEL_TOL,
        min_touches=MIN_TOUCHES,
        use_bias=USE_BIAS,
        bias_ema=BIAS_EMA,
        stop_buffer=STOP_BUFFER,
        atr_mult=ATR_MULT,
        atr_period=ATR_PERIOD,
        rr=RR,
        max_bars=MAX_BARS,
        session_open=SESSION_OPEN,
        session_close=SESSION_CLOSE,
        last_entry=LAST_ENTRY,
        square_off=SQUARE_OFF,
    )


AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def minute_of_day(index):
    """Minutes after midnight (IST wall clock) for each timestamp."""
    return (index.hour * 60 + index.minute).to_numpy()


def session_key(index):
    """An integer per session (the calendar date as a day ordinal)."""
    return (index.normalize().tz_localize(None).asi8 // (86400 * 10**9)).astype(np.int64)


def infer_bar_minutes(df):
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


def resample_ohlcv(df, tf, cfg=None):
    """Resample intraday bars to `tf` ("5m", "15m", "1h" or "D") within each session.

    Labelled by bar start (label="left", closed="left"). A bar is kept only if
    it has every constituent bar its span needs; a daily bar is kept when its
    session has its first and its last bar.
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


def find_pivots(high, low, n):
    """Pivot highs and lows: high[i] is the strict maximum of the highs in
    [i-n, i+n] (no other bar in the window equals it); lows mirror it. Bars
    closer than n to either end can never be pivots. Returns two boolean arrays
    (pivot_high, pivot_low) indexed by the pivot bar i."""
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


def wilder_atr(high, low, close, period):
    """Wilder's ATR, seeded with the simple mean of the first `period` true
    ranges. The first true range is high-low. NaN before `period` true ranges
    exist. Each value uses only bars up to and including its own bar."""
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


def seeded_ema(values, span):
    """EMA with alpha 2/(span+1), seeded with the simple mean of the first `span`
    values. NaN until `span` values exist."""
    out = np.full(len(values), np.nan)
    if len(values) < span:
        return out
    alpha = 2.0 / (span + 1.0)
    out[span - 1] = values[:span].mean()
    for i in range(span, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def build_bias_bars(base, cfg):
    """Bias bars (1h or daily) from intraday bars: close, ema and available_at
    (the time from which the bar may be used). A 1h bar is available from its
    end, capped at the session close; a daily bar from that session's close."""
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


def daily_bias_bars(daily, cfg):
    """Bias bars from the broker's DAILY bars (the 1h timeframe): the same frame
    build_bias_bars makes for "D", labelled 09:15 of the session and available
    from that session's close. `daily` is indexed by the session date (naive or
    zone-aware); only completed sessions should be passed."""
    index = pd.DatetimeIndex(daily.index)
    index = index.tz_localize(cfg.tz) if index.tz is None else index.tz_convert(cfg.tz)
    day = index.normalize()
    out = pd.DataFrame(index=day + pd.Timedelta(minutes=hhmm_to_minutes(cfg.session_open)))
    out["close"] = daily["close"].to_numpy(dtype=float)
    out["ema"] = seeded_ema(daily["close"].to_numpy(dtype=float), cfg.bias_ema)
    out["available_at"] = day + pd.Timedelta(minutes=hhmm_to_minutes(cfg.session_close))
    return out


def bias_direction(bar_index, bias):
    """Bias per trading bar: +1 longs allowed, -1 shorts allowed, 0 none. Uses the
    last bias bar whose available_at is at or before the trading bar's start, so
    the bias bar still in progress is never used. 0 until the EMA exists."""
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
    members: tuple  # pivot bar indices


class LevelBook:
    """Live pivots per side, clustered into levels on demand. A pivot is
    identified by its bar index. Killed pivots never come back."""

    def __init__(self, cfg):
        self.tol = cfg.level_tol
        self.lookback = cfg.level_lookback
        self.live = {"R": {}, "S": {}}

    def add(self, kind, idx, price):
        self.live[kind][idx] = float(price)

    def prune(self, t):
        """Keep only pivots from the last `lookback` bars before bar t (t - i <= lookback)."""
        for side in self.live.values():
            for idx in [i for i in side if t - i > self.lookback]:
                del side[idx]

    def kill(self, kind, members):
        side = self.live[kind]
        for idx in members:
            side.pop(idx, None)

    def levels(self, kind):
        """Cluster live pivots of one side: sorted by price, a pivot joins the
        current cluster while it is within level_tol of the cluster's lowest
        member. Level price is the mean."""
        items = sorted(self.live[kind].items(), key=lambda kv: (kv[1], kv[0]))
        out = []
        cluster = []
        for idx, price in items:
            if cluster and price > cluster[0][1] * (1 + self.tol):
                out.append(self._make(kind, cluster))
                cluster = []
            cluster.append((idx, price))
        if cluster:
            out.append(self._make(kind, cluster))
        return out

    @staticmethod
    def _make(kind, cluster):
        prices = [p for _, p in cluster]
        return Level(
            kind=kind,
            price=float(np.mean(prices)),
            touches=len(cluster),
            members=tuple(sorted(i for i, _ in cluster)),
        )

    def copy(self):
        other = LevelBook.__new__(LevelBook)
        other.tol, other.lookback = self.tol, self.lookback
        other.live = {k: dict(v) for k, v in self.live.items()}
        return other


def pick_signal(high, low, close, resistance, support, min_touches):
    """Sweep signal on one bar, before the bias filter. Returns (side, level):
    side -1 short, +1 long, 0 none. Most touches wins, then the most extreme
    price. A bar that sweeps both sides gives no signal."""
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


# =============================================================================
# THE REPLAY: the research engine's signal path, bar by bar, on one symbol
# =============================================================================
@dataclass
class Signal:
    """A sweep signal on bar t that passed the bias and the ATR checks."""

    t: int
    time: str  # the signal bar's start, "YYYY-MM-DD HH:MM" IST
    side: int  # -1 short, +1 long
    kind: str  # "R" or "S"
    level: float
    touches: int
    members: tuple
    stop: float  # the rule's stop, before tick rounding
    high: float
    low: float
    close: float
    atr: float
    tradable_next: object  # by the bars: the next bar is in the session and opens before
    # last_entry (the research engine's rule); None on the last bar, which the clock decides


@dataclass
class Replay:
    """The level book after bars [0, done) of one frame, kept so the next scan
    can carry on from it. Only used when the frame comes back with those bars
    unchanged and the same traded levels inside them; otherwise rebuilt."""

    cfg: Config
    first: int = 0  # the first bar's start, ns
    done: int = 0
    stamps: np.ndarray = None
    ohlc: np.ndarray = None
    book: LevelBook = None
    kill_next: tuple = None
    traded: frozenset = frozenset()
    last: dict = field(default_factory=dict)  # what the last bar of the latest call showed


def traded_map(entries):
    """{signal bar "YYYY-MM-DD HH:MM": (kind, level price)} from traded-level rows."""
    out = {}
    for row in entries or []:
        try:
            kind = "R" if str(row["side"]).upper() == "SELL" else "S"
            out[str(row["signal_bar"])] = (kind, float(row["level"]))
        except (TypeError, KeyError, ValueError):
            continue
    return out


def traded_members(mark, raw_side, level, resistance, support, tol):
    """The pivots of the level traded on this bar: the level the replay picks
    here when it is the saved one, else the same-side level closest to the
    saved price within tol (the bars were corrected since), else none."""
    kind, price = mark
    if (
        level is not None
        and level.kind == kind
        and math.isclose(level.price, price, rel_tol=1e-9, abs_tol=1e-6)
    ):
        return kind, level.members
    near = [
        lv
        for lv in (resistance if kind == "R" else support)
        if abs(lv.price - price) <= price * tol
    ]
    if not near:
        return None
    best = min(near, key=lambda lv: abs(lv.price - price))
    return kind, best.members


def replay(bars, cfg, bias=None, traded=None, state=None):
    """Replay the research engine's level book and signal check over `bars`.

    The order on bar t is the research engine's (liquidity_sweep/backtest.py):
    the pivots confirmed at bar t-1 are added and pivots older than the lookback
    dropped; a level traded on bar t-1 has its pivots removed; the levels are
    clustered; the sweep signal is checked (then the bias and the ATR); and the
    levels bar t closed beyond are killed. Positions do not change the level
    book in the research engine except through a traded level, so the signal on
    every bar is reported whether or not a trade was open.

    Args:
        bars: completed bars of cfg.timeframe, zone-aware IST start-time index.
        cfg: Config.
        bias: per-bar bias (+1, -1, 0) aligned with bars; None means no bias.
        traded: {signal bar label: (kind, level price)} of the entries that filled.
        state: a Replay from the previous call on the same symbol, or None.

    Returns:
        (signals on the bars replayed in this call, Replay to pass next time).
        The Replay's `last` holds the last bar's raw sweep and bias.
    """
    traded = traded or {}
    idx = bars.index
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    size = len(bars)
    n = cfg.pivot_n
    stamps = idx.as_unit("ns").asi8
    ohlc = np.vstack([o, h, lo, c]) if size else np.empty((4, 0))
    labels = idx.strftime("%Y-%m-%d %H:%M") if size else []
    sess = session_key(idx) if size else np.empty(0, dtype=np.int64)
    start_min = minute_of_day(idx) if size else np.empty(0, dtype=np.int64)
    last_entry = hhmm_to_minutes(cfg.last_entry)
    ph, pl = find_pivots(h, lo, n)
    atr = wilder_atr(h, lo, c, cfg.atr_period)
    if bias is None:
        bias = np.zeros(size, dtype=np.int8)

    start, book, kill_next = 0, LevelBook(cfg), None
    if (
        state is not None
        and state.cfg == cfg
        and 0 < state.done < size
        and state.first == stamps[0]
        and np.array_equal(state.stamps, stamps[: state.done])
        and np.array_equal(state.ohlc, ohlc[:, : state.done])
        and state.traded == _traded_key(traded, labels[state.done - 1])
    ):
        start, book, kill_next = state.done, state.book.copy(), state.kill_next
    out = Replay(cfg=cfg)
    signals = []
    for t in range(start, size):
        if t == size - 1:
            # Kept for the next call, which replays this bar again once it is
            # no longer the last one.
            out.first = int(stamps[0])
            out.done = t
            out.stamps = stamps[:t].copy()
            out.ohlc = ohlc[:, :t].copy()
            out.book = book.copy()
            out.kill_next = kill_next
            out.traded = _traded_key(traded, labels[t - 1]) if t else frozenset()
        # 1. Pivots confirmed at bar t-1 become usable now.
        j = t - 1 - n
        if j >= 0:
            if ph[j]:
                book.add("R", j, h[j])
            if pl[j]:
                book.add("S", j, lo[j])
        book.prune(t)
        # 2. A level traded on the bar before (its entry filled on this bar's open).
        if kill_next is not None:
            book.kill(*kill_next)
            kill_next = None
        resistance = book.levels("R")
        support = book.levels("S")
        # 4. The sweep signal, then the bias and the ATR.
        raw_side, level = pick_signal(h[t], lo[t], c[t], resistance, support, cfg.min_touches)
        side = raw_side
        if side != 0 and cfg.use_bias and bias[t] != side:
            side = 0
        if side != 0 and cfg.atr_mult > 0 and np.isnan(atr[t]):
            side = 0
        if side != 0:
            extra = cfg.atr_mult * atr[t] if cfg.atr_mult > 0 else 0.0
            if side < 0:
                stop = max(h[t] * (1 + cfg.stop_buffer), h[t] + extra)
            else:
                stop = min(lo[t] * (1 - cfg.stop_buffer), lo[t] - extra)
            tradable = None
            if t + 1 < size:
                tradable = bool(sess[t + 1] == sess[t] and start_min[t + 1] < last_entry)
            signals.append(
                Signal(
                    t=t,
                    time=labels[t],
                    side=int(side),
                    kind=level.kind,
                    level=float(level.price),
                    touches=int(level.touches),
                    members=tuple(level.members),
                    stop=float(stop),
                    high=float(h[t]),
                    low=float(lo[t]),
                    close=float(c[t]),
                    atr=float(atr[t]),
                    tradable_next=tradable,
                )
            )
        if t == size - 1:
            out.last = {
                "raw_side": int(raw_side),
                "level": None if level is None else float(level.price),
                "touches": None if level is None else int(level.touches),
                "bias": int(bias[t]),
                "atr": float(atr[t]),
            }
        mark = traded.get(labels[t])
        if mark is not None:
            kill_next = traded_members(mark, raw_side, level, resistance, support, cfg.level_tol)
        # 5. Invalidate levels this bar closed beyond.
        for lv in resistance:
            if c[t] > lv.price * (1 + cfg.level_tol):
                book.kill("R", lv.members)
        for lv in support:
            if c[t] < lv.price * (1 - cfg.level_tol):
                book.kill("S", lv.members)
    if size == 0 and state is not None:
        out = state
    return signals, out


def _traded_key(traded, upto):
    """The traded levels on bars up to and including the label `upto`."""
    return frozenset((k, v) for k, v in traded.items() if k <= upto)


def bias_for(bars, cfg, daily=None):
    """Per-bar bias for the signal bars: from hourly bars built out of the bars
    themselves (5m, 15m), or from completed daily bars (1h). Zeros when the bias
    is off, or when the daily bars are missing on 1h."""
    if not cfg.use_bias:
        return np.zeros(len(bars), dtype=np.int8)
    if cfg.bias_tf == "D":
        if daily is None or daily.empty:
            return np.zeros(len(bars), dtype=np.int8)
        return bias_direction(bars.index, daily_bias_bars(daily, cfg))
    return bias_direction(bars.index, build_bias_bars(bars, cfg))


# =============================================================================
# BARS: completed ones only, and the entry window after each close
# =============================================================================
def history_days(tf=None, cfg=None):
    """Calendar days of signal bars to read: the pivots, the level lookback, the
    ATR and (on 5m and 15m) twice the hourly EMA's warm-up, plus a session, at
    5 sessions a week and 5 days for holidays."""
    tf = tf or TIMEFRAME
    cfg = cfg or config_for_timeframe(
        tf, pivot_n=PIVOT_N, use_bias=USE_BIAS, bias_ema=BIAS_EMA, atr_period=ATR_PERIOD
    )
    if HISTORY_DAYS > 0:
        return HISTORY_DAYS
    bars = 2 * cfg.pivot_n + 1 + cfg.level_lookback + cfg.atr_period
    sessions = math.ceil(bars / bars_per_session(tf, cfg))
    if cfg.use_bias and cfg.bias_tf == "1h":
        sessions = max(sessions, math.ceil(2 * cfg.bias_ema / bars_per_session("1h", cfg)))
    return math.ceil((sessions + 1) * 7 / 5) + 5


def daily_history_days(cfg=None):
    """Calendar days of daily bars for the daily bias (1h): twice the EMA."""
    cfg = cfg or config_for_timeframe("1h", bias_ema=BIAS_EMA)
    if DAILY_HISTORY_DAYS > 0:
        return DAILY_HISTORY_DAYS
    return math.ceil((2 * cfg.bias_ema + 1) * 7 / 5) + 5


def on_day(day, minutes):
    """The IST wall-clock time `minutes` after midnight on `day` (a date)."""
    return datetime(day.year, day.month, day.day) + timedelta(minutes=minutes)


def bar_starts(day):
    """The start of every bar of one session, from 09:15."""
    out, minute = [], hhmm(SESSION_OPEN)
    while minute < hhmm(SESSION_CLOSE):
        out.append(on_day(day, minute))
        minute += BAR_MINUTES
    return out


def bar_end(start):
    """When the bar that started at `start` closes: one timeframe later, or at
    the session close for the short 15:15 bar of the 1h timeframe."""
    close = on_day(start.date(), hhmm(SESSION_CLOSE))
    end = start + timedelta(minutes=BAR_MINUTES)
    return close if start < close < end else end


def entry_window(bar_start):
    """(opens, closes) of the entry window after the bar that started at
    `bar_start` closes, or None when that bar is not traded: the next bar would
    open at or after LAST_ENTRY or the session close. Entries are allowed at
    opens <= now < closes. The window never runs past SQUARE_OFF."""
    day = bar_start.date()
    nxt = bar_end(bar_start)  # the entry bar's open
    if nxt >= on_day(day, hhmm(LAST_ENTRY)) or nxt >= on_day(day, hhmm(SESSION_CLOSE)):
        return None
    opens = nxt + timedelta(seconds=ENTRY_OFFSET_SEC)
    closes = min(opens + timedelta(minutes=ENTRY_WINDOW_MIN), on_day(day, hhmm(SQUARE_OFF)))
    return (opens, closes) if opens < closes else None


def signal_bars(day):
    """Today's bars whose signals can be traded: those with an entry window."""
    return [start for start in bar_starts(day) if entry_window(start)]


def active_bar(now):
    """The start of today's latest bar whose entry window has opened by `now`,
    or None (before the first window opens). Its window may have closed
    already; the bar stays active until the next window opens."""
    best = None
    for start in signal_bars(now.date()):
        if entry_window(start)[0] <= now:
            best = start
        else:
            break
    return best


def time_exit_at(signal_bar):
    """When the time exit is due for an entry on the bar after `signal_bar`
    (a bar start): the entry bar is bar 1, so MAX_BARS bars after its open."""
    return bar_end(signal_bar) + timedelta(minutes=MAX_BARS * BAR_MINUTES)


def _to_ist_naive(index):
    """IST wall-clock times with no zone, from whatever the history call gave.

    A zone-aware stamp is converted (the openalgo SDK gives intraday bars as
    IST-aware bar starts, the backtester the same). A zone-less stamp is read
    as IST when every bar then falls inside the session, else as UTC (an epoch
    turned into a time with no zone).
    """
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        return index.tz_convert(TZ).tz_localize(None)
    if len(index) == 0:
        return index

    def in_session(idx):
        minutes = idx.hour * 60 + idx.minute
        return bool(((minutes >= hhmm(SESSION_OPEN)) & (minutes < hhmm(SESSION_CLOSE))).all())

    if in_session(index):
        return index
    return index + pd.Timedelta(hours=5, minutes=30)


def bar_frame(df):
    """Bars indexed by their START time, IST wall clock with no zone, inside
    the session only (09:15 to before 15:30, Monday to Friday), as the research
    data keeps them. A frame that carries a `timestamp` column instead of a
    time index (epoch seconds, or text) is read from that column."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
        stamps = df["timestamp"]
        if pd.api.types.is_numeric_dtype(stamps):
            index = pd.to_datetime(stamps.to_numpy(), unit="s", utc=True)
        else:
            try:
                index = pd.to_datetime(stamps.to_numpy())
            except (TypeError, ValueError):
                index = pd.to_datetime(stamps.to_numpy(), utc=True)
    elif pd.api.types.is_numeric_dtype(df.index):
        index = pd.to_datetime(np.asarray(df.index), unit="s", utc=True)
    else:
        index = pd.to_datetime(df.index)
    df.index = _to_ist_naive(index)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df.sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")]
    minutes = df.index.hour * 60 + df.index.minute
    inside = (
        (minutes >= hhmm(SESSION_OPEN)) & (minutes < hhmm(SESSION_CLOSE)) & (df.index.dayofweek < 5)
    )
    return df[inside]


def completed_bars(df, now):
    """Bars that have closed by `now`, zone-aware IST: a bar counts only when its
    start + the timeframe (the session close for the short 1h bar) is at or
    before now. The bar still forming is never used."""
    frame = bar_frame(df)
    if frame.empty:
        return frame.tz_localize(TZ)
    index = frame.index
    close = index.normalize() + pd.Timedelta(minutes=hhmm(SESSION_CLOSE))
    end = index + pd.Timedelta(minutes=BAR_MINUTES)
    end = end.where(end <= close, close)
    frame = frame[np.asarray(end <= pd.Timestamp(now))]
    frame.index = frame.index.as_unit("ns")  # the signal code counts in nanoseconds
    return frame.tz_localize(TZ)


def daily_frame(df):
    """Daily bars indexed by their IST trading date.

    A daily bar is stamped at IST midnight or at UTC midnight depending on the
    broker, and the SDK hands the stamp over as a UTC time with no zone. Either
    way the IST calendar date of the stamp is the trading day.
    """
    df = df.copy()
    index = pd.to_datetime(df.index)
    if index.tz is not None:
        index = index.tz_convert(TZ).tz_localize(None)
    else:
        index = index + pd.Timedelta(hours=5, minutes=30)
    df.index = index.normalize()
    df = df.sort_index()[["open", "high", "low", "close"]].astype(float)
    return df[~df.index.duplicated(keep="last")]


def completed_daily(df, today):
    """Daily bars with every bar dated `today` (IST) or later removed: the bar
    for today is still forming during the session and is never used."""
    frame = daily_frame(df)
    return frame[frame.index < pd.Timestamp(today)]


# =============================================================================
# CANDIDATES, SIZE AND PLAN
# =============================================================================
def make_candidate(signal, tick=0.05, rr=None):
    """A candidate dict from the last bar's Signal: the stop rounded to the tick
    away from the entry, and the reward:risk at the reference price (the signal
    bar's close) with the target at rr x risk."""
    rr = RR if rr is None else rr
    side = "SELL" if signal.side < 0 else "BUY"
    if side == "BUY":
        stop = floor_tick(signal.stop, tick)  # away from the entry
        risk = signal.close - stop
    else:
        stop = ceil_tick(signal.stop, tick)
        risk = stop - signal.close
    reward = rr * risk  # the target is rr x risk, so this is rr whenever the risk is real
    rr_ref = round(reward / risk, 6) if risk > 0 else 0.0
    return {
        "side": side,
        "signal_bar": signal.time,
        "ref_entry": round(signal.close, 4),
        "stop": stop,
        "raw_stop": round(signal.stop, 6),
        "level": round(signal.level, 6),
        "kind": signal.kind,
        "touches": signal.touches,
        "rr": rr_ref,
        "sweep_high": signal.high,
        "sweep_low": signal.low,
        "atr": None if not np.isfinite(signal.atr) else round(signal.atr, 6),
    }


def rank_candidates(candidates, order):
    """Most touches first, then the highest reward:risk at the reference price,
    then the order of the symbol list."""
    position = {symbol: n for n, symbol in enumerate(order)}
    return sorted(
        candidates,
        key=lambda c: (
            -int(c.get("touches") or 0),
            -float(c.get("rr") or 0),
            position.get(c["symbol"], len(position)),
        ),
    )


def size_position(entry, stop, lot=1, capital=None, leverage=None, max_loss=None, buffer_pct=None):
    """Return (qty, which cap bound it). qty 0 means the trade is skipped.

    The position value cap is capital x leverage. The loss cap applies only
    when max_loss is above 0.
    """
    capital = CAPITAL if capital is None else capital
    leverage = LEVERAGE if leverage is None else leverage
    max_loss = MAX_LOSS_PER_TRADE if max_loss is None else max_loss
    buffer_pct = SIZING_BUFFER_PCT if buffer_pct is None else buffer_pct
    risk = abs(entry - stop)
    if not (entry > 0 and risk > 0):
        return 0, "invalid prices"
    pad = entry * buffer_pct / 100.0
    qty, sized_by = math.floor(capital * leverage / (entry + pad)), "position value"
    if max_loss > 0:
        by_risk = math.floor(max_loss / (risk + pad))
        if by_risk < qty:
            qty, sized_by = by_risk, "loss cap"
    lot = max(1, int(lot))
    qty = (qty // lot) * lot
    return max(qty, 0), sized_by


def target_from(fill, stop, side, tick=0.05, rr=None):
    """RR x the risk from the fill to the stop, rounded toward the entry."""
    rr = RR if rr is None else rr
    risk = abs(fill - stop)
    if side == "BUY":
        return floor_tick(fill + rr * risk, tick)
    return ceil_tick(fill - rr * risk, tick)


@dataclass
class Plan:
    symbol: str
    side: str
    entry: float  # the live price the order is sized on
    stop: float
    target: float  # at the live price; the trade's target is set again from the fill
    qty: int
    risk: float  # per share, at the live price
    ref_entry: float  # the signal bar's close
    rr: float  # at the reference price
    sized_by: str
    signal_bar: str  # the signal bar's start, "YYYY-MM-DD HH:MM"
    planned_loss: float  # qty x risk
    position_value: float  # qty x entry
    leverage: float = 1.0  # the leverage this entry was sized with
    mode: str = ""  # the trading mode read before sizing: live, analyzer, backtest, unknown
    level: float = 0.0  # the swept level's price
    kind: str = ""  # "R" resistance (a short) or "S" support (a long)
    touches: int = 0


def plan_entry(symbol, cand, price, lot=1, leverage=None, mode="", tick=0.05):
    """Check a candidate against the live price and size it.

    `leverage` is the one chosen for the current trading mode (mode_leverage());
    None means the configured LEVERAGE. Returns (Plan, "") or (None, why).
    """
    leverage = LEVERAGE if leverage is None else float(leverage)
    side, stop = cand["side"], float(cand["stop"])
    if not (price and price > 0):
        return None, "no live price"
    if side == "BUY" and price <= stop:
        return None, f"the price {price:.2f} is already at or below the stop {stop:.2f}"
    if side == "SELL" and price >= stop:
        return None, f"the price {price:.2f} is already at or above the stop {stop:.2f}"
    risk = abs(price - stop)
    qty, sized_by = size_position(price, stop, lot, leverage=leverage)
    if qty < 1:
        limit = f"Rs {CAPITAL * leverage:.0f} position value ({leverage:g}x)"
        if MAX_LOSS_PER_TRADE > 0:
            limit += f", Rs {MAX_LOSS_PER_TRADE:.0f} loss cap"
        return None, (
            f"quantity rounds to 0: price {price:.2f}, stop {risk:.2f} away, {limit}, lot {lot}"
        )
    return Plan(
        symbol=symbol,
        side=side,
        entry=round(float(price), 4),
        stop=stop,
        target=target_from(price, stop, side, tick),
        qty=qty,
        risk=round(risk, 4),
        ref_entry=float(cand["ref_entry"]),
        rr=float(cand["rr"]),
        sized_by=sized_by,
        signal_bar=str(cand.get("signal_bar", "")),
        planned_loss=round(qty * risk, 2),
        position_value=round(qty * price, 2),
        leverage=leverage,
        mode=mode,
        level=float(cand.get("level") or 0.0),
        kind=str(cand.get("kind") or ""),
        touches=int(cand.get("touches") or 0),
    ), ""


LIVE, ANALYZER, BACKTEST, UNKNOWN_MODE = "live", "analyzer", "backtest", "unknown"


def mode_leverage(answer, backtest=None, leverage=None):
    """(mode, leverage) for sizing an entry, from an analyzerstatus answer.

    Live mode (analyze_mode false) and a backtest use the configured LEVERAGE;
    analyzer mode uses 1. A backtest is recognised by OPENALGO_BACKTEST=1,
    which the backtest runner sets, or by the backtester's own answer, mode
    "backtest" (it also says analyze_mode true, so that is checked first). An
    answer that cannot be read sizes at 1, the safe side.
    """
    leverage = LEVERAGE if leverage is None else float(leverage)
    if backtest is None:
        backtest = os.getenv("OPENALGO_BACKTEST") == "1"
    data = answer.get("data") if isinstance(answer, dict) else None
    if not isinstance(data, dict):
        data = {}
    name = str(data.get("mode") or "").strip().lower()
    flag = data.get("analyze_mode")
    if backtest or name == "backtest":
        return BACKTEST, leverage
    if isinstance(answer, dict) and answer.get("status") == "error":
        return UNKNOWN_MODE, 1.0
    if flag is True and name in ("analyze", ""):
        return ANALYZER, 1.0
    if flag is False and name in ("live", ""):
        return LIVE, leverage
    if flag is None and name == "analyze":
        return ANALYZER, 1.0
    return UNKNOWN_MODE, 1.0


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


def watch_actions(value):
    """The closing sides a watch entry names: "BUY", "SELL", or a list of both."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).upper() for v in value if str(v).upper() in ("BUY", "SELL")]


def symbol_specific(reason):
    """Whether a "not scanned" reason is about one symbol rather than the whole bar."""
    return "confirmed only after" in reason


class LiquiditySweepStrategy:
    def __init__(self, client, symbols=None, clock=now_ist, data_dir=DATA_DIR, pause=None):
        self.client = client
        self.symbols = list(symbols if symbols is not None else SYMBOLS)
        self.order = list(self.symbols)  # the configured order: breaks ranking ties
        self.unchecked = []  # symbols whose existence could not be checked yet
        self.clock = clock
        self.pause = HISTORY_PAUSE_SEC if pause is None else pause
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = data_dir / "state.json"
        self.journal_path = data_dir / "journal.jsonl"
        self.meta = {}  # symbol -> (tick, lot)
        self.last_check = None  # the CHECK_MINUTES slot the positions were last checked in
        self.cfg = signal_config()
        self.scan_slot = None  # which scan the signals below belong to
        self.scan_bar = None  # the signal bar that scan expects: a bar start, or None
        self.scan_todo = []  # symbols whose bars are still to be read for it
        self.scan_results = {}  # symbol -> what the scan found
        self.scan_started = None
        self.scan_tried = None
        self.scan_reported = False
        self.scan_failed = set()
        self.scan_stale = {}  # symbol -> its last read, which did not show the signal bar yet
        self.scan_waiting = ""
        self.held_by = ""  # why the signals waited in their window (a trade open), if they did
        self.open_checked = None  # when the open trade was last checked (manage_open)
        self.replays = {}  # symbol -> the level book of its last scan, to carry on from
        self.daily = {}  # symbol -> (date, completed daily bars), the 1h timeframe's bias
        self.price_misses = {}  # symbol -> live price reads that failed
        self.untracked = set()  # symbols with a position not managed here, or unreadable
        self.state_lost = False  # the state file existed but could not be read
        self.unsaved = False  # the state file could not be written on the last try
        self.journal_failed = False
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
            # The signals of the latest bar, best first. Worked out again after every
            # close and after a restart, from completed bars only.
            "candidates": [],
            "signal_bar": None,  # the bar the candidates belong to
            # "SYMBOL YYYY-MM-DD HH:MM" (symbol and signal bar) of every entry sent
            # today: a symbol is entered at most once on one signal bar.
            "taken": [],
            # "SYMBOL YYYY-MM-DD HH:MM" (symbol and signal bar) -> why that signal was
            # dropped. A new signal bar in the same symbol is a new signal.
            "skipped": {},
            # Every entry that filled: symbol, side, signal bar, level price, touches.
            # Kept across days, so the replay kills a traded level where the research
            # engine does; pruned once older than the history depth plus a week.
            "traded_levels": self._keep_traded(previous.get("traded_levels"), day),
            # symbol -> its plan, for an entry whose answer was never confirmed either
            # way. Each counts as one of the day's trades, and the symbol is watched.
            "unknown": {},
            "no_entries": "",  # why no new entry is made for the rest of today, if so
        }

    @staticmethod
    def _keep_traded(rows, day):
        """The traded levels still inside the history depth plus a week of `day`."""
        if not isinstance(rows, list):
            return []
        keep = history_days() + TRADED_KEEP_EXTRA_DAYS  # the replay never reaches further back
        oldest = (datetime.fromisoformat(day) - timedelta(days=keep)).strftime("%Y-%m-%d")
        return [r for r in rows if isinstance(r, dict) and str(r.get("signal_bar", "")) >= oldest]

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
                "broker in the configured symbols is closed at market, one at a time, and no "
                "new entry is made for the rest of today"
            )
            state = self._fresh_day(day)
            # What was traded today is no longer known, so nothing new is entered today.
            state["no_entries"] = "the saved state was lost today, so what was traded is not known"
            return state
        if state.get("date") != day:
            return self._fresh_day(day, state)
        for key, value in self._fresh_day(day).items():
            state.setdefault(key, value)
        return state

    def _save_state(self):
        """Write the state file. Returns False, and says so once, when it cannot be
        written. It never raises: an order being handled must not be cut short by a
        file. No new entry is made while the file cannot be written."""
        try:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, default=str))
            tmp.replace(self.state_path)
        except Exception as e:
            if not self.unsaved:
                self.unsaved = True
                self.notify(
                    f"the state file could not be written ({e}). The open trade is still "
                    "managed, but a restart would not know about it. No new entries until it "
                    f"can be written; check the disk and the {DATA_DIR.name} folder"
                )
            return False
        self.unsaved = False
        return True

    def journal(self, event, **data):
        """Append one row to the journal. It never raises: a journal that cannot be
        written (the file is open elsewhere, the disk is full) must not stop an
        order being handled."""
        try:
            row = {"ts": self.clock().isoformat(timespec="seconds"), "event": event, **data}
            with self.journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception as e:
            if not self.journal_failed:
                self.journal_failed = True
                log(f"the journal could not be written ({e}); trading carries on without it")
            return
        self.journal_failed = False

    def notify(self, text):
        """Log and, when set up, send a Telegram alert. It never raises."""
        try:
            log(text)
            if TELEGRAM_USER:
                self.call("telegram", username=TELEGRAM_USER, message=f"{STRATEGY_NAME}: {text}")
        except Exception:
            pass  # an alert that cannot be sent must not stop an order being handled

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
        mode_name, leverage = mode_leverage(self.call("analyzerstatus"))
        log(
            f"{STRATEGY_NAME}: {len(self.symbols)} symbols on {EXCHANGE} {PRODUCT}: "
            f"{', '.join(self.symbols)}"
        )
        loss_cap = f"Rs {MAX_LOSS_PER_TRADE:.0f}" if MAX_LOSS_PER_TRADE > 0 else "off"
        cfg = self.cfg
        bias = f"{'daily' if cfg.bias_tf == 'D' else '1h'} EMA({cfg.bias_ema})"
        log(
            f"liquidity sweep on {INTERVAL} bars: pivots n={cfg.pivot_n}, levels within "
            f"{cfg.level_tol:g} with at least {cfg.min_touches} touches over the last "
            f"{cfg.level_lookback} bars, bias {bias if cfg.use_bias else 'off'}, stop buffer "
            f"{cfg.stop_buffer:g} and {cfg.atr_mult:g} x ATR({cfg.atr_period}), target "
            f"{cfg.rr:g}R from the fill, time exit after {cfg.max_bars} bars"
        )
        windows = [entry_window(b) for b in signal_bars(self.clock().date())]
        first = f"{windows[0][0]:%H:%M:%S}-{windows[0][1]:%H:%M:%S}" if windows else "none"
        last = f"{windows[-1][0]:%H:%M:%S}-{windows[-1][1]:%H:%M:%S}" if windows else "none"
        log(
            f"intraday {PRODUCT}: an entry window after each close, {len(windows)} a day "
            f"(first {first}, last {last}), no entry on a bar opening at or after "
            f"{LAST_ENTRY}, square-off {SQUARE_OFF}, one trade at a time, "
            f"{MAX_TRADES_PER_DAY or 'no limit of'} trades a day, daily loss limit "
            f"{f'Rs {MAX_DAILY_LOSS:.0f}' if MAX_DAILY_LOSS > 0 else 'off'}; "
            f"position value up to Rs {CAPITAL * LEVERAGE:.0f} (Rs {CAPITAL:.0f} x "
            f"{LEVERAGE:g}) in Live mode and in a backtest, Rs {CAPITAL:.0f} (1x) in analyzer "
            f"mode; loss cap per trade {loss_cap}"
        )
        days = f"{history_days()} days of {INTERVAL} bars"
        if cfg.use_bias and cfg.bias_tf == "D":
            days += f" and {daily_history_days()} days of daily bars"
        log(
            f"each scan reads {days}; {len(self.state.get('traded_levels') or [])} traded "
            "level(s) kept from earlier entries"
        )
        log(
            f"OpenAlgo mode: {mode_name}, so entries are sized at {leverage:g}x for now "
            "(analyzer = sandbox orders, live = real orders). The mode is read again before "
            "every entry"
        )
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
                "broker is not logged in. They are checked again every few minutes and are not "
                "traded until then"
            )
        self.unchecked = still
        return added

    def reconcile(self):
        """Make the saved state agree with the broker's positions."""
        if self.state_lost:
            self.notify(
                "the saved state could not be read. Every position found at the broker in the "
                "configured symbols is closed at market, and no new entry is made for the rest "
                "of today, because what was already traded today is not known"
            )
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
        """Every CHECK_MINUTES: any open position in a configured symbol that this strategy
        is not managing is reported (log and Telegram) and blocks new entries. A
        position on the opposite side in a symbol closed today with an order answer
        unknown is taken over
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
        """Completed signal bars for `symbol` (the forming bar dropped), zone-aware
        IST, or None when they could not be read."""
        start = (now - timedelta(days=history_days())).strftime("%Y-%m-%d")
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
        return completed_bars(bars, now)

    def load_daily(self, symbol, now):
        """Completed daily bars for the daily bias (the 1h timeframe), read once a
        day per symbol (again only when a read fails), or None."""
        today = now.date()
        cached = self.daily.get(symbol)
        if cached is not None and cached[0] == today:
            return cached[1]
        start = (now - timedelta(days=daily_history_days())).strftime("%Y-%m-%d")
        bars = self.call(
            "history",
            quiet=True,
            symbol=symbol,
            exchange=EXCHANGE,
            interval="D",
            start_date=start,
            end_date=now.strftime("%Y-%m-%d"),
        )
        if not isinstance(bars, pd.DataFrame) or bars.empty:
            return None
        frame = completed_daily(bars, today)
        self.daily[symbol] = (today, frame)
        return frame

    def traded_for(self, symbol):
        """{signal bar: (kind, level)} of this strategy's filled entries in `symbol`."""
        rows = [r for r in self.state.get("traded_levels") or [] if r.get("symbol") == symbol]
        return traded_map(rows)

    def read_symbol(self, symbol, now):
        """The bars a scan needs for one symbol: (bars, daily bars or None), or None
        when a read failed."""
        bars = self.load_bars(symbol, now)
        if bars is None:
            return None
        daily = None
        if self.cfg.use_bias and self.cfg.bias_tf == "D":
            time.sleep(self.pause)
            daily = self.load_daily(symbol, now)
            if daily is None:
                return None
        return bars, daily

    # ---------- main cycle ----------
    def step(self):
        now = self.clock()
        today = now.date().isoformat()
        if self.state["date"] != today:
            self.state = self._fresh_day(today, self.state)
            self._save_state()
        bar = active_bar(now)  # the signal bar whose entry window has opened most recently
        scan_key = f"{bar:%Y-%m-%d %H:%M}" if bar else f"{today} none"
        if self.scan_slot != scan_key:
            self.begin_scan(scan_key, bar, now)
        if self.state.get("pending"):
            self.manage_pending(now)
        if self.state.get("open"):
            self.open_checked = now
            self.manage_open(now)
        if self.state.get("sweep"):
            self.run_sweep()
        if self.state.get("unknown"):
            self.watch_unknown()
        if self.state.get("watch"):
            self.check_watch()

        slot = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % CHECK_MINUTES)
        if slot != self.last_check:
            self.last_check = slot
            self.periodic()
        self.scan_signals(now)
        # The time is read again: the reads above (a deferred scan of the whole
        # basket in particular) can take long enough to cross the window's end.
        self.try_enter(self.clock())

    def periodic(self):
        """Every CHECK_MINUTES: symbols not confirmed yet, positions this strategy
        is not managing, and the stop's quantity against the broker position."""
        if self.unchecked:
            added = self.check_symbols()
            if added:
                log(f"now trading {', '.join(added)} as well")
                self.check_positions(added)
                window = entry_window(self.scan_bar) if self.scan_bar is not None else None
                if window and self.clock() < window[1]:  # scanned too, while the window is open
                    self.scan_todo.extend(s for s in added if s not in self.scan_todo)
                    self.scan_reported = False
                self._save_state()
        self.untracked = {s for s in self.untracked if self.position_qty(s) != 0}
        self.scan_positions()
        if self.state.get("open"):
            self.sync_open_qty()

    # ---------- the scan after each close ----------
    def begin_scan(self, slot, bar, now):
        """Start the scan for a new signal bar: the bar that started at `bar` and
        has just closed (None before today's first window opens). The signals of
        the bar before are replaced. It also runs after a restart: the signals
        depend only on completed bars and the traded levels, so they come out the
        same, and the symbols already taken today, or dropped for this bar, are
        kept from the saved state. The bars are read only while this bar's entry
        window is open, once per symbol (again only when a read fails or does not
        show this bar yet)."""
        self.scan_slot = slot
        self.scan_bar = bar
        window = entry_window(bar) if bar is not None else None
        self.scan_todo = list(self.symbols) if window and now < window[1] else []
        self.scan_results = {}
        self.scan_started = self.scan_tried = None
        self.scan_reported = False
        self.scan_failed = set()  # symbols whose bars could not be read for this bar
        self.scan_stale = {}  # symbols whose bars did not show this bar yet
        self.scan_waiting = ""  # why the scan is waiting (an entry being confirmed), if it is
        self.held_by = ""
        self.price_misses = {}
        self.state["candidates"] = []
        self.state["signal_bar"] = f"{bar:%Y-%m-%d %H:%M}" if bar is not None else None
        self._save_state()

    def evaluate_symbol(self, symbol, read, now):
        """What the completed bars say for one symbol: a dict with `result`
        ("candidate", "dropped", "no setup", "stale" or "no data"), or None to read
        its bars again later. `read` is (bars, daily bars or None) from
        read_symbol(). The level book is replayed over the bars with this
        strategy's traded levels, and the signal is the one on the last bar,
        which must be the bar that has just closed."""
        if read is None:
            return None
        bars, daily = read
        cfg = self.cfg
        if len(bars) < 2 * cfg.pivot_n + 2:
            return {
                "result": "no data",
                "why": f"only {len(bars)} completed {INTERVAL} bars, too few for a pivot",
            }
        last = bars.index[-1].tz_localize(None).to_pydatetime()
        last_bar = f"{last:%Y-%m-%d %H:%M}"
        expected = self.scan_bar
        if expected is None or last != expected:
            want = f"{expected:%Y-%m-%d %H:%M}" if expected is not None else "none yet"
            return {
                "result": "stale",
                "why": (
                    f"its latest completed {INTERVAL} bar is {last_bar}, not the bar that has "
                    f"just closed ({want}): the broker has not published it yet, or its data "
                    "is stale, so there is no signal on this bar"
                ),
                "last_bar": last_bar,
            }
        bias = bias_for(bars, cfg, daily)
        signals, state = replay(bars, cfg, bias, self.traded_for(symbol), self.replays.get(symbol))
        self.replays[symbol] = state
        signal = signals[-1] if signals and signals[-1].t == len(bars) - 1 else None
        if signal is None:
            kind, why = self.no_signal_why(state.last)
            return {"result": kind, "why": why, "last_bar": last_bar}
        tick, _lot = self.tick_lot(symbol)
        cand = make_candidate(signal, tick)
        if not cand["rr"] > 0:
            return {
                "result": "dropped",
                "why": f"the stop {cand['stop']:.2f} is at the reference price",
                "last_bar": last_bar,
            }
        return {"result": "candidate", "why": "", "setup": cand, "last_bar": last_bar}

    def no_signal_why(self, last):
        """("no setup" or "dropped", why) for a last bar that gave no signal."""
        cfg = self.cfg
        raw = int(last.get("raw_side") or 0)
        if raw == 0:
            return "no setup", (
                f"no setup: no sweep of a level with at least {cfg.min_touches} touches"
            )
        name = "short" if raw < 0 else "long"
        what = f"a {name} sweep of the {last['level']:.2f} level ({last['touches']} touches)"
        if cfg.use_bias and last.get("bias") != raw:
            tf = "daily" if cfg.bias_tf == "D" else "1h"
            bias = {1: "up", -1: "down"}.get(last.get("bias"), "not set (no EMA yet, or flat)")
            return "dropped", f"{what}, but the {tf} bias is {bias}"
        return "dropped", f"{what}, but ATR({cfg.atr_period}) does not exist yet"

    def scan_hold(self):
        """Why the bars are not read now, or "". An entry being confirmed holds the
        scan, and so does an open trade whose stop is not confirmed at the broker:
        then the stop is watched by this script and nothing may delay it. An open
        trade with its stop working at the broker does not hold it."""
        s = self.state
        if s.get("pending"):
            return f"a {s['pending']['plan']['symbol']} entry is being confirmed"
        trade = s.get("open")
        if trade and (
            not trade.get("stop_order_id") or trade.get("stop_check") or trade.get("exit_retry")
        ):
            return f"a {trade['symbol']} trade is open and its stop is not confirmed at the broker"
        return ""

    def watch_open_trade(self):
        """Between symbol reads: the open trade is checked again (target, time
        exit, square-off, its stop and any exit under way, as in step()) once
        POLL_SECONDS have passed since it was last checked, so reading the
        basket never leaves it unwatched for longer than about one poll.
        Returns scan_hold()'s reason when the read must stop, else ""."""
        if self.state.get("open") and (
            self.open_checked is None
            or (self.clock() - self.open_checked).total_seconds() >= POLL_SECONDS
        ):
            self.open_checked = self.clock()
            self.manage_open(self.open_checked)
        return self.scan_hold()

    def scan_signals(self, now):
        """Read the completed bars of every symbol not read yet for this signal bar
        and note the signals. A symbol whose bars cannot be read, or do not show
        this bar yet, is tried again every SCAN_RETRY_SEC until this bar's entry
        window ends.

        Nothing is read while entries are blocked for the rest of the day, and
        the reading waits while scan_hold() says so. A trade open with its stop
        at the broker does not stop the reading: the signals found are noted and
        wait in the window, and are dropped when it ends unless the trade has
        closed first. Every symbol left unread is journalled with the reason."""
        if not self.scan_todo:
            return
        window = entry_window(self.scan_bar) if self.scan_bar is not None else None
        if window is None or now >= window[1]:
            self.close_scan()
            return
        day_why = self.day_block()
        if day_why:
            self.close_scan(f"not scanned: {day_why}")
            return
        busy = self.scan_hold()
        if busy:
            if self.scan_waiting != busy:
                self.scan_waiting = busy
                log(f"the {self.scan_bar:%H:%M} bar's scan waits: {busy}")
            return
        if self.scan_tried is not None and (now - self.scan_tried).total_seconds() < SCAN_RETRY_SEC:
            return
        if self.scan_started is None:
            self.scan_started = now
        self.scan_tried = now
        left = []
        for n, symbol in enumerate(self.scan_todo):
            if n:
                time.sleep(self.pause)
                hold = self.watch_open_trade()
                if hold:
                    # An exit or a stop needs the script's full attention: the rest
                    # is read once that is settled, while this bar's window lasts.
                    left.extend(self.scan_todo[n:])
                    self.scan_waiting = hold
                    self.scan_tried = None
                    log(f"the {self.scan_bar:%H:%M} bar's scan is paused: {hold}")
                    break
            try:
                found = self.evaluate_symbol(symbol, self.read_symbol(symbol, now), now)
            except Exception as e:
                found = {"result": "no data", "why": f"its bars could not be used ({e})"}
            if found is None:
                self.scan_failed.add(symbol)
                self.scan_stale.pop(symbol, None)
                left.append(symbol)
                continue
            if found["result"] == "stale":
                self.scan_stale[symbol] = found  # the bar may be published a little late
                self.scan_failed.discard(symbol)
                left.append(symbol)
                continue
            self.scan_failed.discard(symbol)
            self.scan_stale.pop(symbol, None)
            self.note_result(symbol, found)
        self.scan_todo = left
        self._save_state()
        if not left:
            self.report_scan()

    def note_result(self, symbol, found, quiet=False):
        """Record what the scan found for one symbol, and log or journal it."""
        self.scan_results[symbol] = found
        if found["result"] == "candidate":
            self.add_candidate(symbol, found["setup"])
        elif found["result"] in ("dropped", "stale"):
            log(f"{symbol}: {found['why']}")
            self.journal(
                "skip",
                symbol=symbol,
                why=found["why"],
                signal_bar=self.state.get("signal_bar"),
                last_bar=found.get("last_bar"),
            )
        elif found["result"] in ("no data", "not scanned") and not quiet:
            log(f"{symbol}: no signal on this bar, {found['why']}")

    def close_scan(self, why=None):
        """End this signal bar's scan with the symbols still unread, each with the
        true reason it was not read, and journal the scan."""
        for symbol in self.scan_todo:
            if symbol in self.scan_stale and not why:
                stale = self.scan_stale[symbol]
                self.note_result(
                    symbol, {**stale, "why": stale["why"] + " (read until the window ended)"}
                )
                continue
            if why:
                reason = why
            elif symbol in self.scan_failed:
                reason = "its bars could not be read before the entry window ended"
            elif self.scan_waiting:
                reason = f"not scanned: {self.scan_waiting} until the entry window ended"
            else:
                reason = "not scanned: it was confirmed only after the entry window ended"
            kind = "no data" if symbol in self.scan_failed and not why else "not scanned"
            # one shared reason is logged once, by report_scan, not once per symbol
            quiet = kind == "not scanned" and not symbol_specific(reason)
            self.note_result(symbol, {"result": kind, "why": reason}, quiet=quiet)
        self.scan_todo = []
        self._save_state()
        self.report_scan()

    @staticmethod
    def skip_key(symbol, signal_bar):
        """A signal is one symbol on one signal bar."""
        return f"{symbol} {signal_bar}"

    def add_candidate(self, symbol, setup):
        cand = {"symbol": symbol, **setup, "status": "waiting", "why": ""}
        key = self.skip_key(symbol, setup.get("signal_bar"))
        if self.is_taken(symbol, setup.get("signal_bar")):
            cand.update(status="taken", why="an entry was already sent for it on this signal bar")
        elif symbol in (self.state.get("unknown") or {}):
            cand.update(status="skipped", why=self.unknown_why(symbol))
        elif key in self.state["skipped"]:
            cand.update(status="skipped", why=self.state["skipped"][key])
        others = [c for c in self.state["candidates"] if c["symbol"] != symbol]
        self.state["candidates"] = rank_candidates(others + [cand], self.order)
        self.journal("candidate", **cand)

    def report_scan(self):
        if self.scan_reported:
            return
        self.scan_reported = True
        ranked = self.state["candidates"]
        bar = self.state.get("signal_bar")
        self.journal(
            "scan",
            day=self.state["date"],
            signal_bar=bar,
            results={s: r["why"] or r["result"] for s, r in self.scan_results.items()},
            ranking=[{"rank": n + 1, **c} for n, c in enumerate(ranked)],
        )
        unread = {r["why"] for r in self.scan_results.values() if r["result"] == "not scanned"}
        read = sum(1 for r in self.scan_results.values() if r["result"] != "not scanned")
        if not ranked:
            if read or not unread:
                log(f"scan of the {bar} bar: no signal in {read} symbols")
            for why in sorted(unread):
                log(f"scan of the {bar} bar: symbols {why}")
            return
        parts = [
            f"{c['symbol']} {'long' if c['side'] == 'BUY' else 'short'} at the {c['level']:.2f} "
            f"level, {c['touches']} touches (entry ~{c['ref_entry']:.2f}, SL {c['stop']:.2f})"
            for c in ranked
        ]
        log(f"scan of the {bar} bar: {len(ranked)} signal(s), best first: " + "; ".join(parts))

    # ---------- choosing the entry ----------
    def entry_block(self, now):
        """(why no entry may be placed now, whether that holds for the current
        signal bar's signals, which are then dropped)."""
        s = self.state
        bar = self.scan_bar
        window = entry_window(bar) if bar is not None else None
        day_why = self.day_block()
        if day_why:
            return day_why, True
        if window is None:
            return "no entry window is open", True
        span = f"{window[0]:%H:%M}-{window[1]:%H:%M}"
        if now >= window[1]:
            return f"outside the entry window {span} for the {bar:%H:%M} bar", True
        if self.unsaved:
            return "the state file cannot be written", False
        if s.get("open"):
            return f"a {s['open']['symbol']} trade is already open", False
        if s.get("pending"):
            return f"a {s['pending']['plan']['symbol']} entry is being confirmed", False
        if s.get("sweep"):
            return f"an order answer in {s['sweep']['symbol']} is still being checked", False
        if now < window[0]:
            return f"before the entry window {span} for the {bar:%H:%M} bar", False
        if self.untracked:
            return f"positions not managed here in {', '.join(sorted(self.untracked))}", False
        return "", False

    def day_block(self):
        """Why no new entry is made for the rest of today, or "". The trade limit
        and the loss limit apply only when set above 0 (both are off by default)."""
        s = self.state
        if s.get("no_entries"):
            return f"no new entries today: {s['no_entries']}"
        # An entry whose answer was never confirmed either way may still exist at
        # the broker, so it counts as one of the day's trades.
        if (
            MAX_TRADES_PER_DAY > 0
            and s["trades"] + len(s.get("unknown") or {}) >= MAX_TRADES_PER_DAY
        ):
            plural = "" if MAX_TRADES_PER_DAY == 1 else "s"
            return f"the day's limit of {MAX_TRADES_PER_DAY} trade{plural} is reached"
        if MAX_DAILY_LOSS > 0 and s["pnl"] <= -MAX_DAILY_LOSS:
            return f"daily loss limit Rs {MAX_DAILY_LOSS:.0f} reached"
        return ""

    def is_taken(self, symbol, signal_bar):
        """Whether an entry was already sent for `symbol` on this signal bar. A bare
        symbol in the list blocks the symbol for the day, the safe reading."""
        taken = self.state.get("taken") or []
        return self.skip_key(symbol, signal_bar) in taken or symbol in taken

    def unknown_why(self, symbol):
        return (
            f"an earlier entry in {symbol} today got no clear answer and may still exist at "
            f"the broker, so {symbol} is not entered again today"
        )

    def can_enter(self, now):
        """Whether a new entry may be placed now."""
        why, _final = self.entry_block(now)
        return not why, why

    def waiting(self):
        """The current signal bar's candidates still eligible, best first."""
        return [c for c in self.state.get("candidates") or [] if c.get("status") == "waiting"]

    def drop(self, cand, why, **data):
        """Drop a candidate for good (signals are never carried to a later bar),
        with the reason."""
        cand.update(status="skipped", why=why)
        self.state["skipped"][self.skip_key(cand["symbol"], cand.get("signal_bar"))] = why
        self._save_state()
        log(f"{cand['symbol']} {cand['side']} signal skipped: {why}")
        self.journal(
            "skip", symbol=cand["symbol"], side=cand["side"], why=why, candidate=dict(cand), **data
        )

    def mark_taken(self, symbol, signal_bar):
        """Record that an entry was sent for `symbol` on this signal bar: it is
        never entered again on the same signal, even after a restart."""
        key = self.skip_key(symbol, signal_bar)
        if key not in self.state["taken"]:
            self.state["taken"].append(key)
        for cand in self.state.get("candidates") or []:
            if cand["symbol"] == symbol and cand.get("signal_bar") == signal_bar:
                cand.update(status="taken", why="")

    def holding(self):
        """The trade that holds the signals of this window back, said in the past
        tense for the journal, or ""."""
        s = self.state
        if s.get("open"):
            return f"a {s['open']['symbol']} trade was open during the window"
        if s.get("pending"):
            return f"a {s['pending']['plan']['symbol']} entry was being confirmed during the window"
        return ""

    def with_held(self, why):
        """`why`, plus the open trade that held the signals in their window, if one did."""
        if not self.held_by:
            return why
        return f"{why} ({self.held_by}, and only one trade is open at a time)"

    def read_leverage(self):
        """(mode, leverage) for the entry about to be sized. The mode is read again
        every time, since it can be switched while the script runs; when it cannot
        be read the entry is sized at 1x and that is logged."""
        mode, leverage = mode_leverage(self.call("analyzerstatus", quiet=True))
        if mode == UNKNOWN_MODE:
            log("the trading mode could not be read; this entry is sized at 1x, the safe side")
        return mode, leverage

    def try_enter(self, now):
        """Enter the best candidate that is still valid on the live price. One
        entry attempt a cycle, and only while nothing is open or pending."""
        waiting = self.waiting()
        if not waiting:
            return
        why, final = self.entry_block(now)
        if why:
            if final:
                for cand in waiting:
                    self.drop(cand, self.with_held(why))
            else:
                self.held_by = self.holding() or self.held_by  # named when they are dropped
            return
        if (
            self.scan_todo
            and self.scan_started is not None
            and (now - self.scan_started).total_seconds() < SCAN_GRACE_SEC
        ):
            return  # wait for every symbol's bars, so the ranking is whole
        ranked = self.state["candidates"]
        sizing = None  # (mode, leverage), read once per attempt, just before sizing
        for n, cand in enumerate(waiting):
            symbol = cand["symbol"]
            if self.is_taken(symbol, cand.get("signal_bar")):
                cand.update(
                    status="taken", why="an entry was already sent for it on this signal bar"
                )
                continue  # never twice on one signal, whatever else says so
            if symbol in (self.state.get("unknown") or {}):
                self.drop(cand, self.unknown_why(symbol))
                continue
            qty = self.position_qty(symbol)
            if qty is None:
                log(f"{symbol}: its position could not be read; the entry is tried again")
                return
            if qty:
                self.drop(cand, f"a position in {symbol} is already open at the broker")
                continue
            price = self.ltp(symbol)
            if price is None:
                misses = self.price_misses[symbol] = self.price_misses.get(symbol, 0) + 1
                if misses < PRICE_READS:
                    return
                self.drop(cand, "its live price could not be read")
                continue
            if sizing is None:
                sizing = self.read_leverage()
            mode, leverage = sizing
            tick, lot = self.tick_lot(symbol)
            plan, why = plan_entry(
                symbol, cand, price, lot, leverage=leverage, mode=mode, tick=tick
            )
            if plan is None:
                self.drop(cand, why, price=price, mode=mode, leverage=leverage)
                continue
            # The reads for this candidate take time too: the window is judged
            # again on the clock as it is now, just before anything is sent.
            now = self.clock()
            late, final = self.entry_block(now)
            if late:
                log(f"{symbol} {plan.side} not entered: {late}")
                if final:
                    for other in self.waiting():
                        self.drop(other, self.with_held(late))
                return
            rank = next((i + 1 for i, c in enumerate(ranked) if c is cand), None)
            self.journal(
                "signal",
                plan=asdict(plan),
                rank=rank,
                candidate=dict(cand),
                mode=mode,
                leverage=leverage,
            )
            log(
                f"{symbol} {plan.side} {plan.qty} @ ~{plan.entry:.2f}, SL {plan.stop:.2f}, "
                f"target ~{plan.target:.2f} (set again from the fill; rank {rank}, "
                f"{plan.touches} touches at the {plan.level:.2f} level, signal bar "
                f"{plan.signal_bar}); position value Rs "
                f"{plan.position_value:.0f} at {leverage:g}x ({mode} mode), loss at the stop "
                f"about Rs {plan.planned_loss:.0f}, sized by {plan.sized_by}"
            )
            if waiting[n + 1 :]:
                self.held_by = f"a {symbol} trade was open during the window"
            for other in waiting[n + 1 :]:
                ignored = (
                    f"{symbol} ranks higher and is being entered; only one trade is open at a time"
                )
                log(f"{other['symbol']} {other['side']} signal ignored: {ignored}")
                self.journal(
                    "signal_ignored",
                    symbol=other["symbol"],
                    side=other["side"],
                    why=ignored,
                    candidate=dict(other),
                )
            self.enter(plan, now)
            return

    # ---------- entry ----------
    def enter(self, plan, now):
        """Place the entry and keep it as `pending` until the broker's answer is
        confirmed. Every entry is blocked on every symbol while it is pending, and
        the symbol is marked as taken on this signal bar before anything is sent."""
        key = self.skip_key(plan.symbol, plan.signal_bar)
        already = key in self.state["taken"]
        self.mark_taken(plan.symbol, plan.signal_bar)
        self.state["pending"] = {
            "order_id": None,
            "plan": asdict(plan),
            "placed": now.isoformat(timespec="seconds"),
        }
        # saved before sending, so a crash mid-send is resolved on restart
        if not self._save_state():
            # Not saved: a restart would not know of the order, so it is not sent.
            self.state["pending"] = None
            if not already:
                self.state["taken"].remove(key)
            for cand in self.state.get("candidates") or []:
                if (
                    cand["symbol"] == plan.symbol
                    and cand.get("signal_bar") == plan.signal_bar
                    and not already
                ):
                    cand.update(status="waiting", why="")
            log(f"{plan.symbol}: the entry is not sent because the state file cannot be written")
            return
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
                # Filled in full by the broker's word; the position check follows.
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
                if self.cancel_orders(ours):
                    p["cancelled"] = True  # found and confirmed cancelled: no longer unknown
                    self._save_state()
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
        unknown = not oid and not p.get("cancelled")
        if unknown:
            # No answer, and no sign of the order or of a position. It may still
            # exist, so the symbol is watched and not entered again today, and it
            # counts toward MAX_TRADES_PER_DAY when that limit is on.
            self.state.setdefault("unknown", {})[plan.symbol] = {
                "plan": p["plan"],
                "placed": p["placed"],
            }
        self._save_state()
        if unknown:
            self.notify(
                f"{plan.symbol} {plan.side} entry: no clear answer came back and neither the "
                f"order nor a position has shown in {ENTRY_CONFIRM_SEC}s, so it is taken as not "
                f"placed. {plan.symbol} is not entered again today, and if the position shows "
                "up later it is taken over and given its stop"
            )
        else:
            self.notify(f"{plan.symbol} {plan.side} entry was not filled; no trade")
        self.journal(
            "entry_failed", symbol=plan.symbol, side=plan.side, status=status, unknown=unknown
        )

    def watch_unknown(self):
        """Every cycle: the symbols whose entry answer was never confirmed. An
        entry order that shows up still working is cancelled. If the position
        shows up after all, on the planned side, it is taken over and given its
        stop; anything else found there is reported and blocks entries."""
        unknown = self.state.get("unknown") or {}
        for symbol, saved in list(unknown.items()):
            try:
                plan = Plan(**saved["plan"])
                placed = datetime.fromisoformat(saved["placed"])
            except (TypeError, KeyError, ValueError):
                plan = placed = None
            if plan is not None:
                late = [o for o in self.open_orders(symbol) or [] if self._ours(o, plan, placed)]
                if late:
                    self.notify(f"{symbol}: the entry order whose answer was lost is cancelled")
                    self.cancel_orders(late)
            qty = self.position_qty(symbol)
            if not qty:
                continue  # flat, or unreadable this cycle
            busy = self.state.get("open") or self.state.get("pending") or self.state.get("sweep")
            if plan is not None and (qty > 0) == (plan.side == "BUY") and not busy:
                del unknown[symbol]
                self.untracked.discard(symbol)
                self.notify(
                    f"{symbol}: the entry whose answer was lost has filled after all "
                    f"(position {qty}); it is taken over and given its stop"
                )
                self.adopt_entry(plan, qty, 0)
            elif symbol not in self.untracked:
                self.handle_unmanaged(symbol, qty)

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
        """Book the fill and place its stop. The stop comes first: the trade is
        saved as "stop not confirmed yet" before the stop is sent, so if this
        script is stopped, or anything here fails, between the fill and the
        stop's answer, the next cycle (or the next start) looks for the stop in
        the order book, adopts one that shows, and places one if none does. The
        journal and the alert come after the stop, and neither can stop it.

        The target is RR x the risk from this fill to the stop, and the time exit
        is due MAX_BARS bars after the entry bar's open. The swept level is kept
        as traded, so the replay kills it where the research engine does."""
        now = self.clock().isoformat(timespec="seconds")
        gapped = fill <= plan.stop if plan.side == "BUY" else fill >= plan.stop
        tick, _lot = self.tick_lot(plan.symbol)
        target = None if gapped else target_from(fill, plan.stop, plan.side, tick)
        try:
            due = time_exit_at(datetime.strptime(plan.signal_bar, "%Y-%m-%d %H:%M"))
            due = due.isoformat(timespec="seconds")
        except (TypeError, ValueError):
            due = None
        self.state["trades"] += 1
        trade = {
            "symbol": plan.symbol,
            "side": plan.side,
            "qty": qty,
            "entry_price": fill,
            "stop": plan.stop,
            "target": target,  # fixed at the fill, like the stop: neither is ever moved
            "time_exit_at": due,
            "signal_bar": plan.signal_bar,
            "level": plan.level,
            "touches": plan.touches,
            "ref_entry": plan.ref_entry,
            "leverage": plan.leverage,
            "mode": plan.mode,
            "stop_order_id": None,
            "stop_type": None,
            "stop_check": not gapped,  # a stop is about to be sent: look for it until confirmed
            "exit_order_id": None,
            "exit_unanswered": False,
            "exit_retry": None,
            "opened": now,
        }
        if not gapped:
            trade["stop_unknown_since"] = now
        self.state["open"] = trade
        if plan.signal_bar and plan.kind:
            self.state.setdefault("traded_levels", []).append(
                {
                    "symbol": plan.symbol,
                    "side": plan.side,
                    "signal_bar": plan.signal_bar,
                    "kind": plan.kind,
                    "level": plan.level,
                    "touches": plan.touches,
                }
            )
        # A new trade in a watched symbol does NOT end the watch: the earlier
        # closing order whose answer was lost can still fill after this trade has
        # closed. While this trade is open the symbol is busy and not checked.
        try:
            self._save_state()
        except Exception as e:  # the stop matters more than the file
            log(f"the state file could not be written ({e}); placing the stop all the same")
        try:
            if gapped:
                self.exit_trade("filled beyond the stop")
            else:
                self.place_stop(trade)
        finally:
            self.journal(
                "entry",
                trade=trade,
                plan=asdict(plan),
                planned_loss=round(qty * abs(fill - plan.stop), 2),
                position_value=round(qty * fill, 2),
                mode=plan.mode,
                leverage=plan.leverage,
            )
            self.notify(
                f"{plan.side} {qty} {plan.symbol} filled @ {fill:.2f}, SL {plan.stop:.2f}, "
                f"target {f'{target:.2f}' if target is not None else 'none (filled beyond the stop)'}, "
                f"time exit {str(due or '')[11:16] or 'none'}; position value Rs "
                f"{qty * fill:.0f} at {plan.leverage:g}x, loss at the stop about Rs "
                f"{qty * abs(fill - plan.stop):.0f}"
            )

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
        # Both order types were clearly refused: no stop exists at the broker.
        trade["stop_check"] = False
        trade.pop("stop_unknown_since", None)
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
        """Every CHECK_MINUTES: the managed quantity follows the broker position,
        and the stop order is resized to cover all of it. Only its quantity
        changes: the stop price is fixed at entry."""
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
        # As at the fill: saved as "stop not confirmed yet" before the new stop is sent.
        trade["stop_check"] = True
        trade["stop_unknown_since"] = self.clock().isoformat(timespec="seconds")
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
        if str(trade.get("opened") or "")[:10] < now.date().isoformat():
            self.exit_trade("left open from an earlier day")  # nothing is held overnight
            return
        if minutes_of(now) >= hhmm(SQUARE_OFF):
            self.exit_trade("square-off time")
            return
        due = trade.get("time_exit_at")
        timed_out = bool(due) and self._since(due) >= 0
        ltp = self.ltp(trade["symbol"])
        if ltp is None:
            if timed_out:
                self.exit_trade(f"time exit after {MAX_BARS} bars")
            return
        long = trade["side"] == "BUY"
        if trade.get("target") is not None and (
            (ltp >= trade["target"]) if long else (ltp <= trade["target"])
        ):
            self.exit_trade("target")
        elif timed_out:
            self.exit_trade(f"time exit after {MAX_BARS} bars")
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
            # Quantity 0, on purpose. With position_size 0 the app closes whatever
            # position it reads itself. If the position went flat between the read
            # above and the app's own read, a smart order carrying a quantity is
            # placed as given and would open the opposite position; with 0 it does
            # nothing.
            quantity=0,
            position_size=0,
        )
        if outcome == NOTHING:
            # The app found nothing to close: flat already, if the position agrees.
            left = self.position_qty(symbol)
            if left == 0:
                return self._book_flat(trade, exit_action, reason)
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
            # Two trades in one symbol on opposite sides can each leave a lost
            # closing order; the watch then keeps both sides.
            sides = set(watch_actions(watch.get(trade["symbol"])))
            sides.add(self.state["sweep"]["action"])
            watch[trade["symbol"]] = sides.pop() if len(sides) == 1 else sorted(sides)
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

    def check_watch(self):
        """Every cycle, for each symbol closed today while an order answer was
        unknown: a position that a late closing order opened there is taken over
        and closed (or, while another trade is open, reported and held back)."""
        if self.state.get("pending"):
            return  # an entry is being confirmed: one thing at a time
        busy = self._busy_symbols()
        for symbol in list(self.state.get("watch") or {}):
            if symbol in busy or (self.state.get("sweep") or {}).get("symbol") == symbol:
                continue  # managed already, or the sweep is looking at it
            qty = self.position_qty(symbol)
            if qty and self.is_reversal(symbol, qty):
                if symbol not in self.untracked or not self.state.get("open"):
                    self.handle_unmanaged(symbol, qty)

    def is_reversal(self, symbol, qty):
        """A position on the side a late closing order would leave, in a symbol this
        strategy closed today while an order answer was unknown."""
        actions = watch_actions((self.state.get("watch") or {}).get(symbol))
        return bool(qty) and (("SELL" in actions and qty < 0) or ("BUY" in actions and qty > 0))

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
    for name, value, expected in BAD_SETTINGS:
        log(f"{name} is set to '{value}', which is not {expected}. Correct it and start again.")
    if BAD_SETTINGS:
        raise SystemExit(1)
    if MAX_TRADES_PER_DAY < 0:
        log("LSW_MAX_TRADES_PER_DAY must be 0 (no limit) or more")
        raise SystemExit(1)
    if min(MAX_DAILY_LOSS, MAX_LOSS_PER_TRADE, SIZING_BUFFER_PCT, STOP_BUFFER, ATR_MULT) < 0:
        log(
            "LSW_MAX_DAILY_LOSS, LSW_MAX_LOSS_PER_TRADE, LSW_SIZING_BUFFER_PCT, "
            "LSW_STOP_BUFFER and LSW_ATR_MULT cannot be below 0 (0 switches each one off)"
        )
        raise SystemExit(1)
    if min(PIVOT_N, MIN_TOUCHES, BIAS_EMA, ATR_PERIOD, MAX_BARS) < 1:
        log(
            "LSW_PIVOT_N, LSW_MIN_TOUCHES, LSW_BIAS_EMA, LSW_ATR_PERIOD and LSW_MAX_BARS "
            "must each be 1 or more"
        )
        raise SystemExit(1)
    if not 0 < LEVEL_TOL < 0.1:
        log("LSW_LEVEL_TOL must be above 0 and below 0.1 (it is a fraction: 0.0015 is 0.15%)")
        raise SystemExit(1)
    if not RR > 0:
        log("LSW_RR must be above 0")
        raise SystemExit(1)
    if not (CAPITAL > 0 and LEVERAGE > 0):
        log("LSW_CAPITAL and LSW_LEVERAGE must be above 0")
        raise SystemExit(1)
    if HISTORY_DAYS < 0 or DAILY_HISTORY_DAYS < 0:
        log("LSW_HISTORY_DAYS and LSW_DAILY_HISTORY_DAYS must be 0 (worked out) or more")
        raise SystemExit(1)
    half = BAR_MINUTES * 30
    if (
        ENTRY_OFFSET_SEC < 0
        or ENTRY_WINDOW_MIN < 1
        or ENTRY_OFFSET_SEC + 60 * ENTRY_WINDOW_MIN > half
    ):
        log(
            "LSW_ENTRY_OFFSET_SEC must be 0 or more and LSW_ENTRY_WINDOW_MIN 1 or more, and "
            f"together they must fit inside the first half of the next {INTERVAL} bar "
            f"({half} seconds at most), so the entry stays near that bar's open"
        )
        raise SystemExit(1)
    if not hhmm(SESSION_OPEN) < hhmm(LAST_ENTRY) <= hhmm(SQUARE_OFF) <= hhmm(SESSION_CLOSE):
        log(
            f"LSW_LAST_ENTRY must be after {SESSION_OPEN}, and LSW_SQUARE_OFF no earlier than "
            f"it and no later than {SESSION_CLOSE}"
        )
        raise SystemExit(1)
    if not signal_bars(now_ist().date()):
        log(
            "no entry window fits before LSW_LAST_ENTRY and LSW_SQUARE_OFF: the first one "
            "opens LSW_ENTRY_OFFSET_SEC seconds after the 09:15 bar closes"
        )
        raise SystemExit(1)
    strategy = LiquiditySweepStrategy(api(api_key=API_KEY, host=HOST, ws_url=WS_URL))
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
