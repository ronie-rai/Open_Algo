#!/usr/bin/env python
"""
Candle Liquidity - 1Hr: hourly 3-candle Supertrend reversal, intraday, many NSE stocks

The same rules as the daily 3 Candle Reversal (three_candle_reversal_multi.py),
on 1-HOUR candles. After every hourly bar closes the setups are worked out
again across the basket, and an entry may be made only in a short window
right after that close. Trades are INTRADAY (MIS), long and short, and nothing
is held overnight. It keeps its own settings (CL1H_*), its own data folder and
its own order tag, so it can run beside the daily strategy.

The three candles
  Candle 1 = the hourly bar before Candle 2.
  Candle 2 = the most recent COMPLETED hourly bar: the one that has just closed.
  Candle 3 = the hour in which the trade is taken.
  NSE hourly bars start at 09:15 and run 09:15, 10:15, 11:15, 12:15, 13:15,
  14:15, plus a short 15:15-15:30 bar. A bar counts as completed only once its
  start + 60 minutes (15:30 for the 15:15 bar) is at or before the current
  time; the bar still forming is always dropped. The history call stamps each
  intraday bar with its START time (the openalgo SDK gives it as an IST time in
  the index, and so does the backtester); a frame that carries the stamp as an
  epoch or text column instead is read from that column.
  Only one of TODAY's bars, the one that closed at the latest hourly close, can
  be a Candle 2. So there is no entry from 09:15 to 10:15 on yesterday's last
  bar: the first possible entry is just after the 09:15 bar closes at 10:15.
  Candle 1 may be the previous session's last bar when Candle 2 is today's
  first bar.

Supertrend
  TradingView's ta.supertrend(3.0, 10) on 1-hour bars (ST_FACTOR, ST_ATR_LEN):
  source hl2, ATR = ta.atr (Wilder's average seeded with the simple average of
  the first 10 true ranges, true range on the first bar = high - low), the
  band carry rules and the direction rule exactly as Pine's built-in. Pine's
  own direction number is -1 when the line is below the price and +1 when it
  is above; this file calls -1 "bullish" and +1 "bearish". It runs over the
  continuous hourly series as the history call returns it, across sessions,
  like a TradingView hourly chart. About 30 calendar days of hourly bars are
  read (HISTORY_DAYS; about 7 bars a session) so the line has settled, and a
  symbol with fewer than 5 x ST_ATR_LEN (50) completed hourly bars is not
  traded.

LONG setup, all true on Candle 2
  1. Supertrend is bearish on Candle 2 (the line is above the price).
  2. Candle 2 high < Candle 1 high, and Candle 2 close < Candle 1 close.
  3. Candle 2 close > Candle 1 low. Only the close matters: a wick below
     Candle 1's low is allowed.
  Trade: BUY at market as soon as that hour's entry window opens.
    reference entry = Candle 2 close
    stop loss       = Candle 2 low
    take profit     = Candle 1 high x (1 - TP_PCT / 100), TP_PCT = 0.5
  Skipped when the take profit is at or below the reference entry.

SHORT setup, the exact mirror
  1. Supertrend is bullish on Candle 2 (the line is below the price).
  2. Candle 2 low > Candle 1 low, and Candle 2 close > Candle 1 close.
  3. Candle 2 close < Candle 1 high (a wick above it is allowed).
  Trade: SELL SHORT at market as soon as that hour's entry window opens.
    reference entry = Candle 2 close
    stop loss       = Candle 2 high
    take profit     = Candle 1 low x (1 + TP_PCT / 100)
  Skipped when the take profit is at or above the reference entry.

Filters
  - ENABLE_LONGS and ENABLE_SHORTS switch each side on or off.
  - MIN_RR (0 = off): skip when (take profit - reference entry) /
    (reference entry - stop) is below it; mirrored for a short. This is
    judged at the reference entry, Candle 2's close.
  - Live price at entry time. The real entry is the current price, not
    Candle 2's close, so just before the order: a long is dropped when the
    current price is at or below the stop, or at or above the take profit; a
    short when it is at or above the stop, or at or below the take profit.
    These two checks use the current price; MIN_RR does not. A candidate
    dropped this way is dropped for that hour (that Candle 2); a new setup in
    the same symbol after a later close is a new signal.
  - A fill at or beyond the stop is closed at once.

Tick rounding
  The stop is rounded AWAY from the entry (a long's stop down, a short's stop
  up), so it is never tighter than Candle 2's low or high. The take profit is
  rounded TOWARD the entry (a long's down, a short's up), so it never asks for
  more than the rule's price. The filters, the reward:risk and the orders all
  use these rounded prices.

Entry windows, exits and the day
  - After each hourly close the window opens ENTRY_OFFSET_MIN (2) minutes
    later, so the broker has published the bar, and lasts ENTRY_WINDOW_MIN
    (10) minutes: 10:17-10:27, 11:17-11:27, 12:17-12:27, 13:17-13:27 and
    14:17-14:27 with the defaults. The bars of every symbol are read once at
    the start of each window (again only when a read fails), with the pause
    between calls the broker's rate limit needs; not on every poll.
  - No entry is made at or after LAST_ENTRY (14:30) or SQUARE_OFF (15:10), and
    a window that would open then does not open at all. So the last usable
    Candle 2 is the 13:15 bar; the 14:15 bar closes at 15:15, after the
    square-off, and the 15:15 bar is never a Candle 2.
  - The signals are fixed per hourly bar once worked out, so the setups of one
    Candle 2 are entered only in the window right after it closed. When the
    window ends, its remaining candidates are dropped (journalled) and the
    next close brings new ones.
  - A Candle 2 that is not the bar that has just closed (the broker has not
    published that bar for the symbol yet, or its data is stale) is skipped
    for that hour and journalled. This replaces the daily strategy's check on
    the age of the last daily bar.
  - The stop and the take profit are fixed at entry and never moved. No
    trailing. (If the broker position turns out larger or smaller than the
    trade, the stop order's QUANTITY is corrected; its price is not.)
  - Stop: one SL-M order at the broker at the stop price (SL only when the
    broker clearly refuses SL-M).
  - Take profit: watched on the last traded price every POLL_SECONDS and
    closed at market when reached.
  - Square-off at SQUARE_OFF (15:10). A trade left from an earlier day is
    closed as soon as it is seen.
  - Intraday only, both sides. NSE shares cannot be held short overnight, and
    this strategy is specified as intraday for longs as well, so both sides
    use MIS and are flat by the square-off.

One trade at a time, and which one
  - Only ONE trade is open across the whole basket. An open trade is the only
    trading reason a valid signal is passed over: while a trade is open or its
    entry is being confirmed, no other signal is taken, and the hour's bars
    are read only once it has closed (so its take profit and stop are watched
    without delay). If it closes inside the window, that hour's setups are
    worked out then and the best one still valid is taken; if it stays open
    for the whole window, the scan is journalled as not done and why.
  - When several stocks qualify on the same Candle 2 they are ranked by
    reward:risk at the reference entry, highest first; equal values keep the
    order of SYMBOLS. The best one still valid on the live price is taken.
  - No limit on trades a day (MAX_TRADES_PER_DAY 0). Once a trade has closed,
    the next valid signal is taken: the next candidate in the same window, or
    a setup after a later close.
  - A symbol is entered at most once on one Candle 2. A symbol that was
    stopped out can be traded again on a later hour's new signal, but never
    twice on the same signal, even after a restart.
  - Off by default, and available as settings: MAX_TRADES_PER_DAY above 0
    caps the day's trades, and MAX_DAILY_LOSS above 0 stops new entries once
    the day's realised loss reaches it. The loss counted is from fill prices;
    brokerage, taxes and positions not opened by this script are not counted.
  - The order-safety blocks below still stop entries whatever the signal: an
    entry being confirmed, a position that cannot be read or is not managed
    here, the SWEEP_SEC check after a close with an unknown answer, a lost
    state file (no entries for the rest of that day) and a state file that
    cannot be written. A symbol whose entry answer was never confirmed either
    way is not entered again that day.
  - Every candidate is journalled with its numbers, its rank and why it was
    or was not taken.

Sizing, and leverage by mode
  quantity = floor(CAPITAL x leverage / entry price), rounded down to the lot
  size. The price is padded by SIZING_BUFFER_PCT first, so a little slippage
  cannot push the position value over the cap.
  The leverage depends on the trading mode, which is read with analyzerstatus
  just before every entry is sized (you can switch modes while the script
  runs), and the mode and leverage used are logged and journalled:
    Live mode (analyze mode off)      LEVERAGE (2.0): CAPITAL 30000 x 2 = Rs 60000
    Backtest                          LEVERAGE (2.0): Rs 60000
    Analyzer mode (sandbox orders)    1x: Rs 30000
    Mode could not be read            1x, the safe side, and logged
  A backtest is recognised by the OPENALGO_BACKTEST=1 that the backtest runner
  sets, or by the backtester's own analyzerstatus answer (mode "backtest").
  CL1H_LEVERAGE changes the Live and backtest value; analyzer mode is always
  1x. OpenAlgo has no MTF product: the 2x here is MIS intraday leverage, and
  it is only available if your broker gives at least 2x intraday margin on
  the stock. The backtester's own MIS margin is 5x, so 2x fits.
  Loss per trade, Rs 4000 at most (MAX_LOSS_PER_TRADE). The stop PRICE is
  always the strategy's own (Candle 2's low for a long, its high for a
  short, rounded away from the entry) and is never moved to fit the rupee
  cap; the QUANTITY is cut instead:
    quantity = floor(4000 / (risk + pad)), risk = |live price - stop|,
    pad = live price x SIZING_BUFFER_PCT / 100,
  rounded down to the lot size. The smaller of this and the position value
  quantity is used, so a tight stop is limited by capital x leverage and a
  wide one by the Rs 4000. At the live price the loss at the stop is
  therefore at most Rs 4000 (the pad keeps it a little under). A signal
  whose quantity comes to 0 (its stop is more than Rs 4000 away for one share
  or one lot) is skipped and journalled. The planned loss is logged and
  journalled with every entry. What the cap cannot hold: a fill worse than
  the live price by more than the pad, an SL-M stop that fills beyond its
  trigger, a price that gaps through the stop, and a stop watched by this
  script rather than held at the broker (see Order safety) all lose more
  than planned. Set CL1H_MAX_LOSS_PER_TRADE to change it; 0 switches the
  rupee cap off and leaves only the position value.

What to expect
  The take profit is 0.5% inside Candle 1's far end. Hourly ranges are small,
  so a setup needs a Candle 1 whose range is well above 0.5% and a Candle 2
  that closes far enough inside it; many hours have no setup at all. With no
  daily trade limit and no daily loss limit by default, a choppy day can
  bring several trades, each risking up to Rs 4000; set MAX_TRADES_PER_DAY or
  MAX_DAILY_LOSS if you want a ceiling for the day. Brokerage and taxes weigh
  far more on a 0.5% move than on a daily one, and the backtest counts them.
  Backtest it before trusting it, and try other values of TP_PCT and MIN_RR
  to see the difference.

Restart
  The signals depend only on completed hourly bars, so after a restart the
  setups of the latest Candle 2 are worked out again and come out the same;
  if that bar's entry window has already ended, nothing is entered until the
  next close. The entries sent today (each kept as symbol and Candle 2), the
  signals dropped for each Candle 2, the entries with no clear answer, the
  trade count and the open trade are kept in state.json, so a restart never
  enters a symbol twice on the same Candle 2, never enters a symbol whose
  entry answer was lost that day, and keeps MAX_TRADES_PER_DAY when it is on.
  The order steps are saved BEFORE they are sent: the entry is saved as
  pending before its order goes out, and a filled trade is saved as "stop not
  confirmed yet" before its stop goes out. If the script is stopped between a
  fill and the stop's answer, the next start reads the order book: a stop that
  is there is adopted, and if none shows within STOP_CONFIRM_SEC of the fill
  one is placed. Until then the stop is watched on the LTP by this script, so
  for up to that long after such a restart there may be no stop at the broker.
  If state.json or the journal cannot be written (the disk is full, the file
  is open in another program) you get one alert, the open trade is still
  managed and its stop still placed, and no new entry is made until the state
  file can be written again.
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
    one (same side, MARKET, the planned quantity, this strategy's tag when the
    book shows one, placed after the attempt) is cancelled. Any other working
    order in that symbol, such as one you placed by hand, is left alone. An
    entry the broker calls complete while the position reads flat keeps
    blocking entries, with an alert, until the position shows it. Whatever the
    outcome, that symbol is not entered again on that Candle 2.
    An unknown entry that shows neither an order nor a position within
    ENTRY_CONFIRM_SEC is taken as not placed, but that symbol is not entered
    again that day (and the entry counts toward MAX_TRADES_PER_DAY when that
    limit is on), and its symbol is then read every cycle: an entry order that
    shows up still working is cancelled, and a position that shows up on the
    planned side is taken over and given its stop, with an alert. An unknown
    entry whose order was found and confirmed cancelled does not count: it is
    treated like a refused one.
  - Stop: sent straight after the fill, before the journal and the alert. SL
    is sent only after SL-M was clearly refused. After an unknown stop answer
    (or a restart before the answer) the order book is read every cycle; a
    stop that shows up is adopted, and a new one is placed only if none has
    shown after STOP_CONFIRM_SEC. Meanwhile the script watches the stop on the LTP. If
    the price reaches it inside that grace period, the position is closed at
    market at once, but only when the order book reads, shows no working
    stop (a stop that shows is adopted and left to work), and the position
    reads open. Any other working closing order in that symbol is cancelled
    and confirmed final first. If the book or the position cannot be read,
    the script waits and tries again every cycle. After such a close the
    sweep below runs.
  - A filled stop is booked as closed only after other closing orders in that
    symbol are cancelled and the position reads flat. If it is not flat (the
    stop was smaller than the position, or a second stop filled), what is
    left is closed at market and entries stay blocked until it is. Every
    CHECK_MINUTES the managed quantity is compared with the broker position
    and the stop is resized.
  - Exit: one closing order at a time. A working exit is left to fill until
    its backoff (EXIT_RETRY_SEC doubling to EXIT_RETRY_MAX_SEC) ends, then
    cancelled and confirmed final before another is sent; after
    EXIT_ALERT_AFTER attempts you get an alert (log and Telegram). After an
    unknown exit answer, a new closing order is sent only when the order book
    shows no working closing order in that symbol and the position is still
    open after EXIT_CONFIRM_SEC. Any working closing order that shows up in
    the book is cancelled first, and that includes one you placed by hand in
    the same symbol, because the order book is account-wide. The closing
    order is a smart order to a position of zero with quantity 0: the app
    closes whatever position it reads itself, and does nothing if the
    position is already flat by then (closed by hand, or by the broker's own
    square-off), so the exit cannot open the opposite position.
  - Sweep: for SWEEP_SEC after a close with an order answer still unknown,
    every cycle, any closing-side order that appears in that symbol is
    cancelled (with an alert), and an opposite position that appears there (a
    late stop or exit filling after the close) is taken over and closed
    through the same exit path, with an alert. For the rest of the day that
    symbol stays watched: its position is read every cycle, and an opposite
    position found there is taken over and closed the same way. A new trade
    in that symbol on a later Candle 2 does not end the watch: while it is
    open the symbol is managed and not checked, and once it has closed the
    watch carries on, so the earlier lost order filling then is still caught.
    If trades on both sides have closed there with an unknown answer, both
    sides are watched. Only one
    position is managed at a time, so one found while another trade is open
    is reported and blocks entries, and is taken over once that trade has
    closed.
  - Every CHECK_MINUTES the position book is read. Any other open position in
    a configured symbol that this strategy is not managing is reported once
    (log and Telegram) and blocks new entries until it is flat. It is never
    traded.

  What is not guaranteed (two faults at once, or a broker that misreports):
  - An unknown order that stays out of the order book for longer than its
    grace period is taken as never placed. If it then appears, two closing
    orders can exist at once. Inside SWEEP_SEC a late closing order is
    cancelled when seen; after it, it is not. An opposite position it opens is
    still found and closed (above), on the next cycle, at market, and with a
    second round of slippage. That watch only knows the symbols closed today
    with an unknown answer; a position in any other configured symbol is
    reported (within CHECK_MINUTES) and blocks entries, but is not closed.
  - The quantity-0 closing order relies on the app's smart order rule for a
    target of zero (close the whole position it reads; do nothing when flat).
    OpenAlgo's sandbox, its backtester and its broker plugins follow that
    rule; a broker plugin that did not would leave the exit unfilled, which
    is retried and alerted, never reversed.
  - The market close inside the stop's grace period can race the lost stop:
    if both fill, the account holds the opposite position until the sweep
    sees it, usually one cycle (POLL_SECONDS), and closing it costs a second
    fill.
  - A fast move through the stop costs more than the planned loss whenever
    the stop is watched by this script rather than held at the broker: after
    an unknown answer, after a refusal, or while the book or the position
    cannot be read. An SL-M order also fills at whatever price the market
    gives once triggered. A stock that opens beyond its stop is skipped only
    if the price is still beyond it when the entry is tried; if it has come
    back inside by then, it is entered.
  - The take profit is watched on the LTP every POLL_SECONDS and closed at
    market, so it can fill below a long's target (above a short's) on a fast
    reversal.
  - Hourly bars: the bars are read once, ENTRY_OFFSET_MIN after the close. A
    symbol whose bar for that hour is not published by then is skipped for the
    hour (journalled), even if the bar arrives a minute later. Bars are
    expected on the NSE grid that starts at 09:15; a broker whose hourly bars
    start elsewhere (09:00, say) never gives the expected Candle 2, so every
    symbol is skipped as stale every hour and nothing trades: the journal says
    so. A bar the broker returns once its hour has ended is taken as final;
    a broker that later corrects it is not followed.
  - Positions opened by hand in a configured symbol are not closed (outside
    the lost state file case below); a hand-placed closing-side order in a
    symbol this script is closing can be cancelled.

  Lost state file: if state.json cannot be read it is kept aside with a
  timestamp, and every open position at the broker in a configured symbol is
  taken over and closed at market, one after another, after any closing
  order still working in that symbol is cancelled. This includes a position
  you opened by hand in one of those symbols: it cannot be told apart. Their
  P&L is not counted. What was already traded today is no longer known, so
  NO new entry is made for the rest of that day (you get an alert saying so);
  entries start again the next trading day. If state.json reads fine, a
  position this script did not open is left alone and only blocks new entries.

  Symbols: one the app says does not exist is skipped. One that could not be
  checked (OpenAlgo unreachable, broker not logged in) is checked again every
  CHECK_MINUTES and traded only once it is confirmed.

Setup
  - Edit the CONFIG block below, or override any value with a CL1H_* entry in
    .env, for example CL1H_SYMBOLS = 'SBIN,TCS,LT' or CL1H_TP_PCT = '0.8'.
    Times are written HH:MM (14:30). A value that cannot be read (a word where
    a number belongs, a time in another form) is named in the log and the
    strategy does not start.
  - Upload at /python, exchange NSE, schedule 09:15 - 15:20, Mon-Fri.
  - Backtest it first: on the /python page, press Backtest on this
    strategy's card, choose the dates, the capital and the costs, and start
    it. The backtest replays one-minute bars, builds the hourly bars from them
    exactly as the live feed stamps them, and sizes at 2x (see Sizing). The
    CL1H_* values in .env apply to the backtest too. The backtester loads
    about 15 days of bars before the first test day, enough for the 50 hourly
    bars the Supertrend needs.
  - Then run it in analyzer mode (sandbox orders, sized at 1x) before going
    live. The log says which mode is active and the leverage each entry used.
  - Telegram alerts are optional: set CL1H_TELEGRAM_USER to your OpenAlgo
    login name.

Every scan, candidate, skip, entry and exit is written to
candle_liquidity_1hr_data/journal.jsonl beside this file, and the open trade
and the day's record are kept in state.json there, so a restart picks them up
again.
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


# =============================================================================
# CONFIG
# =============================================================================
STRATEGY_NAME = _env("CL1H_STRATEGY_NAME", "Candle Liquidity - 1Hr")  # also the order tag
SYMBOLS = _env_list(
    "CL1H_SYMBOLS",
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
EXCHANGE = os.getenv("OPENALGO_STRATEGY_EXCHANGE") or _env("CL1H_EXCHANGE", "NSE")
PRODUCT = "MIS"  # intraday on both sides; nothing is held overnight
INTERVAL = "1h"  # Candle 1, Candle 2 and the Supertrend are on 1-hour bars
BAR_MINUTES = 60
SESSION_OPEN = "09:15"  # NSE hourly bars are anchored here: 09:15, 10:15 ... 14:15
SESSION_CLOSE = "15:30"  # the last bar, 15:15, is a short one and closes here

# Which sides trade
ENABLE_LONGS = _env("CL1H_ENABLE_LONGS", True)
ENABLE_SHORTS = _env("CL1H_ENABLE_SHORTS", True)

# The setup
TP_PCT = _env("CL1H_TP_PCT", 0.5)  # take profit this % inside Candle 1's high (low for a short)
MIN_RR = _env("CL1H_MIN_RR", 0.0)  # 0 = off; else skip a reward:risk below this
ST_ATR_LEN = _env("CL1H_ST_ATR_LEN", 10)  # Supertrend ATR length
ST_FACTOR = _env("CL1H_ST_FACTOR", 3.0)  # Supertrend factor

# Sizing
CAPITAL = _env("CL1H_CAPITAL", 30000.0)
# MIS intraday leverage: position value <= CAPITAL x this. Used in Live mode and in
# a backtest only; in analyzer mode, or when the mode cannot be read, it is 1.
LEVERAGE = _env("CL1H_LEVERAGE", 2.0)
# Rs at the stop per trade: the quantity is cut so the loss at the stop stays within it.
# 0 = off (then only the position value limits the quantity).
MAX_LOSS_PER_TRADE = _env("CL1H_MAX_LOSS_PER_TRADE", 4000.0)
SIZING_BUFFER_PCT = _env("CL1H_SIZING_BUFFER_PCT", 0.1)  # slippage allowance, % of price

# Daily safety caps
MAX_TRADES_PER_DAY = _env("CL1H_MAX_TRADES_PER_DAY", 0)  # 0 = no limit; one trade open at a time
MAX_DAILY_LOSS = _env("CL1H_MAX_DAILY_LOSS", 0.0)  # no new trades after this loss; 0 = off

# Session (IST). After each hourly bar closes, entries are allowed from
# ENTRY_OFFSET_MIN minutes after the close, for ENTRY_WINDOW_MIN minutes. No entry
# is made at or after LAST_ENTRY or SQUARE_OFF, and a window that would open then
# does not open at all.
ENTRY_OFFSET_MIN = _env("CL1H_ENTRY_OFFSET_MIN", 2)  # time for the broker to publish the bar
ENTRY_WINDOW_MIN = _env("CL1H_ENTRY_WINDOW_MIN", 10)
LAST_ENTRY = _env_time("CL1H_LAST_ENTRY", "14:30")
SQUARE_OFF = _env_time("CL1H_SQUARE_OFF", "15:10")

# Plumbing
HISTORY_DAYS = _env("CL1H_HISTORY_DAYS", 30)  # calendar days of hourly bars read after each close
HISTORY_PAUSE_SEC = 0.3  # between history calls, for the broker's rate limit
POLL_SECONDS = 10
CHECK_MINUTES = 5  # how often positions are checked against the broker
SCAN_RETRY_SEC = 30  # a symbol whose hourly bars could not be read is tried again this often
SCAN_GRACE_SEC = 60  # entries wait this long for every symbol's bars, so the ranking is whole
PRICE_READS = 3  # a candidate whose price cannot be read this many times is dropped
ENTRY_TIMEOUT_SEC = 60  # an entry order not final by then is cancelled

# Order answers that are not clear (no reply, or "may have been placed"): how long
# the order book and position are watched before anything new is sent.
ENTRY_CONFIRM_SEC = _env("CL1H_ENTRY_CONFIRM_SEC", 60)  # a lost entry answer
ENTRY_CONFIRM_READS = 3  # position reads after a fill the position does not show yet
STOP_CONFIRM_SEC = _env("CL1H_STOP_CONFIRM_SEC", 60)  # a lost stop-loss answer
EXIT_CONFIRM_SEC = _env("CL1H_EXIT_CONFIRM_SEC", 45)  # a lost exit answer
EXIT_RETRY_SEC = 10  # an exit that does not fill is replaced after 10, 20, 40 ... s
EXIT_RETRY_MAX_SEC = 300
EXIT_ALERT_AFTER = 3  # alert after this many exit attempts
SWEEP_SEC = _env("CL1H_SWEEP_SEC", 300)  # after such a close: late orders are cancelled
TELEGRAM_USER = _env("CL1H_TELEGRAM_USER", "")  # your OpenAlgo login name, optional

API_KEY = os.getenv("OPENALGO_API_KEY")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL") or (
    f"ws://{os.getenv('WEBSOCKET_HOST', '127.0.0.1')}:{os.getenv('WEBSOCKET_PORT', '8765')}"
)

IST = timezone(timedelta(hours=5, minutes=30))
DATA_DIR = Path(__file__).resolve().parent / "candle_liquidity_1hr_data"


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


BULLISH, BEARISH = "bullish", "bearish"


def pine_supertrend(high, low, close, factor=None, atr_len=None):
    """ta.supertrend(factor, atrPeriod) on a series of bars. Bar i uses bars 0..i only.

    Returns (line, direction) as numpy arrays. `direction` is Pine's own
    number: -1 when the line is the lower band (below the price, BULLISH) and
    +1 when it is the upper band (above the price, BEARISH). While ATR is na
    the direction is +1 and the line is not a usable price (0 on bar 0, na
    after it, as Pine gives); see supertrend_trend() for the readable form.

    A literal port of Pine's built-in:
        src = hl2, atr = ta.atr(atrPeriod)
        upperBand = src + factor * atr, lowerBand = src - factor * atr
        lowerBand := lowerBand > nz(lowerBand[1]) or close[1] < nz(lowerBand[1])
                     ? lowerBand : nz(lowerBand[1])
        upperBand := upperBand < nz(upperBand[1]) or close[1] > nz(upperBand[1])
                     ? upperBand : nz(upperBand[1])
        direction := na(atr[1]) ? 1
                   : superTrend[1] == nz(upperBand[1]) ? (close > upperBand ? -1 : 1)
                   : (close < lowerBand ? 1 : -1)
        superTrend := direction == -1 ? lowerBand : upperBand
    A comparison with na is false, which is what Python gives with nan.
    """
    factor = ST_FACTOR if factor is None else factor
    atr_len = ST_ATR_LEN if atr_len is None else atr_len
    high, low, close = (np.asarray(x, dtype=float) for x in (high, low, close))
    n = len(close)
    atr = pine_atr(high, low, close, atr_len)
    line = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    prev_lower = prev_upper = prev_line = np.nan
    for i in range(n):
        src = (high[i] + low[i]) / 2.0
        upper = src + factor * atr[i]
        lower = src - factor * atr[i]
        was_lower = 0.0 if np.isnan(prev_lower) else prev_lower  # nz()
        was_upper = 0.0 if np.isnan(prev_upper) else prev_upper
        prev_close = close[i - 1] if i else np.nan
        if not (lower > was_lower or prev_close < was_lower):
            lower = was_lower
        if not (upper < was_upper or prev_close > was_upper):
            upper = was_upper
        if i == 0 or np.isnan(atr[i - 1]):
            d = 1
        elif prev_line == was_upper:
            d = -1 if close[i] > upper else 1
        else:
            d = 1 if close[i] < lower else -1
        direction[i] = d
        line[i] = lower if d == -1 else upper
        prev_lower, prev_upper, prev_line = lower, upper, line[i]
    return line, direction


def supertrend_trend(high, low, close, factor=None, atr_len=None):
    """The trend on each bar: "bullish" (line below the price), "bearish" (line
    above it), or None while the Supertrend has no value yet."""
    atr_len = ST_ATR_LEN if atr_len is None else atr_len
    line, direction = pine_supertrend(high, low, close, factor, atr_len)
    out = []
    for i in range(len(line)):
        if i < atr_len or not np.isfinite(line[i]):
            out.append(None)  # ATR, or its previous value, is still na
        else:
            out.append(BULLISH if direction[i] == -1 else BEARISH)
    return out


def min_bars():
    """Completed hourly bars needed before a symbol is evaluated at all."""
    return max(5 * ST_ATR_LEN, ST_ATR_LEN + 2)


# =============================================================================
# HOURLY BARS: completed ones only, and the entry window after each close
# =============================================================================
def on_day(day, minutes):
    """The IST wall-clock time `minutes` after midnight on `day` (a date)."""
    return datetime(day.year, day.month, day.day) + timedelta(minutes=minutes)


def hourly_bar_starts(day):
    """The start of every hourly bar of one session: 09:15, 10:15 ... 15:15."""
    out, minute = [], hhmm(SESSION_OPEN)
    while minute < hhmm(SESSION_CLOSE):
        out.append(on_day(day, minute))
        minute += BAR_MINUTES
    return out


def bar_end(start):
    """When the hourly bar that started at `start` closes: an hour later, or at
    the session close for the short 15:15 bar."""
    close = on_day(start.date(), hhmm(SESSION_CLOSE))
    end = start + timedelta(minutes=BAR_MINUTES)
    return close if start < close < end else end


def entry_window(bar_start):
    """(opens, closes) of the entry window after the bar that started at
    `bar_start` closes, or None when it would not open before LAST_ENTRY and
    SQUARE_OFF. Entries are allowed at opens <= now < closes."""
    day = bar_start.date()
    opens = bar_end(bar_start) + timedelta(minutes=ENTRY_OFFSET_MIN)
    closes = min(
        opens + timedelta(minutes=ENTRY_WINDOW_MIN),
        on_day(day, hhmm(LAST_ENTRY)),
        on_day(day, hhmm(SQUARE_OFF)),
    )
    return (opens, closes) if opens < closes else None


def signal_bars(day):
    """Today's hourly bars that can be a Candle 2: those with an entry window."""
    return [start for start in hourly_bar_starts(day) if entry_window(start)]


def active_bar(now):
    """The start of today's latest hourly bar whose entry window has opened by
    `now`, or None (before the first window opens). Its window may have closed
    already; the bar stays active until the next window opens."""
    best = None
    for start in signal_bars(now.date()):
        if entry_window(start)[0] <= now:
            best = start
    return best


def _to_ist_naive(index):
    """IST wall-clock times with no zone, from whatever the history call gave.

    A zone-aware stamp is converted (the openalgo SDK gives intraday bars as
    IST-aware bar starts, the backtester the same). A zone-less stamp is read
    as IST when every bar then falls inside the session, else as UTC (an epoch
    turned into a time with no zone).
    """
    index = pd.DatetimeIndex(index)
    if index.tz is not None:
        return index.tz_convert("Asia/Kolkata").tz_localize(None)
    if len(index) == 0:
        return index

    def in_session(idx):
        minutes = idx.hour * 60 + idx.minute
        return bool(((minutes >= hhmm(SESSION_OPEN)) & (minutes < hhmm(SESSION_CLOSE))).all())

    if in_session(index):
        return index
    return index + pd.Timedelta(hours=5, minutes=30)


def hourly_frame(df):
    """Hourly bars indexed by their START time, IST wall clock with no zone.

    The openalgo SDK sets the index to the bar's start time; a frame that
    carries a `timestamp` column instead (epoch seconds, or text) is read from
    that column.
    """
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
    df = df.sort_index()[["open", "high", "low", "close"]].astype(float)
    return df[~df.index.duplicated(keep="last")]


def completed_hourly(df, now):
    """Hourly bars that have closed by `now`: a bar counts only when its start +
    60 minutes (the session close for the 15:15 bar) is at or before now. The
    bar still forming is never used."""
    frame = hourly_frame(df)
    if frame.empty:
        return frame
    done = np.array([bar_end(t.to_pydatetime()) <= now for t in frame.index], dtype=bool)
    return frame[done]


# =============================================================================
# THE SETUP
# =============================================================================
@dataclass
class Setup:
    side: str  # BUY or SELL
    ref_entry: float  # Candle 2 close
    stop: float
    target: float
    rr: float  # reward:risk at the reference entry
    trend: str  # Supertrend on Candle 2
    candle1: str  # start of the bar, IST, "YYYY-MM-DD HH:MM"
    candle2: str
    c1_high: float
    c1_low: float
    c1_close: float
    c2_high: float
    c2_low: float
    c2_close: float


def _yes(flag):
    return "yes" if flag else "no"


def find_setup(
    bars,
    tick=0.05,
    longs=None,
    shorts=None,
    tp_pct=None,
    min_rr=None,
    factor=None,
    atr_len=None,
):
    """Look for a setup on COMPLETED hourly bars: the last row is Candle 2, the
    row before it Candle 1. Returns (Setup, "") or (None, why).

    `why` starts with "no setup" when the three conditions are not all true,
    and otherwise names the filter that dropped a setup that did form.
    """
    longs = ENABLE_LONGS if longs is None else longs
    shorts = ENABLE_SHORTS if shorts is None else shorts
    tp_pct = TP_PCT if tp_pct is None else tp_pct
    min_rr = MIN_RR if min_rr is None else min_rr
    if len(bars) < 2:
        return None, "no setup: fewer than two completed hourly bars"
    h, lo, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    h1, l1, c1, h2, l2, c2 = h[-2], lo[-2], c[-2], h[-1], lo[-1], c[-1]
    if not all(np.isfinite(v) and v > 0 for v in (h1, l1, c1, h2, l2, c2)):
        return None, "no setup: the last two hourly bars have a missing price"
    trend = supertrend_trend(h, lo, c, factor, atr_len)[-1]
    if trend is None:
        return None, "no setup: not enough hourly history for the Supertrend"

    if trend == BEARISH:
        side = "BUY"
        checks = (
            ("high below Candle 1 high", h2 < h1),
            ("close below Candle 1 close", c2 < c1),
            ("close above Candle 1 low", c2 > l1),
        )
    else:
        side = "SELL"
        checks = (
            ("low above Candle 1 low", l2 > l1),
            ("close above Candle 1 close", c2 > c1),
            ("close below Candle 1 high", c2 < h1),
        )
    name = "long" if side == "BUY" else "short"
    if not all(flag for _text, flag in checks):
        detail = ", ".join(f"{text}: {_yes(flag)}" for text, flag in checks)
        return None, f"no setup: Supertrend {trend}, so a {name} needs Candle 2 {detail}"
    if not (longs if side == "BUY" else shorts):
        return None, f"a {name} setup formed but {name}s are switched off"

    if side == "BUY":
        stop = floor_tick(l2, tick)  # away from the entry: never tighter than the low
        target = floor_tick(h1 * (1 - tp_pct / 100.0), tick)  # toward the entry
        risk, reward = c2 - stop, target - c2
    else:
        stop = ceil_tick(h2, tick)
        target = ceil_tick(l1 * (1 + tp_pct / 100.0), tick)
        risk, reward = stop - c2, c2 - target
    numbers = f"reference entry {c2:.2f}, stop {stop:.2f}, take profit {target:.2f}"
    if not reward > 0:
        where = "at or below" if side == "BUY" else "at or above"
        return None, f"{name} setup dropped: the take profit is {where} the entry ({numbers})"
    if not risk > 0:
        return None, f"{name} setup dropped: the stop is at the entry ({numbers})"
    rr = reward / risk
    if min_rr > 0 and rr < min_rr:
        return None, (
            f"{name} setup dropped: reward:risk {rr:.2f} is below the minimum {min_rr:g} "
            f"({numbers})"
        )
    return Setup(
        side=side,
        ref_entry=round(float(c2), 4),
        stop=stop,
        target=target,
        rr=round(float(rr), 6),
        trend=trend,
        candle1=f"{bars.index[-2]:%Y-%m-%d %H:%M}",
        candle2=f"{bars.index[-1]:%Y-%m-%d %H:%M}",
        c1_high=float(h1),
        c1_low=float(l1),
        c1_close=float(c1),
        c2_high=float(h2),
        c2_low=float(l2),
        c2_close=float(c2),
    ), ""


def rank_candidates(candidates, order):
    """Highest reward:risk first; equal values keep the order of the symbol list."""
    position = {symbol: n for n, symbol in enumerate(order)}
    return sorted(candidates, key=lambda c: (-c["rr"], position.get(c["symbol"], len(position))))


# =============================================================================
# SIZE AND PLAN
# =============================================================================
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


@dataclass
class Plan:
    symbol: str
    side: str
    entry: float  # the live price the order is sized on
    stop: float
    target: float
    qty: int
    risk: float  # per share, at the live price
    ref_entry: float  # Candle 2 close
    rr: float  # at the reference entry
    sized_by: str
    signal_bar: str  # Candle 2's start, "YYYY-MM-DD HH:MM"
    planned_loss: float  # qty x risk
    position_value: float  # qty x entry
    leverage: float = 1.0  # the leverage this entry was sized with
    mode: str = ""  # the trading mode read before sizing: live, analyzer, backtest, unknown


def plan_entry(symbol, setup, price, lot=1, leverage=None, mode=""):
    """Check a setup against the live price and size it.

    `setup` is a candidate dict (symbol, side, stop, target, ref_entry, rr,
    candle2). `leverage` is the one chosen for the current trading mode
    (mode_leverage()); None means the configured LEVERAGE. Returns (Plan, "")
    or (None, why it is dropped).
    """
    leverage = LEVERAGE if leverage is None else float(leverage)
    side, stop, target = setup["side"], float(setup["stop"]), float(setup["target"])
    if not (price and price > 0):
        return None, "no live price"
    if side == "BUY":
        if price <= stop:
            return None, f"the price {price:.2f} is already at or below the stop {stop:.2f}"
        if price >= target:
            return None, (
                f"the price {price:.2f} is already at or above the take profit {target:.2f}"
            )
    else:
        if price >= stop:
            return None, f"the price {price:.2f} is already at or above the stop {stop:.2f}"
        if price <= target:
            return None, (
                f"the price {price:.2f} is already at or below the take profit {target:.2f}"
            )
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
        target=target,
        qty=qty,
        risk=round(risk, 4),
        ref_entry=float(setup["ref_entry"]),
        rr=float(setup["rr"]),
        sized_by=sized_by,
        signal_bar=str(setup.get("candle2", "")),
        planned_loss=round(qty * risk, 2),
        position_value=round(qty * price, 2),
        leverage=leverage,
        mode=mode,
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
    """Whether a "not scanned" reason is about one symbol rather than the whole hour."""
    return "confirmed only after" in reason


class CandleLiquidityStrategy:
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
        self.scan_slot = None  # which hourly scan the signals below belong to
        self.scan_bar = None  # the Candle 2 that scan expects: a bar start, or None
        self.scan_todo = []  # symbols whose hourly bars are still to be read for it
        self.scan_results = {}  # symbol -> what the hourly scan found
        self.scan_started = None
        self.scan_tried = None
        self.scan_reported = False
        self.scan_failed = set()
        self.scan_waiting = ""
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
            # The setups of the latest hourly bar, best first. Worked out again after
            # every hourly close and after a restart, from completed bars only.
            "candidates": [],
            "signal_bar": None,  # the Candle 2 the candidates belong to
            # "SYMBOL YYYY-MM-DD HH:MM" (symbol and Candle 2) of every entry sent today:
            # a symbol is entered at most once on one Candle 2, even across a restart.
            "taken": [],
            # "SYMBOL YYYY-MM-DD HH:MM" (symbol and Candle 2) -> why that signal was
            # dropped. A new Candle 2 in the same symbol is a new signal.
            "skipped": {},
            # symbol -> its plan, for an entry whose answer was never confirmed either
            # way. Each counts as one of the day's trades, and the symbol is watched.
            "unknown": {},
            "no_entries": "",  # why no new entry is made for the rest of today, if so
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
        sides = " and ".join(
            name for name, on in (("longs", ENABLE_LONGS), ("shorts", ENABLE_SHORTS)) if on
        )
        loss_cap = f"Rs {MAX_LOSS_PER_TRADE:.0f}" if MAX_LOSS_PER_TRADE > 0 else "off"
        log(
            f"hourly 3-candle setup with Supertrend({ST_FACTOR:g}, {ST_ATR_LEN}) on {INTERVAL} bars, "
            f"{sides or 'no side'}, take profit {TP_PCT:g}% inside Candle 1, minimum "
            f"reward:risk {MIN_RR:g} (0 = off)"
        )
        windows = ", ".join(
            f"{w[0]:%H:%M}-{w[1]:%H:%M}"
            for w in (entry_window(b) for b in signal_bars(self.clock().date()))
        )
        log(
            f"intraday {PRODUCT}: entry windows after each hourly close {windows or 'none'}, "
            f"square-off {SQUARE_OFF}, one trade at a time, "
            f"{MAX_TRADES_PER_DAY or 'no limit of'} trades a day, daily loss limit "
            f"{f'Rs {MAX_DAILY_LOSS:.0f}' if MAX_DAILY_LOSS > 0 else 'off'}; "
            f"position value up to Rs {CAPITAL * LEVERAGE:.0f} (Rs {CAPITAL:.0f} x "
            f"{LEVERAGE:g}) in Live mode and in a backtest, Rs {CAPITAL:.0f} (1x) in analyzer "
            f"mode; loss cap per trade {loss_cap}"
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
    def load_hourly(self, symbol, now):
        """Completed hourly bars for `symbol` (the forming bar dropped), or None
        when they could not be read."""
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
        return completed_hourly(bars, now)

    # ---------- main cycle ----------
    def step(self):
        now = self.clock()
        today = now.date().isoformat()
        if self.state["date"] != today:
            self.state = self._fresh_day(today, self.state)
            self._save_state()
        bar = active_bar(now)  # the Candle 2 whose entry window has opened most recently
        scan_key = f"{bar:%Y-%m-%d %H:%M}" if bar else f"{today} none"
        if self.scan_slot != scan_key:
            self.begin_scan(scan_key, bar, now)
        if self.state.get("pending"):
            self.manage_pending(now)
        if self.state.get("open"):
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

    # ---------- the hourly scan ----------
    def begin_scan(self, slot, bar, now):
        """Start the scan for a new Candle 2: the hourly bar that started at `bar`
        and has just closed (None before today's first window opens). The setups
        of the bar before are replaced. It also runs after a restart: the signals
        depend only on completed hourly bars, so they come out the same, and the
        symbols already taken today, or dropped for this bar, are kept from the
        saved state. The bars are read only while this bar's entry window is open,
        once per symbol (again only when a read fails)."""
        self.scan_slot = slot
        self.scan_bar = bar
        window = entry_window(bar) if bar is not None else None
        self.scan_todo = list(self.symbols) if window and now < window[1] else []
        self.scan_results = {}
        self.scan_started = self.scan_tried = None
        self.scan_reported = False
        self.scan_failed = set()  # symbols whose bars could not be read for this Candle 2
        self.scan_waiting = ""  # why the scan is waiting (a trade open), if it is
        self.price_misses = {}
        self.state["candidates"] = []
        self.state["signal_bar"] = f"{bar:%Y-%m-%d %H:%M}" if bar is not None else None
        self._save_state()

    def evaluate_symbol(self, symbol, bars, now):
        """What the completed hourly bars say for one symbol: a dict with `result`
        ("candidate", "dropped", "no setup", "stale" or "no data"), or None to read
        its bars again later."""
        if bars is None:
            return None
        if len(bars) < min_bars():
            return {
                "result": "no data",
                "why": f"only {len(bars)} completed hourly bars; {min_bars()} are needed",
            }
        last = bars.index[-1].to_pydatetime()
        last_bar = f"{last:%Y-%m-%d %H:%M}"
        expected = self.scan_bar
        if expected is None or last != expected:
            want = f"{expected:%Y-%m-%d %H:%M}" if expected is not None else "none yet"
            return {
                "result": "stale",
                "why": (
                    f"its latest completed hourly bar is {last_bar}, not the bar that has just "
                    f"closed ({want}): the broker has not published it yet, or its data is "
                    "stale, so there is no signal for this hour"
                ),
                "last_bar": last_bar,
            }
        tick, _lot = self.tick_lot(symbol)
        setup, why = find_setup(bars, tick)
        if setup is None:
            kind = "no setup" if why.startswith("no setup") else "dropped"
            return {"result": kind, "why": why, "last_bar": last_bar}
        return {"result": "candidate", "why": "", "setup": asdict(setup), "last_bar": last_bar}

    def scan_signals(self, now):
        """Read the completed hourly bars of every symbol not read yet for this
        Candle 2 and note the setups. A symbol whose bars cannot be read is tried
        again every SCAN_RETRY_SEC until this bar's entry window ends.

        Nothing is read while entries are blocked for the rest of the day, and
        the reading waits while a trade is open or an entry is being confirmed
        (only one trade is open at a time), so the open trade's take profit and
        stop are watched without delay. If that trade closes inside the window,
        the hour is scanned then. Every symbol left unread is journalled with
        the reason."""
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
        busy = self.busy_reason()
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
            try:
                found = self.evaluate_symbol(symbol, self.load_hourly(symbol, now), now)
            except Exception as e:
                found = {"result": "no data", "why": f"its hourly bars could not be used ({e})"}
            if found is None:
                self.scan_failed.add(symbol)
                left.append(symbol)
                continue
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
            log(f"{symbol}: no signal this hour, {found['why']}")

    def close_scan(self, why=None):
        """End this Candle 2's scan with the symbols still unread, each with the
        true reason it was not read, and journal the scan."""
        for symbol in self.scan_todo:
            if why:
                reason = why
            elif symbol in self.scan_failed:
                reason = "its hourly bars could not be read before the entry window ended"
            elif self.scan_waiting:
                reason = f"not scanned: {self.scan_waiting} for the whole entry window"
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
    def skip_key(symbol, candle2):
        """A signal is one symbol on one Candle 2."""
        return f"{symbol} {candle2}"

    def add_candidate(self, symbol, setup):
        cand = {"symbol": symbol, **setup, "status": "waiting", "why": ""}
        key = self.skip_key(symbol, setup.get("candle2"))
        if self.is_taken(symbol, setup.get("candle2")):
            cand.update(status="taken", why="an entry was already sent for it on this Candle 2")
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
                log(f"hourly scan of the {bar} bar: no setup in {read} symbols")
            for why in sorted(unread):
                log(f"hourly scan of the {bar} bar: symbols {why}")
            return
        parts = [
            f"{c['symbol']} {'long' if c['side'] == 'BUY' else 'short'} RR {c['rr']:.2f} "
            f"(entry ~{c['ref_entry']:.2f}, SL {c['stop']:.2f}, TP {c['target']:.2f})"
            for c in ranked
        ]
        log(
            f"hourly scan of the {bar} bar: {len(ranked)} setup(s), best first: " + "; ".join(parts)
        )

    # ---------- choosing the entry ----------
    def entry_block(self, now):
        """(why no entry may be placed now, whether that holds for the current
        Candle 2's signals, which are then dropped)."""
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

    def busy_reason(self):
        """Why a trade is active (open, or its entry being confirmed), or ""."""
        s = self.state
        if s.get("open"):
            return f"a {s['open']['symbol']} trade is open"
        if s.get("pending"):
            return f"a {s['pending']['plan']['symbol']} entry is being confirmed"
        return ""

    def is_taken(self, symbol, candle2):
        """Whether an entry was already sent for `symbol` on this Candle 2. A bare
        symbol in the list (a state file written before entries were kept per
        Candle 2) blocks the symbol for the day, the safe reading."""
        taken = self.state.get("taken") or []
        return self.skip_key(symbol, candle2) in taken or symbol in taken

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
        """The current Candle 2's candidates still eligible, best first."""
        return [c for c in self.state.get("candidates") or [] if c.get("status") == "waiting"]

    def drop(self, cand, why, **data):
        """Drop a candidate for the rest of its hour (its Candle 2), with the reason."""
        cand.update(status="skipped", why=why)
        self.state["skipped"][self.skip_key(cand["symbol"], cand.get("candle2"))] = why
        self._save_state()
        log(f"{cand['symbol']} {cand['side']} signal skipped: {why}")
        self.journal(
            "skip", symbol=cand["symbol"], side=cand["side"], why=why, candidate=dict(cand), **data
        )

    def mark_taken(self, symbol, candle2):
        """Record that an entry was sent for `symbol` on this Candle 2: it is never
        entered again on the same signal, even after a restart."""
        key = self.skip_key(symbol, candle2)
        if key not in self.state["taken"]:
            self.state["taken"].append(key)
        for cand in self.state.get("candidates") or []:
            if cand["symbol"] == symbol and cand.get("candle2") == candle2:
                cand.update(status="taken", why="")

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
                    self.drop(cand, why)
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
            if self.is_taken(symbol, cand.get("candle2")):
                cand.update(status="taken", why="an entry was already sent for it on this Candle 2")
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
            _tick, lot = self.tick_lot(symbol)
            if sizing is None:
                sizing = self.read_leverage()
            mode, leverage = sizing
            plan, why = plan_entry(symbol, cand, price, lot, leverage=leverage, mode=mode)
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
                        self.drop(other, late)
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
                f"TP {plan.target:.2f} (rank {rank}, RR {plan.rr:.2f} at the reference entry "
                f"{plan.ref_entry:.2f}, Candle 2 {plan.signal_bar}); position value Rs "
                f"{plan.position_value:.0f} at {leverage:g}x ({mode} mode), loss at the stop "
                f"about Rs {plan.planned_loss:.0f}, sized by {plan.sized_by}"
            )
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
        the symbol is marked as taken on this Candle 2 before anything is sent."""
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
                    and cand.get("candle2") == plan.signal_bar
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
        journal and the alert come after the stop, and neither can stop it."""
        now = self.clock().isoformat(timespec="seconds")
        gapped = fill <= plan.stop if plan.side == "BUY" else fill >= plan.stop
        self.state["trades"] += 1
        trade = {
            "symbol": plan.symbol,
            "side": plan.side,
            "qty": qty,
            "entry_price": fill,
            "stop": plan.stop,
            "target": plan.target,  # fixed at entry, like the stop: neither is ever moved
            "signal_bar": plan.signal_bar,
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
                f"{plan.side} {qty} {plan.symbol} filled @ {fill:.2f}, SL {plan.stop:.2f}, TP "
                f"{plan.target:.2f}; position value Rs {qty * fill:.0f} at {plan.leverage:g}x, "
                f"loss at the stop about Rs {qty * abs(fill - plan.stop):.0f}"
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
        log("CL1H_MAX_TRADES_PER_DAY must be 0 (no limit) or more")
        raise SystemExit(1)
    if min(MAX_DAILY_LOSS, MAX_LOSS_PER_TRADE, MIN_RR, SIZING_BUFFER_PCT) < 0:
        log(
            "CL1H_MAX_DAILY_LOSS, CL1H_MAX_LOSS_PER_TRADE, CL1H_MIN_RR and CL1H_SIZING_BUFFER_PCT "
            "cannot be below 0 (0 switches each one off)"
        )
        raise SystemExit(1)
    if ST_ATR_LEN < 1 or not ST_FACTOR > 0:
        log("CL1H_ST_ATR_LEN must be 1 or more and CL1H_ST_FACTOR above 0")
        raise SystemExit(1)
    if not 0 <= TP_PCT < 100:
        log("CL1H_TP_PCT must be from 0 to below 100")
        raise SystemExit(1)
    if not (CAPITAL > 0 and LEVERAGE > 0):
        log("CL1H_CAPITAL and CL1H_LEVERAGE must be above 0")
        raise SystemExit(1)
    if not (ENABLE_LONGS or ENABLE_SHORTS):
        log("both CL1H_ENABLE_LONGS and CL1H_ENABLE_SHORTS are off; nothing to trade")
        raise SystemExit(1)
    if ENTRY_OFFSET_MIN < 0 or ENTRY_WINDOW_MIN < 1 or ENTRY_OFFSET_MIN + ENTRY_WINDOW_MIN > 60:
        log(
            "CL1H_ENTRY_OFFSET_MIN must be 0 or more and CL1H_ENTRY_WINDOW_MIN 1 or more, and "
            "together they must fit inside the hour (60 minutes at most)"
        )
        raise SystemExit(1)
    if not hhmm(SESSION_OPEN) < hhmm(SQUARE_OFF) <= hhmm(SESSION_CLOSE):
        log(f"CL1H_SQUARE_OFF must be after {SESSION_OPEN} and no later than {SESSION_CLOSE}")
        raise SystemExit(1)
    if not signal_bars(now_ist().date()):
        log(
            "no entry window fits before CL1H_LAST_ENTRY and CL1H_SQUARE_OFF: the first one "
            "opens CL1H_ENTRY_OFFSET_MIN minutes after the 09:15 bar closes at 10:15"
        )
        raise SystemExit(1)
    strategy = CandleLiquidityStrategy(api(api_key=API_KEY, host=HOST, ws_url=WS_URL))
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
