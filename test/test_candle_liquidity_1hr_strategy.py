"""Tests for strategies/examples/candle_liquidity_1hr.py.

The hourly copy of the 3-candle Supertrend reversal. The order-safety code is
the daily strategy's, unchanged, and is covered by
test_three_candle_reversal_strategy.py; this file covers what is new: which
hourly bars count as completed (as the real SDK and the backtester stamp
them), the entry window after each hourly close and the latest-entry cutoff,
a stale Candle 2, leverage chosen by the trading mode, the new defaults, the
setup rules on hourly frames, restarts inside an hour, and a headless run
through the real backtest runner.

CANDLE_LIQUIDITY_SCRIPT_UNDER_TEST points the whole file at another copy of
the strategy, for mutation checks on a scratch copy.

Run:
    uv run pytest test/test_candle_liquidity_1hr_strategy.py -v
"""

import importlib.util
import json
import math
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from openalgo import api as sdk_api

from services.python_backtest.client import BacktestClient
from services.python_backtest.data import SymbolSeries, empty_daily
from services.python_backtest.sessions import date_to_day, session_for
from test.test_python_backtest_e2e import run_backtest, write_bars
from test.test_three_candle_reversal_strategy import (
    LONG_C1,
    LONG_C2,
    SHORT_C1,
    SHORT_C2,
    FakeTime,
    HostileBroker,
    trend_history,
)

REPO = Path(__file__).resolve().parent.parent
SCRIPT = Path(
    os.environ.get("CANDLE_LIQUIDITY_SCRIPT_UNDER_TEST")
    or REPO / "strategies" / "examples" / "candle_liquidity_1hr.py"
)


def _load(name="candle_liquidity_1hr"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cl = _load()

TODAY = date(2026, 11, 4)  # a Wednesday
T = lambda h, m, s=0, day=TODAY: datetime(day.year, day.month, day.day, h, m, s)  # noqa: E731
GRID = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]


# =============================================================================
# HOURLY BARS BUILT BY HAND, stamped as the SDK stamps them
# =============================================================================
def grid_starts(last, n):
    """The n NSE hourly bar starts ending with `last` (a naive IST datetime),
    weekdays only, oldest first."""
    out, day = [], last.date()
    while len(out) < n:
        if day.weekday() < 5:
            for hm in reversed(GRID):
                h, m = map(int, hm.split(":"))
                t = datetime(day.year, day.month, day.day, h, m)
                if t <= last:
                    out.append(t)
                if len(out) == n:
                    break
        day -= timedelta(days=1)
    return out[::-1]


def sdk_index(starts):
    """An IST-aware index of bar starts, named timestamp, as openalgo's history() gives."""
    return pd.DatetimeIndex(starts).tz_localize("Asia/Kolkata").rename("timestamp")


def hourly_bars(up, c1, c2, candle2=None, n=70, forming=True):
    """A trend, then Candle 1 and Candle 2 (whose start is `candle2`), then the
    bar still forming with absurd prices, as the live feed returns it."""
    candle2 = candle2 or T(10, 15)
    o, h, lo, c = trend_history(up, n)
    for bar in (c1, c2):
        for column, value in zip((o, h, lo, c), bar, strict=True):
            column.append(float(value))
    starts = grid_starts(candle2, len(c))
    df = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=sdk_index(starts))
    df["volume"] = 1000
    if forming:
        nxt = candle2 + timedelta(hours=1)
        extra = pd.DataFrame(
            {"open": [1.0], "high": [99999.0], "low": [0.5], "close": [99999.0], "volume": [1]},
            index=sdk_index([nxt]),
        )
        df = pd.concat([df, extra])
    return df


def long_bars(**kwargs):
    return hourly_bars(False, LONG_C1, LONG_C2, **kwargs)  # a downtrend: Supertrend bearish


def short_bars(**kwargs):
    return hourly_bars(True, SHORT_C1, SHORT_C2, **kwargs)  # an uptrend: Supertrend bullish


def flat_bars(candle2=None, n=72):
    starts = grid_starts(candle2 or T(10, 15), n)
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=sdk_index(starts)
    )


# =============================================================================
# DEFAULTS AND SETTINGS
# =============================================================================
def test_the_defaults_are_the_users():
    assert cl.STRATEGY_NAME == "Candle Liquidity - 1Hr"
    assert (cl.TP_PCT, cl.CAPITAL, cl.LEVERAGE) == (0.5, 30000.0, 2.0)
    assert (cl.INTERVAL, cl.HISTORY_DAYS, cl.min_bars()) == ("1h", 30, 50)
    assert (cl.ST_FACTOR, cl.ST_ATR_LEN) == (3.0, 10)
    assert (cl.ENTRY_OFFSET_MIN, cl.ENTRY_WINDOW_MIN, cl.LAST_ENTRY) == (2, 10, "14:30")
    assert (cl.SQUARE_OFF, cl.MAX_TRADES_PER_DAY, cl.MAX_DAILY_LOSS) == ("15:10", 0, 0.0)
    assert cl.MAX_LOSS_PER_TRADE == 4000.0
    assert cl.PRODUCT == "MIS" and cl.DATA_DIR.name == "candle_liquidity_1hr_data"


def test_settings_come_from_cl1h_and_never_from_the_daily_strategys_tcr(monkeypatch):
    monkeypatch.setenv("TCR_TP_PCT", "3")
    monkeypatch.setenv("TCR_CAPITAL", "99999")
    monkeypatch.setenv("CL1H_LEVERAGE", "3")
    monkeypatch.setenv("CL1H_ENTRY_WINDOW_MIN", "5")
    monkeypatch.setenv("CL1H_LAST_ENTRY", "13:00")
    other = _load("cl1h_env")
    assert other.BAD_SETTINGS == []
    assert (other.TP_PCT, other.CAPITAL) == (0.5, 30000.0)
    assert (other.LEVERAGE, other.ENTRY_WINDOW_MIN, other.LAST_ENTRY) == (3.0, 5, "13:00")
    assert "TCR_" not in SCRIPT.read_text(encoding="utf-8").replace(
        "three_candle_reversal_multi.py", ""
    )


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("CL1H_ENTRY_WINDOW_MIN", "0", "CL1H_ENTRY_WINDOW_MIN 1 or more"),
        ("CL1H_ENTRY_OFFSET_MIN", "55", "inside the hour"),
        ("CL1H_LAST_ENTRY", "10:10", "no entry window fits"),
        ("CL1H_SQUARE_OFF", "15:45", "no later than 15:30"),
        ("CL1H_MAX_TRADES_PER_DAY", "-1", "must be 0 (no limit) or more"),
        ("CL1H_MAX_LOSS_PER_TRADE", "-5", "cannot be below 0"),
        ("CL1H_LAST_ENTRY", "9.30am", "a time written as HH:MM"),
    ],
)
def test_settings_that_make_no_sense_stop_the_start(monkeypatch, capsys, name, value, message):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENALGO_API_KEY", "k")
    other = _load("cl1h_bad_env")
    made = []
    monkeypatch.setattr(other, "api", lambda **k: made.append(k))
    with pytest.raises(SystemExit) as stop:
        other.main()
    assert stop.value.code == 1 and made == []
    assert message in capsys.readouterr().out


# =============================================================================
# COMPLETED VERSUS FORMING HOURLY BARS
# =============================================================================
def test_bar_end_is_an_hour_later_and_the_session_close_for_the_short_last_bar():
    assert cl.bar_end(T(9, 15)) == T(10, 15)
    assert cl.bar_end(T(14, 15)) == T(15, 15)
    assert cl.bar_end(T(15, 15)) == T(15, 30)
    assert [f"{t:%H:%M}" for t in cl.hourly_bar_starts(TODAY)] == GRID


def test_the_forming_bar_is_dropped_and_a_bar_counts_at_exactly_its_close():
    starts = [T(9, 15), T(10, 15), T(11, 15)]
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}, sdk_index(starts))
    kept = lambda now: [f"{t:%H:%M}" for t in cl.completed_hourly(df, now).index]  # noqa: E731
    assert kept(T(10, 14, 59)) == []
    assert kept(T(10, 15)) == ["09:15"]
    assert kept(T(11, 20)) == ["09:15", "10:15"]  # 11:15 is still forming
    assert kept(T(12, 15)) == ["09:15", "10:15", "11:15"]


def test_the_short_1515_bar_is_complete_at_1530_not_before():
    df = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
        sdk_index([T(14, 15), T(15, 15)]),
    )
    assert list(cl.completed_hourly(df, T(15, 29, 59)).index) == [pd.Timestamp(T(14, 15))]
    assert list(cl.completed_hourly(df, T(15, 30)).index)[-1] == pd.Timestamp(T(15, 15))


def _sdk_history(rows):
    """The real openalgo SDK's history(), fed the JSON the app answers with."""
    client = sdk_api(api_key="k", host="http://openalgo.test")
    reply = {"status": "success", "data": rows}

    def handler(request):
        return httpx.Response(200, json=reply)

    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    return client.history(
        symbol="SBIN",
        exchange="NSE",
        interval="1h",
        start_date="2026-10-05",
        end_date="2026-11-04",
    )


def test_bars_from_the_real_sdk_are_read_as_ist_bar_starts():
    # The app answers with UTC epoch seconds of each bar's start; 09:15 IST is 03:45 UTC.
    epochs = [
        int(pd.Timestamp(T(h, 15)).tz_localize("Asia/Kolkata").timestamp()) for h in (9, 10, 11)
    ]
    rows = [
        {"timestamp": e, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1}
        for e in epochs
    ]
    df = _sdk_history(rows)
    assert isinstance(df, pd.DataFrame), df
    assert str(df.index.tz) == "Asia/Kolkata" and df.index.name == "timestamp"
    kept = cl.completed_hourly(df, T(11, 20))
    assert [f"{t:%Y-%m-%d %H:%M}" for t in kept.index] == ["2026-11-04 09:15", "2026-11-04 10:15"]


@pytest.mark.parametrize("shape", ["epoch column", "naive utc index", "naive ist index", "text"])
def test_other_timestamp_shapes_give_the_same_bars(shape):
    starts = [T(9, 15), T(10, 15), T(11, 15)]
    aware = pd.DatetimeIndex(starts).tz_localize("Asia/Kolkata")
    values = {"open": [1.0] * 3, "high": [2.0] * 3, "low": [0.5] * 3, "close": [1.5] * 3}
    if shape == "epoch column":
        df = pd.DataFrame({"timestamp": [int(t.timestamp()) for t in aware], **values})
    elif shape == "naive utc index":
        df = pd.DataFrame(values, index=aware.tz_convert("UTC").tz_localize(None))
    elif shape == "naive ist index":
        df = pd.DataFrame(values, index=pd.DatetimeIndex(starts))
    else:
        df = pd.DataFrame({"timestamp": [t.isoformat() for t in aware], **values})
    kept = cl.completed_hourly(df, T(11, 20))
    assert [f"{t:%H:%M}" for t in kept.index] == ["09:15", "10:15"]


def _series(days):
    """The backtester's own SymbolSeries, on synthetic one-minute bars."""
    session = session_for("NSE")
    rows = []
    for d in days:
        t0 = session.open_epoch(date_to_day(d))
        rows.extend((t0 + 60 * i, 100.0, 101.0, 99.0, 100.0 + i / 1000, 10, 0) for i in range(375))
    m1 = pd.DataFrame(rows, columns=["epoch", "open", "high", "low", "close", "volume", "oi"])
    return SymbolSeries("SBIN", "NSE", m1, empty_daily())


def test_the_backtesters_hourly_bars_are_read_the_same_way():
    series = _series([date(2026, 11, 3), TODAY])
    epoch = lambda dt: pd.Timestamp(dt).tz_localize("Asia/Kolkata").timestamp()  # noqa: E731
    now = T(11, 20)
    df = series.intraday(60, date(2026, 11, 3), TODAY, epoch(now))
    assert str(df.index.tz) == "Asia/Kolkata"
    assert df.index[-1] == pd.Timestamp(T(11, 15)).tz_localize("Asia/Kolkata")  # forming
    kept = cl.completed_hourly(df, now)
    assert kept.index[-1] == pd.Timestamp(T(10, 15))
    assert kept.index[-3] == pd.Timestamp(T(15, 15, day=date(2026, 11, 3)))  # the short bar
    # the 15:15 bar, at the close
    late = series.intraday(60, TODAY, TODAY, epoch(T(15, 30)))
    assert cl.completed_hourly(late, T(15, 30)).index[-1] == pd.Timestamp(T(15, 15))
    assert cl.completed_hourly(late, T(15, 29)).index[-1] == pd.Timestamp(T(14, 15))


# =============================================================================
# THE ENTRY WINDOW AFTER EACH HOURLY CLOSE
# =============================================================================
def test_one_window_after_each_close_and_none_at_or_after_the_cutoff():
    windows = {
        f"{b:%H:%M}": (f"{w[0]:%H:%M}", f"{w[1]:%H:%M}")
        for b in cl.hourly_bar_starts(TODAY)
        if (w := cl.entry_window(b))
    }
    assert windows == {
        "09:15": ("10:17", "10:27"),
        "10:15": ("11:17", "11:27"),
        "11:15": ("12:17", "12:27"),
        "12:15": ("13:17", "13:27"),
        "13:15": ("14:17", "14:27"),
    }  # 14:15 closes at 15:15, after the square-off; 15:15 is never a Candle 2
    assert cl.entry_window(T(14, 15)) is None and cl.entry_window(T(15, 15)) is None


def test_the_latest_entry_time_cuts_a_window_short_or_removes_it(monkeypatch):
    monkeypatch.setattr(cl, "LAST_ENTRY", "12:20")
    assert cl.entry_window(T(10, 15)) == (T(11, 17), T(11, 27))
    assert cl.entry_window(T(11, 15)) == (T(12, 17), T(12, 20))
    assert cl.entry_window(T(12, 15)) is None
    monkeypatch.setattr(cl, "LAST_ENTRY", "15:30")  # the square-off still rules
    assert cl.entry_window(T(13, 15)) is not None and cl.entry_window(T(14, 15)) is None


def test_offset_and_length_are_settings(monkeypatch):
    monkeypatch.setattr(cl, "ENTRY_OFFSET_MIN", 5)
    monkeypatch.setattr(cl, "ENTRY_WINDOW_MIN", 3)
    assert cl.entry_window(T(9, 15)) == (T(10, 20), T(10, 23))


@pytest.mark.parametrize(
    "now,expected",
    [
        (T(9, 15), None),
        (T(10, 16, 59), None),  # the first close is 10:15; its window opens at 10:17
        (T(10, 17), T(9, 15)),
        (T(11, 16, 59), T(9, 15)),
        (T(11, 17), T(10, 15)),
        (T(14, 50), T(13, 15)),
        (T(15, 25), T(13, 15)),  # the 14:15 bar has no window
    ],
)
def test_the_active_candle_2_is_todays_bar_that_closed_at_the_latest_usable_close(now, expected):
    assert cl.active_bar(now) == expected


# =============================================================================
# THE SETUP ON HOURLY FRAMES
# =============================================================================
def test_long_and_short_setups_on_hourly_bars_with_the_new_take_profit():
    bars = cl.completed_hourly(long_bars(), T(11, 20))
    setup, why = cl.find_setup(bars)
    assert why == "" and setup.side == "BUY" and setup.trend == cl.BEARISH
    assert (setup.candle1, setup.candle2) == ("2026-11-04 09:15", "2026-11-04 10:15")
    assert setup.stop == 95.0 and setup.target == cl.floor_tick(110 * 0.995, 0.05) == 109.45
    assert setup.rr == pytest.approx((109.45 - 97) / 2)
    bars = cl.completed_hourly(short_bars(), T(11, 20))
    setup, _why = cl.find_setup(bars)
    assert setup.side == "SELL" and setup.stop == 105.0
    assert setup.target == cl.ceil_tick(90 * 1.005, 0.05) == 90.45


def test_the_forming_bar_never_changes_the_setup():
    a = cl.find_setup(cl.completed_hourly(long_bars(), T(11, 20)))[0]
    b = cl.find_setup(cl.completed_hourly(long_bars(forming=False), T(11, 20)))[0]
    assert a == b


def test_candle_1_may_be_the_previous_sessions_last_bar():
    bars = cl.completed_hourly(long_bars(candle2=T(9, 15)), T(10, 20))
    setup, _why = cl.find_setup(bars)
    assert setup.candle1 == "2026-11-03 15:15" and setup.candle2 == "2026-11-04 09:15"


@pytest.mark.parametrize(
    "c2,why",
    [
        ((98.0, 111.0, 95.0, 97.0), "high below Candle 1 high: no"),
        ((98.0, 101.0, 95.0, 99.0), "close below Candle 1 close: no"),
        ((98.0, 101.0, 94.0, 95.5), "close above Candle 1 low: no"),
    ],
)
def test_each_long_condition_is_still_needed(c2, why):
    bars = cl.completed_hourly(hourly_bars(False, LONG_C1, c2), T(11, 20))
    setup, reason = cl.find_setup(bars)
    assert setup is None and why in reason


# =============================================================================
# LEVERAGE BY MODE
# =============================================================================
LIVE = {"status": "success", "data": {"mode": "live", "analyze_mode": False, "total_logs": 0}}
ANALYZE = {"status": "success", "data": {"mode": "analyze", "analyze_mode": True, "total_logs": 3}}


def backtester_answer():
    """BacktestClient.analyzerstatus() itself, without an engine behind it."""
    client = BacktestClient.__new__(BacktestClient)
    client._check = lambda: None
    return client.analyzerstatus()


@pytest.mark.parametrize(
    "answer,backtest,mode,leverage",
    [
        (LIVE, False, "live", 2.0),
        (ANALYZE, False, "analyzer", 1.0),
        ("backtester", False, "backtest", 2.0),  # the backtester's own answer
        (ANALYZE, True, "backtest", 2.0),  # OPENALGO_BACKTEST=1
        (None, True, "backtest", 2.0),
        (None, False, "unknown", 1.0),
        ({"status": "error", "message": "Invalid API key"}, False, "unknown", 1.0),
        ({"status": "success", "data": "live"}, False, "unknown", 1.0),
        ({"status": "success", "data": {}}, False, "unknown", 1.0),
        (
            {"status": "success", "data": {"mode": "analyze", "analyze_mode": False}},
            False,
            "unknown",
            1.0,
        ),  # contradicts itself: the safe side
        (42, False, "unknown", 1.0),
    ],
)
def test_leverage_is_2x_live_and_in_a_backtest_and_1x_otherwise(answer, backtest, mode, leverage):
    if answer == "backtester":
        answer = backtester_answer()
        assert answer["data"] == {"analyze_mode": True, "mode": "backtest", "total_logs": 0}
    assert cl.mode_leverage(answer, backtest=backtest) == (mode, leverage)


def test_the_backtest_flag_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENALGO_BACKTEST", "1")
    assert cl.mode_leverage(None) == ("backtest", 2.0)
    monkeypatch.delenv("OPENALGO_BACKTEST")
    assert cl.mode_leverage(ANALYZE) == ("analyzer", 1.0)
    assert cl.mode_leverage(LIVE) == ("live", 2.0)


def test_the_configured_leverage_is_used_live_but_analyzer_stays_1x(monkeypatch):
    monkeypatch.setattr(cl, "LEVERAGE", 3.0)
    assert cl.mode_leverage(LIVE, backtest=False) == ("live", 3.0)
    assert cl.mode_leverage(ANALYZE, backtest=False) == ("analyzer", 1.0)


# =============================================================================
# THE STRATEGY, HOUR BY HOUR, against a fake broker
# =============================================================================
class HourlyBroker(HostileBroker):
    """The daily tests' broker, answering hourly history and a chosen mode."""

    def __init__(self, symbols):
        super().__init__(symbols)
        self.hourly = {}
        self.mode_answer = ANALYZE
        self.mode_calls = 0

    def history(self, symbol, **k):
        self.history_calls.append((symbol, k.get("interval"), k.get("start_date")))
        if symbol in self.history_fail:
            return {"status": "error", "message": "history failed"}
        return self.hourly.get(symbol, flat_bars())

    def analyzerstatus(self, **k):
        self.mode_calls += 1
        if isinstance(self.mode_answer, Exception):
            raise self.mode_answer
        return self.mode_answer


BASKET = ["AAA", "BBB", "CCC", "DDD"]
PRICES = {"AAA": 97.0, "BBB": 103.0, "CCC": 100.0, "DDD": 97.0}


def basket(candle2=None):
    return {
        "AAA": long_bars(candle2=candle2),
        "BBB": short_bars(candle2=candle2),
        "CCC": flat_bars(candle2=candle2),
        "DDD": long_bars(candle2=candle2),
    }


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setattr(cl, "time", FakeTime())
    monkeypatch.delenv("OPENALGO_BACKTEST", raising=False)
    made = {"n": 0}

    def make(symbols=None, now=None, data_dir=None, broker=None, hourly=None, mode=ANALYZE):
        made["n"] += 1
        symbols = list(symbols or BASKET)
        b = broker or HourlyBroker(symbols)
        if broker is None:
            b.hourly.update(basket() if hourly is None else hourly)
            b.ltps.update(PRICES)
            b.mode_answer = mode
        clock = {"now": now or T(11, 17, 5)}
        s = cl.CandleLiquidityStrategy(
            b,
            symbols=symbols,
            clock=lambda: clock["now"],
            data_dir=data_dir or tmp_path / f"desk{made['n']}",
            pause=0,
        )
        s.test_clock = clock
        s.setup()
        return s, b

    return make


def _journal(s, event=None):
    if not s.journal_path.exists():
        return []
    rows = [json.loads(x) for x in s.journal_path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if event is None or r["event"] == event]


def _at(s, h, m, sec=0):
    s.test_clock["now"] = T(h, m, sec, day=s.test_clock["now"].date())
    return s.test_clock["now"]


def _entries(b):
    return [c for c in b.calls if c[0] == "place" and c[4] == "MARKET"]


def test_the_best_setup_of_the_hour_is_entered_inside_its_window(desk):
    s, b = desk()
    s.step()
    assert s.state["signal_bar"] == "2026-11-04 10:15"
    ranked = [(c["symbol"], c["side"]) for c in s.state["candidates"]]
    assert ranked == [("BBB", "SELL"), ("AAA", "BUY"), ("DDD", "BUY")]  # RR 6.275, 6.225, 6.225
    assert s.state["open"]["symbol"] == "BBB"
    assert s.state["open"]["signal_bar"] == "2026-11-04 10:15"
    assert {r["symbol"] for r in _journal(s, "signal_ignored")} == {"AAA", "DDD"}
    assert {(sym, iv) for sym, iv, _start in b.history_calls} == {(x, "1h") for x in BASKET}
    assert {start for _sym, _iv, start in b.history_calls} == {"2026-10-05"}  # 30 days back


def test_nothing_is_entered_before_the_first_close_on_yesterdays_last_bar(desk):
    yesterday_last = T(15, 15, day=date(2026, 11, 3))
    hourly = {k: long_bars(candle2=yesterday_last, forming=False) for k in BASKET}
    for now in (T(9, 15, 5), T(9, 30), T(10, 16, 50)):
        s, b = desk(now=now, hourly=hourly)
        s.step()
        assert b.history_calls == [] and _entries(b) == [] and s.state["candidates"] == []


def test_entries_happen_only_inside_the_window(desk):
    s, b = desk(now=T(11, 16, 0))  # the 11:15 close; its window opens at 11:17
    s.step()  # the 09:15 bar's window (10:17-10:27) has ended; nothing is read for it
    assert _entries(b) == [] and b.history_calls == []
    _at(s, 11, 26, 59)
    s.step()
    assert len(_entries(b)) == 1  # the last second of the window still enters


def test_the_window_end_drops_the_hours_candidates(desk):
    s, b = desk(mode=ANALYZE)
    b.ltps["BBB"] = 106.0  # the short is beyond its stop; AAA is next
    b.refuse["MARKET"] = {"status": "error", "message": "RMS: margin"}
    s.step()
    _at(s, 11, 27, 0)
    s.step()
    assert s.waiting() == []
    dropped = [r for r in _journal(s, "skip") if "outside the entry window" in r["why"]]
    assert dropped and "11:17-11:27 for the 10:15 bar" in dropped[0]["why"]


def test_a_candle_2_that_is_not_the_bar_that_just_closed_is_skipped_and_journalled(desk):
    hourly = basket()
    hourly["BBB"] = short_bars(candle2=T(9, 15), forming=False)  # 10:15 not published yet
    hourly["AAA"] = long_bars(candle2=T(15, 15, day=date(2026, 11, 3)), forming=False)
    s, b = desk(hourly=hourly)
    s.step()
    stale = {r["symbol"]: r for r in _journal(s, "skip") if "not the bar" in r["why"]}
    assert set(stale) == {"AAA", "BBB"}
    assert stale["BBB"]["last_bar"] == "2026-11-04 09:15"
    assert stale["BBB"]["signal_bar"] == "2026-11-04 10:15"
    assert "not published it yet" in stale["BBB"]["why"]
    assert s.state["open"]["symbol"] == "DDD"  # the only fresh setup
    assert s.scan_results["BBB"]["result"] == "stale"


def test_history_is_read_once_per_hourly_close_not_every_poll(desk):
    s, b = desk(hourly={k: flat_bars() for k in BASKET})
    for sec in range(5, 600, 10):
        _at(s, 11, 17 + sec // 60, sec % 60)
        s.step()
    assert len(b.history_calls) == len(BASKET)
    b.hourly = {k: flat_bars(candle2=T(11, 15)) for k in BASKET}
    for minute in (17, 18, 20, 25):
        _at(s, 12, minute)
        s.step()
    assert len(b.history_calls) == 2 * len(BASKET)
    assert s.state["signal_bar"] == "2026-11-04 11:15"


def test_the_signals_are_worked_out_again_after_each_close(desk, monkeypatch):
    monkeypatch.setattr(cl, "MAX_TRADES_PER_DAY", 3)
    hourly = {"AAA": long_bars(), "CCC": flat_bars()}
    s, b = desk(symbols=["AAA", "CCC"], hourly=hourly)
    b.ltps["AAA"] = 94.0  # below the stop: dropped for this hour only
    s.step()
    assert (
        s.state["open"] is None
        and "at or below the stop" in s.state["skipped"]["AAA 2026-11-04 10:15"]
    )
    b.hourly["AAA"] = long_bars(candle2=T(11, 15))  # a new Candle 2 with the same shape
    b.ltps["AAA"] = 97.0
    _at(s, 12, 17, 5)
    s.step()
    assert s.state["open"]["symbol"] == "AAA"
    assert s.state["open"]["signal_bar"] == "2026-11-04 11:15"


def test_no_daily_limit_a_closed_trade_frees_the_next_signal_in_the_same_window(desk):
    s, b = desk()
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    b.ltps["BBB"] = 90.0  # BBB takes its profit; AAA, next in the window, is entered
    s.step()
    assert [e[1] for e in _entries(b)] == ["BBB", "AAA"]
    b.ltps["AAA"] = 110.0
    s.step()
    assert [e[1] for e in _entries(b)] == ["BBB", "AAA", "DDD"]
    b.ltps["DDD"] = 110.0
    s.step()
    s.step()
    assert len(_entries(b)) == 3 and s.state["trades"] == 3 and s.state["open"] is None
    assert {c["status"] for c in s.state["candidates"]} == {"taken"}  # none twice


def test_while_a_trade_is_open_the_other_signals_wait_and_are_journalled(desk):
    s, b = desk()
    s.step()
    assert s.state["open"]["symbol"] == "BBB" and len(_entries(b)) == 1
    ignored = {r["symbol"] for r in _journal(s, "signal_ignored")}
    assert ignored == {"AAA", "DDD"}
    _at(s, 11, 26, 50)
    s.step()
    assert len(_entries(b)) == 1  # only the open trade holds them back
    _at(s, 11, 27, 0)
    s.step()
    whys = {r["symbol"]: r["why"] for r in _journal(s, "skip")}
    assert "outside the entry window" in whys["AAA"] and "outside the entry window" in whys["DDD"]


def test_a_stopped_out_symbol_is_traded_again_on_a_later_hours_signal(desk):
    s, b = desk(symbols=["BBB", "CCC"], hourly={"BBB": short_bars(), "CCC": flat_bars()})
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    b.ltps["BBB"] = 105.5
    b.trigger_stops()  # the stop at the broker fills
    s.step()
    assert s.state["open"] is None and b.pos["BBB"] == 0 and s.state["pnl"] < 0
    _at(s, 11, 25)
    s.step()
    assert len(_entries(b)) == 1  # never twice on the same Candle 2
    b.hourly["BBB"] = short_bars(candle2=T(11, 15))
    b.ltps["BBB"] = 103.0
    _at(s, 12, 17, 5)
    s.step()
    assert [e[1] for e in _entries(b)] == ["BBB", "BBB"]
    assert s.state["open"]["signal_bar"] == "2026-11-04 11:15"
    assert s.state["taken"] == ["BBB 2026-11-04 10:15", "BBB 2026-11-04 11:15"]


def test_a_restart_never_reenters_a_symbol_on_the_same_candle_2(desk, tmp_path):
    data_dir = tmp_path / "shared"
    s, b = desk(symbols=["BBB", "CCC"], data_dir=data_dir, hourly={"BBB": short_bars()})
    s.step()
    b.ltps["BBB"] = 90.0
    s.step()
    assert s.state["open"] is None and len(_entries(b)) == 1
    b.ltps["BBB"] = 103.0
    again, _b = desk(symbols=["BBB", "CCC"], data_dir=data_dir, broker=b, now=T(11, 20))
    again.step()
    assert len(_entries(b)) == 1
    assert again.state["candidates"][0]["status"] == "taken"
    assert "on this Candle 2" in again.state["candidates"][0]["why"]


def test_a_restart_inside_the_window_keeps_the_open_trade_and_enters_nothing_new(desk, tmp_path):
    data_dir = tmp_path / "shared"
    s, b = desk(data_dir=data_dir)
    s.step()
    b.ltps["BBB"] = 90.0
    s.step()
    assert [e[1] for e in _entries(b)] == ["BBB", "AAA"] and s.state["open"]["symbol"] == "AAA"
    again, _b = desk(data_dir=data_dir, broker=b, now=T(11, 20))
    again.step()
    assert len(_entries(b)) == 2 and again.state["open"]["symbol"] == "AAA"
    b.ltps["AAA"] = 110.0
    again.step()  # AAA closes; the hour is scanned again and only DDD is left
    assert [e[1] for e in _entries(b)] == ["BBB", "AAA", "DDD"]
    status = {c["symbol"]: c["status"] for c in again.state["candidates"]}
    assert status == {"BBB": "taken", "AAA": "taken", "DDD": "taken"}


def test_the_hours_bars_are_read_only_once_the_open_trade_has_closed(desk, capsys):
    s, b = desk(now=T(10, 17, 5), hourly=basket(candle2=T(9, 15)))
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    calls = len(b.history_calls)
    b.hourly = basket(candle2=T(10, 15))
    _at(s, 11, 17, 5)
    s.step()
    _at(s, 11, 19, 0)
    s.step()
    assert len(b.history_calls) == calls  # nothing read while BBB is open
    assert "scan waits: a BBB trade is open" in capsys.readouterr().out
    b.ltps["BBB"] = 90.0
    _at(s, 11, 20, 0)
    s.step()  # BBB closes; the hour is read at once and its best signal taken
    assert len(b.history_calls) == calls + len(BASKET)
    # BBB's new Candle 2 (10:15) is a new signal and is judged on its price, not
    # refused as traded today; at 90 it is past its take profit, so AAA goes.
    bbb = [c for c in s.state["candidates"] if c["symbol"] == "BBB"][0]
    assert bbb["status"] == "skipped" and "take profit" in bbb["why"]
    assert [e[1] for e in _entries(b)] == ["BBB", "AAA"]
    assert s.state["open"]["signal_bar"] == "2026-11-04 10:15"


def test_an_hour_with_the_trade_open_throughout_is_journalled_as_not_scanned(desk):
    s, b = desk(now=T(10, 17, 5), hourly=basket(candle2=T(9, 15)))
    s.step()
    calls = len(b.history_calls)
    b.hourly = basket(candle2=T(10, 15))
    for minute in (17, 20, 26, 27, 28):
        _at(s, 11, minute, 5)
        s.step()
    assert len(b.history_calls) == calls
    scan = [r for r in _journal(s, "scan") if r["signal_bar"] == "2026-11-04 10:15"]
    assert len(scan) == 1
    assert set(scan[0]["results"].values()) == {
        "not scanned: a BBB trade is open for the whole entry window"
    }


def test_a_symbol_confirmed_after_the_window_is_not_called_unreadable(desk, capsys):
    class Late(HourlyBroker):
        """DDD cannot be checked until 11:30, after the 10:15 bar's window."""

        clock = staticmethod(lambda: T(11, 17, 5))  # until the strategy's clock is handed over

        def symbol(self, symbol, **k):
            if symbol == "DDD" and self.clock() < T(11, 30):
                return {"status": "error", "message": "broker not logged in"}
            return super().symbol(symbol, **k)

    b = Late(BASKET)
    b.hourly.update(basket())
    b.ltps.update(PRICES)
    s, _ = desk(broker=b, now=T(11, 17, 5))
    b.clock = s.clock
    s.step()
    assert "DDD" not in s.scan_results
    _at(s, 11, 30, 0)
    s.step()
    assert "DDD" in s.symbols and "DDD" not in s.scan_results and s.scan_todo == []
    assert "could not be read before the entry window ended" not in capsys.readouterr().out


def test_the_trade_limit_still_works_when_set_and_then_nothing_is_read(desk, monkeypatch):
    monkeypatch.setattr(cl, "MAX_TRADES_PER_DAY", 1)
    s, b = desk()
    s.step()
    b.ltps["BBB"] = 90.0
    s.step()
    assert len(_entries(b)) == 1
    skipped = [c for c in s.state["candidates"] if c["status"] == "skipped"]
    assert {c["symbol"] for c in skipped} == {"AAA", "DDD"}
    assert all("limit of 1 trade" in c["why"] for c in skipped)
    calls = len(b.history_calls)
    b.hourly = basket(candle2=T(11, 15))
    _at(s, 12, 17, 5)
    s.step()
    assert len(b.history_calls) == calls and len(_entries(b)) == 1  # not even read
    scan = _journal(s, "scan")[-1]
    assert scan["signal_bar"] == "2026-11-04 11:15"
    assert set(scan["results"].values()) == {"not scanned: the day's limit of 1 trade is reached"}


def test_the_daily_loss_limit_is_off_by_default_and_works_when_set(desk, monkeypatch):
    s, b = desk()
    s.state["pnl"] = -250000.0
    s.step()
    assert s.state["open"]["symbol"] == "BBB"  # off: no limit
    monkeypatch.setattr(cl, "MAX_DAILY_LOSS", 4000.0)
    s2, b2 = desk()
    s2.state["pnl"] = -4000.0
    s2.step()
    assert _entries(b2) == [] and b2.history_calls == []
    scan = _journal(s2, "scan")[-1]
    assert set(scan["results"].values()) == {"not scanned: daily loss limit Rs 4000 reached"}


def test_an_entry_with_no_clear_answer_keeps_that_symbol_out_for_the_day(desk):
    s, b = desk()
    b.ghost.add("MARKET")  # the entry never reaches the broker and the answer is lost
    s.step()
    assert s.state["pending"]["plan"]["symbol"] == "BBB" and _entries(b) == [
        ("place", "BBB", "SELL", 290, "MARKET")
    ]
    b.ghost.clear()
    _at(s, 11, 18, 10)  # past ENTRY_CONFIRM_SEC
    s.step()
    assert "BBB" in s.state["unknown"]
    assert s.state["open"]["symbol"] == "AAA"  # no daily limit: the next one is taken
    b.ltps["AAA"] = 110.0
    s.step()  # AAA takes its profit; DDD, the last of this hour, is entered
    b.ltps["DDD"] = 110.0
    s.step()
    assert [e[1] for e in _entries(b)] == ["BBB", "AAA", "DDD"] and s.state["open"] is None
    b.hourly = basket(candle2=T(11, 15))
    b.ltps.update(PRICES)
    _at(s, 12, 17, 5)
    s.step()
    assert s.state["open"]["symbol"] == "AAA"  # BBB ranks first but stays out today
    bbb = [c for c in s.state["candidates"] if c["symbol"] == "BBB"][0]
    assert bbb["status"] == "skipped" and "no clear answer" in bbb["why"]


def test_safety_blocks_still_stop_entries_with_no_daily_limit(desk, tmp_path):
    b = HourlyBroker(BASKET)
    b.hourly.update(basket())
    b.ltps.update(PRICES)
    b.pos["CCC"] = 7  # a position this strategy does not manage
    s, _ = desk(broker=b)
    s.step()
    assert _entries(b) == [] and "CCC" in s.untracked
    assert s.waiting()  # held, not dropped: it can still go once CCC is flat
    b.pos["CCC"] = 0
    _at(s, 11, 20, 0)
    s.step()
    assert s.state["open"]["symbol"] == "BBB"

    d = tmp_path / "lost"
    d.mkdir()
    (d / "state.json").write_text("{not json")
    s2, b2 = desk(data_dir=d)
    s2.step()
    assert _entries(b2) == [] and b2.history_calls == []
    assert "saved state was lost" in s2.state["no_entries"]


# =============================================================================
# THE RS 4000 CAP ON THE LOSS AT THE STOP
# =============================================================================
def test_the_loss_cap_binds_on_a_wide_stop_and_not_on_a_tight_one():
    assert cl.MAX_LOSS_PER_TRADE == 4000.0
    tight = cl.size_position(1000.0, 950.0, leverage=2.0)  # 60000/1001 = 59; 4000/51 = 78
    assert tight == (59, "position value")
    wide = cl.size_position(1000.0, 900.0, leverage=2.0)  # 4000/101 = 39
    assert wide == (39, "loss cap") and 39 * 100 <= 4000
    analyzer = cl.size_position(1000.0, 900.0, leverage=1.0)  # 30000/1001 = 29 < 39
    assert analyzer == (29, "position value")


def test_the_cap_never_moves_the_stop_and_the_planned_loss_stays_within_it():
    cand = {
        "side": "BUY",
        "stop": 900.0,
        "target": 1200.0,
        "ref_entry": 1000.0,
        "rr": 2.0,
        "candle2": "2026-11-04 10:15",
    }
    plan, why = cl.plan_entry("X", cand, 1000.0, leverage=2.0)
    assert why == "" and plan.stop == 900.0 and plan.qty == 39 and plan.sized_by == "loss cap"
    assert plan.planned_loss == 3900.0
    rng = np.random.default_rng(7)
    for _ in range(2000):
        price = float(rng.uniform(20, 20000))
        stop = round(price * (1 - rng.uniform(0.001, 0.2)), 2)
        lot = int(rng.choice([1, 1, 5, 25]))
        cand.update(stop=stop, target=price * 1.5)
        plan, _why = cl.plan_entry("X", cand, price, lot, leverage=float(rng.choice([1.0, 2.0])))
        if plan is not None:
            assert plan.qty * (price - stop) <= 4000 + 1e-6 and plan.planned_loss <= 4000
            assert plan.qty % lot == 0 and plan.qty * price <= 60000 + 1e-6


def test_a_quantity_of_zero_under_the_cap_is_skipped(desk):
    cand = {"side": "BUY", "stop": 5000.0, "target": 12000.0, "ref_entry": 10000.0, "rr": 1.0}
    plan, why = cl.plan_entry("X", cand, 10000.0, leverage=2.0)
    assert plan is None and "quantity rounds to 0" in why and "Rs 4000 loss cap" in why
    cand.update(stop=95.0, target=120.0, ref_entry=100.0)
    plan, why = cl.plan_entry("X", cand, 100.0, lot=1000, leverage=2.0)
    assert plan is None and "lot 1000" in why


def test_a_signal_whose_capped_quantity_is_zero_is_journalled_and_the_next_taken(desk, monkeypatch):
    monkeypatch.setattr(cl, "MAX_LOSS_PER_TRADE", 1.0)  # BBB's 2-rupee stop is too wide
    s, b = desk(symbols=["BBB", "CCC"], hourly={"BBB": short_bars(), "CCC": flat_bars()})
    s.step()
    assert _entries(b) == []
    skip = [r for r in _journal(s, "skip") if r["symbol"] == "BBB"][0]
    assert "quantity rounds to 0" in skip["why"] and skip["leverage"] == 1.0


def test_a_restart_after_the_window_ended_waits_for_the_next_close(desk, tmp_path):
    s, b = desk(now=T(11, 40))
    s.step()
    assert b.history_calls == [] and _entries(b) == []
    b.hourly = basket(candle2=T(11, 15))
    _at(s, 12, 17, 5)
    s.step()
    assert len(_entries(b)) == 1


@pytest.mark.parametrize(
    "mode,backtest_env,leverage",
    [(LIVE, False, 2.0), (ANALYZE, True, 2.0), (ANALYZE, False, 1.0), (None, False, 1.0)],
)
def test_the_entry_is_sized_by_the_mode_read_just_before_it(
    desk, monkeypatch, capsys, mode, backtest_env, leverage
):
    if backtest_env:
        monkeypatch.setenv("OPENALGO_BACKTEST", "1")
    s, b = desk(mode=ANALYZE)
    b.mode_answer = mode if mode is not None else RuntimeError("down")
    before = b.mode_calls
    s.step()
    assert b.mode_calls == before + 1  # read again at the entry, not taken from setup()
    qty = math.floor(30000 * leverage / (103.0 * 1.001))
    assert _entries(b)[0][3] == qty
    signal = _journal(s, "signal")[0]
    assert signal["leverage"] == leverage and signal["plan"]["leverage"] == leverage
    assert _journal(s, "entry")[0]["leverage"] == leverage
    if mode is None:
        assert "sized at 1x, the safe side" in capsys.readouterr().out


def test_switching_to_analyzer_mode_between_entries_sizes_the_next_at_1x(desk, monkeypatch):
    monkeypatch.setattr(cl, "MAX_TRADES_PER_DAY", 2)
    s, b = desk(mode=LIVE)
    s.step()
    assert _entries(b)[0][3] == math.floor(60000 / (103.0 * 1.001))
    b.mode_answer = ANALYZE  # switched while BBB is open
    b.ltps["BBB"] = 90.0
    s.step()  # BBB takes its profit; AAA, next in the same window, is entered
    assert [e[1] for e in _entries(b)] == ["BBB", "AAA"]
    assert _entries(b)[1][3] == math.floor(30000 / (97.0 * 1.001))  # at 1x
    assert [r["mode"] for r in _journal(s, "entry")] == ["live", "analyzer"]


def test_square_off_and_the_last_entry_time_hold(desk):
    s, b = desk(now=T(14, 17, 5))
    b.hourly = basket(candle2=T(13, 15))
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    _at(s, 15, 10, 0)
    s.step()
    assert s.state["open"] is None and b.pos["BBB"] == 0
    b.hourly = basket(candle2=T(14, 15))
    _at(s, 15, 17, 5)
    s.step()
    assert len(_entries(b)) == 1 and s.state["signal_bar"] == "2026-11-04 13:15"


# =============================================================================
# HEADLESS, THROUGH THE REAL BACKTEST RUNNER
# =============================================================================
@pytest.fixture(scope="module")
def random_market(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cl1h_bars")
    start, end = date(2026, 8, 17), date(2026, 9, 25)
    for n, (symbol, base) in enumerate(
        {"SBIN": 800, "TCS": 3500, "LT": 3600, "INFY": 1500, "ITC": 420}.items()
    ):
        write_bars(directory, symbol, "NSE", start, end, base, 20 + n)
    return directory


def _ist(epoch_ms):
    return datetime.fromtimestamp(epoch_ms / 1000, tz=cl.IST).replace(tzinfo=None)


@pytest.mark.timeout(600)
def test_backtest_runs_headless_and_trades_only_in_the_hourly_windows(tmp_path, random_market):
    original = SCRIPT.read_bytes()
    run_dir, result, log = run_backtest(
        tmp_path,
        SCRIPT,
        random_market,
        "2026-09-07",
        "2026-09-18",
        extra_env={"CL1H_SYMBOLS": "SBIN,TCS,LT,INFY,ITC"},
        timeout=500,
    )
    assert SCRIPT.read_bytes() == original
    assert result["status"] == "completed", result.get("message")
    assert "5 symbols on NSE MIS" in log and "OpenAlgo mode: backtest" in log
    rows = [
        json.loads(x)
        for x in (run_dir / "candle_liquidity_1hr_data" / "journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    scans = [r for r in rows if r["event"] == "scan"]
    assert {r["signal_bar"][11:] for r in scans} <= {"09:15", "10:15", "11:15", "12:15", "13:15"}
    assert len(scans) >= 5 * 9  # every usable close of every session
    trades = sorted(result["trades"], key=lambda t: t["entry_time"])
    assert trades, "no trade in the test period"
    for prev, nxt in zip(trades, trades[1:], strict=False):
        assert nxt["entry_time"] >= prev["exit_time"]  # never two positions
    for t in trades:
        opened, closed = _ist(t["entry_time"]), _ist(t["exit_time"])
        assert t["product"] == "MIS" and closed.date() == opened.date()
        assert f"{closed:%H:%M}" <= "15:11"
        assert 17 <= opened.minute < 27 and f"{opened:%H:%M}" < "14:30", t
        assert t["quantity"] * t["entry_price"] <= 60000 + 1e-6
    entries = [r for r in rows if r["event"] == "entry"]
    assert entries and all(r["leverage"] == 2.0 and r["mode"] == "backtest" for r in entries)
    assert all(r["planned_loss"] <= 4000 for r in entries)
    signals = [(r["trade"]["symbol"], r["plan"]["signal_bar"]) for r in entries]
    assert len(signals) == len(set(signals))  # a symbol at most once on one Candle 2
    for r in entries:
        bar = datetime.strptime(r["plan"]["signal_bar"], "%Y-%m-%d %H:%M")
        assert bar + timedelta(minutes=62) <= datetime.fromisoformat(r["ts"])
        assert datetime.fromisoformat(r["ts"]) < bar + timedelta(minutes=72)


# =============================================================================
# ROUND 3: a watched symbol traded again, and the window judged on the clock now
# =============================================================================
def _watched_desk(desk, watch):
    """BBB was closed earlier today with an order answer still unknown, so it is
    watched for a late closing order (`watch` is that order's side)."""
    s, b = desk(symbols=["BBB", "CCC"], hourly={"BBB": short_bars(), "CCC": flat_bars()})
    s.state["watch"] = {"BBB": watch}
    return s, b


def test_a_watch_survives_a_same_side_reentry_and_closes_the_late_reversal(desk):
    # the earlier BBB short was closed; its lost closing BUY would leave a long
    s, b = _watched_desk(desk, "BUY")
    s.step()
    assert s.state["open"]["symbol"] == "BBB" and s.state["watch"] == {"BBB": "BUY"}
    b.ltps["BBB"] = 90.0
    s.step()
    assert s.state["open"] is None and s.state["watch"] == {"BBB": "BUY"}
    b.pos["BBB"] = 290  # the earlier lost closing BUY fills now
    _at(s, 11, 30)
    s.step()
    assert s.state["open"] and s.state["open"].get("recovered")
    s.step()
    assert b.pos["BBB"] == 0 and s.state["open"] is None and s.untracked == set()


def test_a_watch_survives_an_opposite_side_reentry_without_misreading_it(desk):
    # the earlier BBB long was closed; its lost closing SELL would leave a short,
    # the same side as the new BBB short entered now
    s, b = _watched_desk(desk, "SELL")
    s.step()
    assert s.state["open"]["symbol"] == "BBB" and b.pos["BBB"] == -290
    for minute in (18, 19):  # the new short is managed, never taken for a late reversal
        _at(s, 11, minute)
        s.step()
        assert not s.state["open"].get("recovered") and _smarts_sent(b) == 0
    b.ltps["BBB"] = 90.0
    s.step()  # the new short takes its profit cleanly
    assert s.state["open"] is None and b.pos["BBB"] == 0 and _smarts_sent(b) == 1
    for minute in (21, 24):
        _at(s, 11, minute)
        s.step()
    assert s.state["open"] is None and _smarts_sent(b) == 1  # its own close is not misread
    b.pos["BBB"] = -290  # the earlier lost closing SELL fills now
    _at(s, 11, 30)
    s.step()
    assert s.state["open"] and s.state["open"].get("recovered")
    s.step()
    assert b.pos["BBB"] == 0 and s.state["open"] is None


def test_a_watch_without_a_reentry_still_closes_the_late_reversal(desk):
    s, b = desk(symbols=["BBB", "CCC"], hourly={"BBB": flat_bars(), "CCC": flat_bars()})
    s.state["watch"] = {"BBB": "BUY"}
    s.step()
    b.pos["BBB"] = 290
    _at(s, 11, 30)
    s.step()
    assert s.state["open"] and s.state["open"].get("recovered")


def test_trades_on_both_sides_closed_with_unknown_answers_watch_both_sides(desk):
    s, _b = desk(symbols=["BBB", "CCC"], hourly={"BBB": flat_bars(), "CCC": flat_bars()})
    s.state["watch"] = {"BBB": "BUY"}  # an earlier short, closed with an unknown answer
    s.state["open"] = {
        "symbol": "BBB",
        "side": "BUY",
        "qty": 10,
        "entry_price": 100.0,
        "stop_unknown_since": T(11, 0).isoformat(),
    }
    s.close_book(101.0, "target")  # a later long, also with an unknown answer
    assert s.state["watch"] == {"BBB": ["BUY", "SELL"]}
    assert s.is_reversal("BBB", 5) and s.is_reversal("BBB", -5) and not s.is_reversal("BBB", 0)
    assert cl.watch_actions("SELL") == ["SELL"] and cl.watch_actions(None) == []


def _smarts_sent(b):
    return sum(1 for c in b.calls if c[0] == "smart")


def test_no_entry_after_the_window_ends_during_a_deferred_scan(desk):
    s, b = desk(now=T(10, 17, 5), hourly=basket(candle2=T(9, 15)))
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    b.hourly = basket(candle2=T(10, 15))
    _at(s, 11, 26, 50)
    s.step()  # deferred: BBB is open
    n = len(_entries(b))
    real = b.history

    def slow(symbol, **k):  # each history read takes 5 s
        s.test_clock["now"] += timedelta(seconds=5)
        return real(symbol, **k)

    b.history = slow
    b.ltps["BBB"] = 90.0  # BBB takes its profit in this cycle, then the scan runs
    s.step()
    assert s.test_clock["now"] >= T(11, 27) and _entries(b)[n:] == []
    whys = {c["symbol"]: c["why"] for c in s.state["candidates"] if c["status"] == "skipped"}
    assert "outside the entry window 11:17-11:27" in whys["AAA"]


def test_no_entry_when_the_reads_for_a_candidate_cross_the_window_end(desk):
    s, b = desk(now=T(11, 26, 58))
    real = b.quotes

    def slow(symbol, **k):  # the live price read takes 5 s
        s.test_clock["now"] += timedelta(seconds=5)
        return real(symbol, **k)

    b.quotes = slow
    s.step()
    assert _entries(b) == [] and s.waiting() == []
    assert "outside the entry window" in s.state["skipped"]["BBB 2026-11-04 10:15"]
