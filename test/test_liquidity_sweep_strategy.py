"""Tests for strategies/examples/liquidity_sweep_multi.py.

The live copy of the liquidity sweep research backtester (liquidity_sweep/). The
most important tests come first: PARITY, which runs the research engine and the
strategy's own copy of the signal code on the same synthetic bars and checks
they give the same signals, and the lookahead checks. Then the live mechanics
(bars, entry windows, traded levels kept across restarts and days, stop,
target, time exit, square-off, ranking, leverage, the Rs 4000 cap, settings),
the order-safety shell ported from the daily and hourly strategies' tests, and
a headless run through OpenAlgo's real backtest runner for each timeframe.

All data here is synthetic and made up for the tests. None of it is market data.

LIQUIDITY_SWEEP_SCRIPT_UNDER_TEST points the whole file at another copy of the
strategy, for mutation checks on a scratch copy.

Run:
    uv run python -m pytest test/test_liquidity_sweep_strategy.py -o addopts="" -q
"""

import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
from dataclasses import asdict, fields, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services.python_backtest.client import BacktestClient
from services.python_backtest.data import SymbolSeries, empty_daily
from services.python_backtest.sessions import date_to_day, session_for
from test.test_python_backtest_e2e import run_backtest
from test.test_three_candle_reversal_strategy import (
    SDK_CASES,
    UNANSWERED,
    FakeTime,
    HostileBroker,
    _sdk,
)

REPO = Path(__file__).resolve().parent.parent
SCRIPT = Path(
    os.environ.get("LIQUIDITY_SWEEP_SCRIPT_UNDER_TEST")
    or REPO / "strategies" / "examples" / "liquidity_sweep_multi.py"
)
RESEARCH = REPO / "liquidity_sweep"
TZ = "Asia/Kolkata"


def _load(name="liquidity_sweep_multi"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ls = _load()


def _research():
    """The research backtester's modules, imported from liquidity_sweep/ without
    leaving their short names (config, strategy, ...) in sys.modules."""
    names = ("config", "resample", "strategy", "backtest", "synthetic", "scenarios")
    saved = {k: sys.modules.pop(k) for k in names if k in sys.modules}
    paths = [str(RESEARCH), str(RESEARCH / "tests")]
    sys.path[:0] = paths
    no_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # leave it untouched
    try:
        mods = {k: importlib.import_module(k) for k in names}
    finally:
        sys.dont_write_bytecode = no_bytecode
        for p in paths:
            sys.path.remove(p)
        for k in names:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return SimpleNamespace(**mods)


R = _research()


# =============================================================================
# PARITY: the research engine's signal path against the strategy's replay
# =============================================================================
def research_signals(bars, cfg, bias_source):
    """Run the research engine and record, from inside it, every bar it checked
    for a signal (it checks only while flat) and every signal it turned into a
    pending entry (bias, ATR and the next-bar rules passed), with its bar.

    The engine is not changed: its own pick_signal and _Pending are wrapped for
    the length of the run and read the loop's `t` from the calling frame."""
    checked, pending = set(), {}
    real_pick, real_pending = R.backtest.pick_signal, R.backtest._Pending

    def pick(*args):
        checked.add(inspect.currentframe().f_back.f_locals["t"])
        return real_pick(*args)

    def make(side, stop, kind, price, touches, members):
        t = inspect.currentframe().f_back.f_locals["t"]
        pending[t] = (side, kind, price, touches, tuple(members), stop)
        return real_pending(side, stop, kind, price, touches, members)

    R.backtest.pick_signal, R.backtest._Pending = pick, make
    try:
        trades = R.backtest.run_backtest(bars, cfg, "X", bias_source=bias_source)
    finally:
        R.backtest.pick_signal, R.backtest._Pending = real_pick, real_pending
    return checked, pending, trades


def traded_from(trades, bars):
    """The research engine's filled entries as the strategy keeps them: the
    signal bar (the bar before the entry bar), the side and the level price."""
    pos = {ts: i for i, ts in enumerate(bars.index)}
    rows = []
    for t in trades.itertuples():
        signal = bars.index[pos[t.entry_time] - 1]
        rows.append(
            {
                "symbol": "X",
                "side": "SELL" if t.side == "short" else "BUY",
                "signal_bar": f"{signal:%Y-%m-%d %H:%M}",
                "level": float(t.level),
            }
        )
    return rows


def live_cfg(tf, **kw):
    return ls.config_for_timeframe(tf, **kw)


def research_cfg(tf, **kw):
    return R.config.config_for_timeframe(tf, **kw)


def strategy_bars(data5, tf):
    return data5 if tf == "5m" else R.resample.resample_ohlcv(data5, tf)


def live_signals(bars, cfg, data5, traded_rows):
    daily = R.resample.resample_ohlcv(data5, "D") if cfg.bias_tf == "D" else None
    bias = ls.bias_for(bars, cfg, daily)
    signals, _ = ls.replay(bars, cfg, bias, ls.traded_map(traded_rows))
    return signals


def as_tuple(s):
    return (s.side, s.kind, s.level, s.touches, s.members, s.stop)


VARIANTS = {
    "defaults": {},
    "no bias": {"use_bias": False},
    "two touches": {"min_touches": 2},
    "atr 0.5": {"atr_mult": 0.5},
}


def wave_sessions(sessions, seed, period=60, start="2025-06-02"):
    """Synthetic 5m bars (not market data) that touch the same highs and lows
    again and again: a wave of `period` bars around a slowly moving centre, with
    wicks that sweep past the extremes. The research generator below rarely
    clusters three 1h pivots inside the 21-bar 1h lookback; this one does."""
    rng = np.random.default_rng(seed)
    idx = R.synthetic.session_index(pd.bdate_range(start, periods=sessions))
    n = len(idx)
    k = np.arange(n)
    centre = 1000 * (1 + 0.03 * np.sin(2 * np.pi * k / (75 * 24)))
    close = centre + 5 * np.sin(2 * np.pi * k / period) + rng.normal(0, 0.3, n)
    opens = np.r_[close[0], close[:-1]]
    high = np.maximum(opens, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(opens, close) - np.abs(rng.normal(0, 0.3, n))
    wick = rng.random(n) < 0.15
    high[wick] += rng.uniform(1, 4, wick.sum())
    wick = rng.random(n) < 0.15
    low[wick] -= rng.uniform(1, 4, wick.sum())
    return R.synthetic.frame(idx, opens.round(2), high.round(2), low.round(2), close.round(2))


# The 1h runs use pivot_n 2: with the default 5, three pivots fit inside the
# 21-bar lookback only rarely, so the signal path would hardly be exercised.
BASE = {"5m": {}, "15m": {}, "1h": {"pivot_n": 2}}


@pytest.fixture(scope="module")
def synthetic_5m():
    """5m bars per timeframe under test: the research tests' own mean-reverting
    generator for 5m and 15m, the wave above for 1h (160 sessions, enough for
    the daily EMA(50))."""
    return {
        "5m": R.synthetic.random_sessions(sessions=60, seed=7),
        "15m": R.synthetic.random_sessions(sessions=160, seed=13),
        "1h": wave_sessions(160, 5),
    }


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("tf", ["5m", "15m", "1h"])
def test_parity_the_replay_gives_the_research_engines_signals(synthetic_5m, tf, variant):
    data5 = synthetic_5m[tf]
    kw = {**BASE[tf], **VARIANTS[variant]}
    rcfg, lcfg = research_cfg(tf, **kw), live_cfg(tf, **kw)
    bars = strategy_bars(data5, tf)
    checked, pending, trades = research_signals(bars, rcfg, data5)
    assert len(pending) >= 3, "the synthetic data must give the engine signals to compare"
    rows = traded_from(trades, bars)
    signals = live_signals(bars, lcfg, data5, rows)
    # The research engine checks only while flat and keeps only signals whose
    # next bar is tradable; the replay reports every bar.
    live = {s.t: as_tuple(s) for s in signals if s.t in checked and s.tradable_next}
    assert live == pending
    # Every filled research entry is a signal the replay saw on the same bar.
    times = {s.time: s for s in signals}
    for row in rows:
        s = times[row["signal_bar"]]
        assert ("SELL" if s.side < 0 else "BUY") == row["side"] and s.level == row["level"]


@pytest.mark.parametrize("tf", ["5m", "15m", "1h"])
def test_parity_without_the_traded_levels_the_signals_differ(synthetic_5m, tf):
    """Proves the traded levels matter, so the parity above is not vacuous."""
    data5 = synthetic_5m[tf]
    kw = {**BASE[tf], "min_touches": 2, "use_bias": False}
    bars = strategy_bars(data5, tf)
    checked, pending, trades = research_signals(bars, research_cfg(tf, **kw), data5)
    assert len(trades) > 0
    signals = live_signals(bars, live_cfg(tf, **kw), data5, [])
    live = {s.t: as_tuple(s) for s in signals if s.t in checked and s.tradable_next}
    assert live != pending


def test_parity_the_strategys_config_is_the_research_config():
    names = {f.name for f in fields(ls.Config)} & {f.name for f in fields(R.config.Config)}
    assert {"pivot_n", "level_tol", "level_lookback", "bias_tf", "max_bars"} <= names
    for tf in ("5m", "15m", "1h"):
        mine, theirs = (
            asdict(ls.config_for_timeframe(tf)),
            asdict(R.config.config_for_timeframe(tf)),
        )
        assert {k: mine[k] for k in names} == {k: theirs[k] for k in names}
        assert {k: asdict(ls.signal_config())[k] for k in names} == {
            k: asdict(R.config.config_for_timeframe("5m"))[k] for k in names
        }
    assert [ls.config_for_timeframe(tf).level_lookback for tf in ("5m", "15m", "1h")] == [
        225,
        75,
        21,
    ]
    assert [ls.config_for_timeframe(tf).bias_tf for tf in ("5m", "15m", "1h")] == ["1h", "1h", "D"]


def test_parity_the_copied_building_blocks_agree_on_random_data():
    rng = np.random.default_rng(3)
    c = 100 + rng.normal(0, 1, 500).cumsum()
    h, lo = c + rng.uniform(0, 1, 500), c - rng.uniform(0, 1, 500)
    h[100] = h[101]  # a tie is never a pivot
    for n in (1, 3, 5):
        pairs = zip(ls.find_pivots(h, lo, n), R.strategy.find_pivots(h, lo, n), strict=True)
        for mine, theirs in pairs:
            np.testing.assert_array_equal(mine, theirs)
    np.testing.assert_array_equal(ls.wilder_atr(h, lo, c, 14), R.strategy.wilder_atr(h, lo, c, 14))
    np.testing.assert_array_equal(ls.seeded_ema(c, 50), R.strategy.seeded_ema(c, 50))
    data5 = R.synthetic.random_sessions(sessions=12, seed=5)
    for tf in ("15m", "1h", "D"):
        pd.testing.assert_frame_equal(
            ls.resample_ohlcv(data5, tf), R.resample.resample_ohlcv(data5, tf)
        )
    for tf in ("5m", "1h"):
        cfg_l, cfg_r = ls.config_for_timeframe(tf), R.config.config_for_timeframe(tf)
        pd.testing.assert_frame_equal(
            ls.build_bias_bars(data5, cfg_l), R.strategy.build_bias_bars(data5, cfg_r)
        )
    # the daily path (1h) reads the broker's daily bars and builds the same frame
    cfg_l, cfg_r = ls.config_for_timeframe("1h"), R.config.config_for_timeframe("1h")
    daily = R.resample.resample_ohlcv(data5, "D")
    mine = ls.daily_bias_bars(daily, cfg_l)
    theirs = R.strategy.build_bias_bars(data5, cfg_r)
    np.testing.assert_array_equal(mine.index, theirs.index)
    np.testing.assert_array_equal(mine["available_at"], theirs["available_at"])
    np.testing.assert_array_equal(mine["close"], theirs["close"])
    # and the same frame from daily bars stamped at IST midnight with no zone
    naive = daily.copy()
    naive.index = naive.index.tz_localize(None).normalize()
    np.testing.assert_array_equal(ls.daily_bias_bars(naive, cfg_l).index, theirs.index)


def _prefix_daily(data5, end):
    """Daily bars from the 5m bars that had ended by `end`: today's session is
    incomplete, so it is dropped, as an unfinished daily bar must be."""
    return R.resample.resample_ohlcv(data5[data5.index < end], "D")


@pytest.mark.parametrize("tf", ["5m", "15m", "1h"])
def test_lookahead_a_signal_never_changes_when_later_bars_are_removed(synthetic_5m, tf):
    data5 = synthetic_5m[tf]
    cfg = live_cfg(tf, **{**BASE[tf], "min_touches": 2})
    bars = strategy_bars(data5, tf)
    full_daily = R.resample.resample_ohlcv(data5, "D") if cfg.bias_tf == "D" else None
    signals, _ = ls.replay(bars, cfg, ls.bias_for(bars, cfg, full_daily))
    by_bar = {s.t: s for s in signals}
    assert len(by_bar) >= 5
    step = max(1, len(bars) // 60)
    checks = sorted(set(range(2 * cfg.pivot_n + 2, len(bars), step)) | set(by_bar))
    for t in checks:
        part = bars.iloc[: t + 1]
        end = bars.index[t] + pd.Timedelta(minutes=ls.TF_MINUTES[tf])
        daily = _prefix_daily(data5, end) if cfg.bias_tf == "D" else None
        got, _ = ls.replay(part, cfg, ls.bias_for(part, cfg, daily))
        last = [s for s in got if s.t == t]
        want = by_bar.get(t)
        if want is None:
            assert last == [], f"bar {t} signals only when it is the last bar"
        else:
            assert len(last) == 1
            mine = replace(last[0], tradable_next=want.tradable_next)
            assert mine == want, f"bar {t} differs once later bars are removed"


def test_lookahead_the_carried_level_book_equals_a_full_rebuild(synthetic_5m):
    """Scan after scan, the book carried in memory gives what a rebuild from the
    first bar gives, also when a traded level is added on the latest bar."""
    data5 = synthetic_5m["5m"].iloc[:1100]
    cfg = live_cfg("5m", min_touches=2, use_bias=False)
    traded, state, carried = {}, None, {}
    for size in range(40, len(data5) + 1):
        part = data5.iloc[:size]
        got, state = ls.replay(part, cfg, None, traded, state)
        last = [s for s in got if s.t == size - 1]
        if last:
            carried[size - 1] = last[0]
            if len(traded) < 6 and size % 2:  # some signals are taken, some are not
                traded[last[0].time] = (last[0].kind, last[0].level)
    assert len(traded) == 6 and len(carried) > len(traded)
    full, _ = ls.replay(data5, cfg, None, traded)
    full_by_bar = {s.t: replace(s, tradable_next=None) for s in full}
    assert {t: replace(s, tradable_next=None) for t, s in carried.items()} == full_by_bar


def test_lookahead_a_traded_level_added_for_an_older_bar_rebuilds_the_carried_book(synthetic_5m):
    """An entry whose answer came late is adopted after later bars were already
    replayed; its level must still be killed from its own signal bar on."""
    data5 = synthetic_5m["5m"].iloc[:1100]
    cfg = live_cfg("5m", min_touches=2, use_bias=False)
    full, _ = ls.replay(data5, cfg)
    plain = {s.t: s for s in full}
    # a signal whose kill changes a later signal
    for sig in full:
        traded = {sig.time: (sig.kind, sig.level)}
        killed = {s.t: s for s in ls.replay(data5, cfg, None, traded)[0]}
        later = [
            t for t in set(plain) | set(killed) if t > sig.t + 3 and plain.get(t) != killed.get(t)
        ]
        if later:
            break
    else:
        pytest.fail("no signal in the synthetic data whose level matters later")
    upto = sig.t + 3  # the book has been carried three bars past the signal
    _, state = ls.replay(data5.iloc[: upto + 1], cfg)
    carried, _ = ls.replay(data5, cfg, None, traded, state)
    want = [s for s in killed.values() if s.t > upto]
    assert [replace(s, tradable_next=None) for s in carried if s.t > upto] == [
        replace(s, tradable_next=None) for s in sorted(want, key=lambda s: s.t)
    ]


def test_lookahead_a_changed_bar_rebuilds_the_carried_book(synthetic_5m):
    data5 = synthetic_5m["5m"].iloc[:600].copy()
    cfg = live_cfg("5m", min_touches=2, use_bias=False)
    _, state = ls.replay(data5.iloc[:500], cfg)
    edited = data5.copy()
    edited.iloc[100, edited.columns.get_loc("high")] += 5.0  # the broker corrected a bar
    carried, _ = ls.replay(edited, cfg, None, None, state)
    rebuilt, _ = ls.replay(edited, cfg)
    assert carried == rebuilt and carried and carried[0].t < 500  # replayed from bar 0


# =============================================================================
# DEFAULTS AND SETTINGS
# =============================================================================
BASKET17 = [
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
]


def test_the_defaults_are_the_specs():
    assert ls.STRATEGY_NAME == "Liquidity Sweep" and ls.SYMBOLS == BASKET17
    assert (ls.EXCHANGE, ls.PRODUCT, ls.TIMEFRAME, ls.INTERVAL) == ("NSE", "MIS", "5m", "5m")
    assert (ls.PIVOT_N, ls.LEVEL_TOL, ls.MIN_TOUCHES, ls.USE_BIAS, ls.BIAS_EMA) == (
        5,
        0.0015,
        3,
        True,
        50,
    )
    assert (ls.STOP_BUFFER, ls.ATR_MULT, ls.ATR_PERIOD, ls.RR, ls.MAX_BARS) == (
        0.0005,
        0.0,
        14,
        2.0,
        48,
    )
    assert (ls.CAPITAL, ls.LEVERAGE, ls.MAX_LOSS_PER_TRADE) == (30000.0, 2.0, 4000.0)
    assert (ls.MAX_TRADES_PER_DAY, ls.MAX_DAILY_LOSS) == (0, 0.0)
    assert (ls.LAST_ENTRY, ls.SQUARE_OFF) == ("15:00", "15:15")
    assert (ls.ENTRY_OFFSET_SEC, ls.ENTRY_WINDOW_MIN) == (10, 2)
    assert ls.DATA_DIR.name == "liquidity_sweep_data"
    assert ls.signal_config().level_lookback == 225 and ls.signal_config().bias_tf == "1h"


@pytest.mark.parametrize(
    "tf,lookback,bias,window,days",
    [("5m", 225, "1h", 2, 28), ("15m", 75, "1h", 5, 28), ("1h", 21, "D", 10, 17)],
)
def test_each_timeframe_gets_its_lookback_bias_window_and_history(
    monkeypatch, tf, lookback, bias, window, days
):
    monkeypatch.setenv("LSW_TIMEFRAME", tf)
    other = _load(f"lsw_{tf}")
    assert other.BAD_SETTINGS == [] and other.INTERVAL == tf
    cfg = other.signal_config()
    assert (cfg.level_lookback, cfg.bias_tf, other.ENTRY_WINDOW_MIN) == (lookback, bias, window)
    assert other.history_days() == days
    # enough completed bars for the pivots, the lookback, the ATR and the bias
    sessions = days * 5 / 7 - 5 / 7 * 5
    need = 2 * cfg.pivot_n + 1 + cfg.level_lookback + cfg.atr_period
    assert sessions * other.bars_per_session(tf, cfg) >= need
    if bias == "1h":
        assert sessions * 7 >= 2 * cfg.bias_ema
    else:
        assert other.daily_history_days() == 147 and 147 * 5 / 7 - 5 >= 2 * cfg.bias_ema


def test_settings_come_from_lsw_and_never_from_the_other_strategies(monkeypatch):
    for name, value in {
        "TCR_CAPITAL": "99999",
        "CL1H_LEVERAGE": "5",
        "CL1H_SQUARE_OFF": "14:00",
        "LSW_MIN_TOUCHES": "4",
        "LSW_LEVEL_TOL": "0.002",
        "LSW_RR": "1.5",
        "LSW_MAX_BARS": "30",
        "LSW_ATR_MULT": "0.5",
        "LSW_USE_BIAS": "off",
        "LSW_LEVERAGE": "3",
        "LSW_ENTRY_WINDOW_MIN": "1",
        "LSW_SYMBOLS": "sbin, tcs",
    }.items():
        monkeypatch.setenv(name, value)
    other = _load("lsw_env")
    assert other.BAD_SETTINGS == []
    assert (other.CAPITAL, other.SQUARE_OFF) == (30000.0, "15:15")
    assert (other.MIN_TOUCHES, other.LEVEL_TOL, other.RR, other.MAX_BARS) == (4, 0.002, 1.5, 30)
    assert (other.ATR_MULT, other.USE_BIAS, other.LEVERAGE) == (0.5, False, 3.0)
    assert other.ENTRY_WINDOW_MIN == 1 and other.SYMBOLS == ["SBIN", "TCS"]
    cfg = other.signal_config()
    assert (cfg.min_touches, cfg.level_tol, cfg.rr, cfg.max_bars, cfg.use_bias) == (
        4,
        0.002,
        1.5,
        30,
        False,
    )
    text = SCRIPT.read_text(encoding="utf-8")
    assert "CL1H_" not in text and "TCR_" not in text


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("LSW_TIMEFRAME", "4h", "one of 5m, 15m, 1h"),
        ("LSW_ENTRY_WINDOW_MIN", "3", "first half of the next 5m bar"),
        ("LSW_ENTRY_OFFSET_SEC", "-1", "LSW_ENTRY_OFFSET_SEC must be 0 or more"),
        ("LSW_LAST_ENTRY", "9.30am", "a time written as HH:MM"),
        ("LSW_SQUARE_OFF", "15:45", "no later than 15:30"),
        ("LSW_LAST_ENTRY", "15:20", "LSW_SQUARE_OFF no earlier than"),
        ("LSW_MAX_TRADES_PER_DAY", "-1", "must be 0 (no limit) or more"),
        ("LSW_MAX_LOSS_PER_TRADE", "-5", "cannot be below 0"),
        ("LSW_PIVOT_N", "0", "must each be 1 or more"),
        ("LSW_MIN_TOUCHES", "2.5", "a whole number"),
        ("LSW_LEVEL_TOL", "0", "LSW_LEVEL_TOL must be above 0"),
        ("LSW_RR", "0", "LSW_RR must be above 0"),
        ("LSW_USE_BIAS", "maybe", "true or false"),
        ("LSW_CAPITAL", "lots", "a number"),
    ],
)
def test_settings_that_make_no_sense_stop_the_start(monkeypatch, capsys, name, value, message):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENALGO_API_KEY", "k")
    other = _load("lsw_bad_env")
    made = []
    monkeypatch.setattr(other, "api", lambda **k: made.append(k))
    with pytest.raises(SystemExit) as stop:
        other.main()
    assert stop.value.code == 1 and made == []
    assert message in capsys.readouterr().out


# =============================================================================
# COMPLETED VERSUS FORMING BARS
# =============================================================================
TODAY = date(2026, 11, 4)  # a Wednesday
DAY0 = date(2026, 11, 3)
TOMORROW = date(2026, 11, 5)
T = lambda h, m, s=0, day=TODAY: datetime(day.year, day.month, day.day, h, m, s)  # noqa: E731


def sdk_index(starts):
    """An IST-aware index of bar starts, named timestamp, as openalgo's history() gives."""
    return pd.DatetimeIndex(starts).tz_localize(TZ).rename("timestamp")


def _ohlc(starts):
    n = len(starts)
    return pd.DataFrame(
        {"open": [1.0] * n, "high": [2.0] * n, "low": [0.5] * n, "close": [1.5] * n},
        index=sdk_index(starts),
    )


def test_the_forming_5m_bar_is_dropped_and_a_bar_counts_at_exactly_its_close():
    df = _ohlc([T(10, 10), T(10, 15), T(10, 20)])
    kept = lambda now: [f"{t:%H:%M}" for t in ls.completed_bars(df, now).index]  # noqa: E731
    assert kept(T(10, 19, 59)) == ["10:10"]
    assert kept(T(10, 20)) == ["10:10", "10:15"]  # 10:20 is still forming
    assert kept(T(10, 25)) == ["10:10", "10:15", "10:20"]
    assert str(ls.completed_bars(df, T(10, 25)).index.tz) == TZ


@pytest.mark.parametrize(
    "tf,starts,now,kept",
    [
        ("15m", [(10, 0), (10, 15)], (10, 29, 59), ["10:00"]),
        ("15m", [(10, 0), (10, 15)], (10, 30), ["10:00", "10:15"]),
        ("1h", [(14, 15), (15, 15)], (15, 29, 59), ["14:15"]),  # the short bar closes at 15:30
        ("1h", [(14, 15), (15, 15)], (15, 30), ["14:15", "15:15"]),
    ],
)
def test_the_forming_bar_is_dropped_on_every_timeframe(monkeypatch, tf, starts, now, kept):
    monkeypatch.setenv("LSW_TIMEFRAME", tf)
    other = _load(f"lsw_forming_{tf}")
    df = _ohlc([T(*hm) for hm in starts])
    assert [f"{t:%H:%M}" for t in other.completed_bars(df, T(*now)).index] == kept


def test_bars_outside_the_session_are_left_out():
    df = _ohlc([T(9, 10), T(9, 15), T(15, 25), T(15, 30)])
    assert [f"{t:%H:%M}" for t in ls.completed_bars(df, T(16, 0)).index] == ["09:15", "15:25"]


@pytest.mark.parametrize("shape", ["epoch column", "naive utc index", "naive ist index", "text"])
def test_other_timestamp_shapes_give_the_same_bars(shape):
    starts = [T(10, 10), T(10, 15), T(10, 20)]
    aware = pd.DatetimeIndex(starts).tz_localize(TZ)
    values = {"open": [1.0] * 3, "high": [2.0] * 3, "low": [0.5] * 3, "close": [1.5] * 3}
    if shape == "epoch column":
        df = pd.DataFrame({"timestamp": [int(t.timestamp()) for t in aware], **values})
    elif shape == "naive utc index":
        df = pd.DataFrame(values, index=aware.tz_convert("UTC").tz_localize(None))
    elif shape == "naive ist index":
        df = pd.DataFrame(values, index=pd.DatetimeIndex(starts))
    else:
        df = pd.DataFrame({"timestamp": [t.isoformat() for t in aware], **values})
    kept = ls.completed_bars(df, T(10, 22))
    assert [f"{t:%H:%M}" for t in kept.index] == ["10:10", "10:15"]


def _series(days):
    """The backtester's own SymbolSeries, on synthetic one-minute bars."""
    session = session_for("NSE")
    rows = []
    for d in days:
        t0 = session.open_epoch(date_to_day(d))
        rows.extend((t0 + 60 * i, 100.0, 101.0, 99.0, 100.0 + i / 1000, 10, 0) for i in range(375))
    m1 = pd.DataFrame(rows, columns=["epoch", "open", "high", "low", "close", "volume", "oi"])
    return SymbolSeries("SBIN", "NSE", m1, empty_daily())


def test_the_backtesters_5m_bars_are_read_the_same_way():
    series = _series([DAY0, TODAY])
    epoch = lambda dt: pd.Timestamp(dt).tz_localize(TZ).timestamp()  # noqa: E731
    now = T(10, 20, 15)
    df = series.intraday(5, DAY0, TODAY, epoch(now))
    assert df.index[-1] == pd.Timestamp(T(10, 20)).tz_localize(TZ)  # forming
    kept = ls.completed_bars(df, now)
    assert kept.index[-1] == pd.Timestamp(T(10, 15)).tz_localize(TZ)
    assert len(kept) == 75 + 13  # yesterday's whole session and today's 09:15 ... 10:15
    hourly = ls.resample_ohlcv(kept, "1h")
    assert hourly.index[-1] == pd.Timestamp(T(9, 15)).tz_localize(TZ)  # 10:15 hour not complete


# =============================================================================
# THE ENTRY WINDOW AFTER EACH CLOSE
# =============================================================================
def _windows(module, day=TODAY):
    return {f"{b:%H:%M}": module.entry_window(b) for b in module.bar_starts(day)}


def test_5m_windows_open_10s_after_each_close_and_never_for_a_bar_ending_at_1500():
    w = _windows(ls)
    assert w["09:15"] == (T(9, 20, 10), T(9, 22, 10))
    assert w["14:50"] == (T(14, 55, 10), T(14, 57, 10))  # the entry bar opens 14:55 < 15:00
    assert w["14:55"] is None and w["15:25"] is None  # next bar opens 15:00; next session
    assert sum(1 for v in w.values() if v) == 68
    for start, window in w.items():
        if window:
            bar = datetime.strptime(f"{TODAY} {start}", "%Y-%m-%d %H:%M")
            nxt = ls.bar_end(bar)
            assert nxt < window[0] and window[1] <= nxt + timedelta(minutes=2.5)


@pytest.mark.parametrize(
    "tf,first,last,none,count,length",
    [
        ("15m", "09:15", "14:30", "14:45", 22, 5),
        ("1h", "09:15", "13:15", "14:15", 5, 10),
    ],
)
def test_15m_and_1h_windows(monkeypatch, tf, first, last, none, count, length):
    monkeypatch.setenv("LSW_TIMEFRAME", tf)
    other = _load(f"lsw_windows_{tf}")
    w = _windows(other)
    assert sum(1 for v in w.values() if v) == count
    assert w[none] is None and w[last] is not None and w[first] is not None
    for start, window in w.items():
        if window:
            bar = datetime.strptime(f"{TODAY} {start}", "%Y-%m-%d %H:%M")
            nxt = other.bar_end(bar)
            assert window == (
                nxt + timedelta(seconds=10),
                nxt + timedelta(seconds=10 + 60 * length),
            )
            assert window[1] - nxt <= timedelta(minutes=other.BAR_MINUTES / 2)
    if tf == "1h":
        assert w["15:15"] is None


@pytest.mark.parametrize(
    "now,expected",
    [
        (T(9, 15), None),
        (T(9, 20, 9), None),
        (T(9, 20, 10), T(9, 15)),
        (T(10, 20, 15), T(10, 15)),
        (T(10, 25, 9), T(10, 15)),
        (T(14, 58), T(14, 50)),
        (T(15, 20), T(14, 50)),
    ],
)
def test_the_active_signal_bar(now, expected):
    assert ls.active_bar(now) == expected


def test_the_replay_marks_signals_whose_next_bar_cannot_be_traded():
    """At the data level: the research rule, next bar in the same session and
    opening before 15:00."""
    days = [DAY0, TODAY]
    for when, tradable in (((14, 50), True), ((14, 55), False), ((15, 25), False)):
        df = sweep_frame("SELL", sweeps=[T(*when, day=DAY0)], days=days, pivot_day=DAY0)
        cfg = replace(ls.signal_config(), use_bias=False)
        signals, _ = ls.replay(df, cfg)
        assert [(s.time[11:], s.tradable_next) for s in signals] == [
            (f"{when[0]:02d}:{when[1]:02d}", tradable)
        ]


# =============================================================================
# SCENARIO BARS: a level touched 3 (or more) times, then swept
# =============================================================================
PIVOTS = (10, 25, 40, 55, 65)  # bar numbers inside the first session


def grid(days, minutes=5):
    out = []
    for d in days:
        t = T(9, 15, day=d)
        while t < T(15, 30, day=d):
            out.append(t)
            t += timedelta(minutes=minutes)
    return out


def sweep_frame(
    side="SELL", sweeps=(), touches=3, scale=1.0, days=(DAY0, TODAY, TOMORROW), pivot_day=DAY0
):
    """Synthetic 5m bars, flat at 99 (high 99.2, low 98.8: equal highs are never
    pivots), with `touches` pivot highs at 100 (side SELL) or lows at 98 (BUY)
    in the first session, and a sweep bar at each time in `sweeps`: high 100.3
    and close 99.5 for a short, low 97.7 and close 98.5 for a long. Prices are
    multiplied by `scale`. Not market data."""
    starts = grid(days)
    n = len(starts)
    o, h, lo, c = np.full(n, 99.0), np.full(n, 99.2), np.full(n, 98.8), np.full(n, 99.0)
    first = starts.index(T(9, 15, day=pivot_day))
    for k in PIVOTS[:touches]:
        if side == "SELL":
            h[first + k] = 100.0
        else:
            lo[first + k] = 98.0
    for when in sweeps:
        i = starts.index(when)
        if side == "SELL":
            h[i], c[i] = 100.3, 99.5
        else:
            lo[i], c[i] = 97.7, 98.5
    df = pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": 1000.0}, index=sdk_index(starts)
    )
    df[["open", "high", "low", "close"]] *= scale
    return df


def flat_frame(days=(DAY0, TODAY, TOMORROW)):
    return sweep_frame(touches=0, days=days)


STOP_SHORT = ls.ceil_tick(100.3 * 1.0005, 0.05)  # 100.35015 -> 100.40, away from the entry
STOP_LONG = ls.floor_tick(97.7 * 0.9995, 0.05)  # 97.65115 -> 97.65


def test_scenario_bars_give_the_signal_described():
    cfg = replace(ls.signal_config(), use_bias=False)
    df = ls.completed_bars(sweep_frame("SELL", sweeps=[T(10, 15)]), T(10, 20, 15))
    signals, _ = ls.replay(df, cfg)
    assert len(signals) == 1
    s = signals[0]
    assert (s.time, s.side, s.kind, s.level, s.touches) == ("2026-11-04 10:15", -1, "R", 100.0, 3)
    assert s.stop == pytest.approx(100.3 * 1.0005)
    df = ls.completed_bars(sweep_frame("BUY", sweeps=[T(10, 15)], touches=4), T(10, 20, 15))
    s = ls.replay(df, cfg)[0][0]
    assert (s.side, s.kind, s.level, s.touches) == (1, "S", 98.0, 4)


# =============================================================================
# STOP ROUNDING AND TARGET FROM THE FILL
# =============================================================================
def _sig(side, stop, close, **kw):
    base = {
        "t": 0,
        "time": "2026-11-04 10:15",
        "side": side,
        "kind": "R" if side < 0 else "S",
        "level": 100.0,
        "touches": 3,
        "members": (1, 2, 3),
        "stop": stop,
        "high": 100.3,
        "low": 97.7,
        "close": close,
        "atr": float("nan"),
        "tradable_next": None,
    }
    return ls.Signal(**{**base, **kw})


@pytest.mark.parametrize(
    "side,raw,tick,rounded",
    [
        (-1, 100.35015, 0.05, 100.40),
        (-1, 100.35, 0.05, 100.35),  # already on the tick: unchanged
        (-1, 100.351, 0.10, 100.40),
        (1, 97.65115, 0.05, 97.65),
        (1, 97.649, 0.05, 97.60),
        (1, 97.6, 0.05, 97.60),
    ],
)
def test_the_stop_is_rounded_to_the_tick_away_from_the_entry(side, raw, tick, rounded):
    cand = ls.make_candidate(_sig(side, raw, 99.5 if side < 0 else 98.5), tick)
    assert cand["stop"] == pytest.approx(rounded)
    assert cand["stop"] >= raw - 1e-9 if side < 0 else cand["stop"] <= raw + 1e-9
    assert cand["rr"] == 2.0 and cand["raw_stop"] == pytest.approx(raw)


@pytest.mark.parametrize(
    "side,fill,stop,target",
    [
        ("SELL", 99.40, 100.40, 97.40),
        ("SELL", 99.43, 100.40, 97.50),  # 97.49 rounded toward the entry: up
        ("BUY", 98.60, 97.65, 100.50),
        ("BUY", 98.61, 97.65, 100.50),  # 100.53 rounded toward the entry: down
    ],
)
def test_the_target_is_rr_times_the_risk_from_the_fill(side, fill, stop, target):
    assert ls.target_from(fill, stop, side, 0.05) == pytest.approx(target)


# =============================================================================
# THE STRATEGY, BAR BY BAR, against a fake broker
# =============================================================================
LIVE = {"status": "success", "data": {"mode": "live", "analyze_mode": False, "total_logs": 0}}
ANALYZE = {"status": "success", "data": {"mode": "analyze", "analyze_mode": True, "total_logs": 3}}


class SweepBroker(HostileBroker):
    """The daily tests' broker, answering 5m history up to the strategy's clock
    (the forming bar included, as the live feed does) and a chosen mode."""

    def __init__(self, symbols, clock=None):
        super().__init__(symbols)
        self.frames = {}
        self.daily_frames = {}
        self.clock = clock or (lambda: T(10, 20, 15))
        self.mode_answer = ANALYZE
        self.mode_calls = 0

    def history(self, symbol, **k):
        self.history_calls.append((symbol, k.get("interval"), k.get("start_date")))
        if symbol in self.history_fail:
            return {"status": "error", "message": "history failed"}
        if k.get("interval") == "D":
            if symbol not in self.daily_frames:
                return {"status": "error", "message": "No data available"}
            return self.daily_frames[symbol].copy()
        df = self.frames.get(symbol)
        if df is None:
            df = flat_frame()
        now = pd.Timestamp(self.clock()).tz_localize(TZ)
        start = pd.Timestamp(k["start_date"]).tz_localize(TZ)
        return df[(df.index <= now) & (df.index >= start)].copy()

    def analyzerstatus(self, **k):
        self.mode_calls += 1
        if isinstance(self.mode_answer, Exception):
            raise self.mode_answer
        return self.mode_answer


BASKET = ["AAA", "BBB", "CCC"]


@pytest.fixture
def desk(tmp_path, monkeypatch):
    """A strategy on the 5m timeframe with the bias off (the scenarios are too
    short for an EMA(50) of hourly bars), on a SweepBroker."""
    monkeypatch.setattr(ls, "time", FakeTime())
    monkeypatch.setattr(ls, "USE_BIAS", False)
    monkeypatch.delenv("OPENALGO_BACKTEST", raising=False)
    made = {"n": 0}

    def make(symbols=None, now=None, data_dir=None, broker=None, frames=None, mode=ANALYZE):
        made["n"] += 1
        symbols = list(symbols or BASKET)
        clock = {"now": now or T(10, 20, 15)}
        b = broker or SweepBroker(symbols)
        if broker is None:
            b.frames.update(frames if frames is not None else {})
            b.ltp_value = 99.0
            b.mode_answer = mode
        b.clock = lambda: clock["now"]
        s = ls.LiquiditySweepStrategy(
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


def _at(s, h, m, sec=0, day=None):
    s.test_clock["now"] = T(h, m, sec, day=day or s.test_clock["now"].date())
    return s.test_clock["now"]


def _entries(b):
    return [c for c in b.calls if c[0] == "place" and c[4] == "MARKET"]


def short_basket(**kw):
    """AAA and CCC sweep a 3-touch resistance at 10:15, BBB a 4-touch one; CCC
    trades at twice the price."""
    return {
        "AAA": sweep_frame("SELL", sweeps=[T(10, 15)], **kw),
        "BBB": sweep_frame("SELL", sweeps=[T(10, 15)], touches=4, **kw),
        "CCC": sweep_frame("SELL", sweeps=[T(10, 15)], scale=2.0, **kw),
    }


def test_the_best_signal_is_entered_right_after_the_close(desk):
    s, b = desk(frames=short_basket())
    b.ltps.update(AAA=99.5, BBB=99.5, CCC=199.0)
    s.step()
    assert s.state["signal_bar"] == "2026-11-04 10:15"
    ranked = [(c["symbol"], c["touches"]) for c in s.state["candidates"]]
    assert ranked == [("BBB", 4), ("AAA", 3), ("CCC", 3)]  # touches, then the symbol list
    assert s.state["open"]["symbol"] == "BBB" and _entries(b) == [
        ("place", "BBB", "SELL", 301, "MARKET")  # analyzer mode: 1x
    ]
    stops = b.live("BBB")
    assert [(o["pricetype"], o["trigger"]) for o in stops] == [("SL-M", STOP_SHORT)]
    assert {(sym, iv) for sym, iv, _s in b.history_calls} == {(x, "5m") for x in BASKET}
    # 12 days back with the bias off (the desk); 28 with the hourly bias on (tested above)
    assert ls.history_days() == 12
    assert {start for *_x, start in b.history_calls} == {"2026-10-23"}


def test_the_ranking_is_touches_then_reward_risk_then_the_symbol_list():
    order = ["AAA", "BBB", "CCC", "DDD"]
    cands = [
        {"symbol": "CCC", "touches": 3, "rr": 2.0},
        {"symbol": "AAA", "touches": 3, "rr": 2.0},
        {"symbol": "DDD", "touches": 5, "rr": 2.0},
        {"symbol": "BBB", "touches": 3, "rr": 2.5},
    ]
    assert [c["symbol"] for c in ls.rank_candidates(cands, order)] == ["DDD", "BBB", "AAA", "CCC"]


def test_the_target_comes_from_the_fill_and_the_time_exit_is_set(desk):
    s, b = desk(symbols=["AAA"], frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    b.ltps["AAA"] = 99.4  # the live price the entry is sized on, not the close of 99.5
    real = b.placeorder

    def slipped(symbol, action, quantity, price_type, **k):
        if price_type == "MARKET":
            b.ltps[symbol] = 99.3  # the market order fills 0.10 worse than planned
        return real(symbol, action, quantity, price_type, **k)

    b.placeorder = slipped
    s.step()
    trade = s.state["open"]
    assert _journal(s, "signal")[0]["plan"]["entry"] == 99.4
    assert trade["entry_price"] == 99.3 and trade["stop"] == STOP_SHORT
    assert trade["target"] == pytest.approx(99.3 - 2 * (STOP_SHORT - 99.3))  # 97.10, from the fill
    assert trade["time_exit_at"] == "2026-11-04T14:20:00"  # 10:20 + 48 x 5 minutes
    entry = _journal(s, "entry")[0]
    assert entry["planned_loss"] <= 4000 and entry["trade"]["level"] == 100.0


def test_the_time_exit_closes_the_trade_after_max_bars(desk):
    s, b = desk(symbols=["AAA"], frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    b.ltps["AAA"] = 99.5
    s.step()
    assert s.state["open"]
    _at(s, 14, 19, 55)
    s.step()
    assert s.state["open"] is not None
    _at(s, 14, 20, 0)
    s.step()
    assert s.state["open"] is None and b.pos["AAA"] == 0
    assert _journal(s, "exit")[-1]["reason"] == "time exit after 48 bars"


def test_the_square_off_is_at_1515(desk, monkeypatch):
    monkeypatch.setattr(ls, "MAX_BARS", 200)  # so the time exit cannot come first
    s, b = desk(symbols=["AAA"], frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    b.ltps["AAA"] = 99.5
    s.step()
    _at(s, 15, 14, 55)
    s.step()
    assert s.state["open"] is not None
    _at(s, 15, 15, 0)
    s.step()
    assert s.state["open"] is None and b.pos["AAA"] == 0
    assert _journal(s, "exit")[-1]["reason"] == "square-off time"


def test_the_target_is_taken_on_the_live_price(desk):
    s, b = desk(symbols=["AAA"], frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    b.ltps["AAA"] = 99.5
    s.step()
    target = s.state["open"]["target"]
    b.ltps["AAA"] = target + 0.05
    _at(s, 10, 21, 0)
    s.step()
    assert s.state["open"] is not None
    b.ltps["AAA"] = target
    _at(s, 10, 21, 10)
    s.step()
    assert s.state["open"] is None and _journal(s, "exit")[-1]["reason"] == "target"


def test_no_entry_on_a_bar_opening_at_or_after_1500(desk):
    frames = {"AAA": sweep_frame("SELL", sweeps=[T(14, 50), T(14, 55)])}
    s, b = desk(symbols=["AAA"], now=T(14, 55, 15), frames=frames)
    b.ltps["AAA"] = 99.5
    s.step()
    assert len(_entries(b)) == 1 and s.state["open"]["signal_bar"] == "2026-11-04 14:50"
    b.ltps["AAA"] = 97.0  # its target
    s.step()
    assert s.state["open"] is None
    calls = len(b.history_calls)
    b.ltps["AAA"] = 99.5
    for sec in (5, 15, 30, 70, 130):
        _at(s, 15, 0, 0)
        s.test_clock["now"] += timedelta(seconds=sec)
        s.step()
    assert len(_entries(b)) == 1 and len(b.history_calls) == calls  # 14:55 is never read
    assert s.state["signal_bar"] == "2026-11-04 14:50"


def test_no_entry_on_a_signal_from_the_previous_session(desk):
    frames = {"AAA": sweep_frame("SELL", sweeps=[T(15, 25, day=DAY0)])}
    s, b = desk(symbols=["AAA"], now=T(9, 15, 5), frames=frames)
    b.ltps["AAA"] = 99.5
    for when in (T(9, 15, 5), T(9, 19, 0), T(9, 20, 15), T(9, 21, 0)):
        s.test_clock["now"] = when
        s.step()
    assert _entries(b) == [] and s.state["candidates"] == []
    assert s.scan_results["AAA"]["result"] == "no setup"  # the 09:15 bar itself sweeps nothing


def test_a_bar_not_published_yet_is_read_again_inside_the_window(desk):
    frames = {"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])}
    s, b = desk(symbols=["AAA"], frames=frames)
    b.ltps["AAA"] = 99.5
    full = b.frames["AAA"]
    b.frames["AAA"] = full[full.index < pd.Timestamp(T(10, 15)).tz_localize(TZ)]  # late
    s.step()
    assert _entries(b) == [] and s.scan_todo == ["AAA"]
    b.frames["AAA"] = full
    _at(s, 10, 20, 30)
    s.step()
    assert len(_entries(b)) == 1


def test_a_bar_never_published_in_the_window_is_journalled_as_stale(desk):
    frames = {"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])}
    s, b = desk(symbols=["AAA"], frames=frames)
    full = b.frames["AAA"]
    b.frames["AAA"] = full[full.index < pd.Timestamp(T(10, 15)).tz_localize(TZ)]
    for sec in (15, 35, 55, 75, 95, 115, 135):
        _at(s, 10, 20, 0)
        s.test_clock["now"] += timedelta(seconds=sec)
        s.step()
    stale = [r for r in _journal(s, "skip") if "not the bar that has just closed" in r["why"]]
    assert stale and "read until the window ended" in stale[0]["why"]
    assert _entries(b) == []


def test_one_trade_at_a_time_and_a_signal_held_by_it_is_dropped_with_the_reason(desk):
    frames = short_basket()
    frames["CCC"] = sweep_frame("SELL", sweeps=[T(10, 20)], scale=2.0)  # a later signal
    s, b = desk(frames=frames)
    b.ltps.update(AAA=99.5, BBB=99.5, CCC=199.0)
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    ignored = {r["symbol"] for r in _journal(s, "signal_ignored")}
    assert ignored == {"AAA"}
    _at(s, 10, 22, 10)  # the window ends with BBB still open
    s.step()
    drop = [r for r in _journal(s, "skip") if r["symbol"] == "AAA"][-1]
    assert "outside the entry window" in drop["why"] and "only one trade is open" in drop["why"]
    # the next close: read while BBB is open (its stop is at the broker), CCC noted and dropped
    calls = len(b.history_calls)
    _at(s, 10, 25, 15)
    s.step()
    assert len(b.history_calls) == calls + 3
    assert [c["symbol"] for c in s.state["candidates"]] == ["CCC"]
    assert [r["symbol"] for r in _journal(s, "candidate")][-1] == "CCC"
    _at(s, 10, 27, 10)
    s.step()
    drop = [r for r in _journal(s, "skip") if r["symbol"] == "CCC"][-1]
    assert "a BBB trade was open during the window" in drop["why"]
    assert [e[1] for e in _entries(b)] == ["BBB"]
    # never carried to a later bar
    _at(s, 10, 30, 15)
    s.step()
    assert [e[1] for e in _entries(b)] == ["BBB"]


def test_a_trade_that_closes_inside_the_window_frees_the_next_signal(desk):
    frames = {
        "AAA": sweep_frame("SELL", sweeps=[T(10, 15)]),
        "BBB": sweep_frame("SELL", sweeps=[T(10, 20)]),
    }
    s, b = desk(symbols=["AAA", "BBB"], frames=frames)
    b.ltps.update(AAA=99.5, BBB=99.5)
    s.step()
    assert s.state["open"]["symbol"] == "AAA"
    _at(s, 10, 25, 15)
    s.step()  # BBB is found while AAA is open
    assert s.waiting() and s.state["open"]["symbol"] == "AAA"
    b.ltps["AAA"] = 90.0  # AAA takes its target inside BBB's window
    _at(s, 10, 25, 30)
    s.step()
    assert [e[1] for e in _entries(b)] == ["AAA", "BBB"]


def test_the_scan_waits_while_an_entry_is_being_confirmed_or_the_stop_is_unconfirmed(desk):
    s, b = desk(symbols=["AAA"], frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    s.state["pending"] = {
        "order_id": None,
        "plan": {"symbol": "ZZZ"},
        "placed": T(10, 20).isoformat(),
    }
    assert "ZZZ entry is being confirmed" in s.scan_hold()
    s.state["pending"] = None
    s.state["open"] = {"symbol": "ZZZ", "stop_order_id": None}
    assert "stop is not confirmed at the broker" in s.scan_hold()
    s.state["open"] = {"symbol": "ZZZ", "stop_order_id": "O9", "stop_check": False}
    assert s.scan_hold() == ""


# =============================================================================
# TRADED LEVELS: kept in state.json, applied in the replay, across days
# =============================================================================
def _two_sweeps():
    """AAA sweeps the same resistance today at 10:15 and again tomorrow at 09:30."""
    return {"AAA": sweep_frame("SELL", sweeps=[T(10, 15), T(9, 30, day=TOMORROW)])}


def _tomorrow(desk, data_dir, broker=None):
    s, b = desk(
        symbols=["AAA"],
        now=T(9, 35, 15, day=TOMORROW),
        data_dir=data_dir,
        broker=broker,
        frames=_two_sweeps(),
    )
    b.ltps["AAA"] = 99.5
    s.step()
    return s, b


def test_a_traded_level_is_kept_across_a_restart_and_a_day_and_is_not_traded_again(desk, tmp_path):
    d = tmp_path / "shared"
    s, b = desk(symbols=["AAA"], data_dir=d, frames=_two_sweeps())
    b.ltps["AAA"] = 99.5
    s.step()
    assert s.state["open"]["symbol"] == "AAA"
    assert s.state["traded_levels"] == [
        {
            "symbol": "AAA",
            "side": "SELL",
            "signal_bar": "2026-11-04 10:15",
            "kind": "R",
            "level": 100.0,
            "touches": 3,
        }
    ]
    b.ltps["AAA"] = 90.0
    _at(s, 10, 21)
    s.step()
    assert s.state["open"] is None
    saved = json.loads(s.state_path.read_text())
    assert saved["traded_levels"][0]["signal_bar"] == "2026-11-04 10:15"
    # the next day, a restart: the day's record is new, the traded level is not
    again, b2 = _tomorrow(desk, d)
    assert again.state["date"] == "2026-11-05" and again.state["taken"] == []
    assert again.state["traded_levels"] == saved["traded_levels"]
    assert _entries(b2) == [] and again.state["candidates"] == []
    assert again.scan_results["AAA"]["result"] == "no setup"
    # the day roll inside one run carries it too
    s.test_clock["now"] = T(9, 35, 15, day=TOMORROW)
    s.step()
    assert s.state["traded_levels"] == saved["traded_levels"] and s.state["candidates"] == []


def test_without_the_traded_level_the_second_sweep_is_a_signal(desk, tmp_path):
    again, b = _tomorrow(desk, tmp_path / "fresh")
    assert [c["level"] for c in again.state["candidates"]] == [100.0]
    assert len(_entries(b)) == 1


def test_a_signal_that_was_not_taken_does_not_kill_its_level(desk, tmp_path):
    d = tmp_path / "shared"
    s, b = desk(symbols=["AAA"], data_dir=d, frames=_two_sweeps())
    b.ltps["AAA"] = 101.0  # beyond the stop: dropped, no entry
    s.step()
    assert _entries(b) == [] and s.state["traded_levels"] == []
    again, b2 = _tomorrow(desk, d)
    assert len(_entries(b2)) == 1 and again.state["open"]["signal_bar"] == "2026-11-05 09:30"


def test_a_refused_entry_does_not_kill_its_level(desk):
    s, b = desk(symbols=["AAA"], frames=_two_sweeps())
    b.ltps["AAA"] = 99.5
    b.refuse["MARKET"] = {"status": "error", "message": "RMS: margin"}
    s.step()
    assert s.state["open"] is None and s.state["traded_levels"] == []


def test_old_traded_levels_are_pruned_at_the_day_roll(desk, tmp_path):
    s, _b = desk(symbols=["AAA"])
    old = {"symbol": "AAA", "side": "SELL", "signal_bar": "2026-09-01 10:15", "level": 1.0}
    recent = {"symbol": "AAA", "side": "SELL", "signal_bar": "2026-10-20 10:15", "level": 1.0}
    s.state["traded_levels"] = [old, recent]
    fresh = s._fresh_day("2026-11-05", s.state)
    assert fresh["traded_levels"] == [recent]  # 28 days of history plus a week


# =============================================================================
# LEVERAGE BY MODE AND THE RS 4000 CAP
# =============================================================================
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
        ("backtester", False, "backtest", 2.0),
        (ANALYZE, True, "backtest", 2.0),
        (None, True, "backtest", 2.0),
        (None, False, "unknown", 1.0),
        ({"status": "error", "message": "Invalid API key"}, False, "unknown", 1.0),
        ({"status": "success", "data": {}}, False, "unknown", 1.0),
        (
            {"status": "success", "data": {"mode": "analyze", "analyze_mode": False}},
            False,
            "unknown",
            1.0,
        ),
        (42, False, "unknown", 1.0),
    ],
)
def test_leverage_is_2x_live_and_in_a_backtest_and_1x_otherwise(answer, backtest, mode, leverage):
    if answer == "backtester":
        answer = backtester_answer()
    assert ls.mode_leverage(answer, backtest=backtest) == (mode, leverage)


@pytest.mark.parametrize(
    "mode,backtest_env,leverage",
    [(LIVE, False, 2.0), (ANALYZE, True, 2.0), (ANALYZE, False, 1.0), (None, False, 1.0)],
)
def test_the_entry_is_sized_by_the_mode_read_just_before_it(
    desk, monkeypatch, capsys, mode, backtest_env, leverage
):
    if backtest_env:
        monkeypatch.setenv("OPENALGO_BACKTEST", "1")
    s, b = desk(symbols=["AAA"], frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    b.mode_answer = mode if mode is not None else RuntimeError("down")
    b.ltps["AAA"] = 99.5
    before = b.mode_calls
    s.step()
    assert b.mode_calls == before + 1  # read again at the entry, not taken from setup()
    assert _entries(b)[0][3] == math.floor(30000 * leverage / (99.5 * 1.001))
    signal = _journal(s, "signal")[0]
    assert signal["leverage"] == leverage and signal["plan"]["leverage"] == leverage
    entry = _journal(s, "entry")[0]
    assert entry["leverage"] == leverage and entry["trade"]["leverage"] == leverage
    if mode is None:
        assert "sized at 1x, the safe side" in capsys.readouterr().out


def test_the_loss_cap_binds_on_a_wide_stop_and_not_on_a_tight_one():
    assert ls.size_position(1000.0, 950.0, leverage=2.0) == (59, "position value")
    assert ls.size_position(1000.0, 900.0, leverage=2.0) == (39, "loss cap")
    assert ls.size_position(1000.0, 900.0, leverage=1.0) == (29, "position value")


def test_the_cap_never_moves_the_stop_and_the_planned_loss_stays_within_it():
    cand = {"side": "BUY", "stop": 900.0, "ref_entry": 1000.0, "rr": 2.0, "signal_bar": "x"}
    plan, why = ls.plan_entry("X", cand, 1000.0, leverage=2.0)
    assert why == "" and plan.stop == 900.0 and plan.qty == 39 and plan.planned_loss == 3900.0
    rng = np.random.default_rng(7)
    for _ in range(2000):
        price = float(rng.uniform(20, 20000))
        side = str(rng.choice(["BUY", "SELL"]))
        gap = rng.uniform(0.001, 0.2)
        stop = round(price * (1 - gap if side == "BUY" else 1 + gap), 2)
        lot = int(rng.choice([1, 1, 5, 25]))
        cand.update(side=side, stop=stop)
        plan, _why = ls.plan_entry("X", cand, price, lot, leverage=float(rng.choice([1.0, 2.0])))
        if plan is not None:
            assert plan.qty * abs(price - stop) <= 4000 + 1e-6 and plan.planned_loss <= 4000
            assert plan.qty % lot == 0 and plan.qty * price <= 60000 + 1e-6


def test_a_price_at_or_beyond_the_stop_or_a_zero_quantity_is_skipped():
    short = {"side": "SELL", "stop": 100.4, "ref_entry": 99.5, "rr": 2.0}
    assert "at or above the stop" in ls.plan_entry("X", short, 100.4)[1]
    long = {"side": "BUY", "stop": 97.65, "ref_entry": 98.5, "rr": 2.0}
    assert "at or below the stop" in ls.plan_entry("X", long, 97.6)[1]
    wide = {"side": "BUY", "stop": 5000.0, "ref_entry": 10000.0, "rr": 2.0}
    plan, why = ls.plan_entry("X", wide, 10000.0, leverage=2.0)
    assert plan is None and "quantity rounds to 0" in why and "Rs 4000 loss cap" in why


def test_a_skipped_signal_is_journalled_and_the_next_one_taken(desk):
    s, b = desk(frames=short_basket())
    b.ltps.update(AAA=99.5, BBB=101.0, CCC=199.0)  # BBB, ranked first, is beyond its stop
    s.step()
    skip = [r for r in _journal(s, "skip") if r["symbol"] == "BBB"][0]
    assert "at or above the stop" in skip["why"]
    assert s.state["open"]["symbol"] == "AAA"


# =============================================================================
# RESTARTS
# =============================================================================
def test_a_restart_inside_the_window_never_reenters_on_the_same_signal_bar(desk, tmp_path):
    d = tmp_path / "shared"
    s, b = desk(
        symbols=["AAA"], data_dir=d, frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])}
    )
    b.ltps["AAA"] = 99.5
    s.step()
    b.ltps["AAA"] = 90.0
    s.step()
    assert s.state["open"] is None and len(_entries(b)) == 1
    b.ltps["AAA"] = 99.5
    again, _ = desk(symbols=["AAA"], data_dir=d, broker=b, now=T(10, 21, 0))
    again.step()
    assert len(_entries(b)) == 1
    assert again.state["candidates"][0]["status"] == "taken"


def test_a_restart_keeps_the_open_trade_its_target_and_its_time_exit(desk, tmp_path):
    d = tmp_path / "shared"
    s, b = desk(
        symbols=["AAA"], data_dir=d, frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])}
    )
    b.ltps["AAA"] = 99.5
    s.step()
    trade = dict(s.state["open"])
    again, _ = desk(symbols=["AAA"], data_dir=d, broker=b, now=T(10, 30, 0))
    again.step()
    kept = again.state["open"]
    assert (kept["target"], kept["stop"], kept["time_exit_at"]) == (
        trade["target"],
        trade["stop"],
        trade["time_exit_at"],
    )
    assert len(_entries(b)) == 1 and len(b.live("AAA")) == 1


# =============================================================================
# THE ORDER-SAFETY SHELL, LONG AND SHORT (ported from the daily strategy's tests)
# =============================================================================
SYMS = ["AAA", "BBB", "CCC"]
H_NOW = T(10, 20, 30)


class Side:
    """One side of the market for the order-safety tests: entry 100, stop 2 away,
    so the target from a fill at 100 is 4 away."""

    def __init__(self, action):
        long = action == "BUY"
        self.action = action
        self.close = "SELL" if long else "BUY"
        self.sign = 1 if long else -1
        self.stop, self.target = (98.0, 104.0) if long else (102.0, 96.0)
        self.in_profit = 103.0 if long else 97.0

    def candidate(self, symbol):
        return {
            "symbol": symbol,
            "side": self.action,
            "signal_bar": "2026-11-04 10:15",
            "ref_entry": 100.0,
            "stop": self.stop,
            "level": 100.0,
            "kind": "S" if self.action == "BUY" else "R",
            "touches": 3,
            "rr": 2.0,
            "status": "waiting",
            "why": "",
        }

    def plan(self, symbol, cand, price, lot=1, **k):
        return ls.Plan(
            symbol=symbol,
            side=self.action,
            entry=price,
            stop=self.stop,
            target=self.target,
            qty=10,
            risk=2.0,
            ref_entry=100.0,
            rr=2.0,
            sized_by="position value",
            signal_bar="2026-11-04 10:15",
            planned_loss=20.0,
            position_value=1000.0,
            level=100.0,
            kind="S" if self.action == "BUY" else "R",
            touches=3,
        ), ""


@pytest.fixture(params=["BUY", "SELL"], ids=["long", "short"])
def hostile(request, tmp_path, monkeypatch):
    side = Side(request.param)
    fake_time = FakeTime()
    monkeypatch.setattr(ls, "time", fake_time)
    monkeypatch.setattr(ls, "plan_entry", side.plan)
    # a long window, so whenever an entry is refused below it is the safety layer
    # refusing it, not the clock
    monkeypatch.setattr(ls, "ENTRY_WINDOW_MIN", 120)
    monkeypatch.delenv("OPENALGO_BACKTEST", raising=False)
    made = {"n": 0}

    def make(data_dir=None, broker=None):
        made["n"] += 1
        clock = {"now": H_NOW}
        b = broker or SweepBroker(SYMS)
        b.clock = lambda: clock["now"]
        s = ls.LiquiditySweepStrategy(
            b,
            symbols=SYMS,
            clock=lambda: clock["now"],
            data_dir=data_dir or tmp_path / f"run{made['n']}",
            pause=0,
        )
        s.test_clock = clock
        s.side = side
        s.setup()
        s.begin_scan("2026-11-04 10:15", T(10, 15), H_NOW)  # nothing found by the scan
        s.scan_slot = "2026-11-04 10:15"
        s.scan_todo, s.scan_reported = [], True
        s.last_check = H_NOW.replace(second=0)
        return s, b

    make.time = fake_time
    make.side = side
    return make


def _signal(s, *symbols):
    """A signal appears in each symbol; the strategy checks positions and tries to
    enter. Whatever is not entered is forgotten, as a blocked signal is."""
    for symbol in symbols:
        kept = [c for c in s.state["candidates"] if c["symbol"] != symbol]
        s.state["candidates"] = kept + [s.side.candidate(symbol)]
    s.periodic()
    s.try_enter(s.test_clock["now"])
    s.state["candidates"] = [c for c in s.state["candidates"] if c["status"] != "waiting"]


def _advance(s, seconds):
    s.test_clock["now"] += timedelta(seconds=seconds)
    return s.test_clock["now"]


def _placed(b):
    return [c for c in b.calls if c[0] == "place"]


def _smarts(b):
    return sum(1 for c in b.calls if c[0] == "smart")


def _hide(b, hidden_type):
    real_place = b.placeorder

    def place_hidden(symbol, action, quantity, price_type, **k):
        try:
            return real_place(symbol, action, quantity, price_type, **k)
        finally:
            if price_type == hidden_type:
                b.hidden.add(f"O{b.n}")

    b.placeorder = place_hidden
    return real_place


def test_shell_entry_places_the_position_and_one_stop_on_the_closing_side(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    assert b.pos["AAA"] == 10 * side.sign
    assert _placed(b) == [
        ("place", "AAA", side.action, 10, "MARKET"),
        ("place", "AAA", side.close, 10, "SL-M"),
    ]
    assert b.live("AAA")[0]["trigger"] == side.stop
    assert s.state["open"]["target"] == side.target  # 2 x the risk from the fill at 100
    assert s.state["taken"] == ["AAA 2026-11-04 10:15"]


def test_shell_only_one_trade_across_all_symbols(hostile):
    s, b = hostile()
    _signal(s, "AAA", "BBB")
    assert s.state["open"]["symbol"] == "AAA" and b.pos["BBB"] == 0
    ignored = _journal(s, "signal_ignored")
    assert [r["symbol"] for r in ignored] == ["BBB"] and "AAA ranks higher" in ignored[0]["why"]
    _signal(s, "CCC")
    assert b.pos["CCC"] == 0
    assert "AAA trade is already open" in s.can_enter(s.test_clock["now"])[1]


def test_shell_a_pending_entry_blocks_every_other_symbol(hostile):
    s, _b = hostile()
    s.state["pending"] = {"order_id": None, "plan": {"symbol": "AAA"}, "placed": "x"}
    ok, why = s.can_enter(s.test_clock["now"])
    assert not ok and "AAA entry is being confirmed" in why


def test_shell_the_target_is_taken_at_market_after_the_stop_is_cancelled(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    stop_id = s.state["open"]["stop_order_id"]
    b.ltp_value = side.in_profit
    s.manage_open(_advance(s, 10))
    assert s.state["open"] and _smarts(b) == 0
    b.ltp_value = side.target
    s.manage_open(_advance(s, 10))
    assert s.state["open"] is None and b.pos["AAA"] == 0 and b.live("AAA") == []
    kinds = [c[0] for c in b.calls]
    assert b.calls.index(("cancel", stop_id)) < kinds.index("smart")
    exits = _journal(s, "exit")
    assert len(exits) == 1 and exits[0]["reason"] == "target"
    assert exits[0]["pnl"] == pytest.approx(40.0)  # 10 shares x 4


def test_shell_a_fill_at_or_beyond_the_stop_is_closed_at_once(hostile):
    s, b = hostile()
    side = hostile.side
    plan, _ = side.plan("AAA", None, 100.0)
    b.pos["AAA"] = 10 * side.sign
    s.on_filled(plan, 10, side.stop)
    assert s.state["open"] is None and b.pos["AAA"] == 0
    assert not any(c[0] == "place" and c[4] in ("SL", "SL-M") for c in b.calls)


def test_shell_a_slow_exit_is_cancelled_before_it_is_retried(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.hold_market = True
    s.exit_trade("target")
    first_exit = s.state["open"]["exit_order_id"]
    assert first_exit and b.pos["AAA"] == 10 * side.sign
    s.manage_open(s.test_clock["now"])
    assert ("cancel", first_exit) not in b.calls
    s.manage_open(_advance(s, ls.EXIT_RETRY_SEC + 1))
    assert b.orders[first_exit]["order_status"] == "cancelled"
    assert len(b.live("AAA", ("MARKET",))) == 1  # never two exit orders at once
    b.release()
    s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_shell_no_exit_order_is_sent_while_the_stop_cannot_be_cancelled(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.cancel_ok = False
    for _ in range(3):
        assert s.exit_trade("target") is False
    assert _smarts(b) == 0 and b.pos["AAA"] == 10 * side.sign
    b.cancel_ok = True
    s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None and b.live("AAA") == []


@pytest.mark.parametrize("books", [True, False])
def test_shell_a_lost_stop_answer_never_leads_to_two_stops(hostile, books):
    s, b = hostile()
    b.lost_answer = {"SL-M"}
    b.orderbook_fail = not books
    _signal(s, "AAA")
    b.lost_answer = set()
    assert len(b.live("AAA")) == 1 and s.state["open"]["stop_check"]
    s.manage_open(s.test_clock["now"])
    if books:
        assert s.state["open"]["stop_order_id"] == b.live("AAA")[0]["orderid"]
    s.manage_open(_advance(s, ls.STOP_CONFIRM_SEC + 1))
    assert len(b.live("AAA")) == 1
    b.trigger_stops()
    assert b.pos["AAA"] == 0


def test_shell_a_lost_stop_answer_that_shows_up_late_is_adopted_not_doubled(hostile):
    s, b = hostile()
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    _signal(s, "AAA")
    b.placeorder = real_place
    b.lost_answer = set()
    b.hidden.clear()
    for _ in range(3):
        s.step()
    assert len(b.live("AAA")) == 1 and s.state["open"]["stop_order_id"] == "O2"


def test_shell_a_clear_stop_refusal_sends_the_sl_fallback(hostile):
    s, b = hostile()
    b.refuse = {"SL-M": {"status": "error", "message": "SL-M orders are not allowed"}}
    _signal(s, "AAA")
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL"]
    assert s.state["open"]["stop_type"] == "SL" and b.live("AAA")[0]["trigger"] == hostile.side.stop


def test_shell_a_lost_entry_answer_is_taken_over_with_its_stop(hostile):
    s, b = hostile()
    b.lost_answer = {"MARKET"}
    _signal(s, "AAA")
    b.lost_answer = set()
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1
    assert s.state["traded_levels"][0]["symbol"] == "AAA"  # it filled, so it counts as traded
    _signal(s, "BBB")
    assert b.pos["BBB"] == 0


def test_shell_a_lost_entry_answer_still_working_is_cancelled_not_doubled(hostile):
    s, b = hostile()
    side = hostile.side
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    _signal(s, "AAA")
    b.lost_answer = set()
    b.hold_market = False
    assert s.state["pending"] is not None and b.orders["O1"]["order_status"] == "cancelled"
    s.step()
    assert s.state["pending"] is not None
    _advance(s, ls.ENTRY_CONFIRM_SEC + 1)
    s.step()
    assert s.state["pending"] is None and s.state["open"] is None and b.pos["AAA"] == 0
    assert _placed(b) == [("place", "AAA", side.action, 10, "MARKET")]
    assert s.state["traded_levels"] == []  # nothing filled, so the level lives on
    _signal(s, "AAA")
    assert _placed(b) == [("place", "AAA", side.action, 10, "MARKET")]


def test_shell_a_lost_exit_answer_is_resolved_from_the_books(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.smart_lost = True
    b.ltp_value = side.in_profit
    s.exit_trade("target")
    b.smart_lost = False
    assert b.pos["AAA"] == 0 and s.state["open"]["symbol"] == "AAA"
    b.ltp_value = 100.0
    s.manage_open(s.test_clock["now"])
    assert s.state["open"] is None and _smarts(b) == 1


def test_shell_the_closing_order_is_a_smart_order_to_zero_with_quantity_zero(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    s.exit_trade("target")
    assert b.smart_args == [("AAA", side.close, 0, 0)]
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_shell_a_position_that_goes_flat_just_before_the_exit_is_not_reversed(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")

    def closed_by_someone_else():
        b.pos["AAA"] = 0

    b.before_smart = closed_by_someone_else
    b.ltp_value = side.target
    s.manage_open(_advance(s, 10))
    assert b.pos["AAA"] == 0, "the exit opened the opposite position"
    assert s.state["open"] is None and "already flat" in _journal(s, "exit")[0]["reason"]


def test_shell_positions_not_managed_here_block_entries(hostile):
    s, b = hostile()
    assert s.can_enter(s.test_clock["now"]) == (True, "")
    b.pos["CCC"] = 4
    _signal(s, "BBB")
    assert s.untracked == {"CCC"} and b.pos["BBB"] == 0
    b.pos["CCC"] = 0
    s.periodic()
    assert s.untracked == set()


def test_shell_unreadable_state_with_a_working_stop_closes_cleanly(hostile, tmp_path):
    d = tmp_path / "r5"
    s, b = hostile(data_dir=d)
    _signal(s, "AAA")
    s.state_path.write_text(s.state_path.read_text()[:30])
    s2, _ = hostile(data_dir=d, broker=b)
    assert list(d.glob("state.unreadable-*.json")) and s2.state["open"]["recovered"]
    assert "saved state was lost" in s2.state["no_entries"]
    for _ in range(3):
        s2.step()
    assert b.pos["AAA"] == 0 and b.live("AAA") == [] and s2.state["pnl"] == 0.0


def test_shell_a_state_file_that_cannot_be_written_keeps_the_trade_and_blocks_entries(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    good = s.state_path
    s.state_path = good.parent / "missing folder" / "state.json"
    assert s._save_state() is False and s.unsaved
    b.ltp_value = side.target
    s.manage_open(_advance(s, 10))
    assert s.state["open"] is None and b.pos["AAA"] == 0
    b.ltp_value = 100.0
    assert s.can_enter(s.test_clock["now"]) == (False, "the state file cannot be written")
    _signal(s, "BBB")
    assert b.pos["BBB"] == 0
    s.state_path = good
    s._save_state()
    _signal(s, "BBB")
    assert s.state["open"]["symbol"] == "BBB"


def test_shell_an_entry_is_not_sent_when_its_pending_record_cannot_be_saved(hostile):
    s, b = hostile()
    plan, _ = hostile.side.plan("AAA", None, 100.0)
    s.state_path = s.state_path.parent / "missing folder" / "state.json"
    s.enter(plan, s.test_clock["now"])
    assert _placed(b) == [] and s.state["pending"] is None and s.state["taken"] == []


def test_shell_restarts_mid_order_keep_the_fixed_levels(hostile, tmp_path):
    side = hostile.side
    d = tmp_path / "a"
    s, b = hostile(data_dir=d)
    b.hold_market = True
    _signal(s, "AAA")
    b.release(4)
    s2, _ = hostile(data_dir=d, broker=b)
    assert s2.state["open"]["qty"] == 4 and [o["qty"] for o in b.live("AAA")] == [4]
    assert s2.state["open"]["target"] == side.target and s2.state["open"]["stop"] == side.stop


@pytest.mark.parametrize("case", list(SDK_CASES))
def test_shell_send_reads_the_real_sdk_error_answers(case, tmp_path):
    handler, _error_type, _code, expected = SDK_CASES[case]
    client = _sdk(handler)
    s = ls.LiquiditySweepStrategy(client, ["SBIN"], clock=lambda: H_NOW, data_dir=tmp_path)
    for method in ("placeorder", "placesmartorder"):
        kwargs = {"strategy": "t", "symbol": "SBIN", "action": "SELL", "exchange": "NSE"}
        kwargs.update(price_type="MARKET", product="MIS", quantity=1)
        if method == "placesmartorder":
            kwargs["position_size"] = 0
        assert s.send(method, **kwargs)[0] == expected


def test_shell_a_may_have_been_placed_answer_is_unknown(tmp_path):
    class Client:
        def placeorder(self, **k):
            return {"status": "error", "message": UNANSWERED, "error_type": "api_error"}

    s = ls.LiquiditySweepStrategy(Client(), ["X"], clock=lambda: H_NOW, data_dir=tmp_path)
    assert s.place("X", "BUY", 10, "SL-M")[0] == "unknown"


# =============================================================================
# HEADLESS, THROUGH THE REAL BACKTEST RUNNER
# =============================================================================
HEADLESS = {"SBIN": 800.0, "TCS": 3500.0, "LT": 3600.0, "INFY": 1500.0, "ITC": 420.0}


def write_wave_minutes(
    directory, symbol, start, end, base, seed, period_min, drift=0.01, drift_sessions=40, wick=0.01
):
    """Synthetic one-minute bars (not market data) in the backtester's file
    format, swinging around a slow centre every `period_min` minutes with wicks
    past the extremes, so levels get touched and swept; plus a daily file built
    from the same minutes, so the daily EMA has its history on 1h."""
    rng = np.random.default_rng(seed)
    session = session_for("NSE")
    rows, daily, d, k = [], [], start, 0
    while d <= end:
        if d.weekday() < 5:
            t0 = session.open_epoch(date_to_day(d))
            day_rows = []
            for i in range(375):
                centre = base * (1 + drift * np.sin(2 * np.pi * k / (375 * drift_sessions)))
                mid = centre + 0.005 * base * np.sin(2 * np.pi * k / period_min)
                o = mid + rng.normal(0, 0.0002 * base)
                c = mid + rng.normal(0, 0.0002 * base)
                h = max(o, c) + abs(rng.normal(0, 0.0001 * base))
                lo = min(o, c) - abs(rng.normal(0, 0.0001 * base))
                if rng.random() < wick:
                    h += rng.uniform(0.001, 0.004) * base
                if rng.random() < wick:
                    lo -= rng.uniform(0.001, 0.004) * base
                day_rows.append((t0 + 60 * i, o, h, lo, c, int(rng.integers(100, 1000)), 0))
                k += 1
            rows.extend(day_rows)
            arr = np.array(day_rows, dtype=float)
            daily.append(
                (date_to_day(d), arr[0, 1], arr[:, 2].max(), arr[:, 3].min(), arr[-1, 4], 1000, 0)
            )
        d += timedelta(days=1)
    cols = ["epoch", "open", "high", "low", "close", "volume", "oi"]
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_pickle(directory / f"NSE_{symbol}_1m.pkl".upper())
    pd.DataFrame(daily, columns=["day", *cols[1:]]).to_pickle(
        directory / f"NSE_{symbol}_D.pkl".upper()
    )


# timeframe -> (the wave: period in minutes, drift, drift period in sessions, wick
# chance a minute; extra settings; test days). The 1h run uses pivot_n 2, for the
# reason given at BASE above.
HEADLESS_RUNS = {
    "5m": ((120, 0.01, 40, 0.01), {}, ("2026-09-14", "2026-09-18")),
    "15m": ((180, 0.005, 60, 0.01), {}, ("2026-09-07", "2026-09-18")),
    "1h": ((240, 0.005, 60, 0.02), {"LSW_PIVOT_N": "2"}, ("2026-09-07", "2026-09-18")),
}


@pytest.fixture(scope="module")
def wave_market(tmp_path_factory):
    out = {}
    for tf, (wave, _extra, _days) in HEADLESS_RUNS.items():
        directory = tmp_path_factory.mktemp(f"lsw_bars_{tf}")
        for n, (symbol, base) in enumerate(HEADLESS.items()):
            write_wave_minutes(
                directory, symbol, date(2026, 4, 1), date(2026, 9, 25), base, 30 + n, *wave
            )
        out[tf] = directory
    return out


def _ist(epoch_ms):
    return datetime.fromtimestamp(epoch_ms / 1000, tz=ls.IST).replace(tzinfo=None)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("tf", list(HEADLESS_RUNS))
def test_backtest_runs_headless_and_trades_only_in_the_windows(tmp_path, wave_market, tf):
    _period, extra, (start, end) = HEADLESS_RUNS[tf]
    original = SCRIPT.read_bytes()
    run_dir, result, log = run_backtest(
        tmp_path,
        SCRIPT,
        wave_market[tf],
        start,
        end,
        extra_env={"LSW_SYMBOLS": ",".join(HEADLESS), "LSW_TIMEFRAME": tf, **extra},
        timeout=800,
    )
    assert SCRIPT.read_bytes() == original
    assert result["status"] == "completed", result.get("message")
    assert "5 symbols on NSE MIS" in log and "OpenAlgo mode: backtest" in log
    assert f"liquidity sweep on {tf} bars" in log
    assert "error in cycle" not in log and "Traceback" not in log
    rows = [
        json.loads(x)
        for x in (run_dir / "liquidity_sweep_data" / "journal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    minutes = ls.TF_MINUTES[tf]
    window = {"5m": 2, "15m": 5, "1h": 10}[tf]
    scans = [r for r in rows if r["event"] == "scan"]
    assert scans
    for r in scans:  # only bars whose next bar opens before 15:00
        bar = datetime.strptime(r["signal_bar"], "%Y-%m-%d %H:%M")
        assert bar + timedelta(minutes=minutes) < bar.replace(hour=15, minute=0)
    trades = sorted(result["trades"], key=lambda t: t["entry_time"])
    assert trades, "no trade in the test period"
    for prev, nxt in zip(trades, trades[1:], strict=False):
        assert nxt["entry_time"] >= prev["exit_time"]  # never two positions
    for t in trades:
        opened, closed = _ist(t["entry_time"]), _ist(t["exit_time"])
        assert t["product"] == "MIS" and closed.date() == opened.date()
        assert f"{closed:%H:%M:%S}" <= "15:16:00", t
        assert t["quantity"] * t["entry_price"] <= 60000 * 1.01
    entries = [r for r in rows if r["event"] == "entry"]
    assert entries and all(r["leverage"] == 2.0 and r["mode"] == "backtest" for r in entries)
    assert all(r["planned_loss"] <= 4000 for r in entries)
    signals = [r for r in rows if r["event"] == "signal"]
    assert len(signals) == len(entries)
    for r in signals:  # each decision inside its window, right after the signal bar closed
        bar = datetime.strptime(r["plan"]["signal_bar"], "%Y-%m-%d %H:%M")
        opens = bar + timedelta(minutes=minutes, seconds=10)
        assert opens <= datetime.fromisoformat(r["ts"]) < opens + timedelta(minutes=window)
        assert f"{bar + timedelta(minutes=minutes):%H:%M}" < "15:00"
    for r in entries:
        trade = r["trade"]
        risk = abs(trade["entry_price"] - trade["stop"])
        assert abs(abs(trade["target"] - trade["entry_price"]) - 2 * risk) <= 0.06
    levels = [r["trade"]["signal_bar"] for r in entries]
    state = json.loads((run_dir / "liquidity_sweep_data" / "state.json").read_text())
    assert [r["signal_bar"] for r in state["traded_levels"]] == levels[
        -len(state["traded_levels"]) :
    ]


# =============================================================================
# ROUND 2: the open trade is watched while the basket is read
# =============================================================================
SEVENTEEN = ["AAA"] + [f"S{i:02d}" for i in range(16)]


def _slow_history(s, b, seconds=1.2, hook=None):
    """Every history call takes `seconds` of the strategy's clock, as a live one
    does; `hook(now)` runs after each."""
    real = b.history

    def slow(symbol, **k):
        out = real(symbol, **k)
        s.test_clock["now"] += timedelta(seconds=seconds)
        if hook:
            hook(s.test_clock["now"])
        return out

    b.history = slow


def _quote_times(s, b, symbol):
    seen, real = [], b.quotes

    def quotes(**k):
        if k.get("symbol") == symbol:
            seen.append(s.test_clock["now"])
        return real(**k)

    b.quotes = quotes
    return seen


def test_a_target_touched_while_the_basket_is_read_is_taken_within_a_poll(desk):
    """The auditor's case: AAA short open, 17 symbols at 1.2 s a history call; the
    target is touched only from 10:25:14 to 10:25:26, inside the read."""
    s, b = desk(symbols=SEVENTEEN, frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    b.ltps["AAA"] = 99.5
    s.step()
    assert s.state["open"]["symbol"] == "AAA" and s.state["open"]["stop_order_id"]
    target = s.state["open"]["target"]

    def price(now):
        b.ltps["AAA"] = target if T(10, 25, 14) <= now <= T(10, 25, 26) else 99.0

    _slow_history(s, b, hook=price)
    checks = _quote_times(s, b, "AAA")
    for t in (T(10, 25, 0), T(10, 25, 10)):
        s.test_clock["now"] = t
        s.step()
    assert s.state["open"] is None, "the target touched during the read was never seen"
    exit_row = _journal(s, "exit")[-1]
    assert exit_row["reason"] == "target"
    assert datetime.fromisoformat(exit_row["ts"]) <= T(10, 25, 26)
    gaps = [(b2 - a).total_seconds() for a, b2 in zip(checks, checks[1:], strict=False)]
    assert max(gaps) <= ls.POLL_SECONDS + 1.2 + 1e-6  # one poll plus one symbol's read


def test_the_time_exit_and_square_off_are_not_delayed_by_the_read(desk, monkeypatch):
    s, b = desk(symbols=SEVENTEEN, frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])})
    b.ltps["AAA"] = 99.5
    s.step()
    s.state["open"]["time_exit_at"] = "2026-11-04T10:25:15"
    _slow_history(s, b)
    s.test_clock["now"] = T(10, 25, 10)
    s.step()
    exit_row = _journal(s, "exit")[-1]
    assert exit_row["reason"].startswith("time exit")
    assert datetime.fromisoformat(exit_row["ts"]) <= T(10, 25, 15) + timedelta(
        seconds=ls.POLL_SECONDS + 1.2
    )


def test_an_exit_started_mid_read_stops_the_read_and_the_rest_is_read_after(desk):
    frames = {"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])}
    frames["S15"] = sweep_frame("SELL", sweeps=[T(10, 20)])  # the last symbol signals
    s, b = desk(symbols=SEVENTEEN, frames=frames)
    b.ltps.update(AAA=99.5, S15=99.5)
    s.step()
    target = s.state["open"]["target"]
    b.hold_market = True  # the exit order does not fill at once: an exit is under way
    _slow_history(
        s, b, hook=lambda now: b.ltps.update(AAA=target) if now >= T(10, 25, 20) else None
    )
    before = len(b.history_calls)
    s.test_clock["now"] = T(10, 25, 10)
    s.step()
    assert s.state["open"] and s.state["open"]["exit_retry"] == "target"
    assert "S15" in s.scan_todo and "exit" not in [r["event"] for r in _journal(s, "candidate")]
    assert len(b.history_calls) - before < len(SEVENTEEN)  # stopped part-way
    b.release()  # the exit fills
    b.history = SweepBroker.history.__get__(b)
    s.test_clock["now"] = T(10, 25, 50)
    s.step()  # flat now: the rest is read and S15, the best signal left, is entered
    assert s.state["open"] is None or s.state["open"]["symbol"] == "S15"
    assert s.scan_todo == [] and "S15" in s.scan_results
    s.test_clock["now"] = T(10, 26, 30)  # past the grace period
    s.step()
    assert [e[1] for e in _entries(b)] == ["AAA", "S15"]


def test_a_trade_closing_mid_read_enters_only_after_the_whole_basket_is_ranked(desk):
    frames = {
        "AAA": sweep_frame("SELL", sweeps=[T(10, 15)]),
        "S00": sweep_frame("SELL", sweeps=[T(10, 20)]),  # read early, 3 touches
        "S15": sweep_frame("SELL", sweeps=[T(10, 20)], touches=4),  # read last, ranks first
    }
    s, b = desk(symbols=SEVENTEEN, frames=frames)
    b.ltps.update(AAA=99.5, S00=99.5, S15=99.5)
    s.step()
    target = s.state["open"]["target"]
    _slow_history(
        s, b, hook=lambda now: b.ltps.update(AAA=target) if now >= T(10, 25, 14) else None
    )
    s.test_clock["now"] = T(10, 25, 10)
    s.step()  # AAA closes mid-read; the read carries on to the end
    assert _journal(s, "exit")[-1]["reason"] == "target"
    assert [e[1] for e in _entries(b)] == ["AAA", "S15"]  # the whole ranking, not S00
    sent = [r for r in _journal(s, "signal") if r["plan"]["symbol"] == "S15"][0]
    assert sent["rank"] == 1 and datetime.fromisoformat(sent["ts"]) > T(10, 25, 26)


def test_no_order_is_sent_when_a_slow_price_read_crosses_the_window_end(desk):
    """The auditor's case: the clock is read again just before the order."""
    s, b = desk(
        symbols=["AAA"], now=T(10, 22, 5), frames={"AAA": sweep_frame("SELL", sweeps=[T(10, 15)])}
    )
    real = b.quotes

    def slow_quotes(symbol, **k):
        s.test_clock["now"] += timedelta(seconds=8)  # crosses 10:22:10
        return real(symbol, **k)

    b.quotes = slow_quotes
    b.ltps["AAA"] = 99.5
    s.step()
    assert _entries(b) == []
    assert "outside the entry window" in s.state["skipped"]["AAA 2026-11-04 10:15"]


def test_an_exit_under_way_holds_the_read(desk):
    s, _b = desk(symbols=["AAA"])
    s.state["open"] = {"symbol": "ZZZ", "stop_order_id": "O9", "stop_check": False}
    assert s.scan_hold() == ""
    s.state["open"]["exit_retry"] = "target"
    assert "ZZZ trade is open" in s.scan_hold()


def test_a_traded_level_on_corrected_bars_falls_back_to_the_nearest_level():
    lv = lambda price, members: ls.Level("R", price, 3, members)  # noqa: E731
    res = [lv(100.0, (1, 2, 3)), lv(100.5, (4, 5, 6)), lv(99.0, (7, 8, 9))]
    # the picked level is the saved one: its members
    assert ls.traded_members(("R", 100.5), -1, res[1], res, [], 0.0015) == ("R", (4, 5, 6))
    # corrected bars: the pick moved, the saved price is now 100.08 -> nearest within tol
    assert ls.traded_members(("R", 100.08), -1, res[1], res, [], 0.0015) == ("R", (1, 2, 3))
    # no pick at all on that bar now, still the nearest within tol
    assert ls.traded_members(("R", 99.95), 0, None, res, [], 0.0015) == ("R", (1, 2, 3))
    # nothing within tol: nothing is killed; and the other side is never used
    assert ls.traded_members(("R", 101.5), 0, None, res, [], 0.0015) is None
    assert ls.traded_members(("S", 100.0), 0, None, res, [], 0.0015) is None
