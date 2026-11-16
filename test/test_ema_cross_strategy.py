"""Tests for strategies/examples/ema_cross_multi.py.

The strategy module is imported by path: it creates no client and makes no
network call at import. The indicator tests compare it with an independent,
literal reading of TradingView's Pine semantics. The order-safety tests drive
it against fake brokers that lose answers, show orders late, or misreport, and
the last test runs it unchanged through the real backtest runner on synthetic
bars for several symbols.

Run:
    uv run pytest test/test_ema_cross_strategy.py -v
"""

import importlib.util
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from openalgo import api as sdk_api

from test.test_python_backtest_e2e import run_backtest, write_bars

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "strategies" / "examples" / "ema_cross_multi.py"


def _load(name="ema_cross_multi"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st = _load()


def random_ohlc(n, seed, base=1000.0, vol=0.004):
    rng = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.r_[base, c[:-1]] * (1 + rng.normal(0, vol / 4, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return o, h, lo, c


def frame(o, h, lo, c, end=None, start="2026-11-02 09:15"):
    n = len(c)
    if end is not None:
        idx = pd.date_range(end=end, periods=n, freq="5min")
    else:
        idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)


# =============================================================================
# INDEPENDENT PINE REFERENCE (None is na, one bar at a time)
# =============================================================================
def ref_sma(src, length, i):
    if i < length - 1:
        return None
    window = src[i - length + 1 : i + 1]
    if any(v is None for v in window):
        return None
    return sum(window) / length


def ref_seeded(src, length, alpha):
    """Pine: x := na(x[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(x[1])"""
    out, prev = [], None
    for i in range(len(src)):
        value = ref_sma(src, length, i) if prev is None else alpha * src[i] + (1 - alpha) * prev
        out.append(value)
        prev = value
    return out


def ref_ema(src, length):
    return ref_seeded(list(map(float, src)), length, 2.0 / (length + 1))


def ref_atr(h, lo, c, length):
    h, lo, c = (list(map(float, x)) for x in (h, lo, c))
    tr = [h[0] - lo[0]] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))
    ]
    return ref_seeded(tr, length, 1.0 / length)


def ref_cross(a, b, over):
    out = [False]
    for i in range(1, len(a)):
        if None in (a[i], b[i], a[i - 1], b[i - 1]):
            out.append(False)
        elif over:
            out.append(a[i] > b[i] and a[i - 1] <= b[i - 1])
        else:
            out.append(a[i] < b[i] and a[i - 1] >= b[i - 1])
    return out


def arr(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


# =============================================================================
# INDICATORS
# =============================================================================
@pytest.mark.parametrize("length", [9, 13, 21, 50])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_ema_matches_pine(seed, length):
    _o, _h, _lo, c = random_ohlc(400, seed)
    np.testing.assert_allclose(
        st.pine_ema(c, length), arr(ref_ema(c, length)), rtol=1e-10, equal_nan=True
    )


def test_ema_is_na_until_n_bars_then_seeded_with_the_sma():
    _o, _h, _lo, c = random_ohlc(60, 9)
    ema = st.pine_ema(c, 21)
    assert np.isnan(ema[:20]).all() and np.isfinite(ema[20:]).all()
    assert ema[20] == pytest.approx(np.mean(c[:21]), rel=1e-12)
    alpha = 2 / 22
    assert ema[21] == pytest.approx(alpha * c[21] + (1 - alpha) * ema[20], rel=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_atr_matches_pine(seed):
    _o, h, lo, c = random_ohlc(400, seed)
    np.testing.assert_allclose(
        st.pine_atr(h, lo, c, 14), arr(ref_atr(h, lo, c, 14)), rtol=1e-10, equal_nan=True
    )


def test_atr_warmup_follows_pine():
    _o, h, lo, c = random_ohlc(40, 7)
    atr = st.pine_atr(h, lo, c, 14)
    assert np.isnan(atr[:13]).all() and np.isfinite(atr[13:]).all()
    tr = [h[0] - lo[0]] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, 14)
    ]
    assert atr[13] == pytest.approx(np.mean(tr), rel=1e-12)  # bar 0 is high - low


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_crossover_and_crossunder_match_pine(seed):
    rng = np.random.default_rng(seed)
    # rounded to one decimal so equal values (a[1] == b[1]) occur often
    a = np.round(np.cumsum(rng.normal(0, 1, 300)), 1)
    b = np.round(np.cumsum(rng.normal(0, 1, 300)), 1)
    a[:5] = np.nan  # na at the start, as an EMA has
    ra = [None if np.isnan(x) else float(x) for x in a]
    rb = [float(x) for x in b]
    over, under = st.crossover(a, b), st.crossunder(a, b)
    assert list(over) == ref_cross(ra, rb, True)
    assert list(under) == ref_cross(ra, rb, False)
    assert over.sum() >= 3 and under.sum() >= 3
    assert not (over & under).any()


def test_cross_edge_cases():
    # touch then cross: a[1] == b[1] counts; a == b on the bar does not
    assert list(st.crossover([1, 2, 3], [2, 2, 2])) == [False, False, True]
    assert list(st.crossover([1, 2, 2], [2, 2, 2])) == [False, False, False]
    assert list(st.crossunder([3, 2, 1], [2, 2, 2])) == [False, False, True]
    assert list(st.crossover([np.nan, 3.0], [2.0, 2.0])) == [False, False]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_signals_match_the_reference_on_every_bar(seed):
    _o, h, lo, c = random_ohlc(500, seed)
    sig = st.compute_signals(h, lo, c, 9, 21, 13, 21)
    e9, e13, e21 = ref_ema(c, 9), ref_ema(c, 13), ref_ema(c, 21)
    assert list(sig["entry"]) == ref_cross(e9, e21, True)
    assert list(sig["exit"]) == ref_cross(e13, e21, False)
    assert sig["entry"].sum() >= 3 and sig["exit"].sum() >= 3


def test_atr_filter_is_off_by_default_and_only_removes_signals(monkeypatch):
    assert st.ATR_FILTER is False
    _o, h, lo, c = random_ohlc(800, 12)
    plain = st.compute_signals(h, lo, c)["entry"]
    monkeypatch.setattr(st, "ATR_FILTER", True)
    filtered = st.compute_signals(h, lo, c)["entry"]
    atr = arr(ref_atr(h, lo, c, 14))
    atr_sma = st.pine_sma(atr, 50)
    with np.errstate(invalid="ignore"):
        expected = plain & (atr > atr_sma)
    assert list(filtered) == list(expected)
    assert 0 < filtered.sum() < plain.sum()


def test_signals_never_read_ahead():
    _o, h, lo, c = random_ohlc(400, 11)
    full = st.compute_signals(h, lo, c)
    for t in (30, 90, 200, 399):
        part = st.compute_signals(h[: t + 1], lo[: t + 1], c[: t + 1])
        for key in ("entry", "exit"):
            assert (part[key] == full[key][: t + 1]).all()
        # changing every later bar changes nothing up to t
        h2, lo2, c2 = h.copy(), lo.copy(), c.copy()
        h2[t + 1 :] *= 1.5
        lo2[t + 1 :] *= 0.5
        c2[t + 1 :] = c2[t + 1 :][::-1]
        moved = st.compute_signals(h2, lo2, c2)
        for key in ("entry", "exit"):
            assert (moved[key][: t + 1] == full[key][: t + 1]).all()
        np.testing.assert_array_equal(moved["atr"][: t + 1], full["atr"][: t + 1])


# =============================================================================
# EVALUATION: closed bars, restart, long only
# =============================================================================
def _first(mask, after=60):
    return next(int(i) for i in np.flatnonzero(mask) if i > after)


def test_first_evaluation_only_records_and_a_bar_never_signals_twice(tmp_path):
    now = datetime(2026, 11, 4, 10, 0)
    o, h, lo, c = random_ohlc(300, 3)
    bars = frame(o, h, lo, c)
    f = _first(st.compute_signals(h, lo, c)["entry"])

    running = st.EmaCrossStrategy(
        FakeClient(["AAA"], now), ["AAA"], clock=lambda: now, data_dir=tmp_path / "a"
    )
    assert running.evaluate_symbol("AAA", bars.iloc[:f]) is None  # first look: record only
    assert running.evaluate_symbol("AAA", bars.iloc[: f + 1]) == "BUY"  # a fresh crossover
    assert running.evaluate_symbol("AAA", bars.iloc[: f + 1]) is None  # never twice

    restarted = st.EmaCrossStrategy(
        FakeClient(["AAA"], now), ["AAA"], clock=lambda: now, data_dir=tmp_path / "b"
    )
    assert restarted.evaluate_symbol("AAA", bars.iloc[: f + 1]) is None  # restart on a cross
    assert restarted.above["AAA"] is True


def test_a_crossunder_never_signals_a_short(tmp_path):
    now = datetime(2026, 11, 4, 10, 0)
    o, h, lo, c = random_ohlc(600, 5)
    bars = frame(o, h, lo, c)
    sig = st.compute_signals(h, lo, c)
    strat = st.EmaCrossStrategy(
        FakeClient(["AAA"], now), ["AAA"], clock=lambda: now, data_dir=tmp_path
    )
    under = st.crossunder(sig["fast"], sig["slow"])
    seen = []
    for t in range(40, 600):
        seen.append(strat.evaluate_symbol("AAA", bars.iloc[: t + 1]))
    assert set(seen) <= {None, "BUY"}
    assert under[41:].sum() >= 3  # the data has 9/21 crossunders
    buys = [t for t, s in zip(range(40, 600), seen, strict=True) if s == "BUY"]
    assert buys == [int(i) for i in np.flatnonzero(sig["entry"]) if i > 40]


def test_load_bars_drops_the_forming_bar(tmp_path):
    rng = np.random.default_rng(4)
    n = 400
    idx = pd.date_range("2026-11-02 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    c = 1000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[1000, c[:-1]]
    df = pd.DataFrame(
        {"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c},
        index=idx,
    )
    forming = idx[-1].tz_localize(None)
    now = (forming + pd.Timedelta(minutes=2, seconds=30)).to_pydatetime()

    class History:
        def __init__(self, f):
            self.frame = f

        def history(self, **k):
            return self.frame

    strat = st.EmaCrossStrategy(History(df), ["X"], clock=lambda: now, data_dir=tmp_path)
    kept = strat.load_bars("X", now)
    assert kept.index[-1] < forming and kept.index[-1] == idx[-2].tz_localize(None)
    wild = df.copy()
    wild.iloc[-1] = [1, 99999, 0.5, 99999]
    strat.client = History(wild)
    assert strat.load_bars("X", now).equals(kept)  # the forming bar never matters
    after = (forming + pd.Timedelta(minutes=5, seconds=5)).to_pydatetime()
    assert strat.load_bars("X", after).index[-1] == forming


# =============================================================================
# STOP AND SIZE
# =============================================================================
def test_stop_is_signal_close_minus_one_and_a_half_atr():
    o, h, lo, c = random_ohlc(300, 21, base=500)
    bars = frame(o, h, lo, c)
    f = _first(st.compute_signals(h, lo, c)["entry"])
    part = bars.iloc[: f + 1]
    close = float(c[f])
    atr = ref_atr(h[: f + 1], lo[: f + 1], c[: f + 1], 14)[-1]
    plan, why = st.plan_trade("SBIN", part, close, tick=0.05)
    assert why == "" and plan.side == "BUY"
    expected = math.floor((close - 1.5 * atr) / 0.05 + 1e-9) * 0.05
    assert plan.stop == pytest.approx(expected, abs=1e-6)
    assert plan.stop <= close - 1.5 * atr + 1e-9  # rounded down, never closer
    assert plan.atr == pytest.approx(atr, abs=1e-4)
    assert plan.signal_bar == f"{part.index[-1]:%Y-%m-%d %H:%M}"
    assert plan.qty * (plan.entry - plan.stop) <= 2000
    assert plan.qty * plan.entry <= 30000


def test_the_stop_uses_the_signal_close_not_the_entry_price():
    o, h, lo, c = random_ohlc(300, 22, base=500)
    part = frame(o, h, lo, c).iloc[:200]
    a, b = st.plan_trade("X", part, float(c[199])), st.plan_trade("X", part, float(c[199]) + 2)
    assert a[0].stop == b[0].stop and b[0].risk == pytest.approx(a[0].risk + 2, abs=1e-6)


def test_risk_cap_binds_on_a_wide_stop():
    assert st.size_position(100.0, 90.0, 1, 30000, 2000, 0) == (200, "risk")  # capital: 300


def test_capital_cap_binds_on_a_tight_stop():
    assert st.size_position(100.0, 98.0, 1, 30000, 2000, 0) == (300, "capital")  # risk: 1000


def test_default_caps_are_the_users():
    assert (st.CAPITAL, st.MAX_LOSS_PER_TRADE) == (30000.0, 2000.0)
    assert (st.MAX_TRADES_PER_DAY, st.MAX_DAILY_LOSS) == (5, 4000.0)
    assert (st.FAST_LEN, st.SLOW_LEN, st.EXIT_FAST_LEN, st.EXIT_SLOW_LEN) == (9, 21, 13, 21)
    assert (st.ATR_PERIOD, st.STOP_ATR_MULT, st.MIN_STOP_ATR) == (14, 1.5, 0.3)
    assert (st.ENTRY_START, st.LAST_ENTRY, st.SQUARE_OFF) == ("09:30", "14:45", "15:10")
    assert st.STRATEGY_NAME == "EMA Cross 9-21" and st.SWEEP_SEC == 300
    assert st.DATA_DIR.name == "ema_cross_data"
    assert len(st.SYMBOLS) == 17 and "M&M" in st.SYMBOLS


def test_slippage_buffer_only_makes_sizing_more_conservative():
    qty, _ = st.size_position(100.0, 90.0, 1, 30000, 2000, 0.1)
    assert qty == 198  # floor(2000 / 10.1)
    assert qty * (100.1 - 90.0) <= 2000
    assert st.size_position(100.0, 90.0, 75, 30000, 2000, 0)[0] == 150  # lot rounding


def test_quantity_zero_is_skipped_with_a_reason():
    o, h, lo, c = random_ohlc(300, 23, base=40000)  # one share is worth more than the capital
    part = frame(o, h, lo, c).iloc[:200]
    plan, why = st.plan_trade("X", part, float(c[199]))
    assert plan is None and "quantity rounds to 0" in why


def test_entry_at_or_below_the_stop_is_skipped():
    o, h, lo, c = random_ohlc(300, 24, base=500)
    part = frame(o, h, lo, c).iloc[:200]
    plan, _ = st.plan_trade("X", part, float(c[199]))
    skipped, why = st.plan_trade("X", part, plan.stop)
    assert skipped is None and "at or below the stop" in why


def test_min_stop_distance_guard(monkeypatch):
    o, h, lo, c = random_ohlc(300, 25, base=500)
    part = frame(o, h, lo, c).iloc[:200]
    monkeypatch.setattr(st, "STOP_ATR_MULT", 0.2)
    plan, why = st.plan_trade("X", part, float(c[199]))
    assert plan is None and "closer than 0.3 ATR" in why


# =============================================================================
# TAKE PROFIT: EMA13 under EMA21 after the entry bar
# =============================================================================
def _tp_case():
    """A series with: a 9/21 crossover e, a 13/21 crossunder k before e, a 9/21
    crossunder y after e with no 13/21 crossunder in (e, y], and the first
    13/21 crossunder x after e."""
    for seed in range(1, 200):
        o, h, lo, c = random_ohlc(500, seed)
        sig = st.compute_signals(h, lo, c)
        under9 = st.crossunder(sig["fast"], sig["slow"])
        exits = np.flatnonzero(sig["exit"])
        for e in np.flatnonzero(sig["entry"]):
            if e < 80:
                continue
            before = [int(k) for k in exits if k <= e]
            after = [int(x) for x in exits if x > e]
            if not before or not after:
                continue
            x = after[0]
            ys = [int(y) for y in np.flatnonzero(under9) if e < y < x]
            if ys and x + 1 < len(c):
                return frame(o, h, lo, c), int(e), before[-1], ys[0], x
    raise AssertionError("no suitable series")


def _trade(bars, i):
    return {"symbol": "AAA", "side": "BUY", "entry_bar": f"{bars.index[i]:%Y-%m-%d %H:%M}"}


def test_take_profit_fires_on_a_13_21_crossunder_after_the_entry_bar(tmp_path):
    bars, e, _k, _y, x = _tp_case()
    s = st.EmaCrossStrategy(None, ["AAA"], clock=lambda: datetime(2026, 11, 4), data_dir=tmp_path)
    trade = _trade(bars, e)
    for t in range(e, x):
        assert not s.take_profit_due(trade, bars.iloc[: t + 1]), t
    assert s.take_profit_due(trade, bars.iloc[: x + 1])


def test_a_crossunder_at_or_before_the_entry_bar_never_exits(tmp_path):
    bars, e, k, _y, x = _tp_case()
    s = st.EmaCrossStrategy(None, ["AAA"], clock=lambda: datetime(2026, 11, 4), data_dir=tmp_path)
    # the entry bar is the crossunder bar itself
    assert not s.take_profit_due(_trade(bars, k), bars.iloc[: k + 1])
    # a cross from before the entry is still in the history: not counted
    assert not s.take_profit_due(_trade(bars, e), bars.iloc[: e + 1])
    assert not s.take_profit_due(_trade(bars, x), bars.iloc[: x + 1])


def test_the_opposite_9_21_cross_does_not_exit(tmp_path):
    bars, e, _k, y, _x = _tp_case()
    s = st.EmaCrossStrategy(None, ["AAA"], clock=lambda: datetime(2026, 11, 4), data_dir=tmp_path)
    assert not s.take_profit_due(_trade(bars, e), bars.iloc[: y + 1])


def test_take_profit_lengths_are_configurable(tmp_path, monkeypatch):
    bars, e, _k, y, _x = _tp_case()
    s = st.EmaCrossStrategy(None, ["AAA"], clock=lambda: datetime(2026, 11, 4), data_dir=tmp_path)
    monkeypatch.setattr(st, "EXIT_FAST_LEN", 9)  # now the 9/21 crossunder is the exit
    assert s.take_profit_due(_trade(bars, e), bars.iloc[: y + 1])


# =============================================================================
# FAKE CLIENT: one trade at a time, caps, window
# =============================================================================
class FakeClient:
    def __init__(self, symbols, now):
        self.now = now
        self.pos = dict.fromkeys(symbols, 0)
        self.orders = {}
        self.calls = []
        self.n = 0
        self.cancel_ok = True
        idx = pd.date_range(
            end=now.replace(second=0) - timedelta(minutes=5), periods=60, freq="5min"
        )
        self.bars = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1}, index=idx
        )

    def _id(self):
        self.n += 1
        return f"OID{self.n}"

    def symbol(self, symbol, **k):
        return {"status": "success", "data": {"symbol": symbol, "tick_size": 0.05, "lotsize": 1}}

    def analyzerstatus(self):
        return {"status": "success", "data": {"mode": "analyze"}}

    def quotes(self, symbol, **k):
        return {"status": "success", "data": {"ltp": 100.0}}

    def history(self, **k):
        return self.bars

    def openposition(self, symbol, **k):
        return {"status": "success", "quantity": str(self.pos.get(symbol, 0))}

    def placeorder(self, symbol, action, quantity, price_type, **k):
        oid = self._id()
        self.calls.append(("place", symbol, action, quantity, price_type))
        if price_type == "MARKET":
            self.pos[symbol] += quantity if action == "BUY" else -quantity
            self.orders[oid] = "complete"
        else:
            self.orders[oid] = "trigger pending"
        return {"status": "success", "orderid": oid}

    def orderstatus(self, order_id, **k):
        return {
            "status": "success",
            "data": {"order_status": self.orders.get(order_id, "open"), "average_price": 100.0},
        }

    def cancelorder(self, order_id, **k):
        self.calls.append(("cancel", order_id))
        if self.cancel_ok:
            self.orders[order_id] = "cancelled"
        return {"status": "success"}

    def placesmartorder(self, symbol, action, quantity, position_size, **k):
        self.calls.append(("smart", symbol, action, quantity, position_size))
        self.pos[symbol] = position_size
        oid = self._id()
        self.orders[oid] = "complete"
        return {"status": "success", "orderid": oid}


def fake_plan(symbol, bars, entry, tick=0.05, lot=1):
    return st.Plan(symbol, "BUY", entry, 98.0, 10, 2.0, 1.0, 100.0, "risk", "2026-11-04 09:40"), ""


class FakeTime:
    """time for the strategy module: sleep moves a fake monotonic clock on."""

    def __init__(self):
        self.t = 0.0
        self.hooks = []

    def sleep(self, seconds):
        self.t += seconds
        for hook in list(self.hooks):
            hook()

    def monotonic(self):
        return self.t


BAR_1, BAR_2, BAR_3 = (datetime(2026, 11, 4, 9, m) for m in (45, 50, 55))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "time", FakeTime())
    monkeypatch.setattr(st, "plan_trade", fake_plan)
    now = datetime(2026, 11, 4, 10, 0, 6)
    symbols = ["AAA", "BBB", "CCC"]
    client = FakeClient(symbols, now)
    strat = st.EmaCrossStrategy(
        client, symbols=symbols, clock=lambda: now, data_dir=tmp_path, pause=0
    )
    strat.setup()
    script = {}
    monkeypatch.setattr(strat, "evaluate_symbol", lambda symbol, bars: script.get(symbol))
    return strat, client, script, now, tmp_path


def _journal(path):
    return [json.loads(line) for line in (path / "journal.jsonl").read_text().splitlines()]


def test_only_one_trade_across_all_symbols(harness):
    strat, client, script, now, tmp = harness
    script.update({"AAA": "BUY", "BBB": "BUY"})
    strat.on_bar(now, BAR_1)
    assert strat.state["open"]["symbol"] == "AAA" and client.pos["AAA"] == 10
    assert client.pos["BBB"] == 0
    assert [c for c in client.calls if c[0] == "place"] == [
        ("place", "AAA", "BUY", 10, "MARKET"),
        ("place", "AAA", "SELL", 10, "SL-M"),
    ]
    ignored = [r for r in _journal(tmp) if r["event"] == "signal_ignored"]
    assert [(r["symbol"], r["side"]) for r in ignored] == [("BBB", "BUY")]
    assert "AAA trade is already open" in ignored[0]["why"]

    strat.bar_done.clear()
    script.clear()
    script["CCC"] = "BUY"
    strat.on_bar(now, BAR_2)
    assert client.pos["CCC"] == 0
    assert sum(1 for r in _journal(tmp) if r["event"] == "signal_ignored") == 2


def test_a_pending_entry_blocks_every_other_symbol(harness):
    strat, _client, _script, now, _tmp = harness
    strat.state["pending"] = {"order_id": None, "plan": {"symbol": "AAA"}, "placed": "x"}
    ok, why = strat.can_enter(now)
    assert not ok and "AAA entry is being confirmed" in why


def test_take_profit_in_the_bar_cycle_exits_at_market(harness, monkeypatch):
    strat, client, script, now, tmp = harness
    script["AAA"] = "BUY"
    strat.on_bar(now, BAR_2)
    assert strat.state["open"]["entry_bar"] == "2026-11-04 09:40"
    stop_id = strat.state["open"]["stop_order_id"]
    script.clear()
    strat.bar_done.clear()
    monkeypatch.setattr(strat, "take_profit_due", lambda trade, bars: True)
    strat.on_bar(now, BAR_3)
    assert strat.state["open"] is None and client.pos["AAA"] == 0
    kinds = [c[0] for c in client.calls]
    assert client.calls.index(("cancel", stop_id)) < kinds.index("smart")
    assert client.calls[kinds.index("smart")] == ("smart", "AAA", "SELL", 10, 0)
    exits = [r for r in _journal(tmp) if r["event"] == "exit"]
    assert len(exits) == 1 and exits[0]["reason"].startswith("take profit")


def test_take_profit_with_the_real_rule_in_the_bar_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "time", FakeTime())
    bars, e, _k, _y, x = _tp_case()
    now = datetime(2026, 11, 4, 10, 0, 6)
    client = FakeClient(["AAA"], now)
    part = bars.iloc[: x + 1].copy()
    part.index = pd.date_range(end=BAR_3, periods=len(part), freq="5min")
    client.bars = part
    s = st.EmaCrossStrategy(client, ["AAA"], clock=lambda: now, data_dir=tmp_path, pause=0)
    s.setup()

    def open_trade(entry_index):
        trade = {**_trade(part, entry_index), "qty": 10, "entry_price": 100.0, "stop": 1.0}
        trade.update(target=None, stop_order_id=None, stop_check=False, exit_order_id=None)
        trade.update(exit_unanswered=False, exit_retry=None, opened=now.isoformat())
        client.pos["AAA"] = 10
        s.state["open"] = trade

    open_trade(e)
    s.on_bar(now, BAR_3)
    assert s.state["open"] is None and client.pos["AAA"] == 0
    # with the entry bar on the cross bar itself, nothing happens
    open_trade(len(part) - 1)
    s.bar_done.clear()
    s.on_bar(now, BAR_3)
    assert s.state["open"] is not None and client.pos["AAA"] == 10


def test_daily_caps_and_entry_window(harness):
    strat, _client, _script, now, _tmp = harness
    assert strat.can_enter(now) == (True, "")
    assert not strat.can_enter(now.replace(hour=9, minute=29, second=59))[0]
    assert not strat.can_enter(now.replace(hour=14, minute=47, second=5))[0]
    strat.state["trades"] = st.MAX_TRADES_PER_DAY
    assert "trades already taken" in strat.can_enter(now)[1]
    strat.state["trades"] = 0
    strat.state["pnl"] = -st.MAX_DAILY_LOSS
    assert "daily loss limit" in strat.can_enter(now)[1]


def test_daily_trade_cap_stops_the_sixth_entry(harness):
    strat, client, script, now, tmp = harness
    strat.state["trades"] = 5
    script["AAA"] = "BUY"
    strat.on_bar(now, BAR_1)
    assert client.pos["AAA"] == 0 and strat.state["open"] is None
    assert "5 trades already taken" in _journal(tmp)[-1]["why"]


def test_entry_window_is_judged_by_the_bar_close(harness):
    strat, _client, _script, _now, _tmp = harness
    day = datetime(2026, 11, 4)
    last_bar_close = day.replace(hour=14, minute=45)
    assert strat.can_enter(day.replace(hour=14, minute=45, second=5), last_bar_close)[0]
    assert strat.can_enter(day.replace(hour=14, minute=46, second=59), last_bar_close)[0]
    assert not strat.can_enter(day.replace(hour=14, minute=47, second=1), last_bar_close)[0]
    assert not strat.can_enter(
        day.replace(hour=14, minute=50, second=5), day.replace(hour=14, minute=50)
    )[0]
    assert strat.can_enter(
        day.replace(hour=9, minute=30, second=5), day.replace(hour=9, minute=30)
    )[0]
    assert not strat.can_enter(
        day.replace(hour=9, minute=25, second=5), day.replace(hour=9, minute=25)
    )[0]


def test_square_off_closes_the_trade(harness):
    strat, client, script, now, _tmp = harness
    script["AAA"] = "BUY"
    strat.on_bar(now, BAR_1)
    strat.manage_open(now.replace(hour=15, minute=10))
    assert strat.state["open"] is None and client.pos["AAA"] == 0


def test_a_fill_at_or_below_the_stop_is_closed_at_once(harness, monkeypatch):
    strat, client, _script, now, _tmp = harness
    plan = st.Plan("AAA", "BUY", 100.0, 98.0, 10, 2.0, 1.0, 100.0, "risk", "2026-11-04 09:40")
    client.pos["AAA"] = 10
    strat.on_filled(plan, 10, 97.5)
    assert strat.state["open"] is None and client.pos["AAA"] == 0
    assert not any(c[0] == "place" and c[4] in ("SL", "SL-M") for c in client.calls)


def test_unknown_symbols_are_skipped_at_setup(tmp_path):
    class Picky(FakeClient):
        def symbol(self, symbol, **k):
            if symbol == "NOPE":
                return {"status": "error", "message": "Symbol not found"}
            return super().symbol(symbol, **k)

    now = datetime(2026, 11, 4, 10, 0)
    strat = st.EmaCrossStrategy(
        Picky(["AAA", "NOPE"], now), ["AAA", "NOPE"], clock=lambda: now, data_dir=tmp_path
    )
    strat.setup()
    assert strat.symbols == ["AAA"]


def test_a_symbol_that_could_not_be_checked_is_retried_not_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "time", FakeTime())

    class Flaky(FakeClient):
        down = True

        def symbol(self, symbol, **k):
            if symbol == "BBB" and self.down:
                raise ConnectionError("connection refused")
            return super().symbol(symbol, **k)

    now = datetime(2026, 11, 4, 10, 0, 6)
    client = Flaky(["AAA", "BBB"], now)
    strat = st.EmaCrossStrategy(
        client, ["AAA", "BBB"], clock=lambda: now, data_dir=tmp_path, pause=0
    )
    strat.setup()
    assert strat.symbols == ["AAA"] and strat.unchecked == ["BBB"]
    client.down = False
    strat.on_bar(now, BAR_3)
    assert strat.symbols == ["AAA", "BBB"] and strat.unchecked == []


def test_one_bad_history_frame_does_not_stop_the_other_symbols(harness, monkeypatch):
    strat, client, _script, now, _tmp = harness
    seen = []
    monkeypatch.setattr(strat, "evaluate_symbol", lambda symbol, bars: seen.append(symbol))
    good = client.bars

    def history(symbol, **k):
        if symbol == "AAA":
            return pd.DataFrame({"price": [1.0]})
        return good

    client.history = history
    assert strat.on_bar(now, BAR_1)
    assert seen == ["BBB", "CCC"]


def test_sweep_seconds_is_configurable(monkeypatch):
    monkeypatch.setenv("EMAX_SWEEP_SEC", "120")
    monkeypatch.setenv("EMAX_SYMBOLS", "SBIN, tcs")
    other = _load("ema_cross_multi_env")
    assert other.SWEEP_SEC == 120 and other.SYMBOLS == ["SBIN", "TCS"]


# =============================================================================
# SDK ERROR ANSWERS: the real openalgo client, on a mocked transport
# =============================================================================
UNANSWERED = (
    "The connection to the broker dropped after the order was sent, so it may have been "
    "placed. Check the order book before placing it again."
)


def _sdk(handler):
    client = sdk_api(api_key="k", host="http://openalgo.test")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _raise(exc_type, text):
    def handler(request):
        raise exc_type(text, request=request)

    return handler


def _reply(code, **kwargs):
    return lambda request: httpx.Response(code, **kwargs)


SDK_CASES = {
    "timeout": (_raise(httpx.ReadTimeout, "timed out"), "timeout_error", None, "unknown"),
    "504": (_reply(504, text="Gateway Time-out"), "http_error", 504, "unknown"),
    "502 may have been placed": (
        _reply(502, json={"status": "error", "message": UNANSWERED}),
        "http_error",
        502,
        "unknown",
    ),
    "500": (_reply(500, text="Internal Server Error"), "http_error", 500, "unknown"),
    "dropped mid-request": (
        _raise(httpx.RemoteProtocolError, "Server disconnected without sending a response."),
        "http_error",
        None,
        "unknown",
    ),
    "not json": (_reply(200, text="<html>proxy</html>"), "json_error", None, "unknown"),
    "connect refused": (
        _raise(httpx.ConnectError, "connection refused"),
        "connection_error",
        None,
        "refused",
    ),
    "400": (
        _reply(400, json={"status": "error", "message": "Invalid symbol"}),
        "http_error",
        400,
        "refused",
    ),
    "403": (_reply(403, text="Forbidden"), "http_error", 403, "refused"),
    "api error": (
        _reply(200, json={"status": "error", "message": "Insufficient funds"}),
        "api_error",
        200,
        "refused",
    ),
    "api error may have been placed": (
        _reply(200, json={"status": "error", "message": UNANSWERED}),
        "api_error",
        200,
        "unknown",
    ),
    "sent": (_reply(200, json={"status": "success", "orderid": "77"}), None, None, "sent"),
}


@pytest.mark.parametrize("case", list(SDK_CASES))
def test_send_reads_the_real_sdk_error_answers(case, tmp_path):
    handler, error_type, code, expected = SDK_CASES[case]
    client = _sdk(handler)
    raw = client.placeorder(
        strategy="t",
        symbol="SBIN",
        action="BUY",
        exchange="NSE",
        price_type="MARKET",
        product="MIS",
        quantity=1,
    )
    if error_type:
        assert raw["status"] == "error" and raw["error_type"] == error_type
        assert raw.get("code") == code
    s = st.EmaCrossStrategy(
        client, ["SBIN"], clock=lambda: datetime(2026, 11, 4), data_dir=tmp_path
    )
    for method in ("placeorder", "placesmartorder"):
        kwargs = {"strategy": "t", "symbol": "SBIN", "action": "BUY", "exchange": "NSE"}
        kwargs.update(price_type="MARKET", product="MIS", quantity=1)
        if method == "placesmartorder":
            kwargs["position_size"] = 0
        assert s.send(method, **kwargs)[0] == expected


SDK_TIMEOUT = {
    "status": "error",
    "message": "Request timed out. The server took too long to respond.",
    "error_type": "timeout_error",
}
SDK_504 = {
    "status": "error",
    "message": "HTTP 504: <html><body>Gateway Time-out</body></html>",
    "code": 504,
    "error_type": "http_error",
}
SDK_CONNECT = {
    "status": "error",
    "message": "Failed to connect to the server. Please check if the server is running.",
    "error_type": "connection_error",
}
SDK_400 = {
    "status": "error",
    "message": 'HTTP 400: {"status": "error", "message": "SL-M orders are not allowed"}',
    "code": 400,
    "error_type": "http_error",
}
SDK_API = {
    "status": "error",
    "message": "RMS: order rejected",
    "code": 200,
    "error_type": "api_error",
}


@pytest.mark.parametrize(
    "resp,expected",
    [
        (SDK_TIMEOUT, "unknown"),
        (SDK_504, "unknown"),
        ({**SDK_504, "code": 500}, "unknown"),
        (
            {"status": "error", "message": "HTTP error occurred: x", "error_type": "http_error"},
            "unknown",
        ),
        ({"status": "error", "message": "x", "error_type": "unknown_error"}, "unknown"),
        ({"status": "error", "message": "x", "error_type": "json_error"}, "unknown"),
        ({**SDK_400, "message": "HTTP 400: " + UNANSWERED}, "unknown"),
        (SDK_CONNECT, "refused"),
        (SDK_400, "refused"),
        ({**SDK_400, "code": 499}, "refused"),
        (SDK_API, "refused"),
        ({"status": "error", "message": "Insufficient funds"}, "refused"),
        (None, "unknown"),
    ],
)
def test_send_classifies_returned_error_dicts(resp, expected, tmp_path):
    class Client:
        def placeorder(self, **k):
            return resp

    s = st.EmaCrossStrategy(Client(), ["X"], clock=lambda: datetime(2026, 11, 4), data_dir=tmp_path)
    assert s.place("X", "SELL", 10, "SL-M")[0] == expected


# =============================================================================
# ORDER SAFETY: a broker that fills orders and can lose answers
# =============================================================================
class HostileBroker:
    """Orders can be hidden from the books for a while, fill in parts or late,
    and any answer can be lost after the broker accepted the order: raised as
    an exception, or returned as the SDK's own error dict (`lost_reply`)."""

    def __init__(self, symbols, now):
        self.pos = dict.fromkeys(symbols, 0)
        self.orders = {}
        self.n = 0
        self.calls = []
        self.hold_market = False
        self.openpos_fail = 0
        self.openpos_lag = 0
        self.stale = None
        self.orderbook_fail = False
        self.hidden = set()
        self.lost_answer = set()  # placeorder price types whose answer is lost
        self.lost_reply = None  # None: raise; else the dict returned instead
        self.refuse = {}  # placeorder price type -> refusal dict (nothing is created)
        self.smart_lost = False
        self.smart_refuse = False
        self.cancel_ok = True
        self.ltp_value = 100.0
        self.stop_fill = None
        self.status_qty_override = None
        self.positions_extra = []
        self.telegrams = []
        idx = pd.date_range(
            end=now.replace(second=0) - timedelta(minutes=5), periods=60, freq="5min"
        )
        self.bars = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx
        )

    def _new(self, symbol, action, qty, ptype, status):
        self.n += 1
        oid = f"O{self.n}"
        self.orders[oid] = {
            "orderid": oid,
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "filled": 0,
            "pricetype": ptype,
            "order_status": status,
            "exchange": "NSE",
            "product": "MIS",
            "avg": 0.0,
        }
        return oid

    def _fill(self, o, q=None, price=None):
        q = o["qty"] - o["filled"] if q is None else q
        o["filled"] += q
        self.pos[o["symbol"]] += q if o["action"] == "BUY" else -q
        o["avg"] = self.ltp_value if price is None else price
        if o["filled"] >= o["qty"]:
            o["order_status"] = "complete"

    def _lost(self):
        if self.lost_reply is None:
            raise TimeoutError("read timed out")
        return dict(self.lost_reply)

    def release(self, q=None):
        for o in list(self.orders.values()):
            if o["order_status"] == "open":
                self._fill(o, q)

    def trigger_stops(self):
        for o in list(self.orders.values()):
            if o["order_status"] == "trigger pending":
                self._fill(o, price=self.stop_fill)

    def live(self, symbol=None, types=("SL", "SL-M")):
        return [
            o
            for o in self.orders.values()
            if o["order_status"] in ("open", "trigger pending")
            and o["pricetype"] in types
            and (symbol is None or o["symbol"] == symbol)
        ]

    def symbol(self, symbol, **k):
        return {"status": "success", "data": {"tick_size": 0.05, "lotsize": 1}}

    def analyzerstatus(self):
        return {"status": "success", "data": {"mode": "analyze"}}

    def quotes(self, symbol, **k):
        return {"status": "success", "data": {"ltp": self.ltp_value}}

    def history(self, **k):
        return self.bars

    def telegram(self, username, message, **k):
        self.telegrams.append((username, message))
        return {"status": "success"}

    def openposition(self, symbol, **k):
        if self.openpos_fail > 0:
            self.openpos_fail -= 1
            raise TimeoutError("read timed out")
        if self.openpos_lag > 0:
            self.openpos_lag -= 1
            return {"status": "success", "quantity": str(self.stale[symbol])}
        return {"status": "success", "quantity": str(self.pos.get(symbol, 0))}

    def positionbook(self, **k):
        rows = [
            {"symbol": s, "exchange": "NSE", "product": "MIS", "quantity": str(q)}
            for s, q in self.pos.items()
        ]
        return {"status": "success", "data": rows + list(self.positions_extra)}

    def placeorder(self, symbol, action, quantity, price_type, **k):
        self.calls.append(("place", symbol, action, quantity, price_type))
        if price_type in self.refuse:
            return dict(self.refuse[price_type])
        if price_type == "MARKET":
            oid = self._new(symbol, action, quantity, "MARKET", "open")
            if not self.hold_market:
                self._fill(self.orders[oid])
        else:
            oid = self._new(symbol, action, quantity, price_type, "trigger pending")
        if price_type in self.lost_answer:
            return self._lost()
        return {"status": "success", "orderid": oid}

    def placesmartorder(self, symbol, action, quantity, position_size, **k):
        delta = position_size - self.pos[symbol]
        self.calls.append(("smart", symbol, delta))
        if self.smart_refuse:
            return {"status": "error", "message": "RMS: order rejected"}
        if delta == 0:
            return {"status": "success", "message": "Positions Already Matched. No Action needed."}
        oid = self._new(symbol, "BUY" if delta > 0 else "SELL", abs(delta), "MARKET", "open")
        if not self.hold_market:
            self._fill(self.orders[oid])
        if self.smart_lost:
            return self._lost()
        return {"status": "success", "orderid": oid}

    def orderstatus(self, order_id, **k):
        o = self.orders.get(order_id)
        if o is None or order_id in self.hidden:
            return {"status": "error", "message": "Order not found"}
        q = self.status_qty_override if self.status_qty_override is not None else o["qty"]
        return {
            "status": "success",
            "data": {
                "order_status": o["order_status"],
                "average_price": o["avg"],
                "quantity": str(q),
            },
        }

    def orderbook(self, **k):
        if self.orderbook_fail:
            return {"status": "error", "message": "orderbook failed"}
        rows = [
            {**o, "quantity": str(o["qty"])}
            for oid, o in self.orders.items()
            if oid not in self.hidden
        ]
        return {"status": "success", "data": {"orders": rows}}

    def tradebook(self, **k):
        return {"status": "success", "data": []}

    def cancelorder(self, order_id, **k):
        self.calls.append(("cancel", order_id))
        o = self.orders.get(order_id)
        if o and self.cancel_ok and o["order_status"] in ("open", "trigger pending"):
            o["order_status"] = "cancelled"
        return {"status": "success"}


R_BARS = [datetime(2026, 11, 4, 9, 40) + timedelta(minutes=5 * i) for i in range(10)]


@pytest.fixture
def hostile(tmp_path, monkeypatch):
    fake_time = FakeTime()
    monkeypatch.setattr(st, "time", fake_time)
    monkeypatch.setattr(st, "plan_trade", fake_plan)
    made = {"n": 0}

    def make(data_dir=None, broker=None):
        made["n"] += 1
        now = datetime(2026, 11, 4, 10, 0, 6)
        b = broker or HostileBroker(["AAA", "BBB", "CCC"], now)
        clock = {"now": now}
        s = st.EmaCrossStrategy(
            b,
            symbols=["AAA", "BBB", "CCC"],
            clock=lambda: clock["now"],
            data_dir=data_dir or tmp_path / f"run{made['n']}",
            pause=0,
        )
        s.test_clock = clock
        s.setup()
        script = {}
        s.evaluate_symbol = lambda sym, bars: script.get(sym)
        return s, b, script

    make.time = fake_time
    return make


def _advance(s, seconds):
    s.test_clock["now"] += timedelta(seconds=seconds)
    return s.test_clock["now"]


def _rbar(s, i):
    s.bar_done.clear()
    s.on_bar(s.test_clock["now"], R_BARS[i])


def _orders_sent(b):
    return sum(1 for c in b.calls if c[0] in ("place", "smart"))


def _smarts(b):
    return sum(1 for c in b.calls if c[0] == "smart")


def _hide(b, hidden_type):
    """Orders of `hidden_type` stay out of the order book and status until cleared."""
    real_place = b.placeorder

    def place_hidden(symbol, action, quantity, price_type, **k):
        try:
            return real_place(symbol, action, quantity, price_type, **k)
        finally:
            if price_type == hidden_type:
                b.hidden.add(f"O{b.n}")

    b.placeorder = place_hidden
    return real_place


def _alerts(s):
    seen = []
    real = s.notify

    def record(text):
        seen.append(text)
        real(text)

    s.notify = record
    return seen


def test_unreadable_position_after_the_entry_fill_still_manages_it(hostile):
    s, b, script = hostile()
    real = b.orderstatus

    def status_then_fail(order_id, **k):
        r = real(order_id, **k)
        if order_id == "O1" and r["data"]["order_status"] == "complete":
            b.openpos_fail = 1
        return r

    b.orderstatus = status_then_fail
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.orderstatus = real
    assert s.state["open"]["symbol"] == "AAA" and s.state["open"]["qty"] == 10
    assert len(b.live("AAA")) == 1
    script.clear()
    script["BBB"] = "BUY"
    _rbar(s, 1)
    assert b.pos == {"AAA": 10, "BBB": 0, "CCC": 0}


def test_a_slow_exit_is_cancelled_before_it_is_retried(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.hold_market = True
    s.exit_trade("take profit")
    first_exit = s.state["open"]["exit_order_id"]
    assert first_exit and b.pos["AAA"] == 10
    s.manage_open(s.test_clock["now"])  # inside its backoff the working exit is left alone
    assert b.orders[first_exit]["order_status"] == "open"
    assert ("cancel", first_exit) not in b.calls
    s.manage_open(_advance(s, st.EXIT_RETRY_SEC + 1))
    assert b.orders[first_exit]["order_status"] == "cancelled"
    assert len(b.live("AAA", ("MARKET",))) == 1  # never two exit orders at once
    b.release()
    s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_unreadable_position_after_the_exit_order_is_not_booked_closed(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.hold_market = True
    real = b.placesmartorder

    def smart_then_fail(**k):
        r = real(**k)
        b.openpos_fail = 1
        return r

    b.placesmartorder = smart_then_fail
    s.exit_trade("take profit")
    assert s.state["open"]["symbol"] == "AAA"
    b.placesmartorder = real
    b.hold_market = False
    script.clear()
    script["BBB"] = "BUY"
    _rbar(s, 1)
    assert b.pos["BBB"] == 0


@pytest.mark.parametrize("books", [True, False])
def test_a_lost_stop_answer_never_leads_to_two_stops(hostile, books):
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    b.orderbook_fail = not books
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    assert len(b.live("AAA")) == 1  # no SL fallback after a lost answer
    assert s.state["open"]["stop_check"]
    s.manage_open(s.test_clock["now"])
    if books:
        assert s.state["open"]["stop_order_id"] == b.live("AAA")[0]["orderid"]
    else:
        assert s.state["open"]["stop_check"]
    s.manage_open(_advance(s, st.STOP_CONFIRM_SEC + 1))
    assert len(b.live("AAA")) == 1
    b.trigger_stops()
    assert b.pos["AAA"] == 0


def test_a_lost_entry_answer_is_taken_over_with_its_stop(hostile):
    s, b, script = hostile()
    b.lost_answer = {"MARKET"}
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1
    script.clear()
    script["BBB"] = "BUY"
    _rbar(s, 1)
    assert b.pos["BBB"] == 0


def test_a_lost_entry_answer_still_working_is_cancelled_not_doubled(hostile):
    s, b, script = hostile()
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    b.hold_market = False
    assert s.state["pending"] is not None
    assert b.orders["O1"]["order_status"] == "cancelled"
    script.clear()
    s.step()
    assert s.state["pending"] is not None  # inside the grace period nothing is concluded
    _advance(s, st.ENTRY_CONFIRM_SEC + 1)
    s.step()
    assert s.state["pending"] is None and s.state["open"] is None and b.pos["AAA"] == 0
    assert [c for c in b.calls if c[0] == "place"] == [("place", "AAA", "BUY", 10, "MARKET")]


def test_a_lost_entry_answer_leaves_the_users_own_order_alone(hostile):
    s, b, script = hostile()
    manual = b._new("AAA", "BUY", 3, "LIMIT", "open")
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    script["AAA"] = "BUY"
    _rbar(s, 0)
    assert b.orders[manual]["order_status"] == "open"
    ours = [o for o in b.orders.values() if o["pricetype"] == "MARKET"]
    assert [o["order_status"] for o in ours] == ["cancelled"]
    assert s.state["pending"]


def test_a_lost_entry_answer_waits_before_concluding_anything(hostile):
    s, b, script = hostile()
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    real_place = _hide(b, "MARKET")
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.placeorder = real_place
    b.lost_answer = set()
    assert s.state["pending"]
    script.clear()
    s.step()
    assert s.state["pending"]
    b.hidden.clear()
    b.release()  # it filled after all
    s.step()
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1


def test_the_wait_after_a_lost_entry_answer_reads_the_position_late(hostile):
    s, b, script = hostile()
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    hostile.time.hooks.append(b.release)
    script["AAA"] = "BUY"
    _rbar(s, 0)
    hostile.time.hooks.clear()
    assert s.state["open"]["symbol"] == "AAA"
    assert [c for c in b.calls if c[0] == "cancel"] == []


def test_a_lost_exit_answer_is_resolved_from_the_books(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.smart_lost = True
    b.ltp_value = 106.0
    s.exit_trade("take profit")  # the exit filled at 106 but the answer was lost
    b.smart_lost = False
    assert b.pos["AAA"] == 0 and s.state["open"]["symbol"] == "AAA"
    b.ltp_value = 90.0
    s.manage_open(s.test_clock["now"])
    assert s.state["open"] is None and _smarts(b) == 1


def test_a_lost_exit_that_fills_late_is_not_doubled(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.hold_market = True
    b.smart_lost = True
    s.exit_trade("take profit")
    b.smart_lost = False
    hostile.time.hooks.append(b.release)
    s.manage_open(s.test_clock["now"])
    hostile.time.hooks.clear()
    for _ in range(3):
        if s.state["open"]:
            s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_a_lost_exit_not_yet_in_the_book_is_waited_for(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.hold_market = True
    b.smart_lost = True
    s.exit_trade("take profit")
    b.smart_lost = False
    lost = next(
        k for k, o in b.orders.items() if o["pricetype"] == "MARKET" and o["action"] == "SELL"
    )
    b.hidden.add(lost)
    for _ in range(3):
        s.manage_open(s.test_clock["now"])
    assert [c for c in b.calls if c[0] == "smart"] == [("smart", "AAA", -10)]
    b.hidden.clear()
    b.release()
    s.manage_open(s.test_clock["now"])
    assert s.state["open"] is None and s.state["sweep"]


def test_a_lost_exit_answer_is_resent_only_after_the_grace_period(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.smart_lost = True
    b.hold_market = True
    s.exit_trade("take profit")
    b.smart_lost = False
    lost = next(
        k for k, o in b.orders.items() if o["pricetype"] == "MARKET" and o["action"] == "SELL"
    )
    b.orders[lost]["order_status"] = "rejected"
    b.hold_market = False
    s.manage_open(_advance(s, st.EXIT_CONFIRM_SEC - 5))
    assert _smarts(b) == 1
    s.manage_open(_advance(s, 10))
    assert _smarts(b) == 2 and b.pos["AAA"] == 0 and s.state["open"] is None


def test_a_lost_stop_answer_that_shows_up_late_is_adopted_not_doubled(hostile):
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.placeorder = real_place
    b.lost_answer = set()
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL-M"]
    b.hidden.clear()
    for _ in range(3):
        s.step()
    assert len(b.live("AAA")) == 1 and s.state["open"]["stop_order_id"] == "O2"
    b.trigger_stops()
    s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_a_lost_stop_answer_is_replaced_only_after_the_grace_period(hostile):
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    b.orders["O2"]["order_status"] = "rejected"
    s.manage_open(_advance(s, st.STOP_CONFIRM_SEC - 5))
    assert b.live("AAA") == []
    s.manage_open(_advance(s, 10))
    assert len(b.live("AAA")) == 1 and s.state["open"]["stop_order_id"]


def test_a_clear_stop_refusal_sends_the_sl_fallback(hostile):
    s, b, script = hostile()
    b.refuse = {"SL-M": {"status": "error", "message": "SL-M orders are not allowed"}}
    script["AAA"] = "BUY"
    _rbar(s, 0)
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL"]
    assert s.state["open"]["stop_type"] == "SL"


def test_unreadable_state_with_a_working_stop_closes_cleanly(hostile, tmp_path):
    d = tmp_path / "r5"
    s, b, script = hostile(data_dir=d)
    script["AAA"] = "BUY"
    _rbar(s, 0)
    s.state_path.write_text(s.state_path.read_text()[:30])
    s2, _, _ = hostile(data_dir=d, broker=b)
    assert list(d.glob("state.unreadable-*.json"))
    assert s2.state["open"]["recovered"]
    for _ in range(3):
        s2.step()
    assert b.pos["AAA"] == 0 and b.live("AAA") == [] and s2.state["pnl"] == 0.0
    b.trigger_stops()
    assert b.pos["AAA"] == 0


def test_every_position_found_after_a_lost_state_file_is_closed_in_turn(hostile, tmp_path):
    d = tmp_path / "r5b"
    d.mkdir()
    (d / "state.json").write_text("{not json")
    b = HostileBroker(["AAA", "BBB", "CCC"], datetime(2026, 11, 4, 10, 0, 6))
    b.pos.update(AAA=10, BBB=-5)
    s, _, _ = hostile(data_dir=d, broker=b)
    assert s.state["open"]["symbol"] == "AAA" and s.untracked == {"BBB"}
    for _ in range(3):
        s.step()
    assert b.pos == {"AAA": 0, "BBB": 0, "CCC": 0}


def test_restarts_mid_order(hostile, tmp_path):
    d = tmp_path / "a"
    s, b, script = hostile(data_dir=d)
    b.hold_market = True
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.release(4)
    s2, _, _ = hostile(data_dir=d, broker=b)
    assert s2.state["open"]["qty"] == 4 and [o["qty"] for o in b.live("AAA")] == [4]
    assert s2.state["open"]["entry_bar"] == "2026-11-04 09:40"  # kept for the take profit

    d = tmp_path / "b"
    s, b, script = hostile(data_dir=d)
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.hold_market = True
    b.smart_lost = True
    s.exit_trade("take profit")
    b.smart_lost = False
    s2, _, _ = hostile(data_dir=d, broker=b)
    s2.step()
    b.release()
    for _ in range(2):
        if s2.state["open"]:
            s2.step()
    assert b.pos["AAA"] == 0 and s2.state["open"] is None

    d = tmp_path / "c"
    s, b, script = hostile(data_dir=d)
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    b.orderbook_fail = True
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    b.orderbook_fail = False
    b.release()
    s2, _, _ = hostile(data_dir=d, broker=b)
    assert s2.state["open"] and len(b.live("AAA")) == 1


def test_partial_fills(hostile):
    s, b, script = hostile()
    b.hold_market = True
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.release(6)
    _advance(s, 70)
    s.step()
    assert s.state["open"]["qty"] == 6 and [o["qty"] for o in b.live("AAA")] == [6]
    s.exit_trade("take profit")
    b.release(2)
    for _ in range(4):
        if s.state["open"]:
            s.manage_open(s.test_clock["now"])
            b.release()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_the_stop_follows_the_position_size(hostile):
    s, b, script = hostile()
    b.status_qty_override = 7
    real = b.orderstatus

    def fail_position_after_fill(order_id, **k):
        r = real(order_id, **k)
        if r.get("data", {}).get("order_status") == "complete" and order_id == "O1":
            b.openpos_fail = 1
        return r

    b.orderstatus = fail_position_after_fill
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.orderstatus = real
    assert [o["qty"] for o in b.live("AAA")] == [10] and s.state["open"]["qty"] == 10
    b.trigger_stops()
    assert b.pos["AAA"] == 0


def test_the_stop_is_resized_when_the_position_grows(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.pos["AAA"] = 13
    script.clear()
    _rbar(s, 1)
    assert [o["qty"] for o in b.live("AAA")] == [13] and s.state["open"]["qty"] == 13


def test_a_stop_smaller_than_the_position_leaves_a_managed_residual(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.pos["AAA"] += 3
    b.trigger_stops()
    script.clear()
    s.step()
    assert s.state["open"]["residual"] and s.state["open"]["qty"] == 3
    script["BBB"] = "BUY"
    _rbar(s, 1)
    assert b.pos["BBB"] == 0
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_two_stops_filling_never_leaves_an_unmanaged_reversal(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b._new("AAA", "SELL", 10, "SL", "trigger pending")
    b.trigger_stops()
    assert b.pos["AAA"] == -10
    script.clear()
    s.step()
    trade = s.state["open"]
    assert trade["residual"] and trade["side"] == "SELL" and trade["qty"] == 10
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_a_fill_the_position_book_shows_late_is_not_forgotten(hostile):
    s, b, script = hostile()
    b.stale = dict.fromkeys(["AAA", "BBB", "CCC"], 0)
    real = b.orderstatus

    def lag(order_id, **k):
        r = real(order_id, **k)
        if r.get("data", {}).get("order_status") == "complete" and order_id == "O1":
            b.openpos_lag = 1
        return r

    b.orderstatus = lag
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.orderstatus = real
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1


def test_a_complete_entry_with_a_flat_position_keeps_blocking(hostile):
    s, b, script = hostile()
    b.stale = dict.fromkeys(["AAA", "BBB", "CCC"], 0)
    real = b.orderstatus

    def lag(order_id, **k):
        r = real(order_id, **k)
        if order_id == "O1":
            b.openpos_lag = 10
        return r

    b.orderstatus = lag
    script["AAA"] = "BUY"
    _rbar(s, 0)
    assert s.state["pending"] and s.state["pending"].get("alerted")
    script.clear()
    script["BBB"] = "BUY"
    _rbar(s, 1)
    assert b.pos["BBB"] == 0
    b.orderstatus = real
    b.openpos_lag = 0
    s.step()
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1


def test_an_exit_that_never_fills_backs_off_and_alerts(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.hold_market = True
    sent = _orders_sent(b)
    s.exit_trade("take profit")
    for _ in range(30):
        s.manage_open(s.test_clock["now"])
    assert _orders_sent(b) - sent == 1
    alerts = []
    s.notify = alerts.append
    for _ in range(40):
        s.manage_open(_advance(s, 10))
    sends = _orders_sent(b) - sent
    assert 3 <= sends <= 7
    assert any("has not filled after 3 attempts" in a for a in alerts)
    assert len(b.live("AAA", ("MARKET",))) == 1


def test_a_refused_exit_is_retried_with_backoff(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.smart_refuse = True
    s.exit_trade("take profit")
    for _ in range(10):
        s.manage_open(s.test_clock["now"])
    assert _smarts(b) == 1
    b.smart_refuse = False
    s.manage_open(_advance(s, st.EXIT_RETRY_SEC + 1))
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_stop_filled_with_positions_unreadable_books_it_and_blocks(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.trigger_stops()
    b.openpos_fail = 10**6
    sent = _orders_sent(b)
    script.clear()
    for _ in range(5):
        s.step()
    assert s.state["open"] is None and _orders_sent(b) == sent
    assert "AAA" in s.untracked
    assert s.can_enter(datetime(2026, 11, 4, 10, 0, 6))[0] is False


def test_a_stop_is_booked_at_its_fill_price_not_the_trigger(hostile):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.stop_fill = 96.5
    b.trigger_stops()
    script.clear()
    s.step()
    assert s.state["open"] is None and s.state["pnl"] == pytest.approx(-35.0)


def test_positions_not_managed_here_block_entries(hostile):
    s, b, script = hostile()
    assert s.can_enter(datetime(2026, 11, 4, 10, 0, 6)) == (True, "")
    b.pos["CCC"] = 4
    script["BBB"] = "BUY"
    _rbar(s, 0)
    assert s.untracked == {"CCC"} and b.pos["BBB"] == 0
    ok, why = s.can_enter(datetime(2026, 11, 4, 10, 0, 6))
    assert not ok and "CCC" in why
    b.pos["CCC"] = 0
    script.clear()
    _rbar(s, 1)
    assert s.untracked == set()


def test_step_survives_missing_quotes_and_history(hostile):
    s, b, _script = hostile()
    s.evaluate_symbol = st.EmaCrossStrategy.evaluate_symbol.__get__(s)
    b.quotes = lambda **k: None
    b.history = lambda **k: {"status": "error", "message": "boom"}
    for _ in range(3):
        s.step()


# =============================================================================
# ROUND 4: SDK error dicts returned for entry, stop and exit
# =============================================================================
@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_entry_answer_returned_as_an_sdk_error_is_confirmed_not_dropped(hostile, reply):
    s, b, script = hostile()
    b.lost_answer = {"MARKET"}
    b.lost_reply = reply
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    # the order filled at the broker: taken over with its stop, never sent twice
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1
    assert [c for c in b.calls if c[0] == "place" and c[4] == "MARKET"] == [
        ("place", "AAA", "BUY", 10, "MARKET")
    ]


@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_entry_answer_returned_as_an_sdk_error_still_working_is_cancelled(hostile, reply):
    s, b, script = hostile()
    b.lost_answer = {"MARKET"}
    b.lost_reply = reply
    b.hold_market = True
    script["AAA"] = "BUY"
    _rbar(s, 0)
    assert s.state["pending"] and b.orders["O1"]["order_status"] == "cancelled"
    script["BBB"] = "BUY"
    _rbar(s, 1)
    assert b.pos["BBB"] == 0  # every symbol blocked while it is confirmed


@pytest.mark.parametrize("reply", [SDK_CONNECT, SDK_400, SDK_API], ids=["connect", "400", "api"])
def test_a_refused_entry_is_dropped(hostile, reply):
    s, b, script = hostile()
    b.refuse = {"MARKET": reply}
    script["AAA"] = "BUY"
    _rbar(s, 0)
    assert s.state["pending"] is None and s.state["open"] is None
    assert s.can_enter(s.test_clock["now"]) == (True, "")


@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_stop_answer_returned_as_an_sdk_error_sends_no_fallback(hostile, reply):
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    b.lost_reply = reply
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL-M"]  # no SL sent on top
    assert s.state["open"]["stop_check"] and s.state["open"]["stop_unknown_since"]
    s.manage_open(s.test_clock["now"])
    assert s.state["open"]["stop_order_id"] == "O2"  # the stop that exists is adopted


@pytest.mark.parametrize("reply", [SDK_CONNECT, SDK_400], ids=["connect", "400"])
def test_a_stop_refused_by_an_sdk_error_sends_the_sl_fallback(hostile, reply):
    s, b, script = hostile()
    b.refuse = {"SL-M": reply}
    script["AAA"] = "BUY"
    _rbar(s, 0)
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL"]


@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_exit_answer_returned_as_an_sdk_error_is_not_doubled(hostile, reply):
    s, b, script = hostile()
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.smart_lost = True
    b.lost_reply = reply
    b.hold_market = True
    s.exit_trade("take profit")
    b.smart_lost = False
    assert s.state["open"]["exit_unanswered"]
    lost = next(
        k for k, o in b.orders.items() if o["pricetype"] == "MARKET" and o["action"] == "SELL"
    )
    b.hidden.add(lost)  # and it does not show in the books yet
    for _ in range(3):
        s.manage_open(_advance(s, 5))
    assert _smarts(b) == 1
    b.hidden.clear()
    b.release()
    s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None and _smarts(b) == 1
    assert s.state["sweep"]


# =============================================================================
# ROUND 4: notify about positions not managed here
# =============================================================================
def test_the_bar_scan_notifies_about_an_unmanaged_position_once(hostile, monkeypatch):
    monkeypatch.setattr(st, "TELEGRAM_USER", "trader")
    s, b, script = hostile()
    alerts = _alerts(s)
    b.pos["CCC"] = 4
    _rbar(s, 0)
    assert [a for a in alerts if "CCC" in a and "not managing" in a]
    assert any("CCC" in m for _u, m in b.telegrams)
    count = len(alerts)
    _rbar(s, 1)
    assert len(alerts) == count  # reported once, not every bar
    assert b.pos["CCC"] == 4 and not [c for c in b.calls if c[1:2] == ("CCC",)]


def test_a_position_found_at_start_is_notified(hostile, monkeypatch):
    monkeypatch.setattr(st, "TELEGRAM_USER", "trader")
    b = HostileBroker(["AAA", "BBB", "CCC"], datetime(2026, 11, 4, 10, 0, 6))
    b.pos["BBB"] = 7
    s, _, _ = hostile(broker=b)
    assert s.untracked == {"BBB"} and any("BBB" in m for _u, m in b.telegrams)


def _middle_path_close(hostile):
    """A lost stop answer, the stop hidden from the book, and the price through
    the stop inside the grace period: closed at market at once."""
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.placeorder = real_place
    b.lost_answer = set()
    script.clear()
    assert s.state["open"]["stop_check"] and b.pos["AAA"] == 10
    b.ltp_value = 97.0  # through the 98 stop
    s.manage_open(_advance(s, 5))
    return s, b, script


def test_middle_path_a_cross_inside_the_stop_grace_exits_at_once(hostile):
    s, b, _script = _middle_path_close(hostile)
    assert b.pos["AAA"] == 0 and s.state["open"] is None and _smarts(b) == 1
    assert s.state["sweep"]["symbol"] == "AAA"  # the sweep runs after it
    assert s._since(s.state["sweep"]["until"]) == pytest.approx(-(st.SWEEP_SEC), abs=1)
    assert s.can_enter(s.test_clock["now"])[0] is False


def test_middle_path_the_late_stop_is_cancelled_by_the_sweep(hostile):
    s, b, _script = _middle_path_close(hostile)
    alerts = _alerts(s)
    b.hidden.clear()  # the lost stop shows up
    s.step()
    assert b.live("AAA") == [] and b.orders["O2"]["order_status"] == "cancelled"
    assert any("cancelling a closing order" in a for a in alerts)
    assert b.pos["AAA"] == 0


def test_middle_path_a_late_stop_that_fills_is_taken_over_and_closed(hostile):
    s, b, _script = _middle_path_close(hostile)
    alerts = _alerts(s)
    b.hidden.clear()
    b.trigger_stops()  # the lost stop filled after the close: short 10
    assert b.pos["AAA"] == -10
    s.step()
    s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None
    assert any("appeared after the trade was closed" in a for a in alerts)


@pytest.mark.parametrize("fault", ["book", "position"])
def test_middle_path_waits_when_the_book_or_position_cannot_be_read(hostile, fault):
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.placeorder = real_place
    b.lost_answer = set()
    script.clear()
    if fault == "book":
        b.orderbook_fail = True
    else:
        b.openpos_fail = 10**6
    b.ltp_value = 97.0
    for _ in range(3):
        s.manage_open(_advance(s, 5))
    assert _smarts(b) == 0 and b.pos["AAA"] == 10 and s.state["open"]
    b.orderbook_fail, b.openpos_fail = False, 0
    s.manage_open(_advance(s, 5))
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_middle_path_a_visible_stop_is_left_to_work(hostile):
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.lost_answer = set()
    b.ltp_value = 97.0
    s.manage_open(_advance(s, 5))  # the stop shows in the book: adopted, nothing sent
    assert _smarts(b) == 0 and s.state["open"]["stop_order_id"] == "O2"
    b.trigger_stops()
    s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_no_cross_inside_the_stop_grace_sends_nothing(hostile):
    s, b, script = hostile()
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    script["AAA"] = "BUY"
    _rbar(s, 0)
    b.placeorder = real_place
    b.lost_answer = set()
    sent = _orders_sent(b)
    b.ltp_value = 99.0
    for _ in range(3):
        s.manage_open(_advance(s, 5))
    assert _orders_sent(b) == sent and b.pos["AAA"] == 10


# =============================================================================
# ROUND 4: positions that appear during and after the sweep
# =============================================================================
def test_a_long_appearing_during_the_sweep_is_notified_and_blocks(hostile):
    s, b, _script = _middle_path_close(hostile)
    alerts = _alerts(s)
    b.pos["AAA"] = 5  # bought by hand during the sweep: not a reversal
    s.step()
    assert "AAA" in s.untracked and b.pos["AAA"] == 5 and s.state["open"] is None
    assert any("AAA" in a and "not managing" in a for a in alerts)


def test_an_opposite_position_after_the_sweep_is_taken_over_and_closed(hostile):
    s, b, _script = _middle_path_close(hostile)
    s.step()
    _advance(s, st.SWEEP_SEC + 1)
    s.step()
    assert s.state["sweep"] is None
    alerts = _alerts(s)
    b.hidden.clear()
    assert b.orders["O2"]["order_status"] == "trigger pending"  # unseen by the sweep
    b.trigger_stops()  # the lost stop fills only now, after the sweep window
    assert b.pos["AAA"] == -10
    _rbar(s, 1)  # the once-a-bar position scan finds it
    assert s.state["open"]["recovered"] and s.state["open"]["side"] == "SELL"
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None
    assert any("appeared after the trade was closed" in a for a in alerts)


def test_an_opposite_position_waits_for_the_open_trade_then_is_closed(hostile):
    s, b, script = _middle_path_close(hostile)
    _advance(s, st.SWEEP_SEC + 1)
    s.step()
    assert s.state["sweep"] is None
    b.ltp_value = 100.0
    script["BBB"] = "BUY"
    _rbar(s, 1)
    assert s.state["open"]["symbol"] == "BBB"
    script.clear()
    alerts = _alerts(s)
    b.hidden.clear()
    b._fill(b.orders["O2"])  # only the lost AAA stop fills, now
    assert b.pos["AAA"] == -10
    _rbar(s, 2)
    assert "AAA" in s.untracked and s.state["open"]["symbol"] == "BBB"  # one slot: it waits
    assert any("AAA" in a and "not managing" in a for a in alerts)
    s.exit_trade("take profit")
    assert s.state["open"] is None and b.pos["BBB"] == 0
    _rbar(s, 3)
    assert s.state["open"]["symbol"] == "AAA" and s.state["open"]["recovered"]
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


# =============================================================================
# END TO END THROUGH THE REAL BACKTEST RUNNER
# =============================================================================
E2E_SYMBOLS = ["SBIN", "M&M", "TCS", "LT", "BSE", "MARUTI"]
E2E_BASES = (800, 3000, 3500, 3600, 2500, 12000)


@pytest.fixture(scope="module")
def multi_data_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("emax_bars")
    start, end = date(2026, 8, 24), date(2026, 9, 25)
    for n, (symbol, base) in enumerate(zip(E2E_SYMBOLS, E2E_BASES, strict=True)):
        write_bars(directory, symbol, "NSE", start, end, base, 300 + n)
    return directory


def _ist(epoch_ms):
    return datetime.fromtimestamp(epoch_ms / 1000, tz=st.IST).replace(tzinfo=None)


def check_backtest(run_dir, result, symbols):
    """The invariants every run must hold. Returns the journal entries."""
    assert result["status"] == "completed", result.get("message")
    trades = sorted(result["trades"], key=lambda t: t["entry_time"])
    assert trades, "no trade in the test period"
    assert {t["symbol"] for t in trades} <= set(symbols)
    assert all(t["side"] == "LONG" for t in trades)  # long only
    for prev, nxt in zip(trades, trades[1:], strict=False):
        assert nxt["entry_time"] >= prev["exit_time"], (prev, nxt)  # one at a time
    for t in trades:
        assert t["quantity"] * t["entry_price"] <= 30000 + 1e-6
        opened, closed = _ist(t["entry_time"]), _ist(t["exit_time"])
        assert closed.date() == opened.date(), t  # intraday
        assert f"{closed:%H:%M}" <= "15:11", t  # flat by square-off
        assert "09:30" <= f"{opened:%H:%M}" <= "14:47", t
    assert all(o["order_status"] != "open" for o in result["orders"])

    journal = run_dir / "ema_cross_data" / "journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    entries = [r for r in rows if r["event"] == "entry"]
    assert entries and len(entries) == len([r for r in rows if r["event"] == "exit"])
    for r in entries:
        assert r["planned_loss"] <= 2000
        assert r["position_value"] <= 30000
        assert r["trade"]["side"] == "BUY"
        assert r["trade"]["stop"] < r["trade"]["entry_price"]
        assert r["trade"]["entry_bar"] < r["ts"][:16].replace("T", " ")
        assert "09:30" <= r["ts"][11:16] <= "14:47"
    per_day = {}
    for r in entries:
        per_day[r["ts"][:10]] = per_day.get(r["ts"][:10], 0) + 1
    assert max(per_day.values()) <= 5
    return rows


@pytest.mark.timeout(600)
def test_backtest_runs_unchanged_long_only_one_position_at_a_time(tmp_path, multi_data_dir):
    original = SCRIPT.read_bytes()
    run_dir, result, log = run_backtest(
        tmp_path,
        SCRIPT,
        multi_data_dir,
        "2026-09-14",
        "2026-09-25",
        extra_env={"EMAX_SYMBOLS": ",".join(E2E_SYMBOLS)},
        timeout=500,
    )
    assert SCRIPT.read_bytes() == original
    assert "6 symbols on NSE MIS" in log
    rows = check_backtest(run_dir, result, E2E_SYMBOLS)
    reasons = {r["reason"].split(":")[0].split(" (")[0] for r in rows if r["event"] == "exit"}
    assert "take profit" in reasons
