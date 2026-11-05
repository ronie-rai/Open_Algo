"""Tests for strategies/examples/supertrend_fvg_multi.py.

The strategy module is imported by path: it creates no client and makes no
network call at import. The last test runs it unchanged through the real
backtest runner on synthetic bars for several symbols.

Run:
    uv run pytest test/test_supertrend_fvg_strategy.py -v
"""

import importlib.util
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test.test_python_backtest_e2e import run_backtest, write_bars

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "strategies" / "examples" / "supertrend_fvg_multi.py"


def _load():
    spec = importlib.util.spec_from_file_location("supertrend_fvg_multi", SCRIPT)
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


# =============================================================================
# SUPERTREND: an independent, literal reading of the Pine study
# =============================================================================
def pine_reference(o, h, lo, c, periods, multiplier, change_atr):
    """The Pine v4 Supertrend written out with None for na, one bar at a time."""
    n = len(c)
    o, h, lo, c = (list(map(float, x)) for x in (o, h, lo, c))

    def nz(x, replacement):
        return replacement if x is None else x

    # tr (the built-in variable) is tr(false): na when close[1] is na
    tr = [None] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, n)
    ]
    tr_true = [h[0] - lo[0]] + tr[1:]

    def sma(src, length, i):
        if i < length - 1:
            return None
        window = src[i - length + 1 : i + 1]
        if any(v is None for v in window):
            return None
        return sum(window) / length

    atr = []
    if change_atr:
        # atr(length) = rma(tr(true), length)
        alpha = 1.0 / periods
        prev = None
        for i in range(n):
            value = (
                sma(tr_true, periods, i)
                if prev is None
                else alpha * tr_true[i] + (1 - alpha) * prev
            )
            atr.append(value)
            prev = value
    else:
        atr = [sma(tr, periods, i) for i in range(n)]

    up_hist, dn_hist, trend_hist, buys, sells = [], [], [], [], []
    for i in range(n):
        src = (h[i] + lo[i]) / 2
        a = atr[i]
        close_1 = c[i - 1] if i > 0 else None

        up = None if a is None else src - multiplier * a
        up1 = nz(up_hist[i - 1] if i > 0 else None, up)
        if close_1 is not None and up1 is not None and close_1 > up1:
            up = None if up is None else max(up, up1)
        dn = None if a is None else src + multiplier * a
        dn1 = nz(dn_hist[i - 1] if i > 0 else None, dn)
        if close_1 is not None and dn1 is not None and close_1 < dn1:
            dn = None if dn is None else min(dn, dn1)

        trend = 1
        trend_1 = trend_hist[i - 1] if i > 0 else None
        trend = nz(trend_1, trend)
        if trend == -1 and dn1 is not None and c[i] > dn1:
            trend = 1
        elif trend == 1 and up1 is not None and c[i] < up1:
            trend = -1

        up_hist.append(up)
        dn_hist.append(dn)
        trend_hist.append(trend)
        buys.append(trend == 1 and trend_1 == -1)
        sells.append(trend == -1 and trend_1 == 1)

    def arr(values):
        return np.array([np.nan if v is None else v for v in values], dtype=float)

    return {
        "atr": arr(atr),
        "up": arr(up_hist),
        "dn": arr(dn_hist),
        "trend": np.array(trend_hist),
        "buy": np.array(buys),
        "sell": np.array(sells),
    }


@pytest.mark.parametrize("change_atr", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_supertrend_matches_the_pine_study(change_atr, seed):
    o, h, lo, c = random_ohlc(500, seed)
    got = st.supertrend(o, h, lo, c, period=10, multiplier=3.0, source="hl2", change_atr=change_atr)
    ref = pine_reference(o, h, lo, c, 10, 3.0, change_atr)
    np.testing.assert_allclose(got["atr"], ref["atr"], rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(got["up"], ref["up"], rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(got["dn"], ref["dn"], rtol=1e-10, equal_nan=True)
    assert (got["trend"] == ref["trend"]).all()
    assert (got["buy"] == ref["buy"]).all()
    assert (got["sell"] == ref["sell"]).all()
    assert got["buy"].sum() >= 3 and got["sell"].sum() >= 3  # the data exercises both signals


def test_atr_warmup_follows_pine():
    o, h, lo, c = random_ohlc(40, 7)
    rma = st.supertrend(o, h, lo, c, period=10, change_atr=True)["atr"]
    sma = st.supertrend(o, h, lo, c, period=10, change_atr=False)["atr"]
    # ta.atr seeds with the SMA of the first 10 true ranges (bar 0 is high - low)
    assert np.isnan(rma[:9]).all() and np.isfinite(rma[9])
    tr0 = [h[0] - lo[0]] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, 10)
    ]
    assert rma[9] == pytest.approx(np.mean(tr0))
    # sma(tr, 10) with tr na on bar 0 first exists on bar 10
    assert np.isnan(sma[:10]).all() and np.isfinite(sma[10])
    # During the warmup the trend is 1 and nothing signals
    trend = st.supertrend(o, h, lo, c, period=10)["trend"]
    assert (trend[:9] == 1).all()


def test_supertrend_on_a_series_never_reads_ahead():
    o, h, lo, c = random_ohlc(300, 11)
    full = st.supertrend(o, h, lo, c)
    for t in (50, 120, 299):
        part = st.supertrend(o[: t + 1], h[: t + 1], lo[: t + 1], c[: t + 1])
        assert part["trend"][-1] == full["trend"][t]
        assert part["buy"][-1] == full["buy"][t] and part["sell"][-1] == full["sell"][t]


# =============================================================================
# FAIR VALUE GAPS
# =============================================================================
def _gaps(gaps, kind):
    return [g for g in gaps if g.kind == kind]


def test_three_candle_bullish_gap_and_its_fill_state():
    #        0      1      2      3      4      5
    h = np.array([10.0, 11.5, 13.0, 12.8, 12.6, 12.4])
    lo = np.array([9.0, 10.2, 12.0, 11.5, 10.8, 9.9])
    atr = np.ones(6)
    gaps = _gaps(st.detect_fvgs(h, lo, atr, mode="3candle", min_atr=0.05), "bull")
    assert len(gaps) == 1
    g = gaps[0]
    assert (g.bottom, g.top, g.created_idx) == (10.0, 12.0, 2)
    assert g.mid == 11.0
    assert g.mid_hit_idx == 4  # low 10.8 <= 11.0
    assert g.filled_idx == 5  # low 9.9 < 10.0

    # Evaluated at bar 3 the gap was entered but neither its midpoint nor its low was reached
    at3 = _gaps(st.detect_fvgs(h, lo, atr, mode="3candle", min_atr=0.05, end=3), "bull")[0]
    assert at3.mid_hit_idx == -1 and at3.filled_idx == -1
    # ...and bars after the evaluation bar change nothing
    trunc = _gaps(st.detect_fvgs(h[:4], lo[:4], atr[:4], mode="3candle", min_atr=0.05), "bull")[0]
    assert (trunc.mid_hit_idx, trunc.filled_idx) == (at3.mid_hit_idx, at3.filled_idx)


def test_three_candle_bearish_gap():
    h = np.array([21.0, 20.0, 18.0, 18.5, 19.2, 20.5])
    lo = np.array([20.0, 18.5, 17.0, 17.2, 17.5, 18.0])
    gaps = _gaps(st.detect_fvgs(h, lo, np.ones(6), mode="3candle", min_atr=0.05), "bear")
    assert len(gaps) == 1
    g = gaps[0]
    assert (g.bottom, g.top, g.created_idx) == (18.0, 20.0, 2)
    assert g.mid_hit_idx == 4  # high 19.2 >= 19.0
    assert g.filled_idx == 5  # high 20.5 > 20.0


def test_two_candle_mode_uses_consecutive_candles():
    # high[1] < low[2]: a two-candle gap; high[0] = 12 >= low[2] so no three-candle gap
    h = np.array([12.0, 10.0, 12.5])
    lo = np.array([9.0, 9.5, 11.0])
    atr = np.ones(3)
    two = _gaps(st.detect_fvgs(h, lo, atr, mode="2candle", min_atr=0.05), "bull")
    three = _gaps(st.detect_fvgs(h, lo, atr, mode="3candle", min_atr=0.05), "bull")
    assert [(g.bottom, g.top, g.created_idx) for g in two] == [(10.0, 11.0, 2)]
    assert three == []
    # bearish two-candle: low[1] > high[2]
    h2 = np.array([10.0, 11.0, 9.0])
    lo2 = np.array([9.0, 10.0, 8.0])
    bear = _gaps(st.detect_fvgs(h2, lo2, atr, mode="2candle", min_atr=0.05), "bear")
    assert [(g.bottom, g.top) for g in bear] == [(9.0, 10.0)]


def test_gaps_below_the_minimum_size_are_ignored():
    h = np.array([10.0, 11.0, 12.0])
    lo = np.array([9.0, 10.0, 10.02])  # a 0.02 gap
    assert _gaps(st.detect_fvgs(h, lo, np.ones(3), mode="3candle", min_atr=0.05), "bull") == []
    assert len(_gaps(st.detect_fvgs(h, lo, np.ones(3), mode="3candle", min_atr=0.0), "bull")) == 1


def test_unknown_fvg_mode_is_refused():
    with pytest.raises(ValueError):
        st.detect_fvgs(np.ones(3), np.ones(3), np.ones(3), mode="4candle")


@pytest.mark.parametrize("mode", ["3candle", "2candle"])
def test_gaps_and_blocks_never_read_ahead(mode):
    o, h, lo, c = random_ohlc(400, 5, vol=0.006)
    atr = st.atr_rma(h, lo, c, 14)
    min_atr = 0.05 if mode == "3candle" else 0.0  # two-candle gaps are rare on random bars
    full_gaps = st.detect_fvgs(h, lo, atr, mode=mode, min_atr=min_atr)
    assert len(full_gaps) >= 2
    full_blocks = st.detect_order_blocks(o, h, lo, c, atr)
    assert full_blocks
    later_state_differs = False
    for t in range(40, 400, 10):
        cut = st.detect_fvgs(h[: t + 1], lo[: t + 1], atr[: t + 1], mode=mode, min_atr=min_atr)
        at_t = st.detect_fvgs(h, lo, atr, mode=mode, min_atr=min_atr, end=t)
        assert [vars(g) for g in cut] == [vars(g) for g in at_t]
        assert all(g.created_idx <= t and g.filled_idx <= t and g.mid_hit_idx <= t for g in cut)
        full_by_bar = {(g.kind, g.created_idx): g for g in full_gaps}
        later_state_differs |= any(
            vars(g) != vars(full_by_bar[(g.kind, g.created_idx)]) for g in cut
        )
        b_cut = st.detect_order_blocks(
            o[: t + 1], h[: t + 1], lo[: t + 1], c[: t + 1], atr[: t + 1]
        )
        b_at = st.detect_order_blocks(o, h, lo, c, atr, end=t)
        assert [vars(b) for b in b_cut] == [vars(b) for b in b_at]
    # the full series does fill gaps later, so equality above is not vacuous
    assert later_state_differs


def test_order_block_detection():
    # bars 0-9 drift sideways, bar 8 is bearish, bar 10 is a big bullish breakout
    o = np.array([100, 100.5, 100.2, 100.4, 100.1, 100.3, 100.2, 100.4, 100.6, 100.1, 100.2, 103.5])
    c = np.array(
        [100.5, 100.2, 100.4, 100.1, 100.3, 100.2, 100.4, 100.6, 100.1, 100.2, 103.0, 103.8]
    )
    h = np.maximum(o, c) + 0.1
    lo = np.minimum(o, c) - 0.1
    atr = np.full(len(c), 1.0)
    blocks = st.detect_order_blocks(o, h, lo, c, atr, swing_len=10, impulse_atr=1.0)
    bull = [b for b in blocks if b.kind == "bull"]
    assert len(bull) == 1
    b = bull[0]
    assert b.created_idx == 10 and b.ob_idx == 8  # the last bearish candle before the impulse
    assert (b.bottom, b.top) == (lo[8], h[8])
    assert b.mitigated_idx == -1  # bar 11 stays above the block


# =============================================================================
# STOP AND TARGET SELECTION
# =============================================================================
Gap, Block = st.Gap, st.Block


def test_buy_takes_the_nearest_valid_stop_and_the_nearest_target_paying_min_rr():
    gaps = [
        Gap("bull", 99.8, 99.6, 10),  # 0.4 below entry: tighter than 0.3 x ATR 2 = 0.6
        Gap("bull", 99.0, 98.5, 11, filled_idx=20),  # traded through: not a stop
        Gap("bull", 98.0, 97.0, 12),  # the stop: its low, 97
        Gap("bull", 100.5, 99.5, 13),  # not entirely below entry
        Gap("bear", 103.0, 102.0, 14),  # mid 102.5: reward 2.5 < risk 3
        Gap("bear", 104.0, 103.0, 15, mid_hit_idx=30),  # midpoint already reached
        Gap("bear", 106.0, 104.0, 16),  # mid 105: reward 5 >= 3
    ]
    blocks = [Block("bull", 96.5, 96.0, 5, 6), Block("bull", 99.2, 98.4, 7, 8, mitigated_idx=9)]
    levels, why = st.select_levels("BUY", 100.0, 2.0, gaps, blocks, 0.05, 0.3, 1.0)
    assert why == ""
    assert levels["stop"] == 97.0 and levels["target"] == 105.0
    assert levels["rr"] == pytest.approx(5 / 3, abs=0.01)
    assert "FVG" in levels["stop_source"]


def test_buy_stop_can_come_from_an_order_block():
    gaps = [Gap("bull", 98.0, 97.0, 12), Gap("bear", 103.0, 102.0, 14)]
    blocks = [Block("bull", 99.0, 98.2, 7, 8)]
    levels, _ = st.select_levels("BUY", 100.0, 2.0, gaps, blocks, 0.05, 0.3, 1.0)
    assert levels["stop"] == 98.2 and "OB" in levels["stop_source"]
    assert levels["target"] == 102.5  # reward 2.5 >= risk 1.8


def test_sell_mirrors_buy():
    gaps = [
        Gap("bear", 100.3, 100.1, 10),  # 0.1 above entry: too tight
        Gap("bear", 103.0, 102.0, 12),  # the stop: its high, 103
        Gap("bull", 98.0, 97.0, 14),  # mid 97.5: reward 2.5 < 3
        Gap("bull", 96.0, 94.0, 16),  # mid 95: reward 5
        Gap("bull", 97.0, 96.0, 15, mid_hit_idx=20),
    ]
    blocks = [Block("bear", 101.0, 99.5, 3, 4)]  # not entirely above entry
    levels, why = st.select_levels("SELL", 100.0, 2.0, gaps, blocks, 0.05, 0.3, 1.0)
    assert why == ""
    assert levels["stop"] == 103.0 and levels["target"] == 95.0


def test_skips_without_a_stop_or_a_target():
    no_stop, why = st.select_levels(
        "BUY", 100.0, 2.0, [Gap("bear", 106, 104, 1)], [], 0.05, 0.3, 1.0
    )
    assert no_stop is None and "no unfilled bullish" in why
    no_target, why = st.select_levels(
        "BUY", 100.0, 2.0, [Gap("bull", 98, 97, 1)], [], 0.05, 0.3, 1.0
    )
    assert no_target is None and "no bearish FVG" in why
    too_tight, why = st.select_levels(
        "BUY",
        100.0,
        2.0,
        [Gap("bull", 99.9, 99.7, 1), Gap("bear", 106, 104, 2)],
        [],
        0.05,
        0.3,
        1.0,
    )
    assert too_tight is None and "closer than 0.3 ATR" in why
    low_rr, why = st.select_levels(
        "SELL", 100.0, 2.0, [Gap("bear", 103, 102, 1), Gap("bull", 98, 97, 2)], [], 0.05, 0.3, 1.5
    )
    assert low_rr is None and "pays 1.5R" in why


def test_plan_trade_sizes_and_reports_a_skip():
    o, h, lo, c = random_ohlc(600, 21, vol=0.006)
    idx = pd.date_range("2026-11-02 09:15", periods=600, freq="5min")
    bars = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)
    taken = skipped = 0
    for t in range(100, 600, 7):
        part = bars.iloc[: t + 1]
        for side in ("BUY", "SELL"):
            plan, why = st.plan_trade("SBIN", side, part, float(part["close"].iloc[-1]))
            if plan is None:
                assert why
                skipped += 1
                continue
            taken += 1
            assert plan.qty * abs(plan.entry - plan.stop) <= st.MAX_LOSS_PER_TRADE
            assert plan.qty * plan.entry <= st.CAPITAL
            assert (
                (plan.stop < plan.entry < plan.target)
                if side == "BUY"
                else (plan.target < plan.entry < plan.stop)
            )
            assert abs(plan.entry - plan.stop) >= st.MIN_STOP_ATR * plan.atr - 1e-9
            assert plan.rr >= st.MIN_RR
    assert taken > 0 and skipped > 0


# =============================================================================
# SIZING
# =============================================================================
def test_capital_cap_binds_on_a_tight_stop():
    qty, by = st.size_position(100.0, 98.0, 1, capital=25000, max_loss=2000, buffer_pct=0)
    assert (qty, by) == (250, "capital")  # risk would allow 1000


def test_risk_cap_binds_on_a_wide_stop():
    qty, by = st.size_position(100.0, 90.0, 1, capital=25000, max_loss=2000, buffer_pct=0)
    assert (qty, by) == (200, "risk")


def test_lot_rounding_and_zero_quantity():
    assert st.size_position(100.0, 90.0, 75, 25000, 2000, 0)[0] == 150
    assert st.size_position(30000.0, 29990.0, 1, 25000, 2000, 0)[0] == 0  # one share > capital
    assert st.size_position(100.0, 100.0, 1, 25000, 2000, 0)[0] == 0


def test_slippage_buffer_only_makes_sizing_more_conservative():
    qty, _ = st.size_position(100.0, 90.0, 1, 25000, 2000, 0.1)
    assert qty == 198  # floor(2000 / 10.1)
    # a fill 0.1% worse still keeps the loss at the stop under the cap
    assert qty * (100.1 - 90.0) <= 2000


# =============================================================================
# ONE TRADE AT A TIME, FLIP EXITS (fake broker)
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


def fake_plan(symbol, side, bars, entry, tick=0.05, lot=1):
    stop, target = (98.0, 104.0) if side == "BUY" else (102.0, 96.0)
    return st.Plan(
        symbol, side, entry, stop, target, 10, 2.0, 2.0, 1.0, "fvg", "fvg", "risk", "t"
    ), ""


class FakeTime:
    """time for the strategy module: sleep moves a fake monotonic clock on."""

    def __init__(self):
        self.t = 0.0

    def sleep(self, seconds):
        self.t += seconds

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
    strat = st.SupertrendFvgStrategy(
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
    script.update({"AAA": "BUY", "BBB": "SELL"})
    strat.on_bar(now, BAR_1)
    assert strat.state["open"]["symbol"] == "AAA" and client.pos["AAA"] == 10
    assert client.pos["BBB"] == 0
    assert [c for c in client.calls if c[0] == "place"] == [
        ("place", "AAA", "BUY", 10, "MARKET"),
        ("place", "AAA", "SELL", 10, "SL-M"),
    ]
    ignored = [r for r in _journal(tmp) if r["event"] == "signal_ignored"]
    assert [(r["symbol"], r["side"]) for r in ignored] == [("BBB", "SELL")]
    assert "AAA trade is already open" in ignored[0]["why"]

    # next bar: another symbol signals, still ignored
    strat.bar_done.clear()
    script.clear()
    script["CCC"] = "BUY"
    strat.on_bar(now, BAR_2)
    assert client.pos["CCC"] == 0
    assert sum(1 for r in _journal(tmp) if r["event"] == "signal_ignored") == 2


def test_flip_exits_cancelling_the_stop_first_then_enters_the_other_side(harness):
    strat, client, script, now, tmp = harness
    script["AAA"] = "BUY"
    strat.on_bar(now, BAR_2)
    stop_id = strat.state["open"]["stop_order_id"]

    strat.bar_done.clear()
    script["AAA"] = "SELL"
    strat.on_bar(now, BAR_3)
    kinds = [c[0] for c in client.calls]
    cancel_at = client.calls.index(("cancel", stop_id))
    smart_at = kinds.index("smart")
    assert cancel_at < smart_at
    assert client.calls[smart_at] == ("smart", "AAA", "SELL", 10, 0)  # to flat, never reversed
    # the new short is its own order, after the book was flat
    after = client.calls[smart_at + 1 :]
    assert after[0] == ("place", "AAA", "SELL", 10, "MARKET")
    assert client.pos["AAA"] == -10
    assert strat.state["open"]["side"] == "SELL"
    events = [r["event"] for r in _journal(tmp)]
    assert events.count("exit") == 1 and events.count("entry") == 2 and "flip" in events


def test_flip_without_enter_on_flip_only_exits(harness, monkeypatch):
    strat, client, script, now, _tmp = harness
    monkeypatch.setattr(st, "ENTER_ON_FLIP", False)
    script["AAA"] = "BUY"
    strat.on_bar(now, BAR_2)
    strat.bar_done.clear()
    script["AAA"] = "SELL"
    strat.on_bar(now, BAR_3)
    assert client.pos["AAA"] == 0 and strat.state["open"] is None


def test_flip_does_not_exit_while_the_stop_cannot_be_cancelled(harness):
    strat, client, script, now, _tmp = harness
    script["AAA"] = "BUY"
    strat.on_bar(now, BAR_2)
    client.cancel_ok = False
    strat.bar_done.clear()
    script.update({"AAA": "SELL", "BBB": "BUY"})
    strat.on_bar(now, BAR_3)
    assert not any(c[0] == "smart" for c in client.calls)  # exiting now could reverse
    assert client.pos == {"AAA": 10, "BBB": 0, "CCC": 0}
    assert strat.state["open"]["exit_retry"] == "supertrend flip"


def test_daily_caps_and_entry_window(harness):
    strat, _client, _script, now, _tmp = harness
    assert strat.can_enter(now) == (True, "")
    assert not strat.can_enter(now.replace(hour=9, minute=29, second=59))[0]
    assert not strat.can_enter(now.replace(hour=14, minute=45, second=5))[0]
    strat.state["trades"] = st.MAX_TRADES_PER_DAY
    assert "trades already taken" in strat.can_enter(now)[1]
    strat.state["trades"] = 0
    strat.state["pnl"] = -st.MAX_DAILY_LOSS
    assert "daily loss limit" in strat.can_enter(now)[1]


def test_first_evaluation_only_records_the_trend(tmp_path):
    now = datetime(2026, 11, 4, 10, 0)
    o, h, lo, c = random_ohlc(300, 3)
    sig = st.supertrend(o, h, lo, c)
    flips = [int(i) for i in np.flatnonzero(sig["buy"] | sig["sell"]) if i > 60]
    f = flips[0]
    idx = pd.date_range("2026-11-02 09:15", periods=300, freq="5min")
    bars = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=idx)
    expected = "BUY" if sig["buy"][f] else "SELL"

    running = st.SupertrendFvgStrategy(
        FakeClient(["AAA"], now), ["AAA"], clock=lambda: now, data_dir=tmp_path / "a"
    )
    assert running.evaluate_symbol("AAA", bars.iloc[:f]) is None  # first look: trend only
    assert running.evaluate_symbol("AAA", bars.iloc[: f + 1]) == expected  # a fresh flip
    assert running.evaluate_symbol("AAA", bars.iloc[: f + 1]) is None  # never twice

    restarted = st.SupertrendFvgStrategy(
        FakeClient(["AAA"], now), ["AAA"], clock=lambda: now, data_dir=tmp_path / "b"
    )
    assert restarted.evaluate_symbol("AAA", bars.iloc[: f + 1]) is None  # a restart on a flip bar
    assert restarted.trend["AAA"] == (1 if expected == "BUY" else -1)


def test_unknown_symbols_are_skipped_at_setup(tmp_path):
    class Picky(FakeClient):
        def symbol(self, symbol, **k):
            if symbol == "NOPE":
                return {"status": "error", "message": "Symbol not found"}
            return super().symbol(symbol, **k)

    now = datetime(2026, 11, 4, 10, 0)
    strat = st.SupertrendFvgStrategy(
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
    strat = st.SupertrendFvgStrategy(
        client, ["AAA", "BBB"], clock=lambda: now, data_dir=tmp_path, pause=0
    )
    strat.setup()
    assert strat.symbols == ["AAA"] and strat.unchecked == ["BBB"]
    client.down = False
    strat.on_bar(now, BAR_3)
    assert strat.symbols == ["AAA", "BBB"] and strat.unchecked == []


def test_entry_window_is_judged_by_the_bar_close(harness):
    strat, _client, _script, _now, _tmp = harness
    day = datetime(2026, 11, 4)
    last_bar_close = day.replace(hour=14, minute=45)
    # the 14:40-14:45 bar, processed a few seconds after 14:45, may trade
    assert strat.can_enter(day.replace(hour=14, minute=45, second=5), last_bar_close)[0]
    assert strat.can_enter(day.replace(hour=14, minute=46, second=59), last_bar_close)[0]
    # ...but never more than two minutes after the last entry time
    assert not strat.can_enter(day.replace(hour=14, minute=47, second=1), last_bar_close)[0]
    # the bar closing at 14:50 is outside the window
    assert not strat.can_enter(
        day.replace(hour=14, minute=50, second=5), day.replace(hour=14, minute=50)
    )[0]
    # the bar closing at 09:30 is the first one
    assert strat.can_enter(
        day.replace(hour=9, minute=30, second=5), day.replace(hour=9, minute=30)
    )[0]
    assert not strat.can_enter(
        day.replace(hour=9, minute=25, second=5), day.replace(hour=9, minute=25)
    )[0]


def test_one_bad_history_frame_does_not_stop_the_other_symbols(harness, monkeypatch):
    strat, client, _script, now, _tmp = harness
    seen = []
    monkeypatch.setattr(strat, "evaluate_symbol", lambda symbol, bars: seen.append(symbol))
    good = client.bars

    def history(symbol, **k):
        if symbol == "AAA":
            return pd.DataFrame({"price": [1.0]})  # no OHLC columns
        return good

    client.history = history
    assert strat.on_bar(now, BAR_1)
    assert seen == ["BBB", "CCC"]


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
        def __init__(self, frame):
            self.frame = frame

        def history(self, **k):
            return self.frame

    strat = st.SupertrendFvgStrategy(History(df), ["X"], clock=lambda: now, data_dir=tmp_path)
    kept = strat.load_bars("X", now)
    assert kept.index[-1] < forming and kept.index[-1] == idx[-2].tz_localize(None)
    wild = df.copy()
    wild.iloc[-1] = [1, 99999, 0.5, 99999]
    strat.client = History(wild)
    assert strat.load_bars("X", now).equals(kept)  # the forming bar never matters
    # five seconds after it closes, the bar is kept
    after = (forming + pd.Timedelta(minutes=5, seconds=5)).to_pydatetime()
    assert strat.load_bars("X", after).index[-1] == forming


# =============================================================================
# ORDER SAFETY: a broker that fills orders and can lose answers
# (the Auditor's scenarios, kept as regression tests)
# =============================================================================
class Broker:
    """Positions change only when an order fills. MARKET orders fill at once
    unless `hold_market` is set, then they stay open until release()."""

    def __init__(self, symbols, now, books=True):
        self.pos = dict.fromkeys(symbols, 0)
        self.orders = {}
        self.fills = []
        self.n = 0
        self.calls = []
        self.hold_market = False
        self.openpos_fail = 0  # fail the next N openposition calls
        self.raise_after_accept = set()  # order types whose answer is lost after acceptance
        self.smart_raise_after_accept = False
        self.ltp_value = 100.0
        self.books = books
        idx = pd.date_range(
            end=now.replace(second=0) - timedelta(minutes=5), periods=60, freq="5min"
        )
        self.bars = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=idx
        )

    def _fill(self, o):
        q = o["qty"] if o["action"] == "BUY" else -o["qty"]
        self.pos[o["symbol"]] += q
        o["status"] = "complete"
        self.fills.append(
            {"symbol": o["symbol"], "action": o["action"], "average_price": o["fill"]}
        )

    def release(self):
        for o in self.orders.values():
            if o["status"] == "open":
                self._fill(o)

    def trigger_stops(self):
        for o in self.orders.values():
            if o["status"] == "trigger pending":
                self._fill(o)

    def live(self, symbol, kinds=("open", "trigger pending")):
        return [k for k, o in self.orders.items() if o["symbol"] == symbol and o["status"] in kinds]

    def symbol(self, symbol, **k):
        return {"status": "success", "data": {"tick_size": 0.05, "lotsize": 1}}

    def analyzerstatus(self):
        return {"status": "success", "data": {"mode": "analyze"}}

    def quotes(self, symbol, **k):
        return {"status": "success", "data": {"ltp": self.ltp_value}}

    def history(self, **k):
        return self.bars

    def openposition(self, symbol, **k):
        if self.openpos_fail > 0:
            self.openpos_fail -= 1
            raise TimeoutError("read timed out")
        return {"status": "success", "quantity": str(self.pos[symbol])}

    def _new(self, symbol, action, qty, price_type, status, fill):
        self.n += 1
        oid = f"O{self.n}"
        self.orders[oid] = {
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "type": price_type,
            "status": status,
            "fill": fill,
        }
        return oid

    def placeorder(self, symbol, action, quantity, price_type, trigger_price=0, **k):
        self.calls.append(("place", symbol, action, quantity, price_type))
        if price_type == "MARKET":
            oid = self._new(symbol, action, quantity, price_type, "open", self.ltp_value)
            if not self.hold_market:
                self._fill(self.orders[oid])
        else:
            oid = self._new(symbol, action, quantity, price_type, "trigger pending", trigger_price)
        if price_type in self.raise_after_accept:
            raise TimeoutError("read timed out")
        return {"status": "success", "orderid": oid}

    def placesmartorder(self, symbol, action, quantity, position_size, **k):
        delta = position_size - self.pos[symbol]
        self.calls.append(("smart", symbol, delta))
        if delta == 0:
            return {"status": "success", "message": "No action needed"}
        side = "BUY" if delta > 0 else "SELL"
        oid = self._new(symbol, side, abs(delta), "MARKET", "open", self.ltp_value)
        if not self.hold_market:
            self._fill(self.orders[oid])
        if self.smart_raise_after_accept:
            raise TimeoutError("read timed out")
        return {"status": "success", "orderid": oid}

    def orderstatus(self, order_id, **k):
        o = self.orders.get(order_id)
        if o is None:
            return {"status": "error", "message": "not found"}
        avg = o["fill"] if o["status"] == "complete" else 0
        return {
            "status": "success",
            "data": {"order_status": o["status"], "average_price": avg, "quantity": str(o["qty"])},
        }

    def cancelorder(self, order_id, **k):
        o = self.orders[order_id]
        self.calls.append(("cancel", order_id))
        if o["status"] in ("open", "trigger pending"):
            o["status"] = "cancelled"
        return {"status": "success"}

    def orderbook(self, **k):
        if not self.books:
            raise ConnectionError("order book unavailable")
        rows = [
            {
                "orderid": oid,
                "symbol": o["symbol"],
                "exchange": "NSE",
                "product": "MIS",
                "action": o["action"],
                "pricetype": o["type"],
                "quantity": str(o["qty"]),
                "order_status": o["status"],
            }
            for oid, o in self.orders.items()
        ]
        return {"status": "success", "data": {"orders": rows}}

    def tradebook(self, **k):
        if not self.books:
            raise ConnectionError("trade book unavailable")
        return {"status": "success", "data": list(self.fills)}


@pytest.fixture
def broker_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "time", FakeTime())
    monkeypatch.setattr(st, "plan_trade", fake_plan)
    now = datetime(2026, 11, 4, 10, 0, 6)

    def make(books=True, data_dir=None):
        symbols = ["AAA", "BBB", "CCC"]
        b = Broker(symbols, now, books=books)
        s = st.SupertrendFvgStrategy(
            b, symbols=symbols, clock=lambda: now, data_dir=data_dir or tmp_path, pause=0
        )
        s.setup()
        script = {}
        s.evaluate_symbol = lambda sym, bars: script.get(sym)
        return s, b, script, now

    return make


SCENARIO_BARS = [datetime(2026, 11, 4, 9, m) for m in (35, 40, 45, 50)]


def _bar(s, now, i):
    s.bar_done.clear()
    s.on_bar(now, SCENARIO_BARS[i])


def _open_symbol(s):
    return (s.state["open"] or {}).get("symbol")


def test_s1_unreadable_position_after_the_entry_fill_still_manages_it(broker_factory):
    s, b, script, now = broker_factory()
    real = b.orderstatus

    def status_then_fail(order_id, **k):
        r = real(order_id, **k)
        if order_id == "O1" and r["data"]["order_status"] == "complete":
            b.openpos_fail = 1  # the position read right after the fill fails
        return r

    b.orderstatus = status_then_fail
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.orderstatus = real
    assert _open_symbol(s) == "AAA" and s.state["open"]["qty"] == 10
    assert len(b.live("AAA", ("trigger pending",))) == 1  # it has its stop
    script.clear()
    script["BBB"] = "SELL"
    _bar(s, now, 1)
    assert b.pos == {"AAA": 10, "BBB": 0, "CCC": 0}


def test_unreadable_position_with_an_unfinished_entry_blocks_every_symbol(broker_factory):
    s, b, script, now = broker_factory()
    b.hold_market = True
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    assert s.state["pending"] and s.state["open"] is None
    script.clear()
    script["BBB"] = "BUY"
    _bar(s, now, 1)
    assert b.pos["BBB"] == 0 and not any(c[1] == "BBB" for c in b.calls if c[0] == "place")


def test_s2_a_slow_exit_is_cancelled_before_it_is_retried(broker_factory):
    s, b, script, now = broker_factory()
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.hold_market = True
    b.ltp_value = 105.0  # target 104 reached
    s.manage_open(now)
    first_exit = s.state["open"]["exit_order_id"]
    assert first_exit and b.pos["AAA"] == 10
    s.manage_open(now)  # retry: the first exit is cancelled and confirmed first
    assert b.orders[first_exit]["status"] == "cancelled"
    assert len(b.live("AAA", ("open",))) == 1  # never two exit orders at once
    b.release()
    assert b.pos["AAA"] == 0
    s.manage_open(now)
    assert s.state["open"] is None


def test_s3_unreadable_position_after_the_exit_order_is_not_booked_closed(broker_factory):
    s, b, script, now = broker_factory()
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.hold_market = True
    b.ltp_value = 105.0
    real = b.placesmartorder

    def smart_then_fail(**k):
        r = real(**k)
        b.openpos_fail = 1
        return r

    b.placesmartorder = smart_then_fail
    s.manage_open(now)
    assert _open_symbol(s) == "AAA"  # still managed
    b.placesmartorder = real
    b.hold_market = False
    b.ltp_value = 100.0
    script.clear()
    script["BBB"] = "BUY"
    _bar(s, now, 1)
    assert b.pos["BBB"] == 0


@pytest.mark.parametrize("books", [True, False])
def test_s4_a_lost_stop_answer_never_leads_to_two_stops(broker_factory, books):
    s, b, script, now = broker_factory(books=books)
    b.raise_after_accept = {"SL-M"}
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.raise_after_accept = set()
    assert len(b.live("AAA", ("trigger pending",))) == 1
    if books:
        assert s.state["open"]["stop_order_id"] == b.live("AAA", ("trigger pending",))[0]
    else:
        assert s.state["open"]["stop_check"]  # checked again; the script watches the stop
    s.manage_open(now)
    assert len(b.live("AAA", ("trigger pending",))) == 1
    b.trigger_stops()
    assert b.pos["AAA"] == 0


def test_s5_a_lost_entry_answer_is_taken_over_with_its_stop(broker_factory):
    s, b, script, now = broker_factory()
    b.raise_after_accept = {"MARKET"}
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.raise_after_accept = set()
    assert _open_symbol(s) == "AAA" and len(b.live("AAA", ("trigger pending",))) == 1
    script.clear()
    script["BBB"] = "BUY"
    _bar(s, now, 1)
    assert b.pos["BBB"] == 0


def test_a_lost_entry_answer_still_working_is_cancelled_not_doubled(broker_factory):
    s, b, script, now = broker_factory()
    b.raise_after_accept = {"MARKET"}
    b.hold_market = True
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.raise_after_accept = set()
    b.hold_market = False
    assert s.state["pending"] is not None  # the working order was cancelled; checked again
    assert b.orders["O1"]["status"] == "cancelled"
    script.clear()
    s.step()
    assert s.state["pending"] is None and s.state["open"] is None and b.pos["AAA"] == 0
    assert [c for c in b.calls if c[0] == "place"] == [("place", "AAA", "BUY", 10, "MARKET")]


def test_a_lost_exit_answer_is_resolved_from_the_books(broker_factory):
    s, b, script, now = broker_factory()
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.smart_raise_after_accept = True
    b.ltp_value = 106.0
    s.manage_open(now)  # the exit filled at 106 but the answer was lost
    b.smart_raise_after_accept = False
    assert b.pos["AAA"] == 0 and _open_symbol(s) == "AAA"
    b.ltp_value = 90.0
    s.manage_open(now)
    assert s.state["open"] is None
    assert sum(1 for c in b.calls if c[0] == "smart") == 1  # no second exit was sent
    assert s.state["pnl"] == pytest.approx(60.0)  # priced from the trade book, not the LTP


def test_a_lost_exit_answer_while_still_working_is_cancelled_first(broker_factory):
    s, b, script, now = broker_factory()
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.smart_raise_after_accept = True
    b.hold_market = True
    b.ltp_value = 105.0
    s.manage_open(now)
    b.smart_raise_after_accept = False
    lost = b.live("AAA", ("open",))
    assert len(lost) == 1
    s.manage_open(now)  # cancels the lost exit, then sends a new one
    assert b.orders[lost[0]]["status"] == "cancelled"
    assert len(b.live("AAA", ("open",))) == 1
    b.release()
    assert b.pos["AAA"] == 0


def test_s6_step_survives_missing_quotes_and_history(broker_factory):
    s, b, _script, _now = broker_factory()
    s.evaluate_symbol = st.SupertrendFvgStrategy.evaluate_symbol.__get__(s)
    b.quotes = lambda **k: None
    b.history = lambda **k: {"status": "error", "message": "boom"}
    for _ in range(3):
        s.step()


def test_s7_an_unreadable_state_file_is_kept_aside_and_the_position_closed(
    broker_factory, tmp_path
):
    s, b, script, now = broker_factory(data_dir=tmp_path / "state")
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    s.state_path.write_text(s.state_path.read_text()[:40])
    s2 = st.SupertrendFvgStrategy(
        b, symbols=["AAA", "BBB", "CCC"], clock=lambda: now, data_dir=tmp_path / "state", pause=0
    )
    s2.setup()
    assert list((tmp_path / "state").glob("state.unreadable-*.json"))
    assert s2.state["open"]["symbol"] == "AAA" and s2.state["open"]["recovered"]
    s2.manage_open(now)
    assert b.pos["AAA"] == 0
    assert b.live("AAA") == []  # the old stop was cancelled before the close
    assert s2.state["open"] is None and s2.state["pnl"] == 0.0


def test_a_stop_filled_at_the_broker_is_booked_at_its_fill(broker_factory):
    s, b, script, now = broker_factory()
    script["AAA"] = "BUY"
    _bar(s, now, 0)
    b.trigger_stops()  # fills at the trigger, 98
    s.manage_open(now)
    assert s.state["open"] is None and s.state["pnl"] == pytest.approx(-20.0)


# =============================================================================
# END TO END THROUGH THE REAL BACKTEST RUNNER
# =============================================================================
E2E_SYMBOLS = ["SBIN", "M&M", "TCS", "LT"]


@pytest.fixture(scope="module")
def multi_data_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("stfvg_bars")
    start, end = date(2026, 8, 24), date(2026, 9, 25)
    for n, (symbol, base) in enumerate(zip(E2E_SYMBOLS, (800, 3000, 3500, 3600), strict=True)):
        write_bars(directory, symbol, "NSE", start, end, base, 100 + n)
    return directory


@pytest.mark.timeout(500)
def test_backtest_runs_unchanged_and_holds_one_position_at_a_time(tmp_path, multi_data_dir):
    original = SCRIPT.read_bytes()
    run_dir, result, log = run_backtest(
        tmp_path,
        SCRIPT,
        multi_data_dir,
        "2026-09-14",
        "2026-09-25",
        extra_env={"STFVG_SYMBOLS": ",".join(E2E_SYMBOLS)},
        timeout=400,
    )
    assert result["status"] == "completed", result.get("message")
    assert SCRIPT.read_bytes() == original
    assert "4 symbols on NSE MIS" in log

    trades = sorted(result["trades"], key=lambda t: t["entry_time"])
    assert trades, "no trade in the test period"
    assert {t["symbol"] for t in trades} <= set(E2E_SYMBOLS)
    for prev, nxt in zip(trades, trades[1:], strict=False):
        assert nxt["entry_time"] >= prev["exit_time"], (prev, nxt)
    for t in trades:
        assert t["quantity"] * t["entry_price"] <= 25000 + 1e-6

    journal = run_dir / "supertrend_fvg_data" / "journal.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    entries = [r for r in rows if r["event"] == "entry"]
    assert entries and len(entries) == len([r for r in rows if r["event"] == "exit"])
    for r in entries:
        assert r["planned_loss"] <= 2000
        assert r["position_value"] <= 25000
        trade = r["trade"]
        if trade["side"] == "BUY":
            assert trade["stop"] < trade["entry_price"] < trade["target"]
        else:
            assert trade["target"] < trade["entry_price"] < trade["stop"]
    per_day = {}
    for r in entries:
        per_day[r["ts"][:10]] = per_day.get(r["ts"][:10], 0) + 1
    assert max(per_day.values()) <= 5
