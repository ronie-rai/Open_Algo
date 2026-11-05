"""The replay engine behind the /python Backtest button, piece by piece.

The property that matters most is the first: at a simulated moment ``now`` no
view of the market may contain anything that happened after it. The leak test
proves it the strong way, by rewriting every bar after ``now`` and checking
that nothing the strategy could read changed.

Run:
    uv run pytest test/test_python_backtest_engine.py -v
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from services.python_backtest import clock
from services.python_backtest.broker import Broker, Costs
from services.python_backtest.data import (
    MarketData,
    OpenAlgoProvider,
    SymbolSeries,
    empty_daily,
)
from services.python_backtest.engine import BacktestFinished, Engine
from services.python_backtest.metrics import compute
from services.python_backtest.sessions import date_to_day, ist_datetime, session_for

D1 = date(2026, 9, 21)  # a Monday
D2 = date(2026, 9, 22)
FRI = date(2026, 9, 25)
MON = date(2026, 9, 28)


def open_epoch(day: date, minutes_after_open: int = 0) -> int:
    return session_for("NSE").open_epoch(date_to_day(day)) + minutes_after_open * 60


def make_bars(days, base=100.0, step=0.1, seed=None):
    """375 one-minute bars a day. Deterministic unless a seed asks for noise."""
    rng = np.random.default_rng(seed) if seed is not None else None
    rows, price = [], base
    for day in days:
        t0 = open_epoch(day)
        for i in range(375):
            o = price
            c = o + (rng.normal(0, 0.3) if rng is not None else step)
            h, lo = max(o, c) + 0.05, min(o, c) - 0.05
            rows.append((t0 + 60 * i, o, h, lo, c, 100 + i, 0))
            price = c
    return pd.DataFrame(rows, columns=["epoch", "open", "high", "low", "close", "volume", "oi"])


class StubProvider:
    def __init__(self, bars: dict, daily: dict | None = None):
        self.bars = bars
        self.daily = daily or {}

    def load_1m(self, symbol, exchange, start, end):
        return self.bars.get((symbol, exchange), make_bars([])).copy()

    def load_daily(self, symbol, exchange, start, end):
        return self.daily.get((symbol, exchange), empty_daily()).copy()

    def call_readonly(self, method, **kwargs):
        return None


def setup(days=(D1, D2), bars=None, start=None, end=None, capital=100000.0, costs=None):
    bars = bars if bars is not None else make_bars(days)
    market = MarketData(
        StubProvider({("SBIN", "NSE"): bars}), start or days[0], end or days[-1], warmup_days=0
    )
    holder = {}
    broker = Broker(market, capital, costs or Costs(), now_fn=lambda: holder["engine"].now)
    market.get("SBIN", "NSE")
    engine = Engine(broker=broker, market=market, start=start or days[0], end=end or days[-1])
    holder["engine"] = engine
    return market, broker, engine


def no_costs():
    return Costs(
        brokerage_per_order=0, brokerage_pct=0, stt_sell_pct=0, other_charges_pct=0, slippage_bps=0
    )


# =============================================================================
# NO LOOK-AHEAD
# =============================================================================
class TestNoFutureData:
    def _views(self, series: SymbolSeries, now: float):
        out = {}
        for minutes in (1, 3, 5, 15, 60):
            out[minutes] = series.intraday(minutes, D1, D2, now)
        out["D"] = series.daily(D1, D2, now)
        out["price"] = series.price_at(now)
        out["today"] = series.today_bar(now)
        return out

    def test_rewriting_the_future_changes_nothing_visible(self):
        bars = make_bars([D1, D2], seed=7)
        rng = np.random.default_rng(3)
        moments = [open_epoch(D1, 0), open_epoch(D2, 374) + 59] + [
            float(open_epoch(D1) + rng.integers(0, 2 * 86400)) + rng.integers(0, 60)
            for _ in range(40)
        ]
        for now in moments:
            honest = SymbolSeries("SBIN", "NSE", bars, empty_daily())
            future = bars["epoch"] + 60 > now  # bars not complete at `now`
            started = bars["epoch"] <= now
            tampered = bars.copy()
            # Bars that have not started: anything at all. The bar in progress:
            # its open is known, the rest is not.
            for col in ("open", "high", "low", "close"):
                tampered.loc[future & ~started, col] *= 10
            for col in ("high", "low", "close"):
                tampered.loc[future & started, col] *= 10
            tampered.loc[future, "volume"] *= 10
            liar = SymbolSeries("SBIN", "NSE", tampered, empty_daily())

            a, b = self._views(honest, now), self._views(liar, now)
            for key in (1, 3, 5, 15, 60, "D"):
                pd.testing.assert_frame_equal(a[key], b[key], obj=f"{key} at {now}")
            assert a["price"] == b["price"]
            assert a["today"] == b["today"]

    def test_no_bar_is_stamped_after_now(self):
        series = SymbolSeries("SBIN", "NSE", make_bars([D1, D2]), empty_daily())
        now = open_epoch(D2, 42) + 17
        stamp = pd.Timestamp(now, unit="s", tz="UTC")
        for minutes in (1, 5, 15, 60):
            df = series.intraday(minutes, D1, D2, now)
            assert df.index.max() <= stamp
            assert str(df.index.tz) == "Asia/Kolkata"
            assert df.index.name == "timestamp"
            assert list(df.columns) == ["open", "high", "low", "close", "volume", "oi"]

    def test_forming_bar_holds_only_what_has_traded(self):
        bars = make_bars([D1])
        series = SymbolSeries("SBIN", "NSE", bars, empty_daily())
        now = open_epoch(D1, 7) + 30  # 09:22:30, inside the 09:20 five-minute bar
        df = series.intraday(5, D1, D1, now)
        last = df.iloc[-1]
        assert df.index[-1] == pd.Timestamp(open_epoch(D1, 5), unit="s", tz="UTC")
        sub = bars[(bars.epoch >= open_epoch(D1, 5)) & (bars.epoch < open_epoch(D1, 7))]
        current_open = bars.loc[bars.epoch == open_epoch(D1, 7), "open"].item()
        assert last.open == sub.open.iloc[0]
        assert last.high == max(sub.high.max(), current_open)
        assert last.low == min(sub.low.min(), current_open)
        assert last.close == current_open
        assert last.volume == sub.volume.sum()

    def test_one_minute_forming_bar_is_its_open(self):
        bars = make_bars([D1])
        series = SymbolSeries("SBIN", "NSE", bars, empty_daily())
        now = open_epoch(D1, 3) + 10
        df = series.intraday(1, D1, D1, now)
        o = bars.loc[bars.epoch == open_epoch(D1, 3), "open"].item()
        assert len(df) == 4
        assert df.iloc[-1][["open", "high", "low", "close"]].tolist() == [o, o, o, o]
        assert series.price_at(now) == o

    def test_daily_bar_for_today_comes_from_minutes_not_the_source(self):
        bars = make_bars([D1, D2])
        # A source that already knows how today ends must not be believed.
        cheat = pd.DataFrame(
            [(date_to_day(D1), 1, 2, 0.5, 1.5, 9, 0), (date_to_day(D2), 999, 999, 999, 999, 9, 0)],
            columns=["day", "open", "high", "low", "close", "volume", "oi"],
        )
        series = SymbolSeries("SBIN", "NSE", bars, cheat)
        now = open_epoch(D2, 10)
        df = series.daily(D1, D2, now)
        assert len(df) == 2
        assert df.iloc[0].close == 1.5  # yesterday from the source
        assert df.iloc[1].close == series.price_at(now) != 999
        assert df.index[1] == pd.Timestamp(D2)

    def test_before_the_first_bar_there_is_nothing(self):
        series = SymbolSeries("SBIN", "NSE", make_bars([D1]), empty_daily())
        assert series.price_at(open_epoch(D1) - 1) is None
        assert series.intraday(5, D1, D1, open_epoch(D1) - 1).empty


# =============================================================================
# FILLS
# =============================================================================
class TestFills:
    def test_market_fills_at_the_open_in_progress_plus_slippage(self):
        market, broker, engine = setup(costs=Costs(slippage_bps=10))
        engine.now = open_epoch(D1, 5) + 20
        resp = broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="BUY", quantity=2)
        assert resp["status"] == "success"
        order = broker.orders[resp["orderid"]]
        ltp = market.get("SBIN", "NSE").price_at(engine.now)
        assert order.status == "complete"
        assert order.average_price == pytest.approx(ltp * 1.001)
        assert broker.net_qty("SBIN", "NSE", "MIS") == 2

    def test_limit_waits_for_a_later_bar_to_trade_through(self):
        market, broker, engine = setup(costs=no_costs())
        engine.now = open_epoch(D1, 10)
        ltp = market.get("SBIN", "NSE").price_at(engine.now)
        limit = round(ltp + 0.35, 2)  # rising bars: a SELL limit above the market
        resp = broker.place(
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action="SELL",
            quantity=1,
            price_type="LIMIT",
            price=limit,
        )
        order = broker.orders[resp["orderid"]]
        assert order.status == "open"
        engine.sleep(60)  # the 09:25 bar closes, highs reach only ltp + 0.15
        assert order.status == "open"
        engine.sleep(5 * 60)
        assert order.status == "complete"
        assert order.average_price == limit

    def test_limit_gapped_through_fills_at_the_better_open(self):
        bars = make_bars([D1])
        bars.loc[bars.epoch == open_epoch(D1, 11), ["open", "high", "low", "close"]] = [
            90,
            91,
            89,
            90,
        ]
        market, broker, engine = setup(bars=bars, costs=no_costs())
        engine.now = open_epoch(D1, 10) + 30
        resp = broker.place(
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action="BUY",
            quantity=1,
            price_type="LIMIT",
            price=95,
        )
        engine.sleep(120)
        order = broker.orders[resp["orderid"]]
        assert order.status == "complete"
        assert order.average_price == 90  # the open, better than the limit

    def test_pending_order_never_fills_on_a_bar_from_before_it(self):
        bars = make_bars([D1])
        # The bar in progress when the order is placed trades far below it.
        bars.loc[bars.epoch == open_epoch(D1, 10), "low"] = 1
        market, broker, engine = setup(bars=bars, costs=no_costs())
        engine.now = open_epoch(D1, 10) + 30
        resp = broker.place(
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action="BUY",
            quantity=1,
            price_type="LIMIT",
            price=50,
        )
        engine.sleep(45)  # the 09:25 bar completes
        assert broker.orders[resp["orderid"]].status == "open"

    def test_sl_m_triggers_and_fills_at_the_trigger(self):
        market, broker, engine = setup(costs=no_costs())
        engine.now = open_epoch(D1, 10)
        ltp = market.get("SBIN", "NSE").price_at(engine.now)
        trig = round(ltp + 0.5, 2)
        resp = broker.place(
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action="BUY",
            quantity=3,
            price_type="SL-M",
            trigger_price=trig,
        )
        order = broker.orders[resp["orderid"]]
        assert order.status == "trigger pending"
        engine.sleep(10 * 60)
        assert order.status == "complete"
        assert order.average_price == trig

    def test_sl_m_gapped_fills_at_the_worse_open(self):
        bars = make_bars([D1])
        bars.loc[bars.epoch == open_epoch(D1, 12), ["open", "high", "low", "close"]] = [
            80,
            81,
            79,
            80,
        ]
        market, broker, engine = setup(bars=bars, costs=no_costs())
        engine.now = open_epoch(D1, 11)
        resp = broker.place(
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action="SELL",
            quantity=1,
            price_type="SL-M",
            trigger_price=95,
        )
        engine.sleep(3 * 60)
        order = broker.orders[resp["orderid"]]
        assert order.status == "complete"
        assert order.average_price == 80

    def test_cancel_and_modify(self):
        market, broker, engine = setup(costs=no_costs())
        engine.now = open_epoch(D1, 10)
        oid = broker.place(
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action="BUY",
            quantity=1,
            price_type="LIMIT",
            price=1,
        )["orderid"]
        assert broker.modify(oid, price=2, quantity=4)["status"] == "success"
        assert broker.orders[oid].price == 2 and broker.orders[oid].quantity == 4
        assert broker.cancel(oid)["status"] == "success"
        assert broker.orders[oid].status == "cancelled"
        assert broker.cancel(oid)["status"] == "error"

    def test_bad_input_is_an_error_response(self):
        _m, broker, engine = setup()
        engine.now = open_epoch(D1, 10)
        assert (
            broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="HOLD", quantity=1)[
                "status"
            ]
            == "error"
        )
        assert (
            broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="BUY", quantity=0)[
                "status"
            ]
            == "error"
        )
        assert (
            broker.place(
                strategy="s",
                symbol="SBIN",
                exchange="NSE",
                action="BUY",
                quantity=1,
                price_type="LIMIT",
            )["status"]
            == "error"
        )


class TestSmartOrder:
    @pytest.mark.parametrize(
        "current,target,action,qty,expect_qty,expect_net",
        [
            (0, 5, "BUY", 5, 5, 5),
            (5, 5, "BUY", 5, None, 5),  # already there: no order
            (5, -5, "SELL", 5, 10, -5),  # reversal in one order
            (5, 0, "SELL", 5, 5, 0),
            (-3, 0, "BUY", 3, 3, 0),
            (5, 8, "BUY", 3, 3, 8),
            (8, 2, "SELL", 6, 6, 2),
            (0, 0, "SELL", 4, 4, -4),  # OpenAlgo's own rule: no position, no target
        ],
    )
    def test_trades_the_difference(self, current, target, action, qty, expect_qty, expect_net):
        _m, broker, engine = setup(costs=no_costs())
        engine.now = open_epoch(D1, 10)
        if current:
            broker.place(
                strategy="s",
                symbol="SBIN",
                exchange="NSE",
                action="BUY" if current > 0 else "SELL",
                quantity=abs(current),
            )
        before = len(broker.orders)
        resp = broker.place_smart(
            position_size=target,
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action=action,
            quantity=qty,
            product="MIS",
        )
        assert resp["status"] == "success"
        if expect_qty is None:
            assert len(broker.orders) == before
            assert "No action" in resp["message"]
        else:
            assert broker.orders[resp["orderid"]].quantity == expect_qty
        assert broker.net_qty("SBIN", "NSE", "MIS") == expect_net


class TestSquareOff:
    def test_mis_position_is_closed_at_the_square_off_minute(self):
        market, broker, engine = setup(costs=no_costs())
        engine.now = open_epoch(D1, 300)  # 14:15
        broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="BUY", quantity=1)
        oid = broker.place(
            strategy="s",
            symbol="SBIN",
            exchange="NSE",
            action="SELL",
            quantity=1,
            price_type="LIMIT",
            price=10_000,
        )["orderid"]
        engine.sleep(59 * 60)  # 15:14, still held
        assert broker.net_qty("SBIN", "NSE", "MIS") == 1
        engine.sleep(60)  # 15:15
        assert broker.net_qty("SBIN", "NSE", "MIS") == 0
        assert broker.trades[-1]["exit_reason"] == "square-off"
        assert broker.trades[-1]["exit_price"] == pytest.approx(
            market.get("SBIN", "NSE").price_at(engine.now)
        )
        assert broker.orders[oid].status == "cancelled"

    def test_mis_orders_after_square_off_are_rejected(self):
        _m, broker, engine = setup(costs=no_costs())
        engine.now = open_epoch(D1, 361)  # 15:16
        resp = broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="BUY", quantity=1)
        assert broker.orders[resp["orderid"]].status == "rejected"
        cnc = broker.place(
            strategy="s", symbol="SBIN", exchange="NSE", action="BUY", quantity=1, product="CNC"
        )
        assert broker.orders[cnc["orderid"]].status == "complete"


class TestRoundTrips:
    def test_fifo_round_trips_with_costs(self):
        costs = Costs(
            brokerage_per_order=20,
            brokerage_pct=0,
            stt_sell_pct=0,
            other_charges_pct=0,
            slippage_bps=0,
        )
        market, broker, engine = setup(costs=costs)
        engine.now = open_epoch(D1, 10)
        p1 = market.get("SBIN", "NSE").price_at(engine.now)
        broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="BUY", quantity=10)
        engine.sleep(60 * 10)
        p2 = market.get("SBIN", "NSE").price_at(engine.now)
        broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="SELL", quantity=4)
        broker.place(strategy="s", symbol="SBIN", exchange="NSE", action="SELL", quantity=6)
        assert len(broker.trades) == 2
        first = broker.trades[0]
        assert first["side"] == "LONG" and first["quantity"] == 4
        assert first["gross_pnl"] == pytest.approx((p2 - p1) * 4, abs=0.01)
        # entry cost spread per unit (20/10 each), exit order's 20 on its 4
        assert first["costs"] == pytest.approx(4 * 2 + 20, abs=0.01)
        assert broker.total_costs == pytest.approx(60)
        assert broker.equity() == pytest.approx(100000 + (p2 - p1) * 10 - 60, abs=0.01)


# =============================================================================
# THE CLOCK
# =============================================================================
class TestClock:
    def test_sleep_advances_the_clock(self):
        _m, _b, engine = setup()
        t0 = engine.now
        assert ist_datetime(t0).strftime("%Y-%m-%d %H:%M") == "2026-09-21 09:15"
        engine.sleep(15)
        assert engine.now == t0 + 15

    def test_sleep_past_the_close_fast_forwards_to_the_next_open(self):
        _m, _b, engine = setup()
        engine.now = open_epoch(D1, 374) + 50  # 15:29:50
        engine.sleep(30)
        assert ist_datetime(engine.now).strftime("%Y-%m-%d %H:%M:%S") == "2026-09-22 09:15:00"
        assert engine.day_ends[-1][0] == date_to_day(D1)

    def test_weekend_and_missing_days_are_skipped(self):
        _m, _b, engine = setup(days=(FRI, MON))
        engine.now = open_epoch(FRI, 375)
        engine.sleep(1)
        assert ist_datetime(engine.now).strftime("%Y-%m-%d %H:%M") == "2026-09-28 09:15"

    def test_the_end_raises_a_finish_the_strategy_cannot_catch(self):
        _m, _b, engine = setup()
        engine.now = open_epoch(D2, 374)
        caught_by_strategy = False
        with pytest.raises(BacktestFinished):
            try:
                engine.sleep(3600)
            except Exception:  # what a strategy's error handler looks like
                caught_by_strategy = True
        assert not caught_by_strategy
        assert engine.finished and engine.outcome == "completed"
        with pytest.raises(BacktestFinished):
            engine.sleep(1)  # and again, if a bare except swallowed it
        with pytest.raises(BacktestFinished):
            engine.read_clock()

    def test_a_stop_request_finishes_with_results(self, tmp_path):
        _m, broker, engine = setup()
        engine.run_dir = tmp_path
        (tmp_path / "stop").write_text("stop")
        with pytest.raises(BacktestFinished):
            engine.sleep(5)
        assert engine.outcome == "stopped"
        assert (tmp_path / "progress.json").exists()

    def test_patched_datetime_and_time_read_the_simulated_clock(self):
        import datetime as dt
        import time

        _m, _b, engine = setup()
        engine.now = open_epoch(D1, 60) + 5  # 10:15:05 IST
        real_datetime = dt.datetime
        clock.install(engine)
        try:
            from datetime import datetime, timedelta, timezone

            now = datetime.now()
            aware = datetime.now(timezone(timedelta(hours=5, minutes=30)))
            assert now.strftime("%Y-%m-%d %H:%M:%S") == "2026-09-21 10:15:05"
            assert aware.utcoffset() == timedelta(hours=5, minutes=30)
            assert datetime.utcnow().strftime("%H:%M:%S") == "04:45:05"
            assert dt.date.today() == date(2026, 9, 21)
            assert time.time() == engine.now
            assert time.monotonic() == engine.now
            assert time.strftime("%H:%M") == "10:15"
            assert pd.Timestamp.now() == pd.Timestamp("2026-09-21 10:15:05")
            assert pd.Timestamp.today() == pd.Timestamp("2026-09-21 10:15:05")
            assert pd.Timestamp.now("UTC") == pd.Timestamp("2026-09-21 04:45:05", tz="UTC")
            assert pd.Timestamp.utcnow() == pd.Timestamp("2026-09-21 04:45:05")
            # pandas itself is unaffected
            idx = pd.date_range("2026-01-01", periods=3, freq="D")
            frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)
            assert frame.resample("2D").sum()["x"].tolist() == [3.0, 3.0]
            assert pd.to_datetime(["2026-01-02"])[0] == idx[1]
            assert isinstance(real_datetime(2020, 1, 1), datetime)
            assert isinstance(pd.Timestamp("2020-01-01"), datetime)
            assert isinstance(date(2020, 1, 1), dt.date)
            time.sleep(60)
            assert datetime.now().strftime("%H:%M:%S") == "10:16:05"
            with clock.real_time():
                assert dt.datetime is real_datetime
                assert time.time() > 1.7e9 and abs(time.time() - engine.now) > 1
        finally:
            clock.uninstall()
        assert dt.datetime is real_datetime
        drift = pd.Timestamp.now().timestamp() - pd.Timestamp(real_datetime.now()).timestamp()
        assert abs(drift) < 5


# =============================================================================
# METRICS
# =============================================================================
class TestMetrics:
    def test_known_sequence(self):
        trades = [{"pnl": 100.0}, {"pnl": -50.0}, {"pnl": 200.0}, {"pnl": -100.0}]
        day = date_to_day(D1)
        samples = [
            (1000.0, 10_000.0, False),
            (1300.0, 10_100.0, True),
            (1600.0, 9_900.0, True),  # 200 below the 10,100 peak
            (1900.0, 10_150.0, False),
        ]
        out = compute(
            capital=10_000,
            trades=trades,
            equity_samples=samples,
            day_ends=[(day, 10_100.0), (day + 1, 9_999.0), (day + 2, 10_150.0)],
            total_costs=30.0,
        )
        m = out["metrics"]
        assert m["net_pnl"] == 150.0
        assert m["gross_pnl"] == 180.0
        assert m["return_pct"] == 1.5
        assert m["trades"] == 4 and m["winning_trades"] == 2 and m["losing_trades"] == 2
        assert m["win_rate"] == 50.0
        assert m["avg_win"] == 150.0 and m["avg_loss"] == -75.0
        assert m["profit_factor"] == 2.0
        assert m["expectancy"] == 37.5
        assert m["largest_win"] == 200.0 and m["largest_loss"] == -100.0
        assert m["max_drawdown"] == 200.0
        assert m["max_drawdown_pct"] == pytest.approx(200 / 10_100 * 100, abs=0.01)
        assert m["max_drawdown_time"] == 1_600_000
        assert m["exposure_pct"] == 50.0
        assert m["best_day"]["pnl"] == 151.0 and m["worst_day"]["pnl"] == -101.0
        r = [0.01, -101 / 10_100, 151 / 9_999]
        mean = sum(r) / 3
        sd = (sum((x - mean) ** 2 for x in r) / 2) ** 0.5
        assert m["sharpe"] == pytest.approx(mean / sd * 252**0.5, abs=0.01)
        assert [p["drawdown"] for p in out["equity_curve"]] == [0.0, 0.0, -200.0, 0.0]
        assert out["daily"][1] == {"date": "2026-09-22", "pnl": -101.0, "equity": 9999.0}

    def test_no_trades_is_not_an_error(self):
        out = compute(
            capital=5000,
            trades=[],
            equity_samples=[(0.0, 5000.0, False)],
            day_ends=[],
            total_costs=0,
        )
        m = out["metrics"]
        assert m["trades"] == 0 and m["win_rate"] is None and m["profit_factor"] is None
        assert m["net_pnl"] == 0 and m["sharpe"] is None


# =============================================================================
# THE REAL DATA PROVIDER, WITH A STAND-IN CLIENT
# =============================================================================
class FakeSdkClient:
    """Answers the history endpoint the way the app does, and records every call."""

    api_key = "k"

    def __init__(self, db_bars: pd.DataFrame, api_bars: pd.DataFrame):
        self.sources = {"db": db_bars, "api": api_bars}
        self.calls = []

    def _make_request(self, endpoint, payload):
        assert endpoint == "history"
        self.calls.append((payload["source"], payload["start_date"], payload["end_date"]))
        bars = self.sources[payload["source"]]
        lo = open_epoch(date.fromisoformat(payload["start_date"])) - 9 * 3600
        hi = open_epoch(date.fromisoformat(payload["end_date"])) + 15 * 3600
        rows = bars[(bars.epoch >= lo) & (bars.epoch < hi)]
        if rows.empty:
            return {"status": "error", "message": "No data found"}
        records = rows.rename(columns={"epoch": "timestamp"}).to_dict("records")
        return {"status": "success", "data": records}


class TestOpenAlgoProvider:
    def _provider(self, client, tmp_path, today=date(2026, 11, 5)):
        provider = OpenAlgoProvider(lambda: client, tmp_path / "cache", today)
        provider.API_PAUSE_SECONDS = 0
        return provider

    def test_historify_first_then_the_broker_for_what_it_lacks(self, tmp_path):
        days = [date(2026, 9, 21) + timedelta(days=i) for i in range(28)]
        days = [d for d in days if d.weekday() < 5]
        all_bars = make_bars(days)
        # Historify has only the first week; the broker has everything.
        db_only = all_bars[all_bars.epoch < open_epoch(date(2026, 9, 28))]
        client = FakeSdkClient(db_only, all_bars)
        provider = self._provider(client, tmp_path)
        out = provider.load_1m("SBIN", "NSE", date(2026, 9, 21), date(2026, 10, 18))
        assert len(out) == len(all_bars)
        assert [c[0] for c in client.calls[:2]] == ["db", "api"]  # the first chunk
        assert provider.sources_used == {"broker"}

    def test_full_historify_needs_no_broker_and_finished_chunks_are_cached(self, tmp_path):
        days = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)]
        bars = make_bars(days)
        client = FakeSdkClient(bars, make_bars([]))
        provider = self._provider(client, tmp_path)
        first = provider.load_1m("SBIN", "NSE", date(2026, 9, 21), date(2026, 9, 23))
        assert len(first) == len(bars)
        assert [c[0] for c in client.calls] == ["db"]
        again = self._provider(client, tmp_path)
        second = again.load_1m("SBIN", "NSE", date(2026, 9, 21), date(2026, 9, 23))
        pd.testing.assert_frame_equal(first, second)
        assert len(client.calls) == 1  # served from the cache
        assert again.sources_used == {"cache"}

    def test_a_chunk_ending_today_is_never_cached(self, tmp_path):
        bars = make_bars([date(2026, 11, 4), date(2026, 11, 5)])
        client = FakeSdkClient(bars, bars)
        provider = self._provider(client, tmp_path, today=date(2026, 11, 5))
        provider.load_1m("SBIN", "NSE", date(2026, 11, 4), date(2026, 11, 5))
        assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").iterdir())


class TestBusyLoops:
    def test_client_calls_alone_still_hit_the_time_limit(self):
        _m, _b, engine = setup()
        engine.max_wall_seconds = 60
        engine.wall_start -= 61  # the run began a minute ago, in real time
        with pytest.raises(BacktestFinished):
            for _ in range(5000):
                engine.note_call()
        assert engine.outcome == "stopped"
        assert "never pauses" in engine.message

    def test_client_calls_without_sleep_let_time_pass(self):
        _m, _b, engine = setup()
        t0 = engine.now
        for _ in range(20_000):
            engine.note_call()
        assert engine.now == t0 + 1


class TestSquareOffSettings:
    def test_the_sandbox_times_are_used(self):
        from services.python_backtest import sessions

        saved = dict(sessions.SESSIONS)
        try:
            sessions.configure_square_off(
                {"nse_bse_square_off_time": "15:00", "mcx_square_off_time": "bad"}
            )
            assert session_for("NSE").square_off_sec == 15 * 3600
            assert session_for("NFO").square_off_sec == 15 * 3600
            assert session_for("MCX").square_off_sec == 23 * 3600 + 30 * 60
            sessions.configure_square_off({"nse_bse_square_off_time": "08:00"})  # before open
            assert session_for("NSE").square_off_sec == 15 * 3600
        finally:
            sessions.SESSIONS.clear()
            sessions.SESSIONS.update(saved)
        assert session_for("NSE").square_off_sec == 15 * 3600 + 15 * 60


class TestNetworkRules:
    def test_the_app_is_blocked_and_the_loader_window_lets_it_through(self):
        from services.python_backtest import guards

        guards.configure_network(
            ["http://127.0.0.1:5999", "https://my.openalgo.example:443"], [5000]
        )
        assert guards.is_blocked(("127.0.0.1", 5999))
        assert guards.is_blocked(("localhost", 5999))
        assert guards.is_blocked(("::1", 5999, 0, 0))
        assert guards.is_blocked(("::ffff:127.0.0.1", 5000, 0, 0))
        assert guards.is_blocked(("0.0.0.0", 5000))
        assert guards.is_blocked("/run/openalgo.sock")
        assert not guards.is_blocked(("127.0.0.1", 6001))
        assert not guards.is_blocked(("93.184.216.34", 443))
        with guards.allow_app():
            assert not guards.is_blocked(("127.0.0.1", 5999))
        assert guards.is_blocked(("127.0.0.1", 5999))
