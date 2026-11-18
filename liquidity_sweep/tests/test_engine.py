"""ATR, entries, exits, intraday rules, costs and a full trade."""

import numpy as np
import pandas as pd
import pytest

from backtest import net_r_at_costs, run_backtest, trade_r
from config import Config
from resample import resample_ohlcv
from scenarios import BASE, RES, SUP, cfg, flat, scenario, set_bar, sweep, with_level
from strategy import wilder_atr
from synthetic import session_index

TZ = "Asia/Kolkata"
STOP_SHORT = (RES + 0.3) * 1.0005
STOP_LONG = (SUP - 0.3) * (1 - 0.0005)


def ts(s):
    return pd.Timestamp(s, tz=TZ)


# ---------------------------------------------------------------- ATR


def test_wilder_atr_values():
    h = np.array([10, 11, 12, 11, 13], float)
    lo = np.array([9, 10, 10, 10, 11], float)
    c = np.array([9.5, 10.5, 11, 10.5, 12.5], float)
    atr = wilder_atr(h, lo, c, 3)
    tr = [1.0, 1.5, 2.0, 1.0, 2.5]  # tr[1] = max(1, |11-9.5|, |10-9.5|) = 1.5
    assert np.isnan(atr[:2]).all()
    assert atr[2] == pytest.approx(np.mean(tr[:3]))
    assert atr[3] == pytest.approx((atr[2] * 2 + tr[3]) / 3)
    assert atr[4] == pytest.approx((atr[3] * 2 + tr[4]) / 3)


def test_atr_uses_no_future_bars():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(0, 1, 200).cumsum()
    h, lo = c + rng.uniform(0, 1, 200), c - rng.uniform(0, 1, 200)
    full = wilder_atr(h, lo, c, 14)
    for k in (14, 50, 120):
        part = wilder_atr(h[:k], lo[:k], c[:k], 14)
        np.testing.assert_allclose(part, full[:k], equal_nan=True)


@pytest.mark.parametrize("side", ["short", "long"])
def test_atr_buffer_widens_stop_and_target_stays_2r(side):
    df = scenario(side)
    plain = run_backtest(df, cfg(atr_mult=0.0), "X").iloc[0]
    wide = run_backtest(df, cfg(atr_mult=5.0), "X").iloc[0]
    atr = wilder_atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14)[50]
    if side == "short":
        assert plain["stop"] == pytest.approx(STOP_SHORT)
        assert wide["stop"] == pytest.approx(max(STOP_SHORT, RES + 0.3 + 5.0 * atr))
        assert wide["stop"] > plain["stop"]
    else:
        assert plain["stop"] == pytest.approx(STOP_LONG)
        assert wide["stop"] == pytest.approx(min(STOP_LONG, SUP - 0.3 - 5.0 * atr))
        assert wide["stop"] < plain["stop"]
    for t in (plain, wide):
        risk = abs(t["entry_price"] - t["stop"])
        assert abs(t["target"] - t["entry_price"]) == pytest.approx(2.0 * risk)
        assert np.sign(t["target"] - t["entry_price"]) == (1 if side == "long" else -1)


def test_atr_mult_zero_is_plain_buffer_exactly():
    df = scenario("short")
    t = run_backtest(df, cfg(atr_mult=0.0), "X").iloc[0]
    assert t["stop"] == (RES + 0.3) * (1 + 0.0005)


# ---------------------------------------------------------------- exits


@pytest.mark.parametrize("side", ["short", "long"])
def test_stop_first_when_stop_and_target_in_same_bar(side):
    df = scenario(side)
    set_bar(df, 53, h=110.0, lo=90.0)
    t = run_backtest(df, cfg(), "X").iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_time"] == df.index[53]
    assert t["exit_price"] == pytest.approx(STOP_SHORT if side == "short" else STOP_LONG)
    assert t["r_gross"] == pytest.approx(-1.0)


@pytest.mark.parametrize("side", ["short", "long"])
def test_target_exit(side):
    df = scenario(side)
    if side == "short":
        set_bar(df, 53, lo=90.0)
    else:
        set_bar(df, 53, h=110.0)
    t = run_backtest(df, cfg(), "X").iloc[0]
    assert t["exit_reason"] == "target"
    assert t["r_gross"] == pytest.approx(2.0)


def test_gap_past_stop_fills_at_open():
    df = scenario("short")
    set_bar(df, 53, o=101.0, h=101.5, c=101.0)
    t = run_backtest(df, cfg(), "X").iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == 101.0
    assert t["r_gross"] < -1.0


def test_entry_skipped_when_open_already_beyond_stop():
    df = scenario("long")
    set_bar(df, 51, o=97.0, lo=96.9, c=97.9)  # no new sweep, no break
    assert run_backtest(df, cfg(), "X").empty


def test_time_exit_after_max_bars():
    df = scenario("short")
    t = run_backtest(df, cfg(max_bars=5), "X").iloc[0]
    assert t["exit_reason"] == "time"
    assert t["entry_time"] == df.index[51]
    assert t["exit_time"] == df.index[55]  # entry bar counts as bar 1
    assert t["exit_price"] == df["close"].iloc[55]


def test_end_of_data_exit():
    df = scenario("short").iloc[:60]
    t = run_backtest(df, cfg(), "X").iloc[0]
    assert t["exit_reason"] == "end_of_data"
    assert t["exit_time"] == df.index[-1]


def test_new_signal_may_come_from_the_exit_bar():
    df = scenario("short")
    # Bar 53: stop out, and the same bar is a support sweep -> long entry at 54.
    df = with_level(df, "long", pivots=(12, 27, 42))
    set_bar(df, 53, h=100.5, lo=97.6, c=98.4)
    trades = run_backtest(df, cfg(), "X")
    assert list(trades["side"]) == ["short", "long"]
    assert trades.iloc[0]["exit_time"] == df.index[53]
    assert trades.iloc[1]["entry_time"] == df.index[54]


# ---------------------------------------------------------------- intraday rules


def test_session_end_exit_at_1515():
    df = scenario("short")
    t = run_backtest(df, cfg(), "X").iloc[0]
    assert t["exit_reason"] == "session_end"
    assert t["exit_time"] == ts("2026-01-05 15:10")  # the 5m bar ending at 15:15
    assert t["exit_price"] == BASE


def test_session_end_checked_after_stop():
    df = scenario("short")
    i = df.index.get_loc(ts("2026-01-05 15:10"))
    set_bar(df, i, h=110.0)
    t = run_backtest(df, cfg(), "X").iloc[0]
    assert t["exit_reason"] == "stop" and t["exit_time"] == df.index[i]


def _native(minutes: int, days, pivot: int, sweep_at: int) -> pd.DataFrame:
    idx = session_index(list(days), minutes=minutes)
    n = len(idx)
    df = pd.DataFrame(
        {"open": BASE, "high": BASE + 0.2, "low": BASE - 0.2, "close": BASE, "volume": 1000.0},
        index=idx,
    )
    set_bar(df, pivot, h=RES)
    sweep(df, "short", sweep_at)
    assert n > sweep_at
    return df


def test_session_end_on_15m_and_1h():
    cases = (
        ("15m", 15, ("2026-01-05",), 3, 10, "2026-01-05 15:00"),
        ("1h", 60, ("2026-01-05", "2026-01-06"), 1, 8, "2026-01-06 14:15"),
    )
    for tf, minutes, days, pivot, at, exit_label in cases:
        bars = _native(minutes, days, pivot, at)
        trades = run_backtest(bars, cfg(timeframe=tf, pivot_n=1, min_touches=1), "X")
        assert len(trades) == 1
        t = trades.iloc[0]
        assert t["entry_time"] == bars.index[at + 1]
        assert t["exit_reason"] == "session_end"
        assert t["exit_time"] == ts(exit_label)


def _sweep_at_time(hhmm: str):
    df = with_level(flat(("2026-01-05", "2026-01-06")), "short")
    i = df.index.get_loc(ts(f"2026-01-05 {hhmm}"))
    sweep(df, "short", i)
    return df, i


def test_no_entry_into_the_next_session():
    df, i = _sweep_at_time("15:25")  # the session's last bar
    assert run_backtest(df, cfg(), "X").empty


def test_no_entry_on_bar_opening_at_or_after_1500():
    df, i = _sweep_at_time("14:55")  # next bar opens 15:00
    assert run_backtest(df, cfg(), "X").empty
    df, i = _sweep_at_time("14:50")  # next bar opens 14:55: allowed
    t = run_backtest(df, cfg(), "X")
    assert len(t) == 1 and t.iloc[0]["entry_time"] == ts("2026-01-05 14:55")
    assert t.iloc[0]["exit_reason"] == "session_end"


def test_levels_persist_across_sessions():
    df = with_level(flat(("2026-01-05", "2026-01-06")), "short")
    i = df.index.get_loc(ts("2026-01-06 10:00"))
    sweep(df, "short", i)
    t = run_backtest(df, cfg(), "X")
    assert len(t) == 1 and t.iloc[0]["entry_time"] == ts("2026-01-06 10:05")


def test_missing_square_off_bar_flattens_at_next_open():
    df = scenario("short")
    df = df.drop([ts("2026-01-05 15:10")])
    t = run_backtest(df, cfg(), "X").iloc[0]
    assert t["exit_reason"] == "session_end"
    assert t["exit_time"] == ts("2026-01-05 15:15")


def test_bar_size_mismatch_raises():
    with pytest.raises(ValueError):
        run_backtest(resample_ohlcv(flat(), "15m"), cfg(), "X")


# ---------------------------------------------------------------- costs


def test_cost_formula():
    g, n = trade_r(1, 100.0, 102.0, 99.0, 0.0004, 0.0002)
    assert g == pytest.approx(2.0)
    assert n == pytest.approx(2.0 - 0.0006 * 202.0 / 1.0)
    g, n = trade_r(-1, 100.0, 98.0, 101.0, 0.0004, 0.0002)
    assert g == pytest.approx(2.0)
    assert n == pytest.approx(2.0 - 0.0006 * 198.0)


def test_low_cost_repricing():
    trades = run_backtest(scenario("short"), cfg(), "X")
    low = net_r_at_costs(trades)
    t = trades.iloc[0]
    risk = abs(t["entry_price"] - t["stop"])
    c = Config()
    expect = (
        t["r_gross"] - (c.low_fee + c.low_slippage) * (t["entry_price"] + t["exit_price"]) / risk
    )
    assert low.iloc[0] == pytest.approx(expect)
    assert low.iloc[0] > t["r_net"]
    same = net_r_at_costs(trades, c.fee, c.slippage)
    assert same.iloc[0] == pytest.approx(t["r_net"])


# ---------------------------------------------------------------- full trade


def test_full_trade_start_to_finish():
    df = scenario("short")
    set_bar(df, 56, lo=96.0)
    trades = run_backtest(df, cfg(), "SYM")
    assert len(trades) == 1
    t = trades.iloc[0]
    entry = BASE
    stop = (RES + 0.3) * 1.0005
    risk = stop - entry
    target = entry - 2 * risk
    assert t["symbol"] == "SYM" and t["side"] == "short"
    assert t["entry_time"] == df.index[51] and t["entry_price"] == entry
    assert t["stop"] == pytest.approx(stop) and t["target"] == pytest.approx(target)
    assert t["exit_time"] == df.index[56] and t["exit_price"] == pytest.approx(target)
    assert t["exit_reason"] == "target"
    assert t["r_gross"] == pytest.approx(2.0)
    assert t["r_net"] == pytest.approx(2.0 - 0.0006 * (entry + target) / risk)
    assert t["level"] == pytest.approx(RES) and t["touches"] == 3
