"""Tests for strategies/examples/three_candle_reversal_multi.py.

The strategy module is imported by path: it creates no client and makes no
network call at import. The Supertrend tests compare it with an independent,
literal reading of TradingView's Pine semantics. The setup tests use hand-built
daily candles. The order-safety tests drive it, long and short, against a fake
broker that loses answers, shows orders late, or misreports, and the last
tests run it unchanged through the real backtest runner on synthetic bars
shaped so that the setups occur.

THREE_CANDLE_SCRIPT_UNDER_TEST points the whole file at another copy of the
strategy. It exists for mutation checks, which run against a changed scratch
copy and never touch the file in the repository.

Run:
    uv run pytest test/test_three_candle_reversal_strategy.py -v
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

from services.python_backtest.sessions import date_to_day, session_for
from test.test_python_backtest_e2e import run_backtest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = Path(
    os.environ.get("THREE_CANDLE_SCRIPT_UNDER_TEST")
    or REPO / "strategies" / "examples" / "three_candle_reversal_multi.py"
)


def _load(name="three_candle_reversal_multi"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st = _load()

TODAY = date(2026, 11, 4)  # a Wednesday; Candle 2 is Tuesday 3 November
YESTERDAY = "2026-11-03"
NOW = datetime(2026, 11, 4, 9, 20, 6)


# =============================================================================
# DAILY CANDLES BUILT BY HAND
# =============================================================================
# (open, high, low, close). With TP 5%:
#   long:  stop 95, take profit 110 x 0.95 = 104.5, entry 97  -> reward:risk 3.75
#   short: stop 105, take profit 90 x 1.05 = 94.5, entry 103  -> reward:risk 4.25
LONG_C1, LONG_C2 = (100.0, 110.0, 96.0, 98.0), (98.0, 101.0, 95.0, 97.0)
SHORT_C1, SHORT_C2 = (100.0, 104.0, 90.0, 102.0), (102.0, 105.0, 99.0, 103.0)


def trend_history(up, n=70):
    """n daily bars moving 1% a day toward 100, so the Supertrend has a clear side."""
    step = 1.01
    c = 100.0 * (step ** -np.arange(n, 0, -1.0) if up else step ** np.arange(n, 0, -1.0))
    o = np.r_[c[0] / step if up else c[0] * step, c[:-1]]
    h, lo = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    return [list(map(float, x)) for x in (o, h, lo, c)]


def daily_bars(up, c1, c2, last=YESTERDAY, n=70):
    """A trend, then Candle 1 and Candle 2. The last row is dated `last`."""
    o, h, lo, c = trend_history(up, n)
    for bar in (c1, c2):
        for column, value in zip((o, h, lo, c), bar, strict=True):
            column.append(float(value))
    index = pd.bdate_range(end=last, periods=len(c))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}, index=index)


def long_bars(c1=LONG_C1, c2=LONG_C2, **kwargs):
    return daily_bars(False, c1, c2, **kwargs)  # a downtrend: Supertrend bearish


def short_bars(c1=SHORT_C1, c2=SHORT_C2, **kwargs):
    return daily_bars(True, c1, c2, **kwargs)  # an uptrend: Supertrend bullish


def flat_bars(last=YESTERDAY, n=72):
    index = pd.bdate_range(end=last, periods=n)
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)


def with_today(bars, values=(1.0, 99999.0, 0.5, 99999.0), today=TODAY):
    """The same history with a bar dated today on the end, as the feed gives it."""
    extra = pd.DataFrame(
        [dict(zip(("open", "high", "low", "close"), values, strict=True))],
        index=[pd.Timestamp(today)],
    )
    return pd.concat([bars, extra])


def trend_of(bars):
    return st.supertrend_trend(bars["high"], bars["low"], bars["close"])[-1]


def random_daily(n, seed, base=1000.0, flat_start=0):
    """Daily bars with overnight gaps and some flat bars (open = high = low = close)."""
    rng = np.random.default_rng(seed)
    c = base * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = np.r_[base, c[:-1]] * (1 + rng.normal(0, 0.012, n))  # gaps from the last close
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.008, n)))
    flat = rng.random(n) < 0.08
    flat[:flat_start] = True
    for i in np.flatnonzero(flat):
        value = base if i == 0 or i < flat_start else c[i - 1]
        o[i] = h[i] = lo[i] = c[i] = value
    return o, h, lo, c


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


def ref_rma(src, length):
    """Pine: x := na(x[1]) ? ta.sma(src, length) : alpha * src + (1 - alpha) * nz(x[1])"""
    alpha, out, prev = 1.0 / length, [], None
    for i in range(len(src)):
        value = ref_sma(src, length, i) if prev is None else alpha * src[i] + (1 - alpha) * prev
        out.append(value)
        prev = value
    return out


def ref_atr(h, lo, c, length):
    tr = [h[0] - lo[0]] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, len(c))
    ]
    return ref_rma(tr, length)


def nz(value):
    return 0.0 if value is None else value


def gt(a, b):
    """Pine's a > b: false when either side is na."""
    return a is not None and b is not None and a > b


def ref_supertrend(h, lo, c, factor, length):
    """The Pine source of ta.supertrend, line for line. Returns
    (superTrend, direction, upperBand, lowerBand) as lists, None for na."""
    h, lo, c = (list(map(float, x)) for x in (h, lo, c))
    atr = ref_atr(h, lo, c, length)
    uppers, lowers, lines, dirs = [], [], [], []
    for i in range(len(c)):
        src = (h[i] + lo[i]) / 2
        upper = None if atr[i] is None else src + factor * atr[i]
        lower = None if atr[i] is None else src - factor * atr[i]
        prev_lower = nz(lowers[i - 1]) if i else 0.0
        prev_upper = nz(uppers[i - 1]) if i else 0.0
        prev_close = c[i - 1] if i else None
        lower = lower if gt(lower, prev_lower) or gt(prev_lower, prev_close) else prev_lower
        upper = upper if gt(prev_upper, upper) or gt(prev_close, prev_upper) else prev_upper
        prev_line = lines[i - 1] if i else None
        if i == 0 or atr[i - 1] is None:
            direction = 1
        elif prev_line is not None and prev_line == prev_upper:
            direction = -1 if gt(c[i], upper) else 1
        else:
            direction = 1 if gt(lower, c[i]) else -1
        uppers.append(upper)
        lowers.append(lower)
        dirs.append(direction)
        lines.append(lower if direction == -1 else upper)
    return lines, dirs, uppers, lowers


def arr(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


# =============================================================================
# SUPERTREND
# =============================================================================
@pytest.mark.parametrize("factor,length", [(3.0, 10), (2.0, 7), (1.5, 14)])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_supertrend_matches_pine_on_gappy_and_flat_daily_bars(seed, factor, length):
    _o, h, lo, c = random_daily(400, seed)
    line, direction = st.pine_supertrend(h, lo, c, factor, length)
    ref_line, ref_dir, _u, _l = ref_supertrend(h, lo, c, factor, length)
    assert list(direction) == ref_dir
    np.testing.assert_allclose(line, arr(ref_line), rtol=1e-10, equal_nan=True)
    flips = int(np.count_nonzero(np.diff(direction[length:])))
    assert flips >= 2, "the series must turn both ways for the test to mean anything"


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_supertrend_matches_pine_when_the_series_starts_flat(seed):
    # 15 flat bars first: ATR is 0, both bands sit on the price, and the
    # "previous line == previous upper band" test is true on either side.
    _o, h, lo, c = random_daily(200, seed, flat_start=15)
    line, direction = st.pine_supertrend(h, lo, c, 3.0, 10)
    ref_line, ref_dir, _u, _l = ref_supertrend(h, lo, c, 3.0, 10)
    assert list(direction) == ref_dir
    np.testing.assert_allclose(line, arr(ref_line), rtol=1e-10, equal_nan=True)


def test_supertrend_by_hand():
    # ATR length 1 (ATR = true range), factor 1, worked by hand:
    #   bar 0  tr 2    bands 11 / 7    upper carried as 0 (no close[1])  dir +1  line 0
    #   bar 1  tr 3    bands 14 / 8    prev line == prev upper, 11.5 <= 14  dir +1  line 14
    #   bar 2  tr 4.5  upper stays 14, lower 10; 15.5 > 14                  dir -1  line 10
    #   bar 3  tr 6.5  lower stays 10, upper 18.5; 9.5 < 10                 dir +1  line 18.5
    line, direction = st.pine_supertrend(
        [10, 12, 16, 15], [8, 10, 13, 9], [9, 11.5, 15.5, 9.5], 1.0, 1
    )
    assert list(direction) == [1, 1, -1, 1]
    assert list(line) == [0.0, 14.0, 10.0, 18.5]


def test_supertrend_while_atr_is_na():
    _o, h, lo, c = random_daily(60, 21)
    line, direction = st.pine_supertrend(h, lo, c, 3.0, 10)
    trend = st.supertrend_trend(h, lo, c, 3.0, 10)
    assert list(direction[:10]) == [1] * 10  # na(atr[1]) up to and including bar 9
    assert line[0] == 0.0 and np.isnan(line[1:9]).all()  # nz() on bar 0, then na
    atr9 = ref_atr(list(h), list(lo), list(c), 10)[9]
    assert line[9] == pytest.approx((h[9] + lo[9]) / 2 + 3.0 * atr9, rel=1e-12)  # upper band
    assert trend[:10] == [None] * 10 and all(t in ("bullish", "bearish") for t in trend[10:])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_supertrend_direction_names_and_band_carry(seed):
    _o, h, lo, c = random_daily(400, seed)
    line, direction = st.pine_supertrend(h, lo, c, 3.0, 10)
    trend = st.supertrend_trend(h, lo, c, 3.0, 10)
    assert {"bullish", "bearish"} <= set(trend)
    for i in range(11, len(c)):
        if trend[i] == "bullish":
            assert direction[i] == -1 and line[i] <= c[i]  # the line is below the price
            if trend[i - 1] == "bullish":
                assert line[i] >= line[i - 1]  # the lower band never falls in an uptrend
        else:
            assert direction[i] == 1 and line[i] >= c[i]  # the line is above the price
            if trend[i - 1] == "bearish":
                assert line[i] <= line[i - 1]  # the upper band never rises in a downtrend


def test_supertrend_never_reads_ahead():
    _o, h, lo, c = random_daily(300, 7)
    line, direction = st.pine_supertrend(h, lo, c)
    for t in (15, 60, 150, 299):
        part_line, part_dir = st.pine_supertrend(h[: t + 1], lo[: t + 1], c[: t + 1])
        np.testing.assert_array_equal(part_line, line[: t + 1])
        np.testing.assert_array_equal(part_dir, direction[: t + 1])
        h2, lo2, c2 = h.copy(), lo.copy(), c.copy()
        h2[t + 1 :] *= 3.0
        lo2[t + 1 :] *= 0.2
        c2[t + 1 :] = c2[t + 1 :][::-1]
        moved_line, moved_dir = st.pine_supertrend(h2, lo2, c2)
        np.testing.assert_array_equal(moved_line[: t + 1], line[: t + 1])
        np.testing.assert_array_equal(moved_dir[: t + 1], direction[: t + 1])


# =============================================================================
# THE SETUP: every condition, long and short
# =============================================================================
def test_long_setup_with_every_condition_true():
    bars = long_bars()
    assert trend_of(bars) == "bearish"
    setup, why = st.find_setup(bars)
    assert why == "" and setup.side == "BUY" and setup.trend == "bearish"
    assert setup.ref_entry == 97.0  # Candle 2 close
    assert setup.stop == 95.0  # Candle 2 low
    assert setup.target == pytest.approx(104.5)  # Candle 1 high 110 x 0.95
    assert setup.rr == pytest.approx(3.75)
    assert (setup.candle1, setup.candle2) == ("2026-11-02", YESTERDAY)


def test_short_setup_with_every_condition_true():
    bars = short_bars()
    assert trend_of(bars) == "bullish"
    setup, why = st.find_setup(bars)
    assert why == "" and setup.side == "SELL" and setup.trend == "bullish"
    assert setup.ref_entry == 103.0  # Candle 2 close
    assert setup.stop == 105.0  # Candle 2 high
    assert setup.target == pytest.approx(94.5)  # Candle 1 low 90 x 1.05
    assert setup.rr == pytest.approx(4.25)


LONG_BREAKS = {
    "Candle 2 high equals Candle 1 high": (98.0, 110.0, 95.0, 97.0),
    "Candle 2 high above Candle 1 high": (98.0, 111.0, 95.0, 97.0),
    "Candle 2 close equals Candle 1 close": (98.0, 101.0, 95.0, 98.0),
    "Candle 2 close above Candle 1 close": (98.0, 101.0, 95.0, 99.0),
    "Candle 2 close equals Candle 1 low": (98.0, 101.0, 95.0, 96.0),
    "Candle 2 close below Candle 1 low": (98.0, 101.0, 94.0, 95.5),
}
SHORT_BREAKS = {
    "Candle 2 low equals Candle 1 low": (102.0, 105.0, 90.0, 103.0),
    "Candle 2 low below Candle 1 low": (102.0, 105.0, 89.0, 103.0),
    "Candle 2 close equals Candle 1 close": (102.0, 105.0, 99.0, 102.0),
    "Candle 2 close below Candle 1 close": (102.0, 105.0, 99.0, 101.0),
    "Candle 2 close equals Candle 1 high": (102.0, 105.0, 99.0, 104.0),
    "Candle 2 close above Candle 1 high": (102.0, 106.0, 99.0, 104.5),
}


@pytest.mark.parametrize("case", list(LONG_BREAKS))
def test_long_needs_each_candle_condition(case):
    bars = long_bars(c2=LONG_BREAKS[case])
    assert trend_of(bars) == "bearish"  # so the candles are what stops it
    setup, why = st.find_setup(bars)
    assert setup is None and why.startswith("no setup")


@pytest.mark.parametrize("case", list(SHORT_BREAKS))
def test_short_needs_each_candle_condition(case):
    bars = short_bars(c2=SHORT_BREAKS[case])
    assert trend_of(bars) == "bullish"
    setup, why = st.find_setup(bars)
    assert setup is None and why.startswith("no setup")


def test_long_needs_a_bearish_supertrend_and_short_a_bullish_one():
    wrong_long = daily_bars(True, LONG_C1, LONG_C2)  # the long candles in an uptrend
    assert trend_of(wrong_long) == "bullish"
    assert st.find_setup(wrong_long)[0] is None
    wrong_short = daily_bars(False, SHORT_C1, SHORT_C2)
    assert trend_of(wrong_short) == "bearish"
    assert st.find_setup(wrong_short)[0] is None


def test_a_wick_beyond_candle_1_is_allowed_only_the_close_matters():
    setup, _ = st.find_setup(long_bars(c2=(98.0, 101.0, 90.0, 97.0)))  # low far below 96
    assert setup.side == "BUY" and setup.stop == 90.0
    setup, _ = st.find_setup(short_bars(c2=(102.0, 109.0, 99.0, 103.0)))  # high far above 104
    assert setup.side == "SELL" and setup.stop == 109.0


def test_take_profit_at_or_beyond_the_entry_is_dropped():
    # long: 110 x (1 - 12%) = 96.8, below the entry 97
    setup, why = st.find_setup(long_bars(), tp_pct=12.0)
    assert setup is None and "take profit is at or below the entry" in why
    # exactly at the entry: 110 x 0.882 = 97.02, rounded down to the tick = 97.00
    setup, why = st.find_setup(long_bars(), tp_pct=11.8)
    assert setup is None and "take profit is at or below the entry" in why
    assert st.find_setup(long_bars(), tp_pct=11.7)[0].target == pytest.approx(97.1)
    # short: 90 x (1 + 15%) = 103.5, above the entry 103
    setup, why = st.find_setup(short_bars(), tp_pct=15.0)
    assert setup is None and "take profit is at or above the entry" in why
    # exactly at the entry: 90 x 1.1444 = 102.996, rounded up to the tick = 103.00
    setup, why = st.find_setup(short_bars(), tp_pct=14.44)
    assert setup is None and "take profit is at or above the entry" in why
    assert st.find_setup(short_bars(), tp_pct=14.3)[0].target == pytest.approx(102.9)


def test_min_rr_filter_uses_the_reference_entry():
    assert st.MIN_RR == 0.0  # off by default
    assert st.find_setup(long_bars(), min_rr=3.75)[0] is not None  # 3.75 is not below 3.75
    setup, why = st.find_setup(long_bars(), min_rr=3.76)
    assert setup is None and "reward:risk 3.75 is below the minimum 3.76" in why
    assert st.find_setup(short_bars(), min_rr=4.25)[0] is not None
    setup, why = st.find_setup(short_bars(), min_rr=4.3)
    assert setup is None and "reward:risk 4.25 is below the minimum 4.3" in why


def test_each_side_can_be_switched_off():
    assert st.ENABLE_LONGS is True and st.ENABLE_SHORTS is True
    setup, why = st.find_setup(long_bars(), longs=False)
    assert setup is None and "longs are switched off" in why
    assert st.find_setup(long_bars(), shorts=False)[0].side == "BUY"
    setup, why = st.find_setup(short_bars(), shorts=False)
    assert setup is None and "shorts are switched off" in why
    assert st.find_setup(short_bars(), longs=False)[0].side == "SELL"


def test_tp_pct_moves_the_take_profit():
    assert st.TP_PCT == 5.0
    assert st.find_setup(long_bars(), tp_pct=2.0)[0].target == pytest.approx(107.8)
    assert st.find_setup(long_bars(), tp_pct=0.0)[0].target == pytest.approx(110.0)
    assert st.find_setup(short_bars(), tp_pct=2.0)[0].target == pytest.approx(91.8)
    assert st.find_setup(short_bars(), tp_pct=2.0)[0].rr == pytest.approx((103 - 91.8) / 2)


def test_supertrend_settings_reach_the_setup(monkeypatch):
    bars = long_bars()
    calls = []
    real = st.pine_supertrend

    def spy(high, low, close, factor=None, atr_len=None):
        calls.append((factor, atr_len))
        return real(high, low, close, factor, atr_len)

    monkeypatch.setattr(st, "pine_supertrend", spy)
    st.find_setup(bars, factor=2.0, atr_len=7)
    assert calls == [(2.0, 7)]
    assert (st.ST_FACTOR, st.ST_ATR_LEN) == (3.0, 10)
    # too little history for the Supertrend: nothing is signalled
    assert "not enough daily history" in st.find_setup(bars.iloc[-8:])[1]


def test_stop_and_target_are_rounded_to_the_tick_conservatively():
    # long: both rounded down (stop away from the entry, target toward it)
    long_setup, _ = st.find_setup(
        long_bars(c1=(100.0, 110.07, 96.0, 98.0), c2=(98.0, 101.0, 95.03, 97.0)), tick=0.05
    )
    assert long_setup.stop == pytest.approx(95.0) and long_setup.stop <= 95.03
    assert long_setup.target == pytest.approx(104.55) and long_setup.target <= 110.07 * 0.95
    # short: both rounded up
    short_setup, _ = st.find_setup(
        short_bars(c1=(100.0, 104.0, 90.03, 102.0), c2=(102.0, 105.03, 99.0, 103.0)), tick=0.05
    )
    assert short_setup.stop == pytest.approx(105.05) and short_setup.stop >= 105.03
    assert short_setup.target == pytest.approx(94.55) and short_setup.target >= 90.03 * 1.05
    # prices already on the tick are left exactly where they are
    assert st.floor_tick(95.0, 0.05) == 95.0 and st.ceil_tick(105.0, 0.05) == 105.0
    assert st.floor_tick(104.5, 0.05) == 104.5 and st.ceil_tick(94.5, 0.05) == 94.5


# =============================================================================
# NO LOOKAHEAD: today's forming daily bar is never used
# =============================================================================
def test_the_bar_dated_today_is_dropped_whatever_it_holds():
    for bars in (long_bars(), short_bars()):
        clean = st.completed_daily(bars, TODAY)
        wild = st.completed_daily(with_today(bars), TODAY)
        assert wild.equals(clean) and wild.index[-1] == pd.Timestamp(YESTERDAY)
        assert st.find_setup(wild) == st.find_setup(clean)
        assert st.find_setup(wild)[0] is not None
    # the day after, that bar is a completed one and is kept
    later = st.completed_daily(with_today(long_bars()), TODAY + timedelta(days=1))
    assert later.index[-1] == pd.Timestamp(TODAY)


@pytest.mark.parametrize("stamp", ["ist date", "utc midnight", "ist midnight as utc", "tz aware"])
def test_today_is_recognised_however_the_daily_bar_is_stamped(stamp):
    bars = with_today(long_bars())
    if stamp == "ist midnight as utc":
        # IST midnight is 18:30 UTC the day before; the SDK hands it over with no zone
        bars.index = bars.index - pd.Timedelta(hours=5, minutes=30)
        assert bars.index[-1] == pd.Timestamp("2026-11-03 18:30")
    elif stamp == "tz aware":
        bars.index = bars.index.tz_localize("Asia/Kolkata")
    kept = st.completed_daily(bars, TODAY)
    assert kept.index[-1] == pd.Timestamp(YESTERDAY) and len(kept) == len(bars) - 1
    assert list(kept.index) == list(long_bars().index)
    assert st.find_setup(kept)[0].stop == 95.0


def test_later_daily_bars_never_change_an_earlier_days_answer():
    bars = long_bars()
    answer = st.find_setup(bars)
    grown = bars.copy()
    for k in range(1, 6):  # five more days, of anything
        grown = with_today(grown, (500.0 * k, 900.0 * k, 1.0, 700.0 * k), TODAY + timedelta(k))
    assert st.find_setup(st.completed_daily(grown, TODAY)) == answer
    assert st.find_setup(grown.iloc[: len(bars)]) == answer


class History:
    def __init__(self, frame):
        self.frame, self.asked = frame, []

    def history(self, **kwargs):
        self.asked.append(kwargs)
        return self.frame


def test_load_daily_asks_for_daily_bars_and_drops_today(tmp_path):
    client = History(with_today(short_bars()))
    s = st.ThreeCandleStrategy(client, ["X"], clock=lambda: NOW, data_dir=tmp_path)
    kept = s.load_daily("X", NOW)
    assert kept.index[-1] == pd.Timestamp(YESTERDAY)
    asked = client.asked[0]
    assert asked["interval"] == "D" and asked["end_date"] == "2026-11-04"
    assert asked["start_date"] == f"{TODAY - timedelta(days=st.HISTORY_DAYS):%Y-%m-%d}"
    assert st.HISTORY_DAYS == 250
    found = s.evaluate_symbol("X", kept, NOW)
    client.frame = with_today(short_bars(), (7.0, 8.0, 6.0, 7.5))
    assert s.evaluate_symbol("X", s.load_daily("X", NOW), NOW) == found
    assert found["result"] == "candidate" and found["setup"]["side"] == "SELL"


def test_stale_or_short_daily_history_gives_no_signal(tmp_path):
    s = st.ThreeCandleStrategy(None, ["X"], clock=lambda: NOW, data_dir=tmp_path)
    old = s.evaluate_symbol("X", long_bars(last="2026-10-27"), NOW)  # 8 days before today
    assert old["result"] == "no data" and "8 days old" in old["why"]
    fresh = s.evaluate_symbol("X", long_bars(last="2026-10-30"), NOW)  # a long weekend: 5 days
    assert fresh["result"] == "candidate"
    short = s.evaluate_symbol("X", long_bars(n=30), NOW)
    assert short["result"] == "no data" and "50 are needed" in short["why"]
    assert st.min_bars() == 50
    assert s.evaluate_symbol("X", None, NOW) is None  # unreadable: read again later


# =============================================================================
# SIZE AND THE LIVE PRICE
# =============================================================================
def test_default_settings_are_the_users():
    assert (st.CAPITAL, st.LEVERAGE, st.MAX_LOSS_PER_TRADE) == (20000.0, 2.0, 0.0)
    assert (st.MAX_TRADES_PER_DAY, st.MAX_DAILY_LOSS) == (1, 4000.0)
    assert (st.ENTRY_START, st.ENTRY_END, st.SQUARE_OFF) == ("09:20", "09:45", "15:10")
    assert st.STRATEGY_NAME == "3 Candle Reversal" and st.PRODUCT == "MIS"
    assert st.DATA_DIR.name == "three_candle_data" and st.SWEEP_SEC == 300
    assert len(st.SYMBOLS) == 17 and "M&M" in st.SYMBOLS and st.SYMBOLS[0] == "ICICIBANK"


def test_position_value_is_capped_at_capital_times_leverage():
    assert st.size_position(100.0, 95.0, 1, 20000, 2.0, 0, 0) == (400, "position value")
    assert st.size_position(100.0, 95.0, 1, 20000, 1.0, 0, 0) == (200, "position value")
    # the defaults: 20000 x 2 = 40000, with the 0.1% slippage allowance
    qty, sized_by = st.size_position(100.0, 95.0)
    assert (qty, sized_by) == (399, "position value")  # floor(40000 / 100.1)
    assert qty * 100.1 <= 40000 < (qty + 1) * 100.1
    # a short is sized the same way
    assert st.size_position(100.0, 105.0) == (399, "position value")
    # no loss cap by default: a stop 30 away still gets the full size
    assert st.size_position(100.0, 70.0, 1, 20000, 2.0, 0, 0)[0] == 400


def test_quantity_is_rounded_down_to_the_lot():
    assert st.size_position(100.0, 95.0, 75, 20000, 2.0, 0, 0)[0] == 375
    assert st.size_position(100.0, 95.0, 150, 20000, 2.0, 0, 0)[0] == 300
    assert st.size_position(100.0, 95.0, 500, 20000, 2.0, 0, 0)[0] == 0


def test_the_optional_loss_cap_also_limits_quantity():
    assert st.size_position(100.0, 90.0, 1, 20000, 2.0, 2000, 0) == (200, "loss cap")
    assert st.size_position(100.0, 110.0, 1, 20000, 2.0, 2000, 0) == (200, "loss cap")  # short
    assert st.size_position(100.0, 98.0, 1, 20000, 2.0, 2000, 0) == (400, "position value")
    qty, _ = st.size_position(100.0, 90.0, 1, 20000, 2.0, 2000, 0.1)
    assert qty == 198 and qty * (100.1 - 90.0) <= 2000  # floor(2000 / 10.1)


def candidate(symbol, bars, **changes):
    setup, why = st.find_setup(bars)
    assert setup is not None, why
    return {"symbol": symbol, **st.asdict(setup), "status": "waiting", "why": "", **changes}


def test_plan_sizes_on_the_live_price_and_reports_the_loss_at_the_stop():
    plan, why = st.plan_entry("X", candidate("X", long_bars()), 97.5)
    assert why == "" and (plan.side, plan.stop, plan.target) == ("BUY", 95.0, 104.5)
    assert plan.qty == math.floor(40000 / (97.5 * 1.001)) == 409
    assert plan.position_value == pytest.approx(409 * 97.5) and plan.position_value <= 40000
    assert plan.planned_loss == pytest.approx(409 * 2.5)  # from the live price, not 97
    assert plan.ref_entry == 97.0 and plan.rr == pytest.approx(3.75)
    assert plan.signal_bar == YESTERDAY

    plan, why = st.plan_entry("X", candidate("X", short_bars()), 102.0)
    assert why == "" and (plan.side, plan.stop, plan.target) == ("SELL", 105.0, 94.5)
    assert plan.qty == math.floor(40000 / (102.0 * 1.001)) == 391
    assert plan.planned_loss == pytest.approx(391 * 3.0)


@pytest.mark.parametrize(
    "price,expected",
    [
        (95.0, "at or below the stop"),
        (94.0, "at or below the stop"),
        (95.05, None),
        (104.45, None),
        (104.5, "at or above the take profit"),
        (106.0, "at or above the take profit"),
    ],
)
def test_live_price_checks_for_a_long(price, expected):
    plan, why = st.plan_entry("X", candidate("X", long_bars()), price)
    assert (plan is None) == (expected is not None)
    assert expected is None or expected in why


@pytest.mark.parametrize(
    "price,expected",
    [
        (105.0, "at or above the stop"),
        (107.0, "at or above the stop"),
        (104.95, None),
        (94.55, None),
        (94.5, "at or below the take profit"),
        (93.0, "at or below the take profit"),
    ],
)
def test_live_price_checks_for_a_short(price, expected):
    plan, why = st.plan_entry("X", candidate("X", short_bars()), price)
    assert (plan is None) == (expected is not None)
    assert expected is None or expected in why


def test_quantity_zero_is_skipped_with_a_reason(monkeypatch):
    monkeypatch.setattr(st, "CAPITAL", 40.0)  # 80 of position value: less than one share
    plan, why = st.plan_entry("X", candidate("X", long_bars()), 97.0)
    assert plan is None and "quantity rounds to 0" in why
    monkeypatch.setattr(st, "CAPITAL", 20000.0)
    plan, why = st.plan_entry("X", candidate("X", long_bars()), 97.0, lot=1000)
    assert plan is None and "quantity rounds to 0" in why and "lot 1000" in why


def test_ranking_is_by_reward_risk_then_by_the_symbol_list():
    rows = [
        {"symbol": "C", "rr": 2.0},
        {"symbol": "A", "rr": 2.0},
        {"symbol": "D", "rr": 5.0},
        {"symbol": "B", "rr": 2.0},
    ]
    order = ["A", "B", "C", "D"]
    assert [r["symbol"] for r in st.rank_candidates(rows, order)] == ["D", "A", "B", "C"]
    assert [r["symbol"] for r in st.rank_candidates(rows[::-1], order)] == ["D", "A", "B", "C"]
    assert [r["symbol"] for r in st.rank_candidates(rows, ["C", "B", "A", "D"])] == [
        "D",
        "C",
        "B",
        "A",
    ]


def test_settings_can_be_overridden_from_the_environment(monkeypatch):
    for key, value in {
        "TCR_SYMBOLS": "SBIN, tcs",
        "TCR_TP_PCT": "3",
        "TCR_MIN_RR": "1.5",
        "TCR_ENABLE_SHORTS": "false",
        "TCR_CAPITAL": "50000",
        "TCR_LEVERAGE": "1",
        "TCR_MAX_LOSS_PER_TRADE": "1500",
        "TCR_ST_ATR_LEN": "7",
        "TCR_ST_FACTOR": "2",
        "TCR_ENTRY_START": "09:25",
        "TCR_MAX_TRADES_PER_DAY": "2",
        "TCR_SWEEP_SEC": "120",
    }.items():
        monkeypatch.setenv(key, value)
    other = _load("three_candle_env")
    assert other.SYMBOLS == ["SBIN", "TCS"] and other.TP_PCT == 3.0 and other.MIN_RR == 1.5
    assert other.ENABLE_SHORTS is False and other.ENABLE_LONGS is True
    assert (other.CAPITAL, other.LEVERAGE, other.MAX_LOSS_PER_TRADE) == (50000.0, 1.0, 1500.0)
    assert (other.ST_ATR_LEN, other.ST_FACTOR) == (7, 2.0)
    assert other.ENTRY_START == "09:25" and other.MAX_TRADES_PER_DAY == 2
    assert other.SWEEP_SEC == 120 and other.PRODUCT == "MIS"


# =============================================================================
# A FAKE BROKER: fills orders, and can lose answers, hide orders or misreport
# =============================================================================
class HostileBroker:
    """Orders can be hidden from the books for a while, fill in parts or late,
    and any answer can be lost after the broker accepted the order: raised as
    an exception, or returned as the SDK's own error dict (`lost_reply`).

    placesmartorder follows OpenAlgo's real decision table
    (broker/zerodha/api/order_api.py place_smartorder_api, the sandbox and the
    backtester): with a target of zero and no position it places the order as
    given, so a closing order that carries a quantity can open a position."""

    def __init__(self, symbols):
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
        self.ltps = {}  # symbol -> its own last price
        self.stop_fill = None
        self.status_qty_override = None
        self.positions_extra = []
        self.telegrams = []
        self.daily = {}  # symbol -> daily bars; a flat history otherwise
        self.history_fail = set()  # symbols whose history call fails
        self.ghost = set()  # placeorder price types that never reach the broker, answer lost
        self.before_smart = None  # run once after the caller read the position, before the app does
        self.smart_args = []  # (symbol, action, quantity, position_size) as the strategy sent them
        self.history_calls = []

    def _new(self, symbol, action, qty, ptype, status, trigger=0):
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
            "trigger": trigger,
        }
        return oid

    def price(self, symbol):
        return self.ltps.get(symbol, self.ltp_value)

    def _fill(self, o, q=None, price=None):
        q = o["qty"] - o["filled"] if q is None else q
        o["filled"] += q
        self.pos[o["symbol"]] += q if o["action"] == "BUY" else -q
        o["avg"] = self.price(o["symbol"]) if price is None else price
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
        return {"status": "success", "data": {"ltp": self.price(symbol)}}

    def history(self, symbol, **k):
        self.history_calls.append((symbol, k.get("interval")))
        if symbol in self.history_fail:
            return {"status": "error", "message": "history failed"}
        return self.daily[symbol] if symbol in self.daily else with_today(flat_bars())

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
        if price_type in self.ghost:
            return self._lost()
        if price_type == "MARKET":
            oid = self._new(symbol, action, quantity, "MARKET", "open")
            if not self.hold_market:
                self._fill(self.orders[oid])
        else:
            oid = self._new(
                symbol, action, quantity, price_type, "trigger pending", k.get("trigger_price")
            )
        if price_type in self.lost_answer:
            return self._lost()
        return {"status": "success", "orderid": oid}

    def placesmartorder(self, symbol, action, quantity, position_size, **k):
        if self.before_smart:
            hook, self.before_smart = self.before_smart, None
            hook()
        self.smart_args.append((symbol, action, quantity, position_size))
        current, target, qty = self.pos[symbol], int(position_size), int(quantity)
        # the real decision table
        if target == 0 and current == 0:
            side, size = str(action).upper(), qty  # placed as given
        elif target == 0:
            side, size = ("SELL" if current > 0 else "BUY"), abs(current)
        elif current == 0:
            side, size = ("BUY" if target > 0 else "SELL"), abs(target)
        elif target > current:
            side, size = "BUY", target - current
        else:
            side, size = "SELL", current - target
        delta = size if side == "BUY" else -size
        self.calls.append(("smart", symbol, delta))
        if self.smart_refuse:
            return {"status": "error", "message": "RMS: order rejected"}
        if size <= 0:
            return {"status": "success", "message": "No action needed. Position already matched."}
        oid = self._new(symbol, side, size, "MARKET", "open")
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


def _journal(s, event=None):
    path = s.journal_path
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if event is None or r["event"] == event]


def _advance(s, seconds):
    s.test_clock["now"] += timedelta(seconds=seconds)
    return s.test_clock["now"]


def _at(s, hour, minute, second=0):
    s.test_clock["now"] = s.test_clock["now"].replace(hour=hour, minute=minute, second=second)
    return s.test_clock["now"]


def _placed(b):
    return [c for c in b.calls if c[0] == "place"]


def _orders_sent(b):
    return sum(1 for c in b.calls if c[0] in ("place", "smart"))


def _smarts(b):
    return sum(1 for c in b.calls if c[0] == "smart")


# =============================================================================
# THE MORNING: scan, ranking, one trade, the window, the day's limits
# =============================================================================
# AAA long (reward:risk 3.75), BBB short (4.25), CCC nothing, DDD the same long as AAA.
BASKET = ["AAA", "BBB", "CCC", "DDD"]
PRICES = {"AAA": 97.0, "BBB": 103.0, "CCC": 100.0, "DDD": 97.0}


def basket_history():
    return {
        "AAA": with_today(long_bars()),
        "BBB": with_today(short_bars()),
        "CCC": with_today(flat_bars()),
        "DDD": with_today(long_bars()),
    }


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "time", FakeTime())
    made = {"n": 0}

    def make(symbols=None, now=NOW, data_dir=None, broker=None, daily=None):
        made["n"] += 1
        symbols = list(symbols or BASKET)
        b = broker or HostileBroker(symbols)
        if broker is None:
            b.daily.update(basket_history() if daily is None else daily)
            b.ltps.update(PRICES)
        clock = {"now": now}
        s = st.ThreeCandleStrategy(
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


def test_the_scan_ranks_the_setups_and_takes_only_the_best(desk):
    s, b = desk()
    s.step()
    ranked = s.state["candidates"]
    assert [c["symbol"] for c in ranked] == ["BBB", "AAA", "DDD"]  # 4.25, then the tie in order
    assert [round(c["rr"], 2) for c in ranked] == [4.25, 3.75, 3.75]
    # one trade only: the short in BBB, sized on 40000 of position value
    qty = math.floor(40000 / (103.0 * 1.001))
    assert _placed(b) == [
        ("place", "BBB", "SELL", qty, "MARKET"),
        ("place", "BBB", "BUY", qty, "SL-M"),
    ]
    assert b.pos == {"AAA": 0, "BBB": -qty, "CCC": 0, "DDD": 0}
    trade = s.state["open"]
    assert (trade["symbol"], trade["stop"], trade["target"]) == ("BBB", 105.0, 94.5)
    assert b.live("BBB")[0]["trigger"] == 105.0
    assert s.state["taken"] == ["BBB"]

    # every candidate is journalled with its numbers, its rank and what became of it
    found = _journal(s, "candidate")
    assert {r["symbol"] for r in found} == {"AAA", "BBB", "DDD"}
    assert all({"side", "ref_entry", "stop", "target", "rr"} <= set(r) for r in found)
    scan = _journal(s, "scan")
    assert len(scan) == 1 and set(scan[0]["results"]) == set(BASKET)
    assert scan[0]["results"]["CCC"].startswith("no setup")
    assert [(r["rank"], r["symbol"]) for r in scan[0]["ranking"]] == [
        (1, "BBB"),
        (2, "AAA"),
        (3, "DDD"),
    ]
    chosen = _journal(s, "signal")
    assert len(chosen) == 1 and chosen[0]["rank"] == 1 and chosen[0]["plan"]["symbol"] == "BBB"
    ignored = _journal(s, "signal_ignored")
    assert [(r["symbol"], r["side"]) for r in ignored] == [("AAA", "BUY"), ("DDD", "BUY")]
    assert all("BBB ranks higher" in r["why"] for r in ignored)
    entry = _journal(s, "entry")[0]
    assert (
        entry["position_value"] == pytest.approx(qty * 103.0) and entry["position_value"] <= 40000
    )
    assert entry["planned_loss"] == pytest.approx(qty * 2.0)

    # the rest of the day: the others are dropped with the reason, nothing more is sent
    sent = _orders_sent(b)
    for _ in range(3):
        _advance(s, 10)
        s.step()
    assert _orders_sent(b) == sent
    skipped = {r["symbol"]: r["why"] for r in _journal(s, "skip")}
    assert set(skipped) == {"AAA", "DDD"}
    assert all("limit of 1 trade is reached" in why for why in skipped.values())


def test_a_tie_follows_the_order_of_the_symbol_list(desk):
    s, b = desk(symbols=["DDD", "CCC", "AAA"])
    s.step()
    assert [c["symbol"] for c in s.state["candidates"]] == ["DDD", "AAA"]
    assert s.state["open"]["symbol"] == "DDD" and b.pos["AAA"] == 0
    s2, b2 = desk(symbols=["AAA", "CCC", "DDD"])
    s2.step()
    assert s2.state["open"]["symbol"] == "AAA" and b2.pos["DDD"] == 0


def test_a_long_is_bought_with_its_stop_below(desk):
    s, b = desk(symbols=["AAA", "CCC"])
    s.step()
    qty = math.floor(40000 / (97.0 * 1.001))
    assert _placed(b) == [
        ("place", "AAA", "BUY", qty, "MARKET"),
        ("place", "AAA", "SELL", qty, "SL-M"),
    ]
    trade = s.state["open"]
    assert (trade["side"], trade["stop"], trade["target"]) == ("BUY", 95.0, 104.5)
    assert b.live("AAA")[0]["trigger"] == 95.0


def test_entries_happen_only_inside_the_window(desk):
    s, b = desk(now=datetime(2026, 11, 4, 9, 15, 5))
    s.step()
    assert len(s.state["candidates"]) == 3  # the scan runs before the window opens
    for hour, minute, second in ((9, 16, 0), (9, 19, 0), (9, 19, 59)):
        _at(s, hour, minute, second)
        s.step()
    assert _orders_sent(b) == 0 and s.state["open"] is None
    assert "before the entry window" in s.can_enter(s.test_clock["now"])[1]
    _at(s, 9, 20, 0)
    s.step()
    assert s.state["open"]["symbol"] == "BBB"


def test_the_last_second_of_the_window_still_enters_and_the_next_does_not(desk):
    s, b = desk(now=datetime(2026, 11, 4, 9, 45, 0))
    s.step()
    assert s.state["open"]["symbol"] == "BBB"

    late, late_b = desk(now=datetime(2026, 11, 4, 9, 45, 1))
    late.step()
    assert _orders_sent(late_b) == 0 and late.state["open"] is None
    assert len(late.state["candidates"]) == 3  # still scanned and journalled
    skipped = _journal(late, "skip")
    assert {r["symbol"] for r in skipped} == {"AAA", "BBB", "DDD"}
    assert all("outside the entry window 09:20-09:45" in r["why"] for r in skipped)
    for _ in range(3):
        _advance(late, 600)
        late.step()
    assert _orders_sent(late_b) == 0


def _take_profit(s, b):
    """Move BBB's price to the short's take profit and let the strategy see it."""
    b.ltps["BBB"] = 94.0
    _advance(s, 10)
    s.step()
    assert s.state["open"] is None and b.pos["BBB"] == 0
    b.ltps["BBB"] = 103.0


def test_one_trade_a_day_by_default_even_after_it_closes_inside_the_window(desk):
    s, b = desk()
    s.step()
    _take_profit(s, b)
    assert _journal(s, "exit")[0]["reason"] == "target"
    sent = _orders_sent(b)
    for _ in range(5):
        _advance(s, 10)
        s.step()
    assert _orders_sent(b) == sent and s.state["trades"] == 1
    assert b.pos == dict.fromkeys(BASKET, 0)


def test_a_higher_daily_limit_takes_the_next_candidate_after_the_close(desk, monkeypatch):
    monkeypatch.setattr(st, "MAX_TRADES_PER_DAY", 2)
    s, b = desk()
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    _advance(s, 10)
    s.step()
    assert b.pos["AAA"] == 0  # nothing else while BBB is open
    b.ltps["BBB"] = 94.0  # the short's take profit: closed, and the next one is taken
    _advance(s, 10)
    s.step()
    assert b.pos["BBB"] == 0 and _journal(s, "exit")[0]["reason"] == "target"
    assert s.state["open"]["symbol"] == "AAA" and s.state["trades"] == 2
    assert s.state["taken"] == ["BBB", "AAA"]
    # the third is refused by the limit
    b.ltps["AAA"] = 105.0
    _advance(s, 10)
    s.step()
    assert s.state["open"] is None
    _advance(s, 10)
    s.step()
    assert b.pos["DDD"] == 0 and "limit of 2 trades" in _journal(s, "skip")[-1]["why"]


def test_the_next_candidate_is_not_taken_once_the_window_has_ended(desk, monkeypatch):
    monkeypatch.setattr(st, "MAX_TRADES_PER_DAY", 2)
    s, b = desk()
    s.step()
    _at(s, 9, 50)
    _take_profit(s, b)
    sent = _orders_sent(b)
    _advance(s, 10)
    s.step()
    assert _orders_sent(b) == sent and s.state["open"] is None
    assert "outside the entry window" in _journal(s, "skip")[-1]["why"]


def test_a_symbol_is_never_entered_twice_in_a_day(desk, monkeypatch):
    monkeypatch.setattr(st, "MAX_TRADES_PER_DAY", 5)
    s, b = desk(symbols=["BBB", "CCC"])
    s.step()
    _take_profit(s, b)  # closed, and the price is back where the setup is valid
    sent = _orders_sent(b)
    for _ in range(5):
        _advance(s, 10)
        s.step()
    assert _orders_sent(b) == sent and s.state["trades"] == 1
    # a restart works the signals out again, and still does not enter
    s2, _ = desk(symbols=["BBB", "CCC"], data_dir=s.state_path.parent, broker=b)
    s2.test_clock["now"] = s.test_clock["now"]
    for _ in range(3):
        _advance(s2, 10)
        s2.step()
    assert _orders_sent(b) == sent
    assert [(c["symbol"], c["status"]) for c in s2.state["candidates"]] == [("BBB", "taken")]


def test_a_restart_mid_day_does_not_enter_again(desk):
    s, b = desk()
    s.step()
    sent = _orders_sent(b)
    # restart with the trade still open: it is resumed, and the scan is repeated
    s2, _ = desk(data_dir=s.state_path.parent, broker=b)
    assert s2.state["open"]["symbol"] == "BBB"
    for _ in range(3):
        _advance(s2, 10)
        s2.step()
    assert _orders_sent(b) == sent
    assert [(c["symbol"], c["status"]) for c in s2.state["candidates"]] == [
        ("BBB", "taken"),
        ("AAA", "skipped"),
        ("DDD", "skipped"),
    ]
    # restart after the trade closed: the day's count is kept, nothing new is taken
    _take_profit(s2, b)
    sent = _orders_sent(b)
    s3, _ = desk(data_dir=s.state_path.parent, broker=b)
    for _ in range(3):
        _advance(s3, 10)
        s3.step()
    assert _orders_sent(b) == sent and s3.state["trades"] == 1
    assert s3.state["taken"] == ["BBB"] and b.pos == dict.fromkeys(BASKET, 0)


def test_a_restart_before_the_entry_gives_the_same_signals(desk):
    s, b = desk(now=datetime(2026, 11, 4, 9, 16, 0))
    s.step()
    first = json.loads(json.dumps(s.state["candidates"]))
    s2, _ = desk(now=datetime(2026, 11, 4, 9, 18, 0), data_dir=s.state_path.parent, broker=b)
    s2.step()
    assert s2.state["candidates"] == first and _orders_sent(b) == 0


@pytest.mark.parametrize(
    "price,reason",
    [(105.0, "at or above the stop"), (94.5, "at or below the take profit")],
)
def test_a_short_no_longer_valid_on_the_live_price_is_skipped_for_the_next(desk, price, reason):
    s, b = desk()
    b.ltps["BBB"] = price
    s.step()
    assert s.state["open"]["symbol"] == "AAA" and b.pos["BBB"] == 0
    skip = _journal(s, "skip")[0]
    assert skip["symbol"] == "BBB" and reason in skip["why"] and skip["price"] == price
    assert s.state["skipped"]["BBB"] == skip["why"]
    assert _journal(s, "signal")[0]["rank"] == 2


@pytest.mark.parametrize(
    "price,reason",
    [(95.0, "at or below the stop"), (104.5, "at or above the take profit")],
)
def test_a_long_no_longer_valid_on_the_live_price_is_skipped(desk, price, reason):
    s, b = desk(symbols=["AAA", "CCC"])
    b.ltps["AAA"] = price
    s.step()
    assert s.state["open"] is None and _orders_sent(b) == 0
    assert reason in _journal(s, "skip")[0]["why"]
    b.ltps["AAA"] = 97.0  # dropped for the day: a later recovery is not chased
    _advance(s, 10)
    s.step()
    assert _orders_sent(b) == 0


def test_square_off_closes_the_trade(desk):
    s, b = desk()
    s.step()
    _at(s, 15, 9, 59)
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    _at(s, 15, 10, 0)
    s.step()
    assert s.state["open"] is None and b.pos["BBB"] == 0 and b.live("BBB") == []
    assert _journal(s, "exit")[0]["reason"] == "square-off time"


def test_a_trade_left_from_an_earlier_day_is_closed_at_once(desk):
    s, b = desk()
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    s.test_clock["now"] = datetime(2026, 11, 5, 9, 15, 5)  # the next morning
    s.step()
    assert s.state["open"] is None and b.pos["BBB"] == 0 and b.live("BBB") == []
    assert _journal(s, "exit")[0]["reason"] == "left open from an earlier day"


def test_a_new_day_starts_with_a_clean_record(desk):
    s, b = desk()
    s.step()
    _take_profit(s, b)
    assert s.state["trades"] == 1 and s.state["taken"] == ["BBB"]
    s.test_clock["now"] = datetime(2026, 11, 5, 9, 15, 5)
    s.step()
    assert s.state["date"] == "2026-11-05" and s.scan_day == "2026-11-05"
    assert (s.state["trades"], s.state["taken"], s.state["skipped"]) == (0, [], {})
    assert s.state["pnl"] == 0.0


def test_the_daily_loss_limit_stops_new_entries(desk):
    s, b = desk()
    s.state["pnl"] = -st.MAX_DAILY_LOSS
    s.step()
    assert _orders_sent(b) == 0
    assert all("daily loss limit Rs 4000" in r["why"] for r in _journal(s, "skip"))
    assert len(_journal(s, "skip")) == 3


def test_a_side_switched_off_is_not_traded(desk, monkeypatch):
    monkeypatch.setattr(st, "ENABLE_SHORTS", False)
    s, b = desk()
    s.step()
    assert [c["symbol"] for c in s.state["candidates"]] == ["AAA", "DDD"]
    assert s.state["open"]["symbol"] == "AAA" and b.pos["BBB"] == 0
    assert "shorts are switched off" in s.scan_results["BBB"]["why"]


def test_daily_bars_that_cannot_be_read_are_tried_again_before_anything_is_entered(desk):
    s, b = desk()
    b.history_fail = {"BBB"}  # the best setup's bars are late
    s.step()
    assert s.scan_todo == ["BBB"] and [c["symbol"] for c in s.state["candidates"]] == [
        "AAA",
        "DDD",
    ]
    assert _orders_sent(b) == 0  # the ranking is not whole yet: nothing is entered
    _advance(s, 10)
    s.step()
    assert _orders_sent(b) == 0 and b.history_calls.count(("BBB", "D")) == 1  # not every cycle
    b.history_fail = set()
    _advance(s, st.SCAN_RETRY_SEC)
    s.step()
    assert s.scan_todo == [] and s.state["open"]["symbol"] == "BBB"


def test_a_symbol_whose_bars_never_arrive_does_not_hold_the_others_for_long(desk):
    s, b = desk()
    b.history_fail = {"BBB"}
    s.step()
    _advance(s, st.SCAN_GRACE_SEC + 1)
    s.step()
    assert s.state["open"]["symbol"] == "AAA" and s.scan_todo == ["BBB"]


def test_one_bad_history_frame_does_not_stop_the_other_symbols(desk):
    s, b = desk()
    b.daily["AAA"] = pd.DataFrame({"price": [1.0]})
    s.step()
    assert s.scan_results["AAA"]["result"] == "no data"
    assert s.state["open"]["symbol"] == "BBB" and s.scan_todo == []


def test_unknown_symbols_are_skipped_at_setup(desk):
    class Picky(HostileBroker):
        def symbol(self, symbol, **k):
            if symbol == "NOPE":
                return {"status": "error", "message": "Symbol not found"}
            return super().symbol(symbol, **k)

    b = Picky(["AAA", "NOPE"])
    s, _ = desk(symbols=["AAA", "NOPE"], broker=b)
    assert s.symbols == ["AAA"] and s.unchecked == []


def test_a_symbol_that_could_not_be_checked_is_retried_then_scanned(desk):
    class Flaky(HostileBroker):
        down = True

        def symbol(self, symbol, **k):
            if symbol == "BBB" and self.down:
                raise ConnectionError("connection refused")
            return super().symbol(symbol, **k)

    b = Flaky(["AAA", "BBB"])
    b.daily.update(basket_history())
    b.ltps.update(PRICES)
    s, _ = desk(symbols=["AAA", "BBB"], broker=b, now=datetime(2026, 11, 4, 9, 16, 0))
    assert s.symbols == ["AAA"] and s.unchecked == ["BBB"]
    s.step()
    assert [c["symbol"] for c in s.state["candidates"]] == ["AAA"]
    b.down = False
    _advance(s, st.CHECK_MINUTES * 60)
    s.step()
    assert s.symbols == ["AAA", "BBB"] and s.unchecked == []
    assert [c["symbol"] for c in s.state["candidates"]] == ["BBB", "AAA"]


def test_the_stop_and_target_are_never_moved(desk):
    s, b = desk()
    s.step()
    stop_id = s.state["open"]["stop_order_id"]
    for price in (102.0, 100.0, 97.0, 95.0, 99.0, 104.0):  # far in profit, then back
        b.ltps["BBB"] = price
        _advance(s, 301)  # past a position check each time
        s.step()
        trade = s.state["open"]
        assert (trade["stop"], trade["target"], trade["stop_order_id"]) == (105.0, 94.5, stop_id)
    assert [o["trigger"] for o in b.live("BBB")] == [105.0]
    assert [c for c in b.calls if c[0] in ("cancel", "smart")] == []
    assert not hasattr(HostileBroker, "modifyorder")  # nothing to modify an order with


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
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="MIS",
        quantity=1,
    )
    if error_type:
        assert raw["status"] == "error" and raw["error_type"] == error_type
        assert raw.get("code") == code
    s = st.ThreeCandleStrategy(client, ["SBIN"], clock=lambda: NOW, data_dir=tmp_path)
    for method in ("placeorder", "placesmartorder"):
        kwargs = {"strategy": "t", "symbol": "SBIN", "action": "SELL", "exchange": "NSE"}
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

    s = st.ThreeCandleStrategy(Client(), ["X"], clock=lambda: NOW, data_dir=tmp_path)
    assert s.place("X", "BUY", 10, "SL-M")[0] == expected


# =============================================================================
# ORDER SAFETY, LONG AND SHORT: every test below runs once for each side
# =============================================================================
SYMS = ["AAA", "BBB", "CCC"]
H_NOW = datetime(2026, 11, 4, 10, 0, 6)


class Side:
    """One side of the market for the order-safety tests: entry 100, stop 2 away."""

    def __init__(self, action):
        long = action == "BUY"
        self.action = action
        self.close = "SELL" if long else "BUY"
        self.sign = 1 if long else -1
        self.stop, self.target = (98.0, 110.0) if long else (102.0, 90.0)
        self.through_stop = 97.0 if long else 103.0
        self.short_of_stop = 99.0 if long else 101.0
        self.in_profit = 106.0 if long else 94.0
        self.past_target = 111.0 if long else 89.0

    def candidate(self, symbol):
        return {
            "symbol": symbol,
            "side": self.action,
            "ref_entry": 100.0,
            "stop": self.stop,
            "target": self.target,
            "rr": 5.0,
            "candle2": YESTERDAY,
            "status": "waiting",
            "why": "",
        }

    def plan(self, symbol, setup, price, lot=1):
        return st.Plan(
            symbol=symbol,
            side=self.action,
            entry=price,
            stop=self.stop,
            target=self.target,
            qty=10,
            risk=2.0,
            ref_entry=100.0,
            rr=5.0,
            sized_by="position value",
            signal_bar=YESTERDAY,
            planned_loss=20.0,
            position_value=1000.0,
        ), ""


@pytest.fixture(params=["BUY", "SELL"], ids=["long", "short"])
def hostile(request, tmp_path, monkeypatch):
    side = Side(request.param)
    fake_time = FakeTime()
    monkeypatch.setattr(st, "time", fake_time)
    monkeypatch.setattr(st, "plan_entry", side.plan)
    # several entries a day and a long window, so that whenever an entry is
    # refused below it is the safety layer refusing it, not the day's limits
    monkeypatch.setattr(st, "MAX_TRADES_PER_DAY", 5)
    monkeypatch.setattr(st, "ENTRY_END", "14:45")
    made = {"n": 0}

    def make(data_dir=None, broker=None):
        made["n"] += 1
        b = broker or HostileBroker(SYMS)
        clock = {"now": H_NOW}
        s = st.ThreeCandleStrategy(
            b,
            symbols=SYMS,
            clock=lambda: clock["now"],
            data_dir=data_dir or tmp_path / f"run{made['n']}",
            pause=0,
        )
        s.test_clock = clock
        s.side = side
        s.setup()
        s.begin_day(H_NOW.date().isoformat())  # the morning scan found nothing
        s.scan_todo, s.scan_reported = [], True
        s.last_check = H_NOW.replace(second=0)
        return s, b

    make.time = fake_time
    make.side = side
    return make


def _signal(s, *symbols):
    """A setup appears in each symbol; the strategy checks positions and tries to
    enter. Whatever is not entered is forgotten, as a blocked signal is."""
    for symbol in symbols:
        kept = [c for c in s.state["candidates"] if c["symbol"] != symbol]
        s.state["candidates"] = kept + [s.side.candidate(symbol)]
    s.periodic()
    s.try_enter(s.test_clock["now"])
    s.state["candidates"] = [c for c in s.state["candidates"] if c["status"] != "waiting"]


def _check(s):
    """The position check that runs every CHECK_MINUTES."""
    s.periodic()
    s.try_enter(s.test_clock["now"])


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


def _exit_order(b, side):
    return next(
        k for k, o in b.orders.items() if o["pricetype"] == "MARKET" and o["action"] == side.close
    )


def test_entry_places_the_position_and_one_stop_on_the_closing_side(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    assert b.pos["AAA"] == 10 * side.sign
    assert _placed(b) == [
        ("place", "AAA", side.action, 10, "MARKET"),
        ("place", "AAA", side.close, 10, "SL-M"),
    ]
    assert b.live("AAA")[0]["trigger"] == side.stop
    assert s.state["open"]["target"] == side.target and s.state["taken"] == ["AAA"]


def test_only_one_trade_across_all_symbols(hostile):
    s, b = hostile()
    _signal(s, "AAA", "BBB")
    assert s.state["open"]["symbol"] == "AAA" and b.pos["BBB"] == 0
    ignored = _journal(s, "signal_ignored")
    assert [r["symbol"] for r in ignored] == ["BBB"] and "AAA ranks higher" in ignored[0]["why"]
    _signal(s, "CCC")
    assert b.pos["CCC"] == 0
    assert "AAA trade is already open" in s.can_enter(s.test_clock["now"])[1]


def test_a_pending_entry_blocks_every_other_symbol(hostile):
    s, _b = hostile()
    s.state["pending"] = {"order_id": None, "plan": {"symbol": "AAA"}, "placed": "x"}
    ok, why = s.can_enter(s.test_clock["now"])
    assert not ok and "AAA entry is being confirmed" in why


def test_the_target_is_taken_at_market_after_the_stop_is_cancelled(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    stop_id = s.state["open"]["stop_order_id"]
    b.ltp_value = side.in_profit  # not there yet
    s.manage_open(_advance(s, 10))
    assert s.state["open"] and _smarts(b) == 0
    b.ltp_value = side.target  # touched exactly
    s.manage_open(_advance(s, 10))
    assert s.state["open"] is None and b.pos["AAA"] == 0 and b.live("AAA") == []
    kinds = [c[0] for c in b.calls]
    assert b.calls.index(("cancel", stop_id)) < kinds.index("smart")
    assert b.calls[kinds.index("smart")] == ("smart", "AAA", -10 * side.sign)
    exits = _journal(s, "exit")
    assert len(exits) == 1 and exits[0]["reason"] == "target"
    assert exits[0]["pnl"] == pytest.approx(100.0)  # 10 shares x 10


def test_a_fill_at_or_beyond_the_stop_is_closed_at_once(hostile):
    s, b = hostile()
    side = hostile.side
    plan, _ = side.plan("AAA", None, 100.0)
    b.pos["AAA"] = 10 * side.sign
    s.on_filled(plan, 10, side.stop)
    assert s.state["open"] is None and b.pos["AAA"] == 0
    assert not any(c[0] == "place" and c[4] in ("SL", "SL-M") for c in b.calls)


def test_unreadable_position_after_the_entry_fill_still_manages_it(hostile):
    s, b = hostile()
    side = hostile.side
    real = b.orderstatus

    def status_then_fail(order_id, **k):
        r = real(order_id, **k)
        if order_id == "O1" and r["data"]["order_status"] == "complete":
            b.openpos_fail = 1
        return r

    b.orderstatus = status_then_fail
    _signal(s, "AAA")
    b.orderstatus = real
    assert s.state["open"]["symbol"] == "AAA" and s.state["open"]["qty"] == 10
    assert s.state["open"]["side"] == side.action and len(b.live("AAA")) == 1
    _signal(s, "BBB")
    assert b.pos == {"AAA": 10 * side.sign, "BBB": 0, "CCC": 0}


def test_a_slow_exit_is_cancelled_before_it_is_retried(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.hold_market = True
    s.exit_trade("target")
    first_exit = s.state["open"]["exit_order_id"]
    assert first_exit and b.pos["AAA"] == 10 * side.sign
    s.manage_open(s.test_clock["now"])  # inside its backoff the working exit is left alone
    assert b.orders[first_exit]["order_status"] == "open"
    assert ("cancel", first_exit) not in b.calls
    s.manage_open(_advance(s, st.EXIT_RETRY_SEC + 1))
    assert b.orders[first_exit]["order_status"] == "cancelled"
    assert len(b.live("AAA", ("MARKET",))) == 1  # never two exit orders at once
    b.release()
    s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_no_exit_order_is_sent_while_the_stop_cannot_be_cancelled(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.cancel_ok = False  # the broker keeps the stop working
    for _ in range(3):
        assert s.exit_trade("target") is False
    # closing now as well could leave the opposite position once the stop fills
    assert _smarts(b) == 0 and b.pos["AAA"] == 10 * side.sign
    assert len(b.live("AAA")) == 1 and s.state["open"]["exit_retry"] == "target"
    b.cancel_ok = True
    s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None and b.live("AAA") == []


def test_unreadable_position_after_the_exit_order_is_not_booked_closed(hostile):
    s, b = hostile()
    _signal(s, "AAA")
    b.hold_market = True
    real = b.placesmartorder

    def smart_then_fail(**k):
        r = real(**k)
        b.openpos_fail = 1
        return r

    b.placesmartorder = smart_then_fail
    s.exit_trade("target")
    assert s.state["open"]["symbol"] == "AAA"
    b.placesmartorder = real
    b.hold_market = False
    _signal(s, "BBB")
    assert b.pos["BBB"] == 0


@pytest.mark.parametrize("books", [True, False])
def test_a_lost_stop_answer_never_leads_to_two_stops(hostile, books):
    s, b = hostile()
    b.lost_answer = {"SL-M"}
    b.orderbook_fail = not books
    _signal(s, "AAA")
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
    s, b = hostile()
    b.lost_answer = {"MARKET"}
    _signal(s, "AAA")
    b.lost_answer = set()
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1
    assert b.live("AAA")[0]["action"] == hostile.side.close
    _signal(s, "BBB")
    assert b.pos["BBB"] == 0


def test_a_lost_entry_answer_still_working_is_cancelled_not_doubled(hostile):
    s, b = hostile()
    side = hostile.side
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    _signal(s, "AAA")
    b.lost_answer = set()
    b.hold_market = False
    assert s.state["pending"] is not None
    assert b.orders["O1"]["order_status"] == "cancelled"
    s.step()
    assert s.state["pending"] is not None  # inside the grace period nothing is concluded
    _advance(s, st.ENTRY_CONFIRM_SEC + 1)
    s.step()
    assert s.state["pending"] is None and s.state["open"] is None and b.pos["AAA"] == 0
    assert _placed(b) == [("place", "AAA", side.action, 10, "MARKET")]
    # and the symbol is not tried again that day, though its setup is still there
    _signal(s, "AAA")
    assert _placed(b) == [("place", "AAA", side.action, 10, "MARKET")]


def test_a_lost_entry_answer_leaves_the_users_own_order_alone(hostile):
    s, b = hostile()
    manual = b._new("AAA", hostile.side.action, 3, "LIMIT", "open")
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    _signal(s, "AAA")
    assert b.orders[manual]["order_status"] == "open"
    ours = [o for o in b.orders.values() if o["pricetype"] == "MARKET"]
    assert [o["order_status"] for o in ours] == ["cancelled"]
    assert s.state["pending"]


def test_a_lost_entry_answer_waits_before_concluding_anything(hostile):
    s, b = hostile()
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    real_place = _hide(b, "MARKET")
    _signal(s, "AAA")
    b.placeorder = real_place
    b.lost_answer = set()
    assert s.state["pending"]
    s.step()
    assert s.state["pending"]
    b.hidden.clear()
    b.release()  # it filled after all
    s.step()
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1


def test_the_wait_after_a_lost_entry_answer_reads_the_position_late(hostile):
    s, b = hostile()
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    hostile.time.hooks.append(b.release)
    _signal(s, "AAA")
    hostile.time.hooks.clear()
    assert s.state["open"]["symbol"] == "AAA"
    assert [c for c in b.calls if c[0] == "cancel"] == []


def test_a_lost_exit_answer_is_resolved_from_the_books(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.smart_lost = True
    b.ltp_value = side.in_profit
    s.exit_trade("target")  # the exit filled but the answer was lost
    b.smart_lost = False
    assert b.pos["AAA"] == 0 and s.state["open"]["symbol"] == "AAA"
    b.ltp_value = 100.0
    s.manage_open(s.test_clock["now"])
    assert s.state["open"] is None and _smarts(b) == 1


def test_a_lost_exit_that_fills_late_is_not_doubled(hostile):
    s, b = hostile()
    _signal(s, "AAA")
    b.hold_market = True
    b.smart_lost = True
    s.exit_trade("target")
    b.smart_lost = False
    hostile.time.hooks.append(b.release)
    s.manage_open(s.test_clock["now"])
    hostile.time.hooks.clear()
    for _ in range(3):
        if s.state["open"]:
            s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_a_lost_exit_not_yet_in_the_book_is_waited_for(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.hold_market = True
    b.smart_lost = True
    s.exit_trade("target")
    b.smart_lost = False
    b.hidden.add(_exit_order(b, side))
    for _ in range(3):
        s.manage_open(s.test_clock["now"])
    assert [c for c in b.calls if c[0] == "smart"] == [("smart", "AAA", -10 * side.sign)]
    b.hidden.clear()
    b.release()
    s.manage_open(s.test_clock["now"])
    assert s.state["open"] is None and s.state["sweep"]


def test_a_lost_exit_answer_is_resent_only_after_the_grace_period(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.smart_lost = True
    b.hold_market = True
    s.exit_trade("target")
    b.smart_lost = False
    b.orders[_exit_order(b, side)]["order_status"] = "rejected"
    b.hold_market = False
    s.manage_open(_advance(s, st.EXIT_CONFIRM_SEC - 5))
    assert _smarts(b) == 1
    s.manage_open(_advance(s, 10))
    assert _smarts(b) == 2 and b.pos["AAA"] == 0 and s.state["open"] is None


def test_a_lost_stop_answer_that_shows_up_late_is_adopted_not_doubled(hostile):
    s, b = hostile()
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    _signal(s, "AAA")
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
    s, b = hostile()
    b.lost_answer = {"SL-M"}
    _signal(s, "AAA")
    b.lost_answer = set()
    b.orders["O2"]["order_status"] = "rejected"
    s.manage_open(_advance(s, st.STOP_CONFIRM_SEC - 5))
    assert b.live("AAA") == []
    s.manage_open(_advance(s, 10))
    assert len(b.live("AAA")) == 1 and s.state["open"]["stop_order_id"]
    assert b.live("AAA")[0]["trigger"] == hostile.side.stop  # the same price as before


def test_a_clear_stop_refusal_sends_the_sl_fallback(hostile):
    s, b = hostile()
    b.refuse = {"SL-M": {"status": "error", "message": "SL-M orders are not allowed"}}
    _signal(s, "AAA")
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL"]
    assert s.state["open"]["stop_type"] == "SL"
    assert b.live("AAA")[0]["trigger"] == hostile.side.stop


def test_unreadable_state_with_a_working_stop_closes_cleanly(hostile, tmp_path):
    d = tmp_path / "r5"
    s, b = hostile(data_dir=d)
    _signal(s, "AAA")
    s.state_path.write_text(s.state_path.read_text()[:30])
    s2, _ = hostile(data_dir=d, broker=b)
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
    b = HostileBroker(SYMS)
    b.pos.update(AAA=10, BBB=-5)
    s, _ = hostile(data_dir=d, broker=b)
    assert s.state["open"]["symbol"] == "AAA" and s.untracked == {"BBB"}
    for _ in range(3):
        s.step()
    assert b.pos == {"AAA": 0, "BBB": 0, "CCC": 0}


def test_restarts_mid_order(hostile, tmp_path):
    side = hostile.side
    d = tmp_path / "a"
    s, b = hostile(data_dir=d)
    b.hold_market = True
    _signal(s, "AAA")
    b.release(4)
    s2, _ = hostile(data_dir=d, broker=b)
    assert s2.state["open"]["qty"] == 4 and [o["qty"] for o in b.live("AAA")] == [4]
    assert s2.state["open"]["target"] == side.target  # the fixed levels survive the restart
    assert s2.state["open"]["stop"] == side.stop and s2.state["taken"] == ["AAA"]

    d = tmp_path / "b"
    s, b = hostile(data_dir=d)
    _signal(s, "AAA")
    b.hold_market = True
    b.smart_lost = True
    s.exit_trade("target")
    b.smart_lost = False
    s2, _ = hostile(data_dir=d, broker=b)
    s2.step()
    b.release()
    for _ in range(2):
        if s2.state["open"]:
            s2.step()
    assert b.pos["AAA"] == 0 and s2.state["open"] is None

    d = tmp_path / "c"
    s, b = hostile(data_dir=d)
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    b.orderbook_fail = True
    _signal(s, "AAA")
    b.lost_answer = set()
    b.orderbook_fail = False
    b.release()
    s2, _ = hostile(data_dir=d, broker=b)
    assert s2.state["open"] and len(b.live("AAA")) == 1


def test_partial_fills(hostile):
    s, b = hostile()
    b.hold_market = True
    _signal(s, "AAA")
    b.release(6)
    _advance(s, 70)
    s.step()
    assert s.state["open"]["qty"] == 6 and [o["qty"] for o in b.live("AAA")] == [6]
    s.exit_trade("target")
    b.release(2)
    for _ in range(4):
        if s.state["open"]:
            s.manage_open(s.test_clock["now"])
            b.release()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_the_stop_follows_the_position_size(hostile):
    s, b = hostile()
    b.status_qty_override = 7
    real = b.orderstatus

    def fail_position_after_fill(order_id, **k):
        r = real(order_id, **k)
        if r.get("data", {}).get("order_status") == "complete" and order_id == "O1":
            b.openpos_fail = 1
        return r

    b.orderstatus = fail_position_after_fill
    _signal(s, "AAA")
    b.orderstatus = real
    assert [o["qty"] for o in b.live("AAA")] == [7]  # the broker's word, position unread
    _check(s)  # the next position check corrects it
    assert [o["qty"] for o in b.live("AAA")] == [10] and s.state["open"]["qty"] == 10
    b.trigger_stops()
    assert b.pos["AAA"] == 0


def test_the_stop_quantity_is_resized_when_the_position_grows_its_price_is_not(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.pos["AAA"] = 13 * side.sign
    _check(s)
    assert [o["qty"] for o in b.live("AAA")] == [13] and s.state["open"]["qty"] == 13
    assert [o["trigger"] for o in b.live("AAA")] == [side.stop]
    assert (s.state["open"]["stop"], s.state["open"]["target"]) == (side.stop, side.target)


def test_a_stop_smaller_than_the_position_leaves_a_managed_residual(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.pos["AAA"] += 3 * side.sign
    b.trigger_stops()
    s.step()
    assert s.state["open"]["residual"] and s.state["open"]["qty"] == 3
    _signal(s, "BBB")
    assert b.pos["BBB"] == 0
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_two_stops_filling_never_leaves_an_unmanaged_reversal(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b._new("AAA", side.close, 10, "SL", "trigger pending")
    b.trigger_stops()
    assert b.pos["AAA"] == -10 * side.sign
    s.step()
    trade = s.state["open"]
    assert trade["residual"] and trade["side"] == side.close and trade["qty"] == 10
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_a_fill_the_position_book_shows_late_is_not_forgotten(hostile):
    s, b = hostile()
    b.stale = dict.fromkeys(SYMS, 0)
    real = b.orderstatus

    def lag(order_id, **k):
        r = real(order_id, **k)
        if r.get("data", {}).get("order_status") == "complete" and order_id == "O1":
            b.openpos_lag = 1
        return r

    b.orderstatus = lag
    _signal(s, "AAA")
    b.orderstatus = real
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1


def test_a_complete_entry_with_a_flat_position_keeps_blocking(hostile):
    s, b = hostile()
    b.stale = dict.fromkeys(SYMS, 0)
    real = b.orderstatus

    def lag(order_id, **k):
        r = real(order_id, **k)
        if order_id == "O1":
            b.openpos_lag = 10
        return r

    b.orderstatus = lag
    _signal(s, "AAA")
    assert s.state["pending"] and s.state["pending"].get("alerted")
    b.orderstatus = real
    b.openpos_lag = 0
    assert "AAA entry is being confirmed" in s.can_enter(s.test_clock["now"])[1]
    s.step()
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1


def test_an_exit_that_never_fills_backs_off_and_alerts(hostile):
    s, b = hostile()
    _signal(s, "AAA")
    b.hold_market = True
    sent = _orders_sent(b)
    s.exit_trade("target")
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
    s, b = hostile()
    _signal(s, "AAA")
    b.smart_refuse = True
    s.exit_trade("target")
    for _ in range(10):
        s.manage_open(s.test_clock["now"])
    assert _smarts(b) == 1
    b.smart_refuse = False
    s.manage_open(_advance(s, st.EXIT_RETRY_SEC + 1))
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_stop_filled_with_positions_unreadable_books_it_and_blocks(hostile):
    s, b = hostile()
    _signal(s, "AAA")
    b.trigger_stops()
    b.openpos_fail = 10**6
    sent = _orders_sent(b)
    for _ in range(5):
        s.step()
    assert s.state["open"] is None and _orders_sent(b) == sent
    assert "AAA" in s.untracked
    assert s.can_enter(s.test_clock["now"])[0] is False


def test_a_stop_is_booked_at_its_fill_price_not_the_trigger(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.stop_fill = 100.0 - 3.5 * side.sign  # 1.5 through the stop
    b.trigger_stops()
    s.step()
    assert s.state["open"] is None and s.state["pnl"] == pytest.approx(-35.0)
    assert _journal(s, "exit")[0]["reason"] == "stop loss"


def test_positions_not_managed_here_block_entries(hostile):
    s, b = hostile()
    assert s.can_enter(s.test_clock["now"]) == (True, "")
    b.pos["CCC"] = 4
    _signal(s, "BBB")
    assert s.untracked == {"CCC"} and b.pos["BBB"] == 0
    ok, why = s.can_enter(s.test_clock["now"])
    assert not ok and "CCC" in why
    b.pos["CCC"] = 0
    _check(s)
    assert s.untracked == set()


def test_step_survives_missing_quotes_and_history(hostile):
    s, b = hostile()
    s.scan_day = None  # so the morning scan runs again
    b.quotes = lambda **k: None
    b.history = lambda **k: {"status": "error", "message": "boom"}
    for _ in range(3):
        s.step()
    assert s.scan_todo == SYMS and s.state["open"] is None


# ---------- SDK error dicts returned for entry, stop and exit ----------
@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_entry_answer_returned_as_an_sdk_error_is_confirmed_not_dropped(hostile, reply):
    s, b = hostile()
    b.lost_answer = {"MARKET"}
    b.lost_reply = reply
    _signal(s, "AAA")
    b.lost_answer = set()
    # the order filled at the broker: taken over with its stop, never sent twice
    assert s.state["open"]["symbol"] == "AAA" and len(b.live("AAA")) == 1
    assert [c for c in b.calls if c[0] == "place" and c[4] == "MARKET"] == [
        ("place", "AAA", hostile.side.action, 10, "MARKET")
    ]


@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_entry_answer_returned_as_an_sdk_error_still_working_is_cancelled(hostile, reply):
    s, b = hostile()
    b.lost_answer = {"MARKET"}
    b.lost_reply = reply
    b.hold_market = True
    _signal(s, "AAA")
    assert s.state["pending"] and b.orders["O1"]["order_status"] == "cancelled"
    b.lost_answer = set()
    b.hold_market = False
    _signal(s, "BBB")
    assert b.pos["BBB"] == 0  # every symbol blocked while it is confirmed


@pytest.mark.parametrize("reply", [SDK_CONNECT, SDK_400, SDK_API], ids=["connect", "400", "api"])
def test_a_refused_entry_is_dropped(hostile, reply):
    s, b = hostile()
    b.refuse = {"MARKET": reply}
    _signal(s, "AAA")
    assert s.state["pending"] is None and s.state["open"] is None
    assert s.can_enter(s.test_clock["now"]) == (True, "")
    assert s.state["trades"] == 0 and s.state["taken"] == ["AAA"]


@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_stop_answer_returned_as_an_sdk_error_sends_no_fallback(hostile, reply):
    s, b = hostile()
    b.lost_answer = {"SL-M"}
    b.lost_reply = reply
    _signal(s, "AAA")
    b.lost_answer = set()
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL-M"]  # no SL sent on top
    assert s.state["open"]["stop_check"] and s.state["open"]["stop_unknown_since"]
    s.manage_open(s.test_clock["now"])
    assert s.state["open"]["stop_order_id"] == "O2"  # the stop that exists is adopted


@pytest.mark.parametrize("reply", [SDK_CONNECT, SDK_400], ids=["connect", "400"])
def test_a_stop_refused_by_an_sdk_error_sends_the_sl_fallback(hostile, reply):
    s, b = hostile()
    b.refuse = {"SL-M": reply}
    _signal(s, "AAA")
    assert [o["pricetype"] for o in b.live("AAA")] == ["SL"]


@pytest.mark.parametrize("reply", [SDK_TIMEOUT, SDK_504], ids=["timeout", "504"])
def test_exit_answer_returned_as_an_sdk_error_is_not_doubled(hostile, reply):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    b.smart_lost = True
    b.lost_reply = reply
    b.hold_market = True
    s.exit_trade("target")
    b.smart_lost = False
    assert s.state["open"]["exit_unanswered"]
    b.hidden.add(_exit_order(b, side))  # and it does not show in the books yet
    for _ in range(3):
        s.manage_open(_advance(s, 5))
    assert _smarts(b) == 1
    b.hidden.clear()
    b.release()
    s.manage_open(s.test_clock["now"])
    assert b.pos["AAA"] == 0 and s.state["open"] is None and _smarts(b) == 1
    assert s.state["sweep"]


# ---------- positions not managed here ----------
def test_the_position_check_notifies_about_an_unmanaged_position_once(hostile, monkeypatch):
    monkeypatch.setattr(st, "TELEGRAM_USER", "trader")
    s, b = hostile()
    alerts = _alerts(s)
    b.pos["CCC"] = 4
    _check(s)
    assert [a for a in alerts if "CCC" in a and "not managing" in a]
    assert any("CCC" in m for _u, m in b.telegrams)
    count = len(alerts)
    _check(s)
    assert len(alerts) == count  # reported once, not every check
    assert b.pos["CCC"] == 4 and not [c for c in b.calls if c[1:2] == ("CCC",)]


def test_a_position_found_at_start_is_notified(hostile, monkeypatch):
    monkeypatch.setattr(st, "TELEGRAM_USER", "trader")
    b = HostileBroker(SYMS)
    b.pos["BBB"] = 7
    s, _ = hostile(broker=b)
    assert s.untracked == {"BBB"} and any("BBB" in m for _u, m in b.telegrams)


# ---------- a lost stop answer and the price through the stop ----------
def _middle_path_close(hostile):
    """A lost stop answer, the stop hidden from the book, and the price through
    the stop inside the grace period: closed at market at once."""
    s, b = hostile()
    side = hostile.side
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    _signal(s, "AAA")
    b.placeorder = real_place
    b.lost_answer = set()
    assert s.state["open"]["stop_check"] and b.pos["AAA"] == 10 * side.sign
    b.ltp_value = side.through_stop
    s.manage_open(_advance(s, 5))
    return s, b


def test_middle_path_a_cross_inside_the_stop_grace_exits_at_once(hostile):
    s, b = _middle_path_close(hostile)
    assert b.pos["AAA"] == 0 and s.state["open"] is None and _smarts(b) == 1
    assert s.state["sweep"]["symbol"] == "AAA"  # the sweep runs after it
    assert s.state["sweep"]["action"] == hostile.side.close
    assert s._since(s.state["sweep"]["until"]) == pytest.approx(-(st.SWEEP_SEC), abs=1)
    assert s.can_enter(s.test_clock["now"])[0] is False


def test_middle_path_the_late_stop_is_cancelled_by_the_sweep(hostile):
    s, b = _middle_path_close(hostile)
    alerts = _alerts(s)
    b.hidden.clear()  # the lost stop shows up
    s.step()
    assert b.live("AAA") == [] and b.orders["O2"]["order_status"] == "cancelled"
    assert any("cancelling a closing order" in a for a in alerts)
    assert b.pos["AAA"] == 0


def test_middle_path_a_late_stop_that_fills_is_taken_over_and_closed(hostile):
    s, b = _middle_path_close(hostile)
    alerts = _alerts(s)
    b.hidden.clear()
    b.trigger_stops()  # the lost stop filled after the close: the opposite position
    assert b.pos["AAA"] == -10 * hostile.side.sign
    s.step()
    s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None
    assert any("appeared after the trade was closed" in a for a in alerts)


@pytest.mark.parametrize("fault", ["book", "position"])
def test_middle_path_waits_when_the_book_or_position_cannot_be_read(hostile, fault):
    s, b = hostile()
    side = hostile.side
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    _signal(s, "AAA")
    b.placeorder = real_place
    b.lost_answer = set()
    if fault == "book":
        b.orderbook_fail = True
    else:
        b.openpos_fail = 10**6
    b.ltp_value = side.through_stop
    for _ in range(3):
        s.manage_open(_advance(s, 5))
    assert _smarts(b) == 0 and b.pos["AAA"] == 10 * side.sign and s.state["open"]
    b.orderbook_fail, b.openpos_fail = False, 0
    s.manage_open(_advance(s, 5))
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_middle_path_a_visible_stop_is_left_to_work(hostile):
    s, b = hostile()
    b.lost_answer = {"SL-M"}
    _signal(s, "AAA")
    b.lost_answer = set()
    b.ltp_value = hostile.side.through_stop
    s.manage_open(_advance(s, 5))  # the stop shows in the book: adopted, nothing sent
    assert _smarts(b) == 0 and s.state["open"]["stop_order_id"] == "O2"
    b.trigger_stops()
    s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


def test_no_cross_inside_the_stop_grace_sends_nothing(hostile):
    s, b = hostile()
    side = hostile.side
    b.lost_answer = {"SL-M"}
    real_place = _hide(b, "SL-M")
    _signal(s, "AAA")
    b.placeorder = real_place
    b.lost_answer = set()
    sent = _orders_sent(b)
    b.ltp_value = side.short_of_stop
    for _ in range(3):
        s.manage_open(_advance(s, 5))
    assert _orders_sent(b) == sent and b.pos["AAA"] == 10 * side.sign


# ---------- positions that appear during and after the sweep ----------
def test_a_same_side_position_appearing_during_the_sweep_is_notified_and_blocks(hostile):
    s, b = _middle_path_close(hostile)
    alerts = _alerts(s)
    b.pos["AAA"] = 5 * hostile.side.sign  # opened by hand during the sweep: not a reversal
    s.step()
    assert "AAA" in s.untracked and b.pos["AAA"] == 5 * hostile.side.sign
    assert s.state["open"] is None
    assert any("AAA" in a and "not managing" in a for a in alerts)


def test_an_opposite_position_after_the_sweep_is_taken_over_and_closed(hostile):
    s, b = _middle_path_close(hostile)
    side = hostile.side
    s.step()
    _advance(s, st.SWEEP_SEC + 1)
    s.step()
    assert s.state["sweep"] is None
    alerts = _alerts(s)
    b.hidden.clear()
    assert b.orders["O2"]["order_status"] == "trigger pending"  # unseen by the sweep
    b.trigger_stops()  # the lost stop fills only now, after the sweep window
    assert b.pos["AAA"] == -10 * side.sign
    _check(s)  # the position check finds it
    assert s.state["open"]["recovered"] and s.state["open"]["side"] == side.close
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None
    assert any("appeared after the trade was closed" in a for a in alerts)


def test_an_opposite_position_waits_for_the_open_trade_then_is_closed(hostile):
    s, b = _middle_path_close(hostile)
    side = hostile.side
    _advance(s, st.SWEEP_SEC + 1)
    s.step()
    assert s.state["sweep"] is None
    b.ltp_value = 100.0
    _signal(s, "BBB")
    assert s.state["open"]["symbol"] == "BBB"
    alerts = _alerts(s)
    b.hidden.clear()
    b._fill(b.orders["O2"])  # only the lost AAA stop fills, now
    assert b.pos["AAA"] == -10 * side.sign
    _check(s)
    assert "AAA" in s.untracked and s.state["open"]["symbol"] == "BBB"  # one slot: it waits
    assert any("AAA" in a and "not managing" in a for a in alerts)
    s.exit_trade("target")
    assert s.state["open"] is None and b.pos["BBB"] == 0
    _check(s)
    assert s.state["open"]["symbol"] == "AAA" and s.state["open"]["recovered"]
    for _ in range(2):
        s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None


# =============================================================================
# END TO END THROUGH THE REAL BACKTEST RUNNER
# =============================================================================
# A random walk almost never gives a valid 5% take profit, so the daily candles
# are shaped: a steady trend (for the Supertrend's side), and at chosen days a
# Candle 1 / Candle 2 pair that forms the setup, followed by a Candle 3 whose
# intraday path decides how the trade ends. Prices are multiples of the price
# before Candle 1 (P) and, for Candle 3, of its reference entry (E).
#
#   big    Candle 1 range 11%: valid at TP 5%
#   small  Candle 1 range 4.5%: the take profit is beyond the entry only when
#          TP_PCT is lowered
LONG_SHAPES = {
    "big": ((1.0, 1.08, 0.97, 0.99), (0.99, 1.0, 0.975, 0.98)),
    "small": ((1.0, 1.03, 0.985, 0.995), (0.995, 1.0, 0.986, 0.99)),
}
# Candle 3: (open, high, low, close, whether the high comes before the low)
LONG_DAYS = {
    "target": (1.0, 1.06, 0.998, 1.0, False),  # dips a little, then runs through the target
    "stop": (1.0, 1.002, 0.98, 0.985, True),  # falls through Candle 2's low
    "flat": (1.0, 1.004, 0.997, 1.001, True),  # neither: closed at the square-off
    "gap": (0.99, 0.992, 0.98, 0.985, True),  # opens beyond the stop: never entered
}


def mirror(bar):
    """The same candle upside down, for the short side."""
    o, h, lo, c, *rest = bar
    return (2 - o, 2 - lo, 2 - h, 2 - c, *[not flag for flag in rest])


def weekdays(start, end):
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def build_symbol(n_days, up, base, seed, episodes):
    """Daily bars (open, high, low, close, high first) for one symbol.

    `episodes` maps the index of a Candle 3 day to (shape, outcome). An up
    symbol trends up (Supertrend bullish) and gets short setups; a down symbol
    gets long ones.
    """
    rng = np.random.default_rng(seed)
    starts = {i - 2: spec for i, spec in episodes.items()}
    out, price, i = [], float(base), 0
    drift = 0.003 if up else -0.003
    while i < n_days:
        if i in starts:
            shape, outcome = starts[i]
            c1, c2 = LONG_SHAPES[shape]
            c3 = LONG_DAYS[outcome]
            if up:
                c1, c2, c3 = mirror(c1), mirror(c2), mirror(c3)
            out.append((*(price * f for f in c1), not up))
            out.append((*(price * f for f in c2), not up))
            entry = price * c2[3]
            out.append((*(entry * f for f in c3[:4]), c3[4]))
            price = entry * c3[3]
            i += 3
        else:
            o = price
            c = o * (1 + drift + rng.normal(0, 0.002))
            out.append((o, max(o, c) * 1.003, min(o, c) * 0.997, c, c < o))
            price = c
            i += 1
    return out[:n_days]


def day_minutes(o, h, lo, c, high_first, n=375):
    """One-minute bars for a day that opens at o, touches h and lo, and closes at c."""
    first, second = (h, lo) if high_first else (lo, h)
    path = np.interp(np.arange(n + 1), [0, 120, 260, n], [o, first, second, c])
    opens, closes = path[:-1], path[1:]
    return opens, np.maximum(opens, closes), np.minimum(opens, closes), closes


def write_symbol(directory, symbol, days, bars, minutes_from):
    """The daily file for every day, and one-minute bars from `minutes_from` on.
    The one-minute bars add up to exactly the daily bar of their day."""
    directory.mkdir(parents=True, exist_ok=True)
    session = session_for("NSE")
    daily = pd.DataFrame(
        {
            "day": [date_to_day(d) for d in days],
            "open": [b[0] for b in bars],
            "high": [b[1] for b in bars],
            "low": [b[2] for b in bars],
            "close": [b[3] for b in bars],
            "volume": 100000,
            "oi": 0,
        }
    )
    daily.to_pickle(directory / (f"NSE_{symbol}_D".upper() + ".pkl"))
    rows = []
    for d, bar in list(zip(days, bars, strict=True))[minutes_from:]:
        t0 = session.open_epoch(date_to_day(d))
        o, h, lo, c = day_minutes(*bar)
        rows.extend((t0 + 60 * i, o[i], h[i], lo[i], c[i], 500, 0) for i in range(len(o)))
    columns = ["epoch", "open", "high", "low", "close", "volume", "oi"]
    pd.DataFrame(rows, columns=columns).to_pickle(directory / (f"NSE_{symbol}_1m".upper() + ".pkl"))


def make_market(directory, universe, data_start, test_start, test_end):
    """Write files for `universe` = {symbol: (up, base, {Candle 3 offset from the
    first test day: (shape, outcome)})}. Returns (days, {symbol: daily bars})."""
    days = weekdays(data_start, test_end)
    t0 = days.index(test_start)
    market = {}
    for n, (symbol, (up, base, plan)) in enumerate(universe.items()):
        episodes = {t0 + offset: spec for offset, spec in plan.items()}
        bars = build_symbol(len(days), up, base, 500 + n, episodes)
        write_symbol(directory, symbol, days, bars, max(0, t0 - 12))
        market[symbol] = bars
    return days, market


E2E_START, E2E_END = date(2026, 9, 7), date(2026, 9, 22)  # 12 sessions
E2E_UNIVERSE = {
    "SBIN": (False, 800, {0: ("big", "target"), 5: ("big", "stop"), 9: ("small", "flat")}),
    "TCS": (True, 3500, {1: ("big", "target"), 6: ("big", "flat"), 10: ("small", "stop")}),
    "LT": (False, 3600, {1: ("big", "flat"), 7: ("big", "gap")}),
    "M&M": (True, 3000, {2: ("big", "stop"), 8: ("big", "target")}),
    "BSE": (False, 2500, {3: ("big", "flat"), 11: ("big", "stop")}),
    "MARUTI": (True, 12000, {4: ("big", "gap")}),
}
# What the default settings must trade: (session, symbol, side, how it ends)
E2E_EXPECTED = [
    (0, "SBIN", "BUY", "target"),
    (1, "TCS", "SELL", "target"),  # LT qualifies the same morning and ranks lower
    (2, "M&M", "SELL", "stop loss"),
    (3, "BSE", "BUY", "square-off time"),
    (5, "SBIN", "BUY", "stop loss"),
    (6, "TCS", "SELL", "square-off time"),
    (8, "M&M", "SELL", "target"),
    (11, "BSE", "BUY", "stop loss"),
]


@pytest.fixture(scope="module")
def shaped_market(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tcr_bars")
    days, market = make_market(directory, E2E_UNIVERSE, date(2026, 4, 20), E2E_START, E2E_END)
    return directory, days, market


def _ist(epoch_ms):
    return datetime.fromtimestamp(epoch_ms / 1000, tz=st.IST).replace(tzinfo=None)


def read_journal(run_dir):
    path = run_dir / "three_candle_data" / "journal.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_backtest(run_dir, result, days, market, tp_pct=5.0, max_per_day=1, cap=40000.0):
    """The invariants every run must hold. Returns (journal rows, entries, exits)."""
    assert result["status"] == "completed", result.get("message")
    trades = sorted(result["trades"], key=lambda t: t["entry_time"])
    assert trades, "no trade in the test period"
    assert {t["symbol"] for t in trades} <= set(market)
    assert {t["side"] for t in trades} == {"LONG", "SHORT"}  # both sides trade
    assert all(t["product"] == "MIS" for t in trades)
    for prev, nxt in zip(trades, trades[1:], strict=False):
        assert nxt["entry_time"] >= prev["exit_time"], (prev, nxt)  # never two positions
    for t in trades:
        assert t["quantity"] * t["entry_price"] <= cap + 1e-6, t
        opened, closed = _ist(t["entry_time"]), _ist(t["exit_time"])
        assert closed.date() == opened.date(), t  # opened and closed the same day
        assert f"{closed:%H:%M}" <= "15:11", t  # flat by the square-off
        assert "09:20" <= f"{opened:%H:%M}" <= "09:45", t  # only in the entry window
    assert all(o["order_status"] != "open" for o in result["orders"])

    rows = read_journal(run_dir)
    entries = [r for r in rows if r["event"] == "entry"]
    exits = [r for r in rows if r["event"] == "exit"]
    assert entries and len(entries) == len(exits) == len(trades)
    per_day, seen = {}, set()
    for r in entries:
        trade, plan = r["trade"], r["plan"]
        day = date.fromisoformat(r["ts"][:10])
        i = days.index(day)
        c1, c2 = market[trade["symbol"]][i - 2], market[trade["symbol"]][i - 1]
        assert plan["signal_bar"] == days[i - 1].isoformat()  # Candle 2 is the day before
        assert plan["ref_entry"] == pytest.approx(c2[3], abs=1e-3)
        if trade["side"] == "BUY":
            assert trade["stop"] < trade["entry_price"] < trade["target"], r
            assert trade["stop"] == pytest.approx(st.floor_tick(c2[2], 0.05), abs=1e-6)
            want = st.floor_tick(c1[1] * (1 - tp_pct / 100), 0.05)
        else:
            assert trade["target"] < trade["entry_price"] < trade["stop"], r
            assert trade["stop"] == pytest.approx(st.ceil_tick(c2[1], 0.05), abs=1e-6)
            want = st.ceil_tick(c1[2] * (1 + tp_pct / 100), 0.05)
        assert trade["target"] == pytest.approx(want, abs=1e-6)
        assert r["position_value"] <= cap
        assert r["planned_loss"] == pytest.approx(
            trade["qty"] * abs(trade["entry_price"] - trade["stop"]), abs=0.02
        )
        assert "09:20" <= r["ts"][11:16] <= "09:45"
        per_day[day] = per_day.get(day, 0) + 1
        assert (day, trade["symbol"]) not in seen  # never twice in a symbol in a day
        seen.add((day, trade["symbol"]))
    assert max(per_day.values()) <= max_per_day
    return rows, entries, exits


@pytest.mark.timeout(600)
def test_backtest_runs_unchanged_and_trades_both_sides(tmp_path, shaped_market):
    directory, days, market = shaped_market
    original = SCRIPT.read_bytes()
    run_dir, result, log = run_backtest(
        tmp_path,
        SCRIPT,
        directory,
        E2E_START.isoformat(),
        E2E_END.isoformat(),
        extra_env={"TCR_SYMBOLS": ",".join(E2E_UNIVERSE)},
        timeout=500,
    )
    assert SCRIPT.read_bytes() == original
    assert "6 symbols on NSE MIS" in log
    rows, entries, exits = check_backtest(run_dir, result, days, market)
    t0 = days.index(E2E_START)
    got = [
        (
            days.index(date.fromisoformat(e["ts"][:10])) - t0,
            e["trade"]["symbol"],
            e["trade"]["side"],
            x["reason"],
        )
        for e, x in zip(entries, exits, strict=True)
    ]
    assert got == E2E_EXPECTED
    # the morning both TCS and LT qualified: the higher reward:risk was taken, the other journalled
    morning = days[t0 + 1].isoformat()
    ignored = [r for r in rows if r["event"] == "signal_ignored" and r["ts"][:10] == morning]
    assert [r["symbol"] for r in ignored] == ["LT"] and "TCS ranks higher" in ignored[0]["why"]
    # a stock that opened beyond its stop was not entered
    gaps = [r for r in rows if r["event"] == "skip" and "already at or" in r["why"]]
    assert {r["symbol"] for r in gaps} == {"LT", "MARUTI"}
    # the small setups need a lower TP_PCT: at 5% they are dropped by the take profit filter
    dropped = [r for r in rows if r["event"] == "skip" and "take profit is at or" in r["why"]]
    assert {r["symbol"] for r in dropped} >= {"SBIN", "TCS"}
    assert result["metrics"]["trades"] == len(E2E_EXPECTED)


@pytest.mark.timeout(600)
def test_backtest_with_a_lower_tp_pct_takes_more_setups(tmp_path, shaped_market):
    directory, days, market = shaped_market
    run_dir, result, _log = run_backtest(
        tmp_path,
        SCRIPT,
        directory,
        E2E_START.isoformat(),
        E2E_END.isoformat(),
        extra_env={"TCR_SYMBOLS": ",".join(E2E_UNIVERSE), "TCR_TP_PCT": "1"},
        timeout=500,
    )
    _rows, entries, _exits = check_backtest(run_dir, result, days, market, tp_pct=1.0)
    t0 = days.index(E2E_START)
    traded = {days.index(date.fromisoformat(e["ts"][:10])) - t0 for e in entries}
    assert {9, 10} <= traded  # the days whose setups only qualify with the lower TP_PCT
    assert len(entries) > len(E2E_EXPECTED)


# =============================================================================
# ROUND 2: the stop survives a kill, a failed journal and a failed state file
# =============================================================================
class Killed(BaseException):
    """The process dies (kill, power cut, host restart). Not an Exception: nothing catches it."""


def _killed_while_sending_the_stop(hostile, data_dir, reached_broker):
    """The entry fills, then the process dies inside the stop's placeorder call.
    Returns a NEW strategy started on the same data directory, and the broker."""
    s, b = hostile(data_dir=data_dir)
    real = b.placeorder

    def die(symbol, action, quantity, price_type, **k):
        if price_type != "MARKET":
            if reached_broker:
                real(symbol, action, quantity, price_type, **k)
            raise Killed()
        return real(symbol, action, quantity, price_type, **k)

    b.placeorder = die
    with pytest.raises(Killed):
        _signal(s, "AAA")
    b.placeorder = real
    saved = json.loads(s.state_path.read_text())["open"]
    assert saved["stop_check"] is True and saved["stop_unknown_since"]  # saved before the send
    assert saved["stop_order_id"] is None
    return hostile(data_dir=data_dir, broker=b)


def test_killed_while_sending_the_stop_that_reached_the_broker(hostile, tmp_path):
    s, b = _killed_while_sending_the_stop(hostile, tmp_path / "k1", True)
    side = hostile.side
    assert b.pos["AAA"] == 10 * side.sign and len(b.live("AAA")) == 1
    s.step()
    trade = s.state["open"]
    assert trade["stop_order_id"] == "O2" and not trade["stop_check"]  # found and tracked
    assert "stop_unknown_since" not in trade and len(b.live("AAA")) == 1
    b.ltp_value = side.target
    s.manage_open(_advance(s, 10))
    assert s.state["open"] is None and b.pos["AAA"] == 0
    assert b.live("AAA") == []  # nothing left working to fire later
    b.trigger_stops()
    assert b.pos["AAA"] == 0


def test_killed_while_sending_the_stop_before_it_is_adopted_the_exit_still_cancels_it(
    hostile, tmp_path
):
    s, b = _killed_while_sending_the_stop(hostile, tmp_path / "k2", True)
    b.orderbook_fail = True  # the restart cannot read the book yet
    s.step()
    assert s.state["open"]["stop_order_id"] is None and s.state["open"]["stop_check"]
    b.ltp_value = hostile.side.target
    s.manage_open(_advance(s, 10))
    assert s.state["open"] and _smarts(b) == 0  # no exit while the book cannot be read
    b.orderbook_fail = False
    s.manage_open(_advance(s, 10))
    assert s.state["open"] is None and b.pos["AAA"] == 0 and b.live("AAA") == []


def test_killed_while_sending_the_stop_that_never_reached_the_broker(hostile, tmp_path):
    s, b = _killed_while_sending_the_stop(hostile, tmp_path / "k3", False)
    side = hostile.side
    assert b.pos["AAA"] == 10 * side.sign and b.live("AAA") == []
    s.step()
    s.manage_open(_advance(s, st.STOP_CONFIRM_SEC - 5))
    assert b.live("AAA") == [] and s.state["open"]["stop_check"]  # inside the grace: watched
    s.manage_open(_advance(s, 10))
    live = b.live("AAA")
    assert [(o["action"], o["qty"], o["trigger"]) for o in live] == [(side.close, 10, side.stop)]
    assert s.state["open"]["stop_order_id"] == live[0]["orderid"]
    s.manage_open(_advance(s, 120))
    assert len(b.live("AAA")) == 1  # and never a second one


def test_killed_before_the_stop_and_the_price_crosses_it_is_closed_by_the_script(hostile, tmp_path):
    s, b = _killed_while_sending_the_stop(hostile, tmp_path / "k4", False)
    b.ltp_value = hostile.side.through_stop
    s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None
    assert s.state["sweep"]["symbol"] == "AAA"  # a stop that turns up late is swept


def test_the_stop_is_placed_before_the_journal_and_a_failing_journal_cannot_skip_it(hostile):
    s, b = hostile()
    side = hostile.side
    order = []
    real_place, real_journal = b.placeorder, s.journal

    def place(symbol, action, quantity, price_type, **k):
        order.append(f"place {price_type}")
        return real_place(symbol, action, quantity, price_type, **k)

    def locked(event, **data):
        if event == "entry":
            order.append("journal entry")
            raise PermissionError("journal.jsonl is in use")
        return real_journal(event, **data)

    b.placeorder, s.journal = place, locked
    with pytest.raises(PermissionError):
        _signal(s, "AAA")
    assert order == ["place MARKET", "place SL-M", "journal entry"]
    assert b.pos["AAA"] == 10 * side.sign and len(b.live("AAA")) == 1
    assert s.state["open"]["stop_order_id"] == "O2"


def test_a_journal_that_cannot_be_written_never_raises(hostile, capsys):
    s, b = hostile()
    s.journal_path = s.journal_path.parent  # a folder: it cannot be opened for writing
    _signal(s, "AAA")
    assert s.state["open"]["stop_order_id"] and len(b.live("AAA")) == 1
    s.exit_trade("target")
    assert s.state["open"] is None and b.pos["AAA"] == 0
    out = capsys.readouterr().out
    assert out.count("the journal could not be written") == 1  # said once, in plain words


def test_a_raising_save_at_the_fill_cannot_skip_the_stop(hostile):
    s, b = hostile()
    side = hostile.side
    plan, _ = side.plan("AAA", None, 100.0)
    b.pos["AAA"] = 10 * side.sign

    def broken():
        raise OSError("disk full")

    s._save_state = broken
    with pytest.raises(OSError):
        s.on_filled(plan, 10, 100.0)
    live = b.live("AAA")
    assert [(o["action"], o["trigger"]) for o in live] == [(side.close, side.stop)]
    assert s.state["open"]["stop_order_id"] == live[0]["orderid"]


def test_a_state_file_that_cannot_be_written_keeps_the_trade_managed_and_blocks_entries(
    hostile, monkeypatch
):
    monkeypatch.setattr(st, "TELEGRAM_USER", "trader")
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    good = s.state_path
    s.state_path = good.parent / "missing folder" / "state.json"  # every write now fails
    assert s._save_state() is False and s.unsaved
    assert sum("state file could not be written" in m for _u, m in b.telegrams) == 1
    b.ltp_value = side.target
    s.manage_open(_advance(s, 10))  # the trade is still managed and closed
    assert s.state["open"] is None and b.pos["AAA"] == 0 and b.live("AAA") == []
    assert sum("state file could not be written" in m for _u, m in b.telegrams) == 1  # once
    b.ltp_value = 100.0
    assert s.can_enter(s.test_clock["now"]) == (False, "the state file cannot be written")
    sent = _orders_sent(b)
    _signal(s, "BBB")
    assert _orders_sent(b) == sent and b.pos["BBB"] == 0
    s.state_path = good
    s._save_state()
    _signal(s, "BBB")
    assert s.state["open"]["symbol"] == "BBB"


def test_an_entry_is_not_sent_when_its_pending_record_cannot_be_saved(hostile):
    s, b = hostile()
    plan, _ = hostile.side.plan("AAA", None, 100.0)
    s.state_path = s.state_path.parent / "missing folder" / "state.json"
    s.enter(plan, s.test_clock["now"])
    assert _orders_sent(b) == 0 and s.state["pending"] is None and s.state["taken"] == []


def test_a_stop_refused_both_ways_is_watched_by_the_script_with_no_sweep(hostile):
    s, b = hostile()
    refusal = {"status": "error", "message": "stop orders are not allowed"}
    b.refuse = {"SL-M": refusal, "SL": refusal}
    alerts = _alerts(s)
    _signal(s, "AAA")
    trade = s.state["open"]
    assert b.live("AAA") == [] and not trade["stop_check"] and "stop_unknown_since" not in trade
    assert any("watched by this script only" in a for a in alerts)
    b.ltp_value = hostile.side.through_stop
    s.manage_open(_advance(s, 10))
    assert s.state["open"] is None and b.pos["AAA"] == 0 and s.state["sweep"] is None


def test_a_normal_fill_leaves_no_unconfirmed_stop_and_no_sweep(hostile):
    s, b = hostile()
    _signal(s, "AAA")
    trade = s.state["open"]
    assert not trade["stop_check"] and "stop_unknown_since" not in trade
    s.exit_trade("target")
    assert s.state["open"] is None and s.state["sweep"] is None


def test_killed_while_resizing_the_stop_is_recovered_like_a_fill(hostile, tmp_path):
    d = tmp_path / "k5"
    s, b = hostile(data_dir=d)
    side = hostile.side
    _signal(s, "AAA")
    b.pos["AAA"] = 13 * side.sign
    real = b.placeorder

    def die(symbol, action, quantity, price_type, **k):
        raise Killed()

    b.placeorder = die
    with pytest.raises(Killed):
        _check(s)  # cancels the 10 stop, then dies sending the 13 one
    b.placeorder = real
    assert b.live("AAA") == []
    s2, _ = hostile(data_dir=d, broker=b)
    assert s2.state["open"]["stop_check"] and s2.state["open"]["stop_unknown_since"]
    s2.manage_open(_advance(s2, st.STOP_CONFIRM_SEC + 1))
    assert [(o["qty"], o["trigger"]) for o in b.live("AAA")] == [(13, side.stop)]


# =============================================================================
# ROUND 2: the closing order cannot open the opposite position
# =============================================================================
def test_the_closing_order_is_a_smart_order_to_zero_with_quantity_zero(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    s.exit_trade("target")
    assert b.smart_args == [("AAA", side.close, 0, 0)]  # action, quantity 0, position_size 0
    assert b.pos["AAA"] == 0 and s.state["open"] is None
    assert [c for c in b.calls if c[0] == "smart"] == [("smart", "AAA", -10 * side.sign)]


def test_a_position_that_goes_flat_just_before_the_exit_is_not_reversed(hostile):
    s, b = hostile()
    side = hostile.side
    _signal(s, "AAA")
    stop_id = s.state["open"]["stop_order_id"]

    def closed_by_someone_else():
        b.pos["AAA"] = 0  # by hand, or the broker's own square-off

    b.before_smart = closed_by_someone_else
    b.ltp_value = side.target
    s.manage_open(_advance(s, 10))
    # under the real smart order rule a quantity would have been placed as given
    assert b.pos["AAA"] == 0, "the exit opened the opposite position"
    assert [o for o in b.orders.values() if o["pricetype"] == "MARKET"][1:] == []
    assert s.state["open"] is None and b.orders[stop_id]["order_status"] == "cancelled"
    assert "already flat" in _journal(s, "exit")[0]["reason"]
    for _ in range(3):
        s.step()
    assert b.pos == dict.fromkeys(SYMS, 0)


def test_the_fake_broker_follows_the_real_smart_order_table():
    b = HostileBroker(["X"])
    # flat, target 0: the order is placed as given (the surprise the strategy must avoid)
    b.placesmartorder(symbol="X", action="SELL", quantity=7, position_size=0)
    assert b.pos["X"] == -7
    # flat, target 0, quantity 0: nothing
    b.pos["X"] = 0
    r = b.placesmartorder(symbol="X", action="SELL", quantity=0, position_size=0)
    assert b.pos["X"] == 0 and "orderid" not in r
    # a position, target 0: closed whatever the request says
    b.pos["X"] = 5
    b.placesmartorder(symbol="X", action="BUY", quantity=0, position_size=0)
    assert b.pos["X"] == 0
    b.pos["X"] = -5
    b.placesmartorder(symbol="X", action="SELL", quantity=99, position_size=0)
    assert b.pos["X"] == 0
    # a target: the difference is traded
    b.placesmartorder(symbol="X", action="BUY", quantity=1, position_size=4)
    assert b.pos["X"] == 4
    b.placesmartorder(symbol="X", action="BUY", quantity=1, position_size=-2)
    assert b.pos["X"] == -2


# =============================================================================
# ROUND 2: a lost state file ends the day's entries
# =============================================================================
def test_after_a_lost_state_file_nothing_new_is_entered_that_day(desk, monkeypatch):
    monkeypatch.setattr(st, "TELEGRAM_USER", "trader")
    monkeypatch.setattr(st, "MAX_TRADES_PER_DAY", 5)
    s, b = desk()
    s.step()
    assert s.state["open"]["symbol"] == "BBB"
    s.state_path.write_text("{ not json")
    s2, _ = desk(data_dir=s.state_path.parent, broker=b)
    assert any("no new entry is made for the rest of today" in m for _u, m in b.telegrams)
    for _ in range(6):
        _advance(s2, 10)
        s2.step()
    assert b.pos == dict.fromkeys(BASKET, 0) and b.live() == []  # taken over and closed
    market = [o for o in b.orders.values() if o["pricetype"] == "MARKET"]
    assert [(o["symbol"], o["action"]) for o in market] == [("BBB", "SELL"), ("BBB", "BUY")]
    assert "no new entries today" in s2.can_enter(s2.test_clock["now"])[1]
    skipped = {r["symbol"]: r["why"] for r in _journal(s2, "skip")}
    assert set(skipped) == {"AAA", "BBB", "DDD"}
    assert all("the saved state was lost today" in why for why in skipped.values())
    # another restart the same day reads a good file and is still blocked
    s3, _ = desk(data_dir=s.state_path.parent, broker=b)
    assert s3.state_lost is False
    for _ in range(3):
        _advance(s3, 10)
        s3.step()
    assert len(market) == len([o for o in b.orders.values() if o["pricetype"] == "MARKET"])
    # the next trading day entries are allowed again
    s3.test_clock["now"] = datetime(2026, 11, 5, 9, 30, 0)
    s3.step()
    assert s3.state["no_entries"] == ""
    assert "no new entries today" not in s3.can_enter(s3.test_clock["now"])[1]


# =============================================================================
# ROUND 2: an entry whose answer is never confirmed uses up a trade
# =============================================================================
def _unknown_entry(desk, created):
    """BBB's entry answer is lost. `created`: the order exists at the broker,
    working and out of sight; otherwise it never reached the broker."""
    s, b = desk()
    if created:
        b.lost_answer = {"MARKET"}
        b.hold_market = True
        real_place = _hide(b, "MARKET")
    else:
        b.ghost = {"MARKET"}
    s.step()
    if created:
        b.placeorder = real_place
    b.lost_answer, b.ghost = set(), set()
    assert s.state["pending"]["plan"]["symbol"] == "BBB"
    _advance(s, st.ENTRY_CONFIRM_SEC + 1)
    s.step()
    assert s.state["pending"] is None
    return s, b


def test_an_unknown_entry_counts_as_the_days_trade(desk):
    s, b = _unknown_entry(desk, created=False)
    assert list(s.state["unknown"]) == ["BBB"] and s.state["trades"] == 0
    failed = _journal(s, "entry_failed")
    assert failed[0]["symbol"] == "BBB" and failed[0]["unknown"] is True
    for _ in range(5):
        _advance(s, 10)
        s.step()
    assert b.pos == dict.fromkeys(BASKET, 0)  # AAA and DDD are not entered
    assert _placed(b) == [("place", "BBB", "SELL", 387, "MARKET")]
    skipped = {r["symbol"]: r["why"] for r in _journal(s, "skip")}
    assert set(skipped) == {"AAA", "DDD"}
    assert all("limit of 1 trade is reached" in why for why in skipped.values())
    # and a restart keeps the count
    s2, _ = desk(data_dir=s.state_path.parent, broker=b)
    s2.test_clock["now"] = s.test_clock["now"]
    s2.step()
    assert list(s2.state["unknown"]) == ["BBB"] and len(_placed(b)) == 1


def test_an_unknown_entry_leaves_room_only_if_the_daily_limit_allows(desk, monkeypatch):
    monkeypatch.setattr(st, "MAX_TRADES_PER_DAY", 2)
    s, b = _unknown_entry(desk, created=False)
    _advance(s, 10)
    s.step()
    assert s.state["open"]["symbol"] == "AAA"  # one unknown + one trade = the limit of 2
    b.ltps["AAA"] = 105.0  # its take profit
    _advance(s, 10)
    s.step()
    assert s.state["open"] is None and b.pos["DDD"] == 0
    _advance(s, 10)
    s.step()
    assert b.pos["DDD"] == 0 and "limit of 2 trades" in _journal(s, "skip")[-1]["why"]


def test_an_unknown_entry_that_fills_late_is_taken_over_with_its_stop(desk):
    s, b = _unknown_entry(desk, created=True)
    assert b.pos == dict.fromkeys(BASKET, 0)
    b.hidden.clear()
    b.release()  # the lost order fills after it was taken as not placed
    assert b.pos["BBB"] == -387
    alerts = _alerts(s)
    _advance(s, 10)
    s.step()  # found on the very next cycle, not at the next position check
    trade = s.state["open"]
    assert trade["symbol"] == "BBB" and trade["side"] == "SELL" and trade["qty"] == 387
    assert [(o["action"], o["trigger"]) for o in b.live("BBB")] == [("BUY", 105.0)]
    assert s.state["unknown"] == {} and s.state["trades"] == 1
    assert any("has filled after all" in a for a in alerts)
    for _ in range(3):
        _advance(s, 10)
        s.step()
    assert b.pos["AAA"] == 0 and b.pos["DDD"] == 0  # still one position only


def test_an_unknown_entry_order_that_shows_up_working_is_cancelled(desk):
    s, b = _unknown_entry(desk, created=True)
    b.hidden.clear()  # it shows in the book, still working
    _advance(s, 10)
    s.step()
    assert b.orders["O1"]["order_status"] == "cancelled" and b.pos["BBB"] == 0
    assert list(s.state["unknown"]) == ["BBB"]  # still counted: nothing else is entered
    _advance(s, 10)
    s.step()
    assert len(_placed(b)) == 1


def test_an_unknown_entry_found_and_cancelled_in_time_does_not_count(desk):
    s, b = desk()
    b.lost_answer = {"MARKET"}
    b.hold_market = True
    s.step()  # the order is in the book, working: cancelled and confirmed
    b.lost_answer, b.hold_market = set(), False
    assert b.orders["O1"]["order_status"] == "cancelled"
    _advance(s, st.ENTRY_CONFIRM_SEC + 1)
    s.step()
    assert s.state["unknown"] == {} and s.state["taken"][0] == "BBB"
    _advance(s, 10)
    s.step()
    assert s.state["open"]["symbol"] == "AAA"  # the day's trade is still available


# =============================================================================
# ROUND 2: a daily bar one session stale
# =============================================================================
def test_a_symbol_whose_daily_bar_is_a_session_behind_the_basket_is_skipped(desk):
    daily = basket_history()
    daily["BBB"] = with_today(short_bars(last="2026-11-02"))  # yesterday's bar not published
    s, b = desk(daily=daily)
    s.step()
    assert s.newest_bar == YESTERDAY
    assert s.state["open"]["symbol"] == "AAA" and b.pos["BBB"] == 0
    skip = _journal(s, "skip")[0]
    assert skip["symbol"] == "BBB"
    assert "its latest daily bar is 2026-11-02, older than 2026-11-03" in skip["why"]
    assert s.state["skipped"]["BBB"] == skip["why"]


def test_a_stale_symbol_read_first_is_still_dropped_once_a_fresher_one_is_seen(desk):
    daily = basket_history()
    daily["AAA"] = with_today(long_bars(last="2026-11-02"))
    s, b = desk(symbols=["AAA", "CCC"], daily=daily)  # CCC has no setup but a fresh bar
    s.step()
    assert _orders_sent(b) == 0 and s.state["open"] is None
    assert "older than 2026-11-03" in s.state["skipped"]["AAA"]


def test_when_the_whole_basket_ends_on_the_same_day_nothing_is_called_stale(desk):
    daily = {k: with_today(v.iloc[:-1].iloc[:-1]) for k, v in basket_history().items()}
    assert all(v.index[-2] == pd.Timestamp("2026-11-02") for v in daily.values())
    s, _b = desk(daily=daily)
    s.step()
    assert s.newest_bar == "2026-11-02" and not any(
        "older than" in why for why in s.state["skipped"].values()
    )


# =============================================================================
# ROUND 2: settings a trader got wrong, and a setup() that cannot be tripped
# =============================================================================
BAD = {
    "TCR_CAPITAL": ("abc", "a number"),
    "TCR_MAX_TRADES_PER_DAY": ("two", "a whole number"),
    "TCR_ST_ATR_LEN": ("10.5", "a whole number"),
    "TCR_ENTRY_START": ("9.20am", "a time written as HH:MM"),
    "TCR_SQUARE_OFF": ("25:00", "a time written as HH:MM"),
    "TCR_ENABLE_SHORTS": ("maybe", "true or false"),
    "TCR_TP_PCT": ("nan", "a number"),
}


def test_a_bad_setting_does_not_raise_at_import_and_is_named_at_start(monkeypatch, capsys):
    for name, (value, _expected) in BAD.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENALGO_API_KEY", "k")
    other = _load("three_candle_bad_env")  # no exception: import has no side effects
    assert {name for name, _v, _e in other.BAD_SETTINGS} == set(BAD)
    assert (other.CAPITAL, other.ENTRY_START, other.ENABLE_SHORTS) == (20000.0, "09:20", True)
    made = []
    monkeypatch.setattr(other, "api", lambda **k: made.append(k))
    with pytest.raises(SystemExit) as stop:
        other.main()
    assert stop.value.code == 1 and made == []  # stopped before any client was made
    out = capsys.readouterr().out
    for name, (value, expected) in BAD.items():
        assert f"{name} is set to '{value}', which is not {expected}" in out
    assert "Traceback" not in out and "ValueError" not in out


def test_good_settings_in_their_usual_spellings_are_accepted(monkeypatch):
    for name, value in {
        "TCR_ENTRY_START": "9:25",
        "TCR_ENABLE_LONGS": "No",
        "TCR_MAX_TRADES_PER_DAY": "2.0",
        "TCR_CAPITAL": " 25000 ",
    }.items():
        monkeypatch.setenv(name, value)
    other = _load("three_candle_good_env")
    assert other.BAD_SETTINGS == []
    assert (other.ENTRY_START, other.ENABLE_LONGS) == ("09:25", False)
    assert (other.MAX_TRADES_PER_DAY, other.CAPITAL) == (2, 25000.0)
    assert other.hhmm(other.ENTRY_START) == 9 * 60 + 25


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("TCR_MAX_TRADES_PER_DAY", "0", "TCR_MAX_TRADES_PER_DAY must be 1 or more"),
        ("TCR_MAX_DAILY_LOSS", "-5", "cannot be below 0"),
        ("TCR_ENTRY_END", "15:30", "the times must run"),
    ],
)
def test_settings_that_make_no_sense_stop_the_start_with_a_reason(
    monkeypatch, capsys, name, value, message
):
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENALGO_API_KEY", "k")
    other = _load("three_candle_range_env")
    with pytest.raises(SystemExit) as stop:
        other.main()
    assert stop.value.code == 1 and message in capsys.readouterr().out


def test_a_daily_loss_limit_of_zero_switches_it_off(desk, monkeypatch):
    monkeypatch.setattr(st, "MAX_DAILY_LOSS", 0.0)
    s, _b = desk()
    s.state["pnl"] = -250000.0
    s.step()
    assert s.state["open"]["symbol"] == "BBB"  # 0 is "off", not "already reached"


@pytest.mark.parametrize(
    "answer",
    [
        "analyze",
        ["analyze"],
        {"status": "success", "data": "analyze"},
        {"status": "success", "data": None},
        {"status": "success"},
        None,
        42,
    ],
)
def test_setup_survives_any_shape_of_mode_answer(tmp_path, monkeypatch, answer):
    monkeypatch.setattr(st, "time", FakeTime())
    b = HostileBroker(SYMS)
    b.analyzerstatus = lambda: answer
    s = st.ThreeCandleStrategy(b, symbols=SYMS, clock=lambda: NOW, data_dir=tmp_path, pause=0)
    s.setup()
    assert s.symbols == SYMS


# =============================================================================
# ROUND 2: a symbol closed with an unknown answer is read every cycle
# =============================================================================
def test_a_late_reversal_after_the_sweep_is_found_on_the_next_cycle(hostile):
    s, b = _middle_path_close(hostile)
    side = hostile.side
    _advance(s, st.SWEEP_SEC + 1)
    s.step()
    assert s.state["sweep"] is None and s.state["watch"] == {"AAA": side.close}
    b.hidden.clear()
    b.trigger_stops()  # the lost stop fires long after the sweep
    assert b.pos["AAA"] == -10 * side.sign
    _advance(s, 10)  # well inside the same CHECK_MINUTES slot
    s.step()
    assert (s.state["open"] or {}).get("recovered") or b.pos["AAA"] == 0
    s.step()
    assert b.pos["AAA"] == 0 and s.state["open"] is None
