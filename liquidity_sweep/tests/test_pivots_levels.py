"""Pivots, the level book, invalidation and sweep signals."""

import numpy as np
import pytest

from backtest import run_backtest
from config import config_for_timeframe
from scenarios import RES, SUP, cfg, flat, scenario, set_bar, sweep, with_level
from strategy import Level, LevelBook, find_pivots, pick_signal


def test_pivot_is_strict_maximum():
    high = np.array([1, 2, 3, 5, 3, 2, 1, 2, 3, 2, 1], float)
    low = np.array([5, 4, 3, 2, 3.5, 4, 3, 4, 5, 6, 7], float)
    ph, pl = find_pivots(high, low, 2)
    assert ph.tolist() == [i in (3, 8) for i in range(11)]
    assert pl.tolist() == [i in (3, 6) for i in range(11)]


def test_equal_highs_in_window_are_not_pivots():
    high = np.array([1, 2, 5, 5, 2, 1, 0], float)
    ph, _ = find_pivots(high, -high, 2)
    assert not ph.any()


def test_bars_near_the_edges_are_never_pivots():
    high = np.array([9, 1, 1, 1, 9], float)
    ph, _ = find_pivots(high, -high, 2)
    assert not ph.any()


def test_pivot_used_only_after_its_confirmation_bar():
    """Two pivots at 100 and a third at 100.05 (confirmed at bar 45).

    Bar 45 sweeps the three-touch level (mean 100.0167) but the third pivot is
    not usable yet, so only two touches count: no signal. Bar 46 repeats the
    sweep and does signal, so the entry is at bar 47.
    """
    df = with_level(flat(), "short", pivots=(10, 25))
    set_bar(df, 40, h=100.05)
    for i in (45, 46):
        set_bar(df, i, h=100.03, c=99.0)
    trades = run_backtest(df, cfg(), "X")
    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == df.index[47]
    assert trades.iloc[0]["touches"] == 3


def test_level_clustering_mean_and_touches():
    book = LevelBook(config_for_timeframe("5m"))
    for i, p in enumerate([100.0, 100.1, 100.12, 101.0]):
        book.add("R", i, p)
    levels = book.levels("R")
    assert [lv.touches for lv in levels] == [3, 1]
    assert levels[0].price == pytest.approx((100.0 + 100.1 + 100.12) / 3)
    assert levels[0].members == (0, 1, 2)
    assert levels[1].price == 101.0


def test_cluster_members_are_all_within_tolerance_of_each_other():
    book = LevelBook(config_for_timeframe("5m"))  # tol 0.15%
    for i, p in enumerate([100.0, 100.1, 100.2, 100.3]):
        book.add("S", i, p)
    levels = book.levels("S")
    for lv in levels:
        prices = [[100.0, 100.1, 100.2, 100.3][m] for m in lv.members]
        assert max(prices) <= min(prices) * 1.0015
    assert [lv.touches for lv in levels] == [2, 2]


def test_lookback_window_drops_old_pivots():
    c = config_for_timeframe("5m")
    book = LevelBook(c)
    book.add("R", 0, 100.0)
    book.add("R", 1, 100.0)
    book.prune(c.level_lookback)  # pivot 0 is exactly lookback bars old: kept
    assert sorted(book.live["R"]) == [0, 1]
    book.prune(c.level_lookback + 1)  # now older than lookback: dropped
    assert sorted(book.live["R"]) == [1]


def test_lookback_in_engine_old_level_no_longer_trades():
    df = scenario("short", sweep_at=50)
    trades_in = run_backtest(df, cfg(), "X")
    assert len(trades_in) == 1
    # Same pivots, but the lookback is too short to still hold the first pivot.
    trades_out = run_backtest(df, cfg(level_lookback=39), "X")
    assert trades_out.empty


def test_level_dies_on_close_beyond_tolerance():
    df = scenario("short", sweep_at=60)
    set_bar(df, 50, h=100.2, c=100.0 * 1.0015 + 0.01)  # closes beyond 100 by more than tol
    assert run_backtest(df, cfg(), "X").empty


def test_close_beyond_within_tolerance_does_not_kill():
    df = scenario("short", sweep_at=60)
    set_bar(df, 50, h=100.1, c=100.1)  # beyond, but by less than tol
    trades = run_backtest(df, cfg(), "X")
    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == df.index[61]


def test_support_dies_on_close_below_tolerance():
    df = scenario("long", sweep_at=60)
    set_bar(df, 50, lo=97.7, c=98.0 * (1 - 0.0015) - 0.01)
    assert run_backtest(df, cfg(), "X").empty


def test_traded_level_dies():
    df = scenario("short", sweep_at=50)
    set_bar(df, 52, h=100.4, lo=96.0)  # stop hit (stop first) -> flat again
    sweep(df, "short", 55)  # same level swept again
    trades = run_backtest(df, cfg(), "X")
    assert len(trades) == 1


def test_untraded_level_survives_a_dropped_signal():
    """A sweep whose entry is refused (open beyond stop) leaves the level alive."""
    df = scenario("short", sweep_at=50)
    set_bar(df, 51, o=101.0, h=101.0, c=100.1)  # opens beyond stop: skip; no new signal, no break
    sweep(df, "short", 55)
    trades = run_backtest(df, cfg(), "X")
    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == df.index[56]


def _lv(kind, price, touches):
    return Level(kind, price, touches, tuple(range(touches)))


def test_signal_short_and_long_and_min_touches():
    res = [_lv("R", 100.0, 3)]
    sup = [_lv("S", 98.0, 3)]
    assert pick_signal(100.3, 99.0, 99.5, res, sup, 3)[0] == -1
    assert pick_signal(99.5, 97.7, 98.5, res, sup, 3)[0] == 1
    assert pick_signal(100.3, 99.0, 99.5, res, sup, 4)[0] == 0
    assert pick_signal(100.3, 99.0, 100.1, res, sup, 3)[0] == 0  # closed beyond, no sweep
    assert pick_signal(99.9, 99.0, 99.5, res, sup, 3)[0] == 0  # did not reach the level


def test_signal_prefers_touches_then_extreme():
    res = [_lv("R", 100.0, 3), _lv("R", 100.5, 3), _lv("R", 100.2, 4)]
    side, lv = pick_signal(101.0, 99.0, 99.5, res, [], 3)
    assert side == -1 and lv.price == 100.2
    side, lv = pick_signal(101.0, 99.0, 99.5, res[:2], [], 3)
    assert lv.price == 100.5
    sup = [_lv("S", 98.0, 3), _lv("S", 97.5, 3)]
    side, lv = pick_signal(99.0, 97.0, 98.5, [], sup, 3)
    assert side == 1 and lv.price == 97.5


def test_bar_sweeping_both_sides_is_skipped():
    res = [_lv("R", 100.0, 3)]
    sup = [_lv("S", 98.0, 3)]
    assert pick_signal(100.3, 97.7, 99.0, res, sup, 3) == (0, None)


def test_engine_sweep_both_sides_and_min_touches():
    for side in ("short", "long"):
        trades = run_backtest(scenario(side), cfg(), "X")
        assert len(trades) == 1 and trades.iloc[0]["side"] == side
        assert trades.iloc[0]["level"] == pytest.approx(RES if side == "short" else SUP)
        assert run_backtest(scenario(side), cfg(min_touches=4), "X").empty
