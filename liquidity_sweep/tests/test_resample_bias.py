"""Session-aware resampling and the higher-timeframe bias."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from config import config_for_timeframe
from resample import resample_ohlcv
from strategy import bias_direction, build_bias_bars, seeded_ema
from synthetic import frame, random_sessions, session_index

TZ = "Asia/Kolkata"


def ramp(days, start=100.0, step=0.01):
    idx = session_index(days)
    c = start + step * np.arange(len(idx))
    o = c - step / 2
    return frame(idx, o, c + 0.05, o - 0.05, c, np.arange(len(idx)) + 1.0)


def ts(s):
    return pd.Timestamp(s, tz=TZ)


def test_15m_aggregation_values():
    df = ramp(["2026-01-05"])
    out = resample_ohlcv(df, "15m")
    first = df.iloc[0:3]
    row = out.iloc[0]
    assert out.index[0] == ts("2026-01-05 09:15")
    assert row["open"] == first["open"].iloc[0]
    assert row["high"] == first["high"].max()
    assert row["low"] == first["low"].min()
    assert row["close"] == first["close"].iloc[-1]
    assert row["volume"] == first["volume"].sum()
    assert len(out) == 25


def test_1h_bars_anchored_at_0915_with_short_last_bar():
    df = ramp(["2026-01-05", "2026-01-06"])
    out = resample_ohlcv(df, "1h")
    day1 = [t.strftime("%H:%M") for t in out.index if t.date() == pd.Timestamp("2026-01-05").date()]
    assert day1 == ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]
    last = out.loc[ts("2026-01-05 15:15")]
    src = df.loc[ts("2026-01-05 15:15") : ts("2026-01-05 15:25")]
    assert len(src) == 3
    assert last["close"] == src["close"].iloc[-1]
    assert last["volume"] == src["volume"].sum()


def test_no_bar_spans_two_sessions():
    df = ramp(["2026-01-05", "2026-01-06"])
    for tf in ("15m", "1h"):
        out = resample_ohlcv(df, tf)
        day2_open = out.loc[ts("2026-01-06 09:15")]
        assert day2_open["open"] == df.loc[ts("2026-01-06 09:15"), "open"]
        last1 = out[out.index.date == pd.Timestamp("2026-01-05").date()].iloc[-1]
        assert last1["close"] == df.loc[ts("2026-01-05 15:25"), "close"]


def test_partial_bars_dropped_at_both_ends_and_on_holes():
    df = ramp(["2026-01-05", "2026-01-06"])
    trimmed = df.loc[ts("2026-01-05 09:20") : ts("2026-01-06 10:05")]
    out15 = resample_ohlcv(trimmed, "15m")
    assert out15.index[0] == ts("2026-01-05 09:30")  # 09:15 bar missing its first 5m bar
    assert out15.index[-1] == ts("2026-01-06 09:45")  # 10:00 bar incomplete at the end
    out1h = resample_ohlcv(trimmed, "1h")
    assert out1h.index[0] == ts("2026-01-05 10:15")
    assert out1h.index[-1] == ts("2026-01-05 15:15")  # day-2 09:15 bar is incomplete
    holed = df.drop(ts("2026-01-05 11:40"))
    out_h = resample_ohlcv(holed, "1h")
    assert ts("2026-01-05 11:15") not in out_h.index
    assert ts("2026-01-05 10:15") in out_h.index
    assert ts("2026-01-05 11:30") not in resample_ohlcv(holed, "15m").index


def test_daily_bars_need_session_open_and_close():
    df = ramp(["2026-01-05", "2026-01-06", "2026-01-07"])
    cut = df.loc[ts("2026-01-05 10:00") : ts("2026-01-07 14:00")]
    daily = resample_ohlcv(cut, "D")
    assert list(daily.index) == [ts("2026-01-06 09:15")]
    assert daily.iloc[0]["close"] == df.loc[ts("2026-01-06 15:25"), "close"]


def test_1h_bias_bar_available_only_after_it_closes():
    c = replace(config_for_timeframe("5m"), bias_ema=3)
    df = ramp(["2026-01-05", "2026-01-06"])
    bias = build_bias_bars(df, c)
    assert bias.loc[ts("2026-01-05 09:15"), "available_at"] == ts("2026-01-05 10:15")
    assert bias.loc[ts("2026-01-05 15:15"), "available_at"] == ts("2026-01-05 15:30")


def test_1h_bias_ignores_the_hour_in_progress():
    """Day 1 rises (bias long). Day 2's first hour collapses: the bias turns short
    only from 10:15, never on a 5m bar inside the collapsing hour."""
    days = ["2026-01-05", "2026-01-06"]
    df = ramp(days)
    day2 = df.index.date == pd.Timestamp("2026-01-06").date()
    first_hour = day2 & (df.index < ts("2026-01-06 10:15"))
    drop = np.linspace(0, -20, first_hour.sum())
    for col in ("open", "high", "low", "close"):
        df.loc[first_hour, col] = df.loc[first_hour, col] + drop
    later = day2 & ~first_hour
    for col in ("open", "high", "low", "close"):
        df.loc[later, col] = df.loc[later, col] - 20
    c = replace(config_for_timeframe("5m"), bias_ema=3)
    b = pd.Series(bias_direction(df.index, build_bias_bars(df, c)), index=df.index)
    assert (b[first_hour] == 1).all()
    assert b[ts("2026-01-06 10:15")] == -1


def test_daily_bias_uses_only_closed_sessions():
    days = list(pd.bdate_range("2026-01-05", periods=6).strftime("%Y-%m-%d"))
    df = ramp(days, step=0.01)
    last_day = df.index.date == pd.Timestamp(days[-2]).date()
    for col in ("open", "high", "low", "close"):
        df.loc[last_day, col] = df.loc[last_day, col] - 30
    nxt = df.index.date == pd.Timestamp(days[-1]).date()
    for col in ("open", "high", "low", "close"):
        df.loc[nxt, col] = df.loc[nxt, col] - 30
    c = replace(config_for_timeframe("1h"), bias_ema=3)
    b = pd.Series(bias_direction(df.index, build_bias_bars(df, c)), index=df.index)
    assert (b[last_day] == 1).all()  # the collapsing day still sees yesterday's close
    assert (b[nxt] == -1).all()  # usable from the next session
    first_day = df.index.date == pd.Timestamp(days[0]).date()
    assert (b[first_day] == 0).all()  # EMA(3) does not exist yet


def test_no_bias_until_ema_exists():
    c = config_for_timeframe("5m")  # EMA(50) on 1h bars: 50 hours = 7+ sessions
    df = random_sessions(sessions=10)
    bias = build_bias_bars(df, c)
    b = bias_direction(df.index, bias)
    first_ok = bias["available_at"][~bias["ema"].isna()].iloc[0]
    assert (b[df.index < first_ok] == 0).all()
    assert (b[df.index >= first_ok] != 0).any()


def test_seeded_ema():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = seeded_ema(v, 3)
    assert np.isnan(e[:2]).all()
    assert e[2] == pytest.approx(2.0)
    assert e[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)
    assert e[4] == pytest.approx(0.5 * 5 + 0.5 * e[3])
