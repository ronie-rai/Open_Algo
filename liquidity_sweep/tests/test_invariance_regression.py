"""Prefix invariance and the saved regression fixture."""

import pandas as pd
import pytest

from backtest import run_backtest
from config import config_for_timeframe
from regression_fixture import FIXTURE, regression_trades
from resample import resample_ohlcv
from synthetic import random_sessions

DATA = random_sessions(sessions=60, seed=7)


def _run(data5, tf, **kw):
    bars = data5 if tf == "5m" else resample_ohlcv(data5, tf)
    return run_backtest(bars, config_for_timeframe(tf, **kw), "SYN", bias_source=data5), bars


def _key(df):
    cols = ["side", "entry_time", "entry_price", "stop", "exit_time", "exit_price", "exit_reason"]
    return sorted(tuple(r) for r in df[cols].round(8).itertuples(index=False))


@pytest.mark.parametrize(
    "tf,kw",
    [
        ("5m", {}),
        ("5m", {"use_bias": False}),
        ("5m", {"min_touches": 1, "atr_mult": 0.5}),
        ("15m", {"min_touches": 1}),
        ("1h", {"min_touches": 1, "use_bias": False}),
    ],
)
def test_prefix_invariance(tf, kw):
    full, _ = _run(DATA, tf, **kw)
    compared = 0
    for frac in (0.35, 0.5, 0.63, 0.81, 0.92):
        cut = DATA.index[int(len(DATA) * frac)] + pd.Timedelta(minutes=2)  # mid-bar of higher tfs
        part_data = DATA[DATA.index < cut]
        part, bars = _run(part_data, tf, **kw)
        if bars.empty:
            continue
        last = bars.index[-1]
        closed = part[part["exit_reason"] != "end_of_data"]
        expected = full[full["exit_time"] <= last]
        assert _key(closed) == _key(expected)
        compared += len(closed)
    assert compared > 0, "the test compared no trades"


def test_regression_fixture():
    got = regression_trades()
    want = pd.read_csv(FIXTURE)
    assert len(got) == len(want) and len(want) > 0
    for col in ("symbol", "side", "exit_reason"):
        assert list(got[col]) == list(want[col])
    for col in ("entry_time", "exit_time"):
        assert list(got[col].astype(str)) == list(want[col].astype(str))
    for col in (
        "entry_price",
        "stop",
        "target",
        "exit_price",
        "r_gross",
        "r_net",
        "level",
        "touches",
    ):
        pd.testing.assert_series_equal(
            got[col].astype(float).reset_index(drop=True),
            want[col].astype(float).reset_index(drop=True),
            rtol=1e-9,
            check_names=False,
        )
