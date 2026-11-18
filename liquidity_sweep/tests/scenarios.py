"""Hand-built sweep scenarios on flat 5m bars. Not market data."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from config import config_for_timeframe
from synthetic import frame, session_index

BASE = 99.0
RES = 100.0
SUP = 98.0
PIVOTS = (10, 25, 40)


def flat(days=("2026-01-05",)) -> pd.DataFrame:
    """Flat bars: open = close = 99, high 99.2, low 98.8 (equal highs/lows are never pivots)."""
    idx = session_index(list(days))
    n = len(idx)
    return frame(
        idx, np.full(n, BASE), np.full(n, BASE + 0.2), np.full(n, BASE - 0.2), np.full(n, BASE)
    )


def set_bar(df: pd.DataFrame, i: int, o=None, h=None, lo=None, c=None) -> None:
    for col, val in (("open", o), ("high", h), ("low", lo), ("close", c)):
        if val is not None:
            df.iloc[i, df.columns.get_loc(col)] = val


def with_level(df: pd.DataFrame, side: str, pivots=PIVOTS, price=None) -> pd.DataFrame:
    """Add pivot highs at RES (side 'short') or pivot lows at SUP (side 'long')."""
    for i in pivots:
        if side == "short":
            set_bar(df, i, h=RES if price is None else price)
        else:
            set_bar(df, i, lo=SUP if price is None else price)
    return df


def sweep(df: pd.DataFrame, side: str, i: int, extreme=None, close=None) -> None:
    """A bar that wicks through the level and closes back inside."""
    if side == "short":
        set_bar(
            df, i, h=RES + 0.3 if extreme is None else extreme, c=99.5 if close is None else close
        )
    else:
        set_bar(
            df, i, lo=SUP - 0.3 if extreme is None else extreme, c=98.5 if close is None else close
        )


def scenario(side: str, sweep_at: int = 50, days=("2026-01-05",)) -> pd.DataFrame:
    df = with_level(flat(days), side)
    sweep(df, side, sweep_at)
    return df


def cfg(**kw):
    """5m config with the bias off (scenarios are too short for EMA(50))."""
    base = config_for_timeframe("5m", use_bias=False)
    return replace(base, **kw) if kw else base
