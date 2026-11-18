"""Every parameter of the liquidity sweep backtest, in one frozen dataclass."""

from __future__ import annotations

from dataclasses import dataclass, replace

# Basket of NSE equities (OpenAlgo symbols).
BASKET = (
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
)

# Recent listings with short histories; reported separately.
RECENT_LISTINGS = frozenset({"LENSKART", "GROWW", "TONBO"})

# Minutes per bar for each supported timeframe. "D" is one bar per session.
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}

# Lookback measured in sessions instead of 24/7 calendar days.
LOOKBACK_SESSIONS = 3


@dataclass(frozen=True)
class Config:
    """Backtest parameters. Times are IST wall clock, "HH:MM"."""

    timeframe: str = "5m"
    pivot_n: int = 5
    level_tol: float = 0.0015
    level_lookback: int = 225
    min_touches: int = 3
    use_bias: bool = True
    bias_tf: str = "1h"
    bias_ema: int = 50
    stop_buffer: float = 0.0005
    atr_mult: float = 0.0
    atr_period: int = 14
    rr: float = 2.0
    max_bars: int = 48
    fee: float = 0.0004
    slippage: float = 0.0002
    low_fee: float = 0.0002
    low_slippage: float = 0.0
    is_frac: float = 0.7
    # NSE session settings.
    exchange: str = "NSE"
    tz: str = "Asia/Kolkata"
    session_open: str = "09:15"
    session_close: str = "15:30"
    last_entry: str = "15:00"  # no entry on a bar opening at or after this time
    square_off: str = "15:15"  # close at the last bar ending at or before this time


def hhmm_to_minutes(value: str) -> int:
    """Convert "HH:MM" to minutes after midnight."""
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def bars_per_session(tf: str, cfg: Config | None = None) -> int:
    """Number of session-anchored bars of `tf` in one full session (short last bar counts)."""
    cfg = cfg or Config()
    span = hhmm_to_minutes(cfg.session_close) - hhmm_to_minutes(cfg.session_open)
    minutes = TF_MINUTES[tf]
    return -(-span // minutes)


def config_for_timeframe(tf: str, **overrides) -> Config:
    """Config for a timeframe: 3-session lookback and the matching bias timeframe.

    5m -> lookback 225, bias 1h; 15m -> 75, bias 1h; 1h -> 21, bias daily ("D").
    """
    if tf not in ("5m", "15m", "1h"):
        raise ValueError(f"Unsupported timeframe {tf!r}; use 5m, 15m or 1h")
    base = Config()
    cfg = replace(
        base,
        timeframe=tf,
        level_lookback=LOOKBACK_SESSIONS * bars_per_session(tf, base),
        bias_tf="D" if tf == "1h" else "1h",
    )
    return replace(cfg, **overrides) if overrides else cfg
