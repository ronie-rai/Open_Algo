"""Trade statistics and plain markdown tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

EXIT_REASONS = ("stop", "target", "time", "session_end", "end_of_data")


def equity_order(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades in the order their results became known (exit time, then entry, then symbol)."""
    if trades.empty:
        return trades
    return trades.sort_values(["exit_time", "entry_time", "symbol"], kind="stable")


def max_drawdown(r: pd.Series) -> float:
    """Largest fall of cumulative R from its running peak (peak starts at 0)."""
    if r.empty:
        return 0.0
    equity = np.concatenate([[0.0], r.cumsum().to_numpy()])
    return float((np.maximum.accumulate(equity) - equity).max())


def summarize(trades: pd.DataFrame, r_col: str = "r_net") -> dict:
    """Statistics for a set of trades. Wins are trades with positive net R."""
    count = len(trades)
    out = {
        "trades": count,
        "win_rate": np.nan,
        "avg_r_net": np.nan,
        "avg_r_gross": np.nan,
        "total_r": 0.0,
        "profit_factor": np.nan,
        "max_dd_r": 0.0,
        "longs": 0,
        "shorts": 0,
    }
    for reason in EXIT_REASONS:
        out[reason] = 0
    if count == 0:
        return out
    ordered = equity_order(trades)
    r = ordered[r_col]
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    out.update(
        win_rate=float((r > 0).mean()),
        avg_r_net=float(r.mean()),
        avg_r_gross=float(ordered["r_gross"].mean()),
        total_r=float(r.sum()),
        profit_factor=float(gains / losses) if losses > 0 else (np.inf if gains > 0 else np.nan),
        max_dd_r=max_drawdown(r),
        longs=int((ordered["side"] == "long").sum()),
        shorts=int((ordered["side"] == "short").sum()),
    )
    counts = ordered["exit_reason"].value_counts()
    for reason in EXIT_REASONS:
        out[reason] = int(counts.get(reason, 0))
    return out


def split_trades(trades: pd.DataFrame, split: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """In-sample / out-of-sample / full, by entry time."""
    if trades.empty:
        return {"IS": trades, "OOS": trades, "Full": trades}
    return {
        "IS": trades[trades["entry_time"] < split],
        "OOS": trades[trades["entry_time"] >= split],
        "Full": trades,
    }


def fmt(value, kind: str = "num") -> str:
    """Format one table cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    if kind == "pct":
        return f"{100 * value:.1f}%"
    if kind == "int":
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def summary_table(parts: dict[str, pd.DataFrame]) -> str:
    """IS / OOS / Full statistics as one markdown table."""
    headers = [
        "Sample",
        "Trades",
        "Win rate",
        "Avg R net",
        "Avg R gross",
        "Total R",
        "Profit factor",
        "Max DD (R)",
        "Longs",
        "Shorts",
        *EXIT_REASONS,
    ]
    rows = []
    for name, df in parts.items():
        s = summarize(df)
        rows.append(
            [
                name,
                fmt(s["trades"]),
                fmt(s["win_rate"], "pct"),
                fmt(s["avg_r_net"]),
                fmt(s["avg_r_gross"]),
                fmt(s["total_r"]),
                fmt(s["profit_factor"]),
                fmt(s["max_dd_r"]),
                fmt(s["longs"]),
                fmt(s["shorts"]),
                *[fmt(s[r]) for r in EXIT_REASONS],
            ]
        )
    return md_table(headers, rows)
