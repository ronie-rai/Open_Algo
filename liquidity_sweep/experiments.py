"""Grid of 9 runs: timeframe {5m, 15m, 1h} x atr_mult {0, 0.5, 1.0}, bias on, min_touches 3.

No other parameter is varied. Research only, not financial advice.

Usage (from the liquidity_sweep folder):
    uv run --no-project --python .venv python experiments.py [--symbols SBIN,TCS] [--refresh]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from backtest import net_r_at_costs
from config import config_for_timeframe
from data import split_time
from metrics import equity_order, fmt, md_table, split_trades, summarize
from run import DISCLAIMER, OUTPUT_DIR, load_or_explain, parse_symbols, run_basket

TIMEFRAMES = ("5m", "15m", "1h")
ATR_MULTS = (0.0, 0.5, 1.0)

NOTE = (
    "9 configurations were tested, so the best one is likely flattered by chance; "
    "the out-of-sample results are what matter, and rows with few trades mean nothing."
)


def mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def row_for(tf: str, atr_mult: float, trades: pd.DataFrame, split: pd.Timestamp) -> list[str]:
    parts = split_trades(trades, split)
    full = summarize(parts["Full"])
    if trades.empty:
        risk_pct = cost_r = float("nan")
    else:
        risk = (trades["entry_price"] - trades["stop"]).abs()
        risk_pct = float(np.median(100 * risk / trades["entry_price"]))
        cost_r = float(np.median(trades["r_gross"] - trades["r_net"]))
    return [
        tf,
        f"{atr_mult:g}",
        fmt(len(parts["IS"])),
        fmt(len(parts["OOS"])),
        fmt(full["win_rate"], "pct"),
        fmt(full["avg_r_gross"]),
        fmt(mean_or_nan(parts["IS"]["r_net"]) if len(parts["IS"]) else float("nan")),
        fmt(mean_or_nan(parts["OOS"]["r_net"]) if len(parts["OOS"]) else float("nan")),
        fmt(full["avg_r_net"]),
        fmt(mean_or_nan(net_r_at_costs(parts["Full"]))),
        fmt(mean_or_nan(net_r_at_costs(parts["OOS"]))),
        fmt(full["profit_factor"]),
        fmt(risk_pct) + "%" if not np.isnan(risk_pct) else "-",
        fmt(cost_r),
    ]


HEADERS = [
    "Timeframe",
    "atr_mult",
    "Trades IS",
    "Trades OOS",
    "Win rate (full)",
    "Avg R gross (full)",
    "Avg R net IS",
    "Avg R net OOS",
    "Avg R net full",
    "Avg R net low-cost full",
    "Avg R net low-cost OOS",
    "Profit factor (full)",
    "Median risk % of price",
    "Median cost (R)",
]


def plot_grid(results: dict, split: pd.Timestamp, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TIMEFRAMES), figsize=(16, 5), sharey=False)
    for ax, tf in zip(axes, TIMEFRAMES, strict=True):
        for atr in ATR_MULTS:
            trades = results[(tf, atr)]
            label = f"atr_mult {atr}"
            if trades.empty:
                ax.plot([], [], label=f"{label} (no trades)")
                continue
            ordered = equity_order(trades)
            ax.plot(
                ordered["exit_time"].dt.tz_localize(None), ordered["r_net"].cumsum(), label=label
            )
        ax.axvline(split.tz_localize(None), linestyle="--", color="grey", label="IS / OOS split")
        ax.set_title(f"{tf} (bias {'daily' if tf == '1h' else '1h'})")
        ax.set_ylabel("Cumulative net R (basket)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Liquidity sweep experiment grid (research only).")
    parser.add_argument(
        "--symbols", default=None, help="comma separated, default the 17-stock basket"
    )
    parser.add_argument("--refresh", action="store_true", help="download the data again")
    args = parser.parse_args(argv)

    data, _info, (start, end) = load_or_explain(parse_symbols(args.symbols), args.refresh)
    split = split_time(start, end, config_for_timeframe("5m").is_frac)
    cache: dict = {}
    results = {}
    rows = []
    for tf in TIMEFRAMES:
        for atr in ATR_MULTS:
            cfg = config_for_timeframe(tf, atr_mult=atr, use_bias=True, min_touches=3)
            trades = run_basket(data, cfg, cache)
            results[(tf, atr)] = trades
            rows.append(row_for(tf, atr, trades, split))

    table = md_table(HEADERS, rows)
    text = "\n".join(
        [
            "# Liquidity sweep experiments, NSE basket",
            "",
            DISCLAIMER,
            "",
            f"Data {start} to {end}, {len(data)} symbols, IS / OOS split at {split:%Y-%m-%d %H:%M} IST (entry time).",
            "",
            table,
            "",
            NOTE,
            "",
        ]
    )
    print(table)
    print()
    print(NOTE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "experiments.md").write_text(text)
    plot_grid(results, split, OUTPUT_DIR / "experiments_equity.png")
    print(f"\nWrote {OUTPUT_DIR / 'experiments.md'} and experiments_equity.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
