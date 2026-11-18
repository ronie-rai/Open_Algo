"""Run the liquidity sweep backtest on the NSE basket: baseline, no bias, min_touches=1.

Research only, not financial advice.

Usage (from the liquidity_sweep folder):
    uv run --no-project --python .venv python run.py [--timeframe 5m] [--atr-mult 0]
        [--symbols SBIN,TCS] [--refresh]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from backtest import TRADE_COLUMNS, net_r_at_costs, run_backtest
from config import BASKET, RECENT_LISTINGS, Config, config_for_timeframe
from data import load_basket, no_data_message, range_bounds, split_time
from metrics import equity_order, fmt, md_table, split_trades, summarize, summary_table
from resample import resample_ohlcv

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
DISCLAIMER = "Research only, not financial advice. Results are historical and include no tuning."


def parse_symbols(value: str | None) -> list[str]:
    if not value:
        return list(BASKET)
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def load_or_explain(symbols: list[str], refresh: bool) -> tuple[dict, dict, tuple[date, date]]:
    """Load the basket, or print how to provide data and exit with status 1."""
    print("Loading 5-minute data...")
    data, info, (start, end) = load_basket(symbols, refresh=refresh)
    if not data:
        print()
        print(no_data_message(start, end))
        sys.exit(1)
    return data, info, (start, end)


def bars_for(data5: pd.DataFrame, tf: str) -> pd.DataFrame:
    return data5 if tf == "5m" else resample_ohlcv(data5, tf)


def run_basket(
    data: dict[str, pd.DataFrame], cfg: Config, cache: dict | None = None
) -> pd.DataFrame:
    """Run one config on every symbol (one position at a time per symbol) and pool."""
    cache = cache if cache is not None else {}
    frames = []
    for sym, data5 in data.items():
        key = (sym, cfg.timeframe)
        if key not in cache:
            cache[key] = bars_for(data5, cfg.timeframe)
        trades = run_backtest(cache[key], cfg, symbol=sym, bias_source=data5)
        if not trades.empty:
            frames.append(trades)
    if not frames:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def short_history(bars: pd.DataFrame, start: date, end: date) -> bool:
    return bars.index[0].normalize() > range_bounds(start, end)[0] + pd.Timedelta(days=30)


def per_symbol_table(trades: pd.DataFrame, data: dict, start: date, end: date) -> str:
    rows = []
    for sym in data:
        t = trades[trades["symbol"] == sym] if not trades.empty else trades
        s = summarize(t)
        flags = []
        if sym in RECENT_LISTINGS:
            flags.append("recent listing")
        if short_history(data[sym], start, end):
            flags.append(f"data from {data[sym].index[0]:%Y-%m-%d}")
        rows.append(
            [
                sym,
                fmt(s["trades"]),
                fmt(s["win_rate"], "pct"),
                fmt(s["avg_r_net"]),
                fmt(s["total_r"]),
                ", ".join(flags),
            ]
        )
    return md_table(["Symbol", "Trades", "Win rate", "Avg R net", "Total R", "Note"], rows)


def plot_equity(
    results: dict[str, pd.DataFrame], split: pd.Timestamp, path: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, trades in results.items():
        if trades.empty:
            ax.plot([], [], label=f"{name} (no trades)")
            continue
        ordered = equity_order(trades)
        ax.plot(ordered["exit_time"].dt.tz_localize(None), ordered["r_net"].cumsum(), label=name)
    ax.axvline(split.tz_localize(None), linestyle="--", color="grey", label="IS / OOS split")
    ax.set_title(title)
    ax.set_ylabel("Cumulative net R (basket)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def first_trade_note(results: dict[str, pd.DataFrame]) -> list[str]:
    lines = []
    for name, t in results.items():
        if not t.empty:
            lines.append(f"- {name}: first entry {t['entry_time'].min():%Y-%m-%d %H:%M}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Liquidity sweep backtest on the NSE basket (research only)."
    )
    parser.add_argument("--timeframe", default="5m", choices=["5m", "15m", "1h"])
    parser.add_argument("--atr-mult", type=float, default=0.0)
    parser.add_argument(
        "--symbols", default=None, help="comma separated, default the 17-stock basket"
    )
    parser.add_argument("--refresh", action="store_true", help="download the data again")
    args = parser.parse_args(argv)

    symbols = parse_symbols(args.symbols)
    data, info, (start, end) = load_or_explain(symbols, args.refresh)
    base = config_for_timeframe(args.timeframe, atr_mult=args.atr_mult)
    split = split_time(start, end, base.is_frac)
    variants = {
        "baseline": base,
        "no_bias": replace(base, use_bias=False),
        "min_touches_1": replace(base, min_touches=1),
    }

    cache: dict = {}
    results = {name: run_basket(data, cfg, cache) for name, cfg in variants.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Liquidity sweep backtest, NSE basket, {args.timeframe}, atr_mult {args.atr_mult}",
        "",
        DISCLAIMER,
        "",
        f"- Data range: {start} to {end} (12 full months ending at the last complete month)",
        f"- In-sample / out-of-sample split (entry time): {split:%Y-%m-%d %H:%M} IST",
        f"- Bias: {'1h' if base.bias_tf == '1h' else 'daily'} EMA({base.bias_ema}); "
        "no trades until the EMA exists (no extra warm-up data is loaded)",
        f"- Costs per side: fee {base.fee} + slippage {base.slippage}; low-cost {base.low_fee} + {base.low_slippage}",
        f"- Symbols with data: {len(data)} of {len(symbols)}",
        "",
        "## Data",
        "",
    ]
    lines.append(
        md_table(
            ["Symbol", "Source", "First bar", "Last bar", "Bars"],
            [
                [
                    s,
                    info[s].source,
                    f"{data[s].index[0]:%Y-%m-%d}" if s in data else "-",
                    f"{data[s].index[-1]:%Y-%m-%d}" if s in data else "-",
                    str(len(data[s])) if s in data else "0",
                ]
                for s in symbols
            ],
        )
    )
    skipped = [s for s in symbols if s not in data]
    if skipped:
        lines += ["", "Skipped (no data): " + ", ".join(skipped)]
    lines += ["", "First entry per variant (shows the EMA warm-up):", *first_trade_note(results)]

    for name, trades in results.items():
        parts = split_trades(trades, split)
        print(f"\n== {name} ({args.timeframe}, atr_mult {args.atr_mult}) ==")
        table = summary_table(parts)
        print(table)
        low = {k: net_r_at_costs(v).mean() if len(v) else float("nan") for k, v in parts.items()}
        established = (
            trades[~trades["symbol"].isin(RECENT_LISTINGS)] if not trades.empty else trades
        )
        recent = trades[trades["symbol"].isin(RECENT_LISTINGS)] if not trades.empty else trades
        lines += [
            "",
            f"## {name}",
            "",
            table,
            "",
            "Avg R net at low costs: " + ", ".join(f"{k} {fmt(v)}" for k, v in low.items()),
            "",
            "Established symbols vs recent listings (full sample):",
            "",
            summary_table({"Established": established, "Recent listings": recent}),
            "",
            "Per symbol (full sample):",
            "",
            per_symbol_table(trades, data, start, end),
        ]

    all_trades = []
    for name, trades in results.items():
        t = trades.copy()
        t.insert(0, "variant", name)
        all_trades.append(t)
    out = pd.concat(all_trades, ignore_index=True)
    out = out[["variant", *TRADE_COLUMNS]]
    out.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    plot_equity(
        results, split, OUTPUT_DIR / "equity.png", f"Liquidity sweep, NSE basket, {args.timeframe}"
    )
    print(f"\nWrote {OUTPUT_DIR / 'report.md'}, trades.csv and equity.png")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
