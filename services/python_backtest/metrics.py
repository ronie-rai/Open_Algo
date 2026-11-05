"""The numbers a trader reads off a finished backtest. Pure functions, no I/O."""

from __future__ import annotations

import math

from .sessions import day_to_date

MAX_CURVE_POINTS = 2000


def _r(value, digits=2):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return round(float(value), digits)


def compute(
    *,
    capital: float,
    trades: list[dict],
    equity_samples: list[tuple[float, float, bool]],
    day_ends: list[tuple[int, float]],
    total_costs: float,
) -> dict:
    """Summary metrics, the equity curve and the daily P&L of a run.

    ``equity_samples`` are (epoch, equity, holding a position) every few
    minutes; ``day_ends`` are (IST day number, equity at the close).
    """
    capital = float(capital)
    final_equity = equity_samples[-1][1] if equity_samples else capital
    net = final_equity - capital

    pnls = [float(t["pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins), -sum(losses)

    # Drawdown over the sampled curve. The samples fall every five minutes of
    # each session, the close included, plus one after the final square-off.
    points = [(t, e) for t, e, _ in equity_samples]
    peak = capital
    max_dd, max_dd_pct, max_dd_time = 0.0, 0.0, None
    curve = []
    for t, e in points:
        peak = max(peak, e)
        dd = e - peak
        if dd < max_dd:
            max_dd, max_dd_time = dd, t
            max_dd_pct = dd / peak * 100 if peak > 0 else 0.0
        curve.append({"time": int(t * 1000), "equity": round(e, 2), "drawdown": round(dd, 2)})

    daily = []
    prev = capital
    returns = []
    for d, e in day_ends:
        pnl = e - prev
        daily.append(
            {"date": day_to_date(d).isoformat(), "pnl": round(pnl, 2), "equity": round(e, 2)}
        )
        if prev > 0:
            returns.append(pnl / prev)
        prev = e

    sharpe = None
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        if var > 0:
            sharpe = mean / math.sqrt(var) * math.sqrt(252)

    in_market = sum(1 for _, _, held in equity_samples if held)
    exposure = in_market / len(equity_samples) * 100 if equity_samples else 0.0
    best = max(daily, key=lambda x: x["pnl"]) if daily else None
    worst = min(daily, key=lambda x: x["pnl"]) if daily else None

    metrics = {
        "starting_capital": _r(capital),
        "final_equity": _r(final_equity),
        "net_pnl": _r(net),
        "gross_pnl": _r(net + total_costs),
        "total_costs": _r(total_costs),
        "return_pct": _r(net / capital * 100 if capital else 0.0),
        "trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": _r(len(wins) / len(trades) * 100) if trades else None,
        "avg_win": _r(gross_win / len(wins)) if wins else None,
        "avg_loss": _r(-gross_loss / len(losses)) if losses else None,
        "profit_factor": _r(gross_win / gross_loss) if gross_loss > 0 else None,
        "expectancy": _r(sum(pnls) / len(pnls)) if pnls else None,
        "largest_win": _r(max(wins)) if wins else None,
        "largest_loss": _r(min(losses)) if losses else None,
        "max_drawdown": _r(abs(max_dd)),
        "max_drawdown_pct": _r(abs(max_dd_pct)),
        "max_drawdown_time": int(max_dd_time * 1000) if max_dd_time is not None else None,
        "sharpe": _r(sharpe),
        "exposure_pct": _r(exposure),
        "trading_days": len(day_ends),
        "best_day": best,
        "worst_day": worst,
    }
    return {
        "metrics": metrics,
        "equity_curve": downsample(curve, MAX_CURVE_POINTS),
        "daily": daily,
    }


def downsample(points: list[dict], limit: int) -> list[dict]:
    """At most `limit` points, keeping the first, the last and the deepest drawdown."""
    if len(points) <= limit:
        return points
    keep = {0, len(points) - 1}
    trough = min(range(len(points)), key=lambda i: points[i]["drawdown"])
    keep.add(trough)
    step = (len(points) - 1) / (limit - 3)
    keep.update(int(round(i * step)) for i in range(limit - 3))
    return [points[i] for i in sorted(keep)][:limit]
