"""The deterministic regression dataset and its trades under the default config.

Regenerate the saved fixture only when a rule change is intended:
    uv run --no-project --python .venv python tests/regression_fixture.py --write
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

from backtest import run_backtest  # noqa: E402
from config import Config  # noqa: E402
from synthetic import random_sessions  # noqa: E402

FIXTURE = HERE / "fixtures" / "regression_trades.csv"


def regression_trades():
    return run_backtest(random_sessions(sessions=60, seed=7), Config(), "SYN")


if __name__ == "__main__":
    if "--write" in sys.argv:
        trades = regression_trades()
        trades.to_csv(FIXTURE, index=False)
        print(f"wrote {len(trades)} trades to {FIXTURE}")
