"""Smoke tests of run.py and experiments.py on synthetic data, written to a temp folder."""

from datetime import date

import pandas as pd

import experiments
import run
from backtest import TRADE_COLUMNS
from data import SymbolData
from synthetic import random_sessions

START, END = date(2025, 11, 1), date(2026, 10, 31)


def fake_basket(symbols, refresh):
    data = {
        "SBIN": random_sessions(60, seed=7, start="2025-11-03"),
        "TONBO": random_sessions(30, seed=8, start="2026-03-02"),
    }
    info = {
        s: SymbolData(s, data.get(s), "synthetic test data" if s in data else "none")
        for s in symbols
    }
    return {s: d for s, d in data.items() if s in symbols}, info, (START, END)


def test_run_writes_report_trades_and_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "load_basket", fake_basket)
    monkeypatch.setattr(run, "OUTPUT_DIR", tmp_path)
    assert run.main(["--symbols", "SBIN,TONBO,TCS"]) == 0
    trades = pd.read_csv(tmp_path / "trades.csv")
    assert list(trades.columns) == ["variant", *TRADE_COLUMNS]
    assert set(trades["variant"]) <= {"baseline", "no_bias", "min_touches_1"}
    report = (tmp_path / "report.md").read_text()
    assert "recent listing" in report and "Skipped (no data): TCS" in report
    assert "not financial advice" in report
    assert (tmp_path / "equity.png").stat().st_size > 0


def test_experiments_writes_table_and_chart(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "load_basket", fake_basket)
    monkeypatch.setattr(experiments, "OUTPUT_DIR", tmp_path)
    assert experiments.main(["--symbols", "SBIN"]) == 0
    text = (tmp_path / "experiments.md").read_text()
    assert text.count("\n| 5m") == 3 and text.count("\n| 15m") == 3 and text.count("\n| 1h") == 3
    assert "9 configurations were tested" in text
    assert (tmp_path / "experiments_equity.png").stat().st_size > 0
