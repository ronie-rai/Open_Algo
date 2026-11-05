"""Run one strategy through one backtest. A subprocess entry point:

    python -u -m services.python_backtest.runner --config <run_dir>/config.json

The strategy's own output goes to stdout, which the parent points at
``run.log``. Progress lands in ``progress.json`` while it runs, and everything
a trader reads afterwards in ``result.json``. The OpenAlgo API key arrives in
the environment, never in a file, and is taken out of it before the strategy
starts: only the data loader holds it. ``guards`` keeps the strategy away from
the app itself (see that module for what is and is not covered).
"""

from __future__ import annotations

import argparse
import atexit
import collections
import io
import json
import logging
import os
import runpy
import sys
import time
from datetime import date, datetime
from pathlib import Path

# pandas and numpy first, while datetime and time are still the real ones.
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

from . import clock, guards
from .broker import Broker, Costs
from .client import BacktestClient
from .data import FileProvider, MarketData, OpenAlgoProvider, describe_sources
from .engine import BacktestFinished, Engine, write_json_atomic
from .metrics import compute
from .sessions import IST, configure_square_off, ist_datetime, parse_day

logger = logging.getLogger("python_backtest")

DEFAULT_LOG_LIMIT_BYTES = 8 * 1024 * 1024
LOG_TAIL_LINES = 2000


# =============================================================================
# OUTPUT
# =============================================================================
class CappedWriter(io.TextIOBase):
    """Passes output through up to a limit, then keeps only the last lines.

    A strategy that prints every cycle prints thousands of times a simulated
    day; a year of that would be a log nobody can open. The head is written as
    it comes, the tail is written at the end, and a line in between says how
    much was left out.
    """

    def __init__(self, target, limit_bytes: int):
        self.target = target
        self.limit = int(limit_bytes)
        self.written = 0
        self.dropped_lines = 0
        self.tail = collections.deque(maxlen=LOG_TAIL_LINES)
        self._partial = ""

    def writable(self):
        return True

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        if self.written < self.limit:
            self.written += len(text)
            self.target.write(text)
            return len(text)
        data = self._partial + text
        lines = data.split("\n")
        self._partial = lines.pop()
        for line in lines:
            if len(self.tail) == self.tail.maxlen:
                self.dropped_lines += 1
            self.tail.append(line)
        return len(text)

    def flush(self):
        try:
            self.target.flush()
        except (OSError, ValueError):
            pass

    def finish(self):
        if self.tail or self._partial:
            if self.dropped_lines:
                self.target.write(
                    f"\n[backtest] {self.dropped_lines} lines were left out to keep this log "
                    f"small. The last {len(self.tail)} lines follow.\n"
                )
            for line in self.tail:
                self.target.write(line + "\n")
            if self._partial:
                self.target.write(self._partial + "\n")
            self.tail.clear()
            self._partial = ""
        self.flush()


# =============================================================================
# SAFETY
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DUMMY_API_KEY = "backtest-no-real-key"
_secret = {"api_key": ""}  # the real key, for the data loader only


def take_api_key() -> None:
    """Keep the real API key here and leave the strategy a stand-in.

    A strategy reads OPENALGO_API_KEY to build its client, which the backtest
    replaces anyway, so a stand-in keeps every script that checks for the key
    working while the real one stays out of the strategy's reach.
    """
    _secret["api_key"] = os.environ.get("OPENALGO_API_KEY", "")
    os.environ["OPENALGO_API_KEY"] = DUMMY_API_KEY


class _AllowDataCalls:
    """The real clock and a way through to the app, for the data loader only."""

    def __enter__(self):
        self._clock = clock.real_time()
        self._clock.__enter__()
        self._allow = guards.allow_app()
        self._allow.__enter__()
        return self

    def __exit__(self, *exc):
        self._allow.__exit__(*exc)
        return self._clock.__exit__(*exc)


def _app_addresses(config: dict) -> tuple[list[str], list[int]]:
    """The URLs and local ports at which this process could reach OpenAlgo."""
    urls = [
        config.get("host"),
        os.getenv("HOST_SERVER"),
        os.getenv("MCP_LOOPBACK_URL"),
        os.getenv("WEBSOCKET_URL"),
    ]
    ports = []
    for name, default in (("FLASK_PORT", 5000), ("WEBSOCKET_PORT", 8765), ("ZMQ_PORT", 5555)):
        try:
            ports.append(int(os.getenv(name) or default))
        except ValueError:
            ports.append(default)
    return [u for u in urls if u], ports


def apply_resource_limits() -> None:
    """The same limits a live strategy gets (POSIX only), read from the environment."""
    try:
        import resource
    except ImportError:
        return

    def limit(name, value):
        try:
            which = getattr(resource, name)
            _soft, hard = resource.getrlimit(which)
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError, AttributeError):
            pass

    mb = int(os.environ.get("OPENALGO_STRATEGY_MEM_MB") or 0)
    if mb:
        limit("RLIMIT_AS", mb * 1024 * 1024)
    cpu = int(os.environ.get("OPENALGO_STRATEGY_CPU_SEC") or 0)
    if cpu:
        limit("RLIMIT_CPU", cpu)
    files = int(os.environ.get("OPENALGO_STRATEGY_NOFILE") or 0)
    if files:
        limit("RLIMIT_NOFILE", files)


# =============================================================================
# THE RUN
# =============================================================================
def _today_ist() -> date:
    return datetime.now(IST).date()


def build_provider(config: dict):
    data_dir = config.get("data_dir")
    if data_dir:
        return FileProvider(data_dir)
    from openalgo import api as real_api  # the real SDK, before it is replaced

    host = (config.get("host") or os.getenv("HOST_SERVER") or "http://127.0.0.1:5000").rstrip("/")

    def make_client():
        return real_api(api_key=_secret["api_key"], host=host, timeout=120.0)

    return OpenAlgoProvider(
        make_client, config.get("cache_dir"), _today_ist(), real_time=_AllowDataCalls
    )


def install_fake_openalgo(factory) -> None:
    """``from openalgo import api`` returns the backtest client; the rest of the SDK stays real."""
    try:
        import openalgo
    except ImportError:
        import types

        openalgo = types.ModuleType("openalgo")
        sys.modules["openalgo"] = openalgo
    openalgo.api = factory


def run(config: dict) -> dict:
    """Run the backtest a config describes and return the result document."""
    run_dir = Path(config["run_dir"])
    script = Path(config["script"])
    start = parse_day(config["start_date"])
    end = parse_day(config["end_date"])
    exchange = str(config.get("exchange") or "NSE").upper()
    capital = float(config.get("capital") or 100000)
    costs = Costs(
        brokerage_per_order=float(config.get("brokerage_per_order", 20.0)),
        brokerage_pct=float(config.get("brokerage_pct", 0.03)),
        stt_sell_pct=float(config.get("stt_sell_pct", 0.025)),
        other_charges_pct=float(config.get("other_charges_pct", 0.005)),
        slippage_bps=float(config.get("slippage_bps", 2.0)),
    )
    wall_began = time.monotonic()
    configure_square_off(config.get("square_off_times") or {})

    provider = build_provider(config)
    market = MarketData(
        provider,
        start,
        end,
        warmup_days=int(config.get("warmup_days", 15)),
        daily_warmup_days=int(config.get("daily_warmup_days", 120)),
    )
    holder = {}
    broker = Broker(market, capital, costs, now_fn=lambda: holder["engine"].now)
    engine = Engine(
        broker=broker,
        market=market,
        start=start,
        end=end,
        exchange=exchange,
        run_dir=run_dir,
        max_wall_seconds=float(config.get("max_wall_seconds", 3600)),
    )
    holder["engine"] = engine

    def runaway():
        # The strategy swallows BacktestFinished in a bare except and loops on.
        # Its run is over either way: save what it did and leave.
        clock.uninstall()
        logger.info("The strategy kept running after the test period ended; it was stopped")
        _write_and_exit(run_dir, build_result(config, engine, broker, market, provider, wall_began))

    engine.on_runaway = runaway

    def factory(*args, **kwargs):
        api_key = kwargs.get("api_key", args[0] if args else None)
        return BacktestClient(engine, market, broker, provider, api_key=api_key)

    os.environ["OPENALGO_BACKTEST"] = "1"
    for key, value in (config.get("env") or {}).items():
        os.environ[str(key)] = str(value)

    install_fake_openalgo(factory)
    urls, ports = _app_addresses(config)
    guards.configure_network(urls, ports)
    guards.install_network_guard()
    guards.install_process_guard()
    source_dir = Path(config["source_dir"]) if config.get("source_dir") else None
    guards.install_import_guard(REPO_ROOT, script.parent, source_dir)

    logger.info(
        f"Backtest of {script.name} from {start} to {end} on {exchange}, capital {capital:,.0f}, "
        f"brokerage {costs.brokerage_per_order:g} per order, slippage {costs.slippage_bps:g} bps"
    )
    sys.argv = [str(script)]
    outcome, message = None, ""
    clock.install(engine)
    try:
        runpy.run_path(str(script), run_name="__main__")
        outcome = "completed"
        message = "The strategy finished on its own before the end of the test period."
    except BacktestFinished:
        pass
    except SystemExit as exc:
        code = exc.code
        if engine.finished:
            pass
        elif code in (None, 0):
            outcome = "completed"
            message = "The strategy stopped on its own before the end of the test period."
        else:
            outcome = "failed"
            message = (
                "The strategy exited with an error before the backtest finished. "
                "The log below shows what it printed."
            )
    except KeyboardInterrupt:
        outcome, message = "stopped", "The backtest was interrupted."
    except Exception as exc:
        clock.uninstall()
        logger.exception("The strategy raised an error")
        outcome = "failed"
        message = (
            f"The strategy stopped with an error: {str(exc)[:300]}. The log below has the details."
        )
    finally:
        clock.uninstall()

    if not engine.finished:
        sim_end = engine.now
        try:
            engine.finish(outcome or "completed", message)
        except BacktestFinished:
            pass
        engine.message = message
        stopped_at = ist_datetime(sim_end)
        if outcome == "completed":
            engine.message = f"{message} It stopped at {stopped_at:%d %b %Y %H:%M}."

    return build_result(config, engine, broker, market, provider, wall_began)


def build_result(config, engine, broker, market, provider, wall_began) -> dict:
    report = compute(
        capital=broker.capital,
        trades=broker.trades,
        equity_samples=engine.equity_samples,
        day_ends=engine.day_ends,
        total_costs=broker.total_costs,
    )
    trades = [
        {
            **t,
            "entry_time": int(t["entry_time"] * 1000),
            "exit_time": int(t["exit_time"] * 1000),
        }
        for t in broker.trades
    ]
    orders = broker.orders_report()
    for o in orders:
        o["placed_at"] = int(o["placed_at"] * 1000)
        o["filled_at"] = int(o["filled_at"] * 1000) if o["filled_at"] else None
        o.pop("day", None)

    return {
        "status": engine.outcome,
        "message": engine.message,
        "config": {k: config.get(k) for k in CONFIG_ECHO},
        **report,
        "trades": trades,
        "orders": orders[-5000:],
        "orders_total": len(orders),
        "data": {
            "symbols": market.summary(),
            "sources": describe_sources(provider),
            "load_seconds": round(market.load_seconds, 1),
        },
        "simulated_until": f"{ist_datetime(engine.now):%Y-%m-%d %H:%M}",
        "sleep_calls": engine.sleep_calls,
        "wall_seconds": round(time.monotonic() - wall_began, 1),
        "finished_at": time.time(),
    }


CONFIG_ECHO = (
    "strategy_id",
    "strategy_name",
    "file_name",
    "start_date",
    "end_date",
    "exchange",
    "capital",
    "brokerage_per_order",
    "brokerage_pct",
    "stt_sell_pct",
    "other_charges_pct",
    "slippage_bps",
)


def _write_and_exit(run_dir: Path, result: dict) -> None:
    write_json_atomic(run_dir / "result.json", result)
    logger.info(f"Backtest {result.get('status')}. {result.get('message') or ''}".strip())
    # The strategy's own exit handlers, so files it keeps open are flushed and
    # closed as they would be at a normal exit. After the result, so a handler
    # that hangs or fails cannot cost it.
    try:
        clock.uninstall()
        atexit._run_exitfuncs()
    except BaseException:
        logger.info("An exit handler of the strategy failed; its results are saved")
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, CappedWriter):
            stream.finish()
        try:
            stream.flush()
        except (OSError, ValueError, AttributeError):
            pass
    for stream in (sys.__stdout__, sys.__stderr__):
        try:
            stream.flush()
        except (OSError, ValueError, AttributeError):
            pass
    # A strategy may leave non-daemon threads behind; the result is written, so
    # nothing after this is worth waiting for.
    os._exit(0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backtest a /python strategy")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config.get("run_dir") or config_path.parent)
    config["run_dir"] = str(run_dir)

    out = CappedWriter(sys.stdout, int(config.get("log_limit_bytes", DEFAULT_LOG_LIMIT_BYTES)))
    sys.stdout = out
    sys.stderr = out
    handler = logging.StreamHandler(out)
    handler.setFormatter(logging.Formatter("[backtest] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    apply_resource_limits()
    take_api_key()
    try:
        result = run(config)
    except Exception as exc:
        clock.uninstall()
        logger.exception("The backtest could not run")
        result = {
            "status": "failed",
            "message": f"The backtest could not run: {str(exc)[:300]}",
            "finished_at": time.time(),
        }
    _write_and_exit(run_dir, result)
    return 0


if __name__ == "__main__":
    main()
