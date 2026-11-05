"""Start, watch and stop backtest runs for the /python blueprint.

A run is a directory, ``strategies/backtests/<strategy_id>/<run_id>/``:

- ``config.json``  what was asked for (never the API key, which travels in
  the child's environment, as it does for a live strategy)
- a copy of the strategy file, so whatever it writes beside itself lands
  here and not beside the live script
- ``run.log``      the strategy's output
- ``progress.json`` and ``result.json``, written by the runner
- ``stop``          created to ask the runner to finish early

The runner is a subprocess, so a request never waits for a backtest: start
returns at once and the page polls status. Children are reaped by polling
(``reap``), never by a blocking wait, and every wait here sleeps in short
steps so the eventlet hub keeps running.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from utils.logging import get_logger

from .engine import time_limit_message, write_json_atomic
from .sessions import IST

logger = get_logger(__name__)

BACKTESTS_DIR = Path("strategies") / "backtests"
CACHE_DIR = BACKTESTS_DIR / "_cache"
RUNNER_MODULE = "services.python_backtest.runner"

MAX_RUNS_KEPT = 20  # per strategy, oldest removed first
MAX_CONCURRENT = 2  # across all strategies; each one is a busy CPU core
MAX_RANGE_DAYS = 366
MAX_WALL_SECONDS = 3600
CACHE_MAX_AGE_DAYS = 30
STOP_GRACE_SECONDS = 6.0
LOG_TAIL_BYTES = 64 * 1024

_STRATEGY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


@dataclass
class _Active:
    strategy_id: str
    run_id: str
    process: subprocess.Popen | None  # None while the launch is being claimed
    started: float


_RUNS: dict[str, _Active] = {}
_LOCK = threading.Lock()  # held only for dictionary bookkeeping, never across I/O
_shutting_down = False


# =============================================================================
# VALIDATION
# =============================================================================
def valid_strategy_id(strategy_id: str) -> bool:
    return (
        bool(strategy_id) and ".." not in strategy_id and bool(_STRATEGY_ID_RE.match(strategy_id))
    )


def valid_run_id(run_id: str) -> bool:
    return bool(run_id) and bool(_RUN_ID_RE.match(run_id))


def _number(body: dict, key: str, default: float, low: float, high: float, label: str):
    raw = body.get(key, default)
    if raw in (None, ""):
        raw = default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"{label} must be a number."
    if not (low <= value <= high):
        return None, f"{label} must be between {low:g} and {high:g}."
    return value, ""


def validate_request(body: dict, today: date | None = None) -> tuple[dict | None, str]:
    """The run parameters from a request body, or a message saying what to fix."""
    today = today or datetime.now(IST).date()
    body = body or {}
    try:
        start = datetime.strptime(str(body.get("start_date", "")), "%Y-%m-%d").date()
        end = datetime.strptime(str(body.get("end_date", "")), "%Y-%m-%d").date()
    except ValueError:
        return None, "Choose a start date and an end date."
    if end > today:
        return None, "The end date cannot be in the future."
    if start > end:
        return None, "The start date must be on or before the end date."
    if (end - start).days + 1 > MAX_RANGE_DAYS:
        return None, f"A backtest can cover at most {MAX_RANGE_DAYS} days. Shorten the range."

    capital, err = _number(body, "capital", 100000, 1000, 1e10, "Capital")
    if err:
        return None, err
    brokerage, err = _number(body, "brokerage_per_order", 20, 0, 1000, "Brokerage per order")
    if err:
        return None, err
    slippage, err = _number(body, "slippage_bps", 2, 0, 500, "Slippage")
    if err:
        return None, err
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "capital": capital,
        "brokerage_per_order": brokerage,
        "slippage_bps": slippage,
    }, ""


# =============================================================================
# PATHS
# =============================================================================
def _strategy_dir(strategy_id: str) -> Path:
    return BACKTESTS_DIR / strategy_id


def _run_dir(strategy_id: str, run_id: str) -> Path | None:
    if not (valid_strategy_id(strategy_id) and valid_run_id(run_id)):
        return None
    path = _strategy_dir(strategy_id) / run_id
    try:
        if BACKTESTS_DIR.resolve() not in path.resolve().parents:
            return None
    except OSError:
        return None
    return path


def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _log_tail(path: Path, max_lines: int = 200) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - LOG_TAIL_BYTES))
            data = fh.read()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if size > LOG_TAIL_BYTES and lines:
        lines = lines[1:]  # the first line is probably cut
    return "\n".join(lines[-max_lines:])


# =============================================================================
# PROCESS BOOKKEEPING
# =============================================================================
def reap() -> None:
    """Collect finished runs, and kill any that has outlived the wall-clock limit."""
    with _LOCK:
        active = list(_RUNS.values())
    now = time.monotonic()
    for run in active:
        proc = run.process
        if proc is None:
            continue
        if proc.poll() is not None:
            with _LOCK:
                if _RUNS.get(run.strategy_id) is run:
                    del _RUNS[run.strategy_id]
            continue
        if now - run.started > MAX_WALL_SECONDS + 120:
            logger.warning(
                f"Backtest {run.run_id} of {run.strategy_id} outlived its time limit; ending it"
            )
            _terminate(proc)
            _write_result_if_missing(
                run.strategy_id, run.run_id, "stopped", time_limit_message(MAX_WALL_SECONDS // 60)
            )


def _is_running(strategy_id: str, run_id: str) -> bool:
    with _LOCK:
        run = _RUNS.get(strategy_id)
    if run is None or run.run_id != run_id:
        return False
    return run.process is None or run.process.poll() is None


def active_run_id(strategy_id: str) -> str | None:
    reap()
    with _LOCK:
        run = _RUNS.get(strategy_id)
    return run.run_id if run else None


def _wait_exit(proc: subprocess.Popen, timeout: float) -> bool:
    """Poll for exit in short sleeps, so the eventlet hub is never blocked."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)
    return proc.poll() is not None


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
    except OSError:
        return
    if not _wait_exit(proc, 3.0):
        try:
            proc.kill()
        except OSError:
            return
        _wait_exit(proc, 2.0)


def _write_result_if_missing(strategy_id: str, run_id: str, status: str, message: str) -> None:
    """Record why a run ended when the runner itself could not, so the page does not guess."""
    run_dir = _run_dir(strategy_id, run_id)
    if run_dir is None or (run_dir / "result.json").exists():
        return
    try:
        write_json_atomic(
            run_dir / "result.json",
            {"status": status, "message": message, "finished_at": time.time()},
        )
    except Exception:
        logger.exception(f"Could not record the end of backtest {run_id}")


def _popen_platform_args() -> dict:
    if os.name == "nt":
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP, "startupinfo": info}
    return {"start_new_session": True}


# =============================================================================
# HOUSEKEEPING
# =============================================================================
def _prune_runs(strategy_id: str, keep_also: str | None = None) -> None:
    base = _strategy_dir(strategy_id)
    try:
        runs = sorted(
            (p for p in base.iterdir() if p.is_dir() and valid_run_id(p.name)), reverse=True
        )
    except OSError:
        return
    active = active_run_id(strategy_id)
    # `runs` already holds the run just started, so this keeps MAX_RUNS_KEPT in all.
    for path in runs[MAX_RUNS_KEPT:]:
        if path.name in (active, keep_also):
            continue
        shutil.rmtree(path, ignore_errors=True)


def _prune_cache() -> None:
    cutoff = time.time() - CACHE_MAX_AGE_DAYS * 86400
    try:
        for path in CACHE_DIR.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
    except OSError:
        return


# =============================================================================
# THE OPERATIONS
# =============================================================================
def start(
    *,
    strategy_id: str,
    strategy_name: str,
    script_path: Path,
    exchange: str,
    params: dict,
    api_key: str,
    host: str,
    python_executable: str,
    extra_env: dict | None = None,
    square_off_times: dict | None = None,
) -> tuple[bool, dict | str, int]:
    """Launch a run. Returns (ok, {"run_id": ...} or a message, HTTP status)."""
    if _shutting_down:
        return False, "The server is shutting down, so the backtest was not started.", 503
    if not valid_strategy_id(strategy_id):
        return False, "Invalid strategy ID", 400
    if not script_path.exists():
        return False, "The strategy file was not found. Upload it again.", 404
    if not api_key:
        return (
            False,
            "Create an API key on the API Key page first. The backtest uses it to read "
            "historical data.",
            400,
        )

    reap()
    now_ist = datetime.now(IST)
    run_id = f"{now_ist:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    claim = _Active(strategy_id, run_id, None, time.monotonic())
    with _LOCK:
        # Checked again in the hold that claims: stop_all sets the flag under
        # this lock, so a start that gets here after it can launch nothing.
        if _shutting_down:
            return False, "The server is shutting down, so the backtest was not started.", 503
        if strategy_id in _RUNS:
            return (
                False,
                "A backtest of this strategy is already running. Wait for it to finish or stop it.",
                409,
            )
        if len(_RUNS) >= MAX_CONCURRENT:
            return (
                False,
                f"{MAX_CONCURRENT} backtests are already running. Wait for one to finish.",
                409,
            )
        _RUNS[strategy_id] = claim

    try:
        run_dir = _strategy_dir(strategy_id) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        script_copy = run_dir / script_path.name
        shutil.copy2(script_path, script_copy)
        config = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "file_name": script_path.name,
            "run_id": run_id,
            "run_dir": str(run_dir.resolve()),
            "script": str(script_copy.resolve()),
            "cache_dir": str(CACHE_DIR.resolve()),
            "exchange": exchange,
            "host": host,
            "source_dir": str(script_path.parent.resolve()),
            "square_off_times": square_off_times or {},
            "max_wall_seconds": MAX_WALL_SECONDS,
            "created_at": now_ist.isoformat(timespec="seconds"),
            **params,
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env.update(extra_env or {})
        env.update(
            {
                "OPENALGO_API_KEY": api_key,
                "HOST_SERVER": host,
                "STRATEGY_ID": strategy_id,
                "STRATEGY_NAME": strategy_name,
                "OPENALGO_STRATEGY_EXCHANGE": exchange,
                "OPENALGO_BACKTEST": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
            }
        )
        cmd = [
            python_executable,
            "-u",
            "-m",
            RUNNER_MODULE,
            "--config",
            str((run_dir / "config.json").resolve()),
        ]
        # The child inherits the log handle; the parent's copy is closed at
        # once so a run never pins a descriptor here.
        with open(run_dir / "run.log", "w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=str(Path.cwd()),
                env=env,
                **_popen_platform_args(),
            )
    except Exception:
        logger.exception(f"Could not start a backtest of {strategy_id}")
        with _LOCK:
            if _RUNS.get(strategy_id) is claim:
                del _RUNS[strategy_id]
        return False, "The backtest could not be started. Check the server log for details.", 500

    with _LOCK:
        claim.process = process
    logger.info(f"Backtest {run_id} of {strategy_id} started with PID {process.pid}")
    _prune_runs(strategy_id, keep_also=run_id)
    _prune_cache()
    return True, {"run_id": run_id}, 200


_INTERRUPTED = (
    "This backtest ended before it could save its results, most likely because the server "
    "restarted while it ran. Run it again."
)


def status(strategy_id: str, run_id: str, include_result: bool = True) -> dict | None:
    """What the results page shows for one run, or None if there is no such run."""
    run_dir = _run_dir(strategy_id, run_id)
    if run_dir is None or not run_dir.is_dir():
        return None
    reap()
    config = _read_json(run_dir / "config.json") or {}
    progress = _read_json(run_dir / "progress.json")
    result = _read_json(run_dir / "result.json")
    running = _is_running(strategy_id, run_id)
    if running:
        state, message = "running", ""
        if (run_dir / "stop").exists():
            state = "stopping"
    elif result:
        state, message = result.get("status") or "completed", result.get("message") or ""
    else:
        state, message = "stopped", _INTERRUPTED
    out = {
        "run_id": run_id,
        "strategy_id": strategy_id,
        "status": state,
        "message": message,
        "config": {
            k: v
            for k, v in config.items()
            if k not in ("run_dir", "script", "cache_dir", "host", "source_dir")
        },
        "progress": progress,
        "log_tail": _log_tail(run_dir / "run.log"),
    }
    if include_result and result and not running:
        out["result"] = result
    return out


def list_runs(strategy_id: str, limit: int = 50) -> list[dict]:
    """Past and current runs of a strategy, newest first, with their headline numbers."""
    if not valid_strategy_id(strategy_id):
        return []
    base = _strategy_dir(strategy_id)
    try:
        dirs = sorted(
            (p for p in base.iterdir() if p.is_dir() and valid_run_id(p.name)), reverse=True
        )
    except OSError:
        return []
    reap()
    rows = []
    for path in dirs[:limit]:
        config = _read_json(path / "config.json") or {}
        result = _read_json(path / "result.json") or {}
        running = _is_running(strategy_id, path.name)
        metrics = result.get("metrics") or {}
        rows.append(
            {
                "run_id": path.name,
                "status": "running" if running else (result.get("status") or "stopped"),
                "created_at": config.get("created_at"),
                "start_date": config.get("start_date"),
                "end_date": config.get("end_date"),
                "capital": config.get("capital"),
                "net_pnl": metrics.get("net_pnl"),
                "return_pct": metrics.get("return_pct"),
                "trades": metrics.get("trades"),
                "win_rate": metrics.get("win_rate"),
                "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            }
        )
    return rows


def stop(strategy_id: str, run_id: str) -> tuple[bool, str]:
    """Ask a run to finish early, and make sure it does.

    The runner notices the stop file within a moment and saves the results of
    the time it ran. If it has not exited after the grace period it is
    terminated, and its results are whatever it last saved.
    """
    run_dir = _run_dir(strategy_id, run_id)
    if run_dir is None or not run_dir.is_dir():
        return False, "Backtest not found"
    with _LOCK:
        run = _RUNS.get(strategy_id)
    if run is None or run.run_id != run_id or run.process is None:
        return True, "This backtest is not running."
    try:
        (run_dir / "stop").write_text("stop", encoding="utf-8")
    except OSError:
        logger.exception(f"Could not ask backtest {run_id} to stop")
    if not _wait_exit(run.process, STOP_GRACE_SECONDS):
        _terminate(run.process)
        _write_result_if_missing(
            strategy_id,
            run_id,
            "stopped",
            "You stopped this backtest before it could save its results.",
        )
    reap()
    return True, "The backtest was stopped."


def stop_all(grace_seconds: float = 3.0) -> None:
    """Shutdown: ask every run to stop, then end whatever is left."""
    global _shutting_down
    with _LOCK:
        _shutting_down = True
        runs = [r for r in _RUNS.values() if r.process is not None]
    if not runs:
        return
    for run in runs:
        try:
            (_strategy_dir(run.strategy_id) / run.run_id / "stop").write_text(
                "stop", encoding="utf-8"
            )
        except OSError:
            pass
    deadline = time.monotonic() + grace_seconds
    for run in runs:
        _wait_exit(run.process, max(0.0, deadline - time.monotonic()))
    for run in runs:
        if run.process.poll() is None:
            _terminate(run.process)
    reap()
