"""The simulated clock and what happens as it moves.

The strategy only ever sees time through ``now()`` and ``sleep()``. A sleep
moves the clock forward; on the way every one-minute bar that completes is
offered to the pending orders, the MIS square-off fires at its minute, and
equity is sampled every five minutes. Outside the session (nights, weekends,
days the data has no bars for) a sleep fast-forwards to the next open. Past
the last day of the test the run ends with ``BacktestFinished``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

from .sessions import (
    DAY,
    IST_OFFSET,
    date_to_day,
    day_of,
    day_to_date,
    ist_datetime,
    sec_of_day,
    session_for,
)

logger = logging.getLogger("python_backtest")

# Captured before anything is patched: the engine's own sense of real time.
_real_monotonic = time.monotonic
_real_time = time.time

EQUITY_SAMPLE_SECONDS = 300


class BacktestFinished(BaseException):  # noqa: N818 - a signal, not an error
    """The test period is over (or the run was stopped).

    A BaseException, so a strategy's ``except Exception`` cannot swallow it.
    """


class Engine:
    def __init__(
        self,
        *,
        broker,
        market,
        start: date,
        end: date,
        exchange: str = "NSE",
        run_dir: str | Path | None = None,
        max_wall_seconds: float = 3600,
        max_sleep_calls: int = 20_000_000,
    ):
        self.broker = broker
        self.market = market
        self.start = start
        self.end = end
        self.session = session_for(exchange)
        self.run_dir = Path(run_dir) if run_dir else None
        self.max_wall_seconds = float(max_wall_seconds)
        self.max_sleep_calls = int(max_sleep_calls)

        self.start_day = date_to_day(start)
        self.end_day = date_to_day(end)
        self.end_epoch = self.session.close_epoch(self.end_day)

        first = self._next_session_day(self.start_day)
        self.now = float(self.session.open_epoch(first if first is not None else self.start_day))
        self.start_epoch = self.now

        self.finished = False
        self.outcome = None  # "completed" | "stopped" | "failed"
        self.message = ""
        self.equity_samples: list[tuple[float, float, bool]] = []
        self.day_ends: list[tuple[int, float]] = []
        self.sleep_calls = 0
        self.on_runaway = None  # called once if the finish signal keeps being swallowed
        self.calls_since_sleep = 0
        self._calls_after_finish = 0
        self.reads_since_sleep = 0
        self.wall_start = _real_monotonic()
        self._last_progress_wall = 0.0
        self._last_stop_check = 0.0
        self._progress_day = None
        self._sample_equity(self.now)

    # ---------- sessions ----------
    def _is_session_day(self, day: int) -> bool:
        if self.market.has_any_data():
            return day in self.market.data_days
        if self.session.every_day:
            return True
        return day_to_date(day).weekday() < 5

    def _next_session_day(self, day: int) -> int | None:
        """The first session day at or after `day`, within the test, or None."""
        while day <= self.end_day:
            if self._is_session_day(day):
                return day
            day += 1
        return None

    def _in_session(self, t: float) -> bool:
        day = day_of(t)
        sec = sec_of_day(t)
        return (
            self.session.open_sec <= sec < self.session.close_sec
            and self.start_day <= day <= self.end_day
            and self._is_session_day(day)
        )

    def _next_open_after(self, t: float) -> float | None:
        day = day_of(t)
        if sec_of_day(t) >= self.session.open_sec or not self._is_session_day(day):
            day += 1
        nxt = self._next_session_day(max(day, self.start_day))
        return None if nxt is None else float(self.session.open_epoch(nxt))

    # ---------- what the patched stdlib calls ----------
    def after_finish(self) -> None:
        """Raise BacktestFinished again; a strategy that keeps swallowing it is ended outright."""
        self._calls_after_finish += 1
        if self._calls_after_finish > 5000 and self.on_runaway is not None:
            hook, self.on_runaway = self.on_runaway, None
            hook()
        raise BacktestFinished()

    def read_clock(self) -> float:
        if self.finished:
            self.after_finish()
        self.reads_since_sleep += 1
        if self.reads_since_sleep % 10_000 == 0:
            self._check_wall()
            # A loop that polls the clock without ever sleeping would spin
            # forever on a clock only sleep moves. Let time pass for it.
            if self.reads_since_sleep >= 200_000:
                self.reads_since_sleep = 0
                self.advance(1.0)
        return self.now

    def note_call(self) -> float:
        """A client call. A loop that only calls the client still has to let time pass."""
        if self.finished:
            self.after_finish()
        self.calls_since_sleep += 1
        if self.calls_since_sleep % 1000 == 0:
            self._check_wall()
            self._check_stop_request()
            if self.calls_since_sleep >= 20_000:
                self.calls_since_sleep = 0
                self.advance(1.0)
        return self.now

    def sleep(self, seconds) -> None:
        if self.finished:
            self.after_finish()
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        self.sleep_calls += 1
        self.reads_since_sleep = 0
        self.calls_since_sleep = 0
        if self.sleep_calls > self.max_sleep_calls:
            self.finish(
                "stopped",
                "The backtest was stopped because the strategy looped more times than "
                "the limit allows. Make sure it sleeps between cycles.",
            )
        self._check_wall()
        self._check_stop_request()
        self.advance(max(0.0, seconds))

    # ---------- moving the clock ----------
    def advance(self, seconds: float) -> None:
        target = self.now + seconds
        while self.now < target or not self._in_session(self.now):
            if not self._in_session(self.now):
                nxt = self._next_open_after(self.now)
                if nxt is None or nxt >= self.end_epoch:
                    self.finish("completed")
                self.now = nxt
                if target <= nxt:
                    break
                continue
            close = float(self.session.close_epoch(day_of(self.now)))
            step = min(target, close)
            self._process_until(step)
            self.now = step
        self._progress()

    def _process_until(self, t: float) -> None:
        """Run every minute boundary in (now, t]."""
        m = (int(self.now) // 60 + 1) * 60
        close_sec = self.session.close_sec % DAY
        broker = self.broker
        while m <= t:
            self.now = float(m)
            broker.on_bar_closed(m - 60)
            if broker.pending or broker.positions:
                broker.square_off_mis(m)
            if m % EQUITY_SAMPLE_SECONDS == 0:
                self._sample_equity(m)
            if sec_of_day(m) == close_sec:
                self._record_day_end(day_of(m - 1))
            m += 60

    def _sample_equity(self, t: float) -> None:
        self.equity_samples.append(
            (float(t), self.broker.equity(), self.broker.has_open_positions())
        )

    def _record_day_end(self, day: int) -> None:
        equity = self.broker.equity()
        if self.day_ends and self.day_ends[-1][0] == day:
            self.day_ends[-1] = (day, equity)
        else:
            self.day_ends.append((day, equity))

    # ---------- ending ----------
    def finish(self, outcome: str, message: str = "") -> None:
        """End the run: square off what is open, record the last day, stop the strategy."""
        if not self.finished:
            self.broker.end_of_test()
            self._sample_equity(self.now)
            if self._in_session(self.now):
                self._record_day_end(day_of(self.now))  # stopped part way through a day
            elif self.day_ends and self.day_ends[-1][0] == day_of(self.now - 1):
                self._record_day_end(self.day_ends[-1][0])  # costs of the final square-off
            self.finished = True
            self.outcome = outcome
            self.message = message
            self._write_progress(force=True)
        raise BacktestFinished()

    def _check_wall(self) -> None:
        if _real_monotonic() - self.wall_start > self.max_wall_seconds:
            minutes = int(self.max_wall_seconds // 60)
            self.finish("stopped", time_limit_message(minutes))

    def _check_stop_request(self) -> None:
        if self.run_dir is None:
            return
        wall = _real_monotonic()
        if wall - self._last_stop_check < 0.5:
            return
        self._last_stop_check = wall
        if (self.run_dir / "stop").exists():
            self.finish("stopped", "You stopped this backtest. The results cover the time it ran.")

    # ---------- progress ----------
    def percent(self) -> float:
        span = max(1.0, self.end_epoch - self.start_epoch)
        return max(0.0, min(100.0, (self.now - self.start_epoch) / span * 100))

    def _progress(self) -> None:
        day = day_of(self.now)
        wall = _real_monotonic()
        if day != self._progress_day or wall - self._last_progress_wall > 2.0:
            self._progress_day = day
            self._write_progress()

    def _write_progress(self, force: bool = False) -> None:
        if self.run_dir is None:
            return
        self._last_progress_wall = _real_monotonic()
        data = {
            "status": "finished" if self.finished else "running",
            "sim_time": f"{ist_datetime(self.now):%Y-%m-%d %H:%M}",
            "percent": round(100.0 if self.finished else self.percent(), 1),
            "days_done": len(self.day_ends),
            "trades": len(self.broker.trades),
            "orders": len(self.broker.orders),
            "equity": round(self.broker.equity(), 2),
            "updated_at": _real_time(),
        }
        write_json_atomic(self.run_dir / "progress.json", data)


def time_limit_message(minutes: int) -> str:
    return (
        f"The backtest was stopped after {minutes} minutes, the most one run may take. Either "
        "the date range is too long for this strategy, or the strategy never pauses between "
        "cycles. Try a shorter range, or add a time.sleep to its loop. The results cover the "
        "time it ran."
    )


def write_json_atomic(path: Path, data) -> None:
    """Write JSON through a temporary file and a rename, so a reader never sees half."""
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, default=str)
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                # Windows refuses to replace a file another process has open.
                _sleep_real(0.05)
    except OSError:
        logger.warning(f"Could not write {path.name}")


_real_sleep = time.sleep


def _sleep_real(seconds: float) -> None:
    _real_sleep(seconds)


def trading_days_between(start: date, end: date) -> int:
    days = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            days += 1
        d += timedelta(days=1)
    return days


__all__ = ["BacktestFinished", "DAY", "Engine", "IST_OFFSET", "write_json_atomic"]
