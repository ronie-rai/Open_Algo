"""Backtest a /python strategy, unchanged, against historical data.

The strategy runs exactly as it does live, as a subprocess of its own, but
inside a replayed market:

- ``clock`` replaces ``datetime.now``, ``time.time``, ``time.monotonic`` and
  ``time.sleep`` so the script lives on a simulated IST clock that a sleep
  advances, and that skips the hours the market is shut.
- ``data`` loads one-minute bars (Historify first, then the broker) and answers
  ``history`` and ``quotes`` as of the simulated moment, never later.
- ``broker`` stands in for the OpenAlgo client: orders fill against the
  one-minute bars, positions and costs are tracked, nothing ever reaches a
  real broker.
- ``metrics`` turns the run into the numbers a trader reads.
- ``runner`` is the subprocess entry point and ``manager`` is what the /python
  blueprint uses to start, watch and stop runs.

Nothing in this package imports Flask or the app's database, so the runner
starts in a fraction of a second and can be tested on its own.
"""
