"""Put a strategy on the simulated clock without changing a line of it.

``install(engine)`` replaces, on the ``datetime`` and ``time`` modules:

- ``datetime.datetime`` and ``datetime.date`` with subclasses whose ``now``,
  ``utcnow``, ``today`` and ``fromtimestamp`` read the simulated clock. A
  metaclass keeps ``isinstance(x, datetime)`` true for real datetimes too
  (pandas Timestamps among them), so a strategy's type checks still pass.
- ``time.time``, ``time.monotonic`` (and their ``_ns`` forms), ``time.sleep``,
  ``time.localtime``, ``time.gmtime``, ``time.strftime``, ``time.ctime`` and
  ``time.asctime``.
- ``pandas.Timestamp.now``, ``today`` and ``utcnow``.

Not covered: ``numpy.datetime64("now")`` and ``pandas.Timestamp("now")``
(the string form), which read the real clock inside compiled code.

Simulated time is IST wall-clock time: a naive ``datetime.now()`` returns IST,
as it does on an Indian server, whatever zone this machine is in.

It must run after pandas and numpy are imported, so their internals keep the
real classes, and before the strategy is imported, so its
``from datetime import datetime`` picks up the replacement.
``uninstall()`` puts everything back; ``real_time()`` does so for the length
of a ``with`` block, which is how the data loader talks to the network with
working timeouts.
"""

from __future__ import annotations

import datetime as _dt_module
import time as _time_module
from contextlib import contextmanager

import pandas as _pd

from .sessions import IST_OFFSET

_RealDatetime = _dt_module.datetime
_RealDate = _dt_module.date
_IST = _dt_module.timezone(_dt_module.timedelta(seconds=IST_OFFSET))
_UTC = _dt_module.UTC

_ORIGINALS = {
    "time": _time_module.time,
    "time_ns": _time_module.time_ns,
    "monotonic": _time_module.monotonic,
    "monotonic_ns": _time_module.monotonic_ns,
    "sleep": _time_module.sleep,
    "localtime": _time_module.localtime,
    "gmtime": _time_module.gmtime,
    "strftime": _time_module.strftime,
    "ctime": _time_module.ctime,
    "asctime": _time_module.asctime,
}

_PD_ORIGINALS = {name: _pd.Timestamp.__dict__[name] for name in ("now", "today", "utcnow")}

_engine = None
_installed = False


def _now() -> float:
    return _engine.read_clock()


class _DateMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _RealDate)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _RealDate)


class _DatetimeMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, _RealDatetime)

    def __subclasscheck__(cls, sub):
        return issubclass(sub, _RealDatetime)


class SimDate(_RealDate, metaclass=_DateMeta):
    @classmethod
    def today(cls):
        d = _RealDatetime.fromtimestamp(_now(), _IST)
        return cls(d.year, d.month, d.day)


class SimDatetime(_RealDatetime, metaclass=_DatetimeMeta):
    @classmethod
    def _wrap(cls, d):
        return cls(
            d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, d.tzinfo, fold=d.fold
        )

    @classmethod
    def now(cls, tz=None):
        epoch = _now()
        if tz is None:
            return cls._wrap(_RealDatetime.fromtimestamp(epoch, _IST).replace(tzinfo=None))
        return cls._wrap(_RealDatetime.fromtimestamp(epoch, tz))

    @classmethod
    def today(cls):
        return cls.now()

    @classmethod
    def utcnow(cls):
        return cls._wrap(_RealDatetime.fromtimestamp(_now(), _UTC).replace(tzinfo=None))

    @classmethod
    def fromtimestamp(cls, t, tz=None):
        if tz is None:
            return cls._wrap(_RealDatetime.fromtimestamp(t, _IST).replace(tzinfo=None))
        return cls._wrap(_RealDatetime.fromtimestamp(t, tz))


def _ist_struct(secs=None):
    return _ORIGINALS["gmtime"]((_now() if secs is None else secs) + IST_OFFSET)


def _gmtime(secs=None):
    return _ORIGINALS["gmtime"](_now() if secs is None else secs)


def _strftime(fmt, t=None):
    return _ORIGINALS["strftime"](fmt, _ist_struct() if t is None else t)


def _asctime(t=None):
    return _ORIGINALS["asctime"](_ist_struct() if t is None else t)


def _ctime(secs=None):
    return _ORIGINALS["asctime"](_ist_struct(secs))


_REPLACEMENTS = {
    "time": lambda: _now(),
    "time_ns": lambda: int(_now() * 1e9),
    "monotonic": lambda: _now(),
    "monotonic_ns": lambda: int(_now() * 1e9),
    "sleep": lambda seconds: _engine.sleep(seconds),
    "localtime": _ist_struct,
    "gmtime": _gmtime,
    "strftime": _strftime,
    "ctime": _ctime,
    "asctime": _asctime,
}


def _pd_now(cls, tz=None):
    ts = _pd.Timestamp(_now(), unit="s", tz="UTC")
    if tz is None:
        return ts.tz_convert("Asia/Kolkata").tz_localize(None)
    return ts.tz_convert(tz)


def _pd_utcnow(cls):
    return _pd.Timestamp(_now(), unit="s")


_PD_REPLACEMENTS = {
    "now": classmethod(_pd_now),
    "today": classmethod(_pd_now),
    "utcnow": classmethod(_pd_utcnow),
}


def _apply() -> None:
    _dt_module.datetime = SimDatetime
    _dt_module.date = SimDate
    for name, fn in _REPLACEMENTS.items():
        setattr(_time_module, name, fn)
    for name, fn in _PD_REPLACEMENTS.items():
        setattr(_pd.Timestamp, name, fn)


def _restore() -> None:
    _dt_module.datetime = _RealDatetime
    _dt_module.date = _RealDate
    for name, fn in _ORIGINALS.items():
        setattr(_time_module, name, fn)
    for name, fn in _PD_ORIGINALS.items():
        setattr(_pd.Timestamp, name, fn)


def install(engine) -> None:
    global _engine, _installed
    _engine = engine
    _apply()
    _installed = True


def uninstall() -> None:
    global _installed
    _restore()
    _installed = False


@contextmanager
def real_time():
    """The real clock for the length of the block, the simulated one after."""
    if not _installed:
        yield
        return
    _restore()
    try:
        yield
    finally:
        _apply()
