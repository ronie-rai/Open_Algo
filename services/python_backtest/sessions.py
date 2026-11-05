"""Exchange sessions and IST time arithmetic, in epoch seconds.

Everything in the replay is kept as integer epoch seconds (UTC) and read as IST
wall-clock time where a trader would read it. IST has no daylight saving, so a
fixed +05:30 offset is exact, and because that offset is a whole number of
minutes, one-minute bar boundaries in epoch seconds are also IST minute
boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

IST_OFFSET = 19800  # +05:30 in seconds
IST = timezone(timedelta(seconds=IST_OFFSET))
DAY = 86400

_EPOCH_DATE = date(1970, 1, 1)


@dataclass(frozen=True)
class Session:
    """One exchange's trading hours, as seconds after IST midnight."""

    open_sec: int
    close_sec: int
    square_off_sec: int | None  # MIS auto square-off, None where there is none
    every_day: bool = False  # crypto trades through weekends

    def open_epoch(self, day: int) -> int:
        return day * DAY - IST_OFFSET + self.open_sec

    def close_epoch(self, day: int) -> int:
        return day * DAY - IST_OFFSET + self.close_sec


def _hm(text: str) -> int:
    h, m = text.split(":")
    return int(h) * 3600 + int(m) * 60


_EQUITY = Session(_hm("09:15"), _hm("15:30"), _hm("15:15"))
_CURRENCY = Session(_hm("09:00"), _hm("17:00"), _hm("16:45"))
_MCX = Session(_hm("09:00"), _hm("23:30"), _hm("23:30"))
_NCDEX = Session(_hm("09:00"), _hm("17:00"), _hm("17:00"))
_CRYPTO = Session(0, DAY, None, every_day=True)

# MIS square-off times follow the sandbox engine's defaults
# (sandbox/squareoff_manager.py), so a backtest closes positions when the
# sandbox would.
SESSIONS = {
    "NSE": _EQUITY,
    "BSE": _EQUITY,
    "NFO": _EQUITY,
    "BFO": _EQUITY,
    "NSE_INDEX": _EQUITY,
    "BSE_INDEX": _EQUITY,
    "CDS": _CURRENCY,
    "BCD": _CURRENCY,
    "MCX": _MCX,
    "NCO": _MCX,
    "NCDEX": _NCDEX,
    "CRYPTO": _CRYPTO,
}


#: Which sandbox setting governs which exchanges' MIS square-off.
SQUARE_OFF_SETTINGS = {
    "nse_bse_square_off_time": ("NSE", "BSE", "NFO", "BFO", "NSE_INDEX", "BSE_INDEX"),
    "cds_bcd_square_off_time": ("CDS", "BCD"),
    "mcx_square_off_time": ("MCX", "NCO"),
    "ncdex_square_off_time": ("NCDEX",),
}


def configure_square_off(times: dict) -> None:
    """Apply the operator's sandbox square-off times, keyed by sandbox setting name.

    A value that is not HH:MM, or falls outside the exchange's session, is
    ignored and the default kept.
    """
    from dataclasses import replace

    for key, exchanges in SQUARE_OFF_SETTINGS.items():
        value = times.get(key)
        if not value:
            continue
        try:
            sec = _hm(str(value).strip())
        except (ValueError, TypeError):
            continue
        for exch in exchanges:
            session = SESSIONS[exch]
            if session.open_sec < sec <= session.close_sec:
                SESSIONS[exch] = replace(session, square_off_sec=sec)


def session_for(exchange: str | None) -> Session:
    return SESSIONS.get(str(exchange or "NSE").upper(), _EQUITY)


def day_of(epoch: float) -> int:
    """IST day number (days since 1970-01-01 in IST) of an epoch second."""
    return int((epoch + IST_OFFSET) // DAY)


def day_to_date(day: int) -> date:
    return _EPOCH_DATE + timedelta(days=int(day))


def date_to_day(value: date) -> int:
    return (value - _EPOCH_DATE).days


def sec_of_day(epoch: float) -> int:
    return int((epoch + IST_OFFSET) % DAY)


def ist_midnight_epoch(value: date) -> int:
    return date_to_day(value) * DAY - IST_OFFSET


def ist_datetime(epoch: float) -> datetime:
    """Naive IST wall-clock datetime of an epoch second."""
    return datetime.fromtimestamp(epoch, IST).replace(tzinfo=None)


def parse_day(value) -> date | None:
    """A date from 'YYYY-MM-DD', a date or a datetime. None if unreadable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return date(value.year, value.month, value.day)
    if isinstance(value, date):
        return date(value.year, value.month, value.day)
    text = str(value).strip()[:10]
    try:
        y, m, d = text.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None
