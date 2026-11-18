"""5-minute NSE data for the basket: Historify first, then the local OpenAlgo API.

Range: 12 full months ending at the last complete month (computed from today's
date in IST). Each symbol is cached to data/<SYMBOL>_5m.csv and its source and
range are recorded in data/source.txt.

The API key is read only from the environment variable OPENALGO_API_KEY. It is
never logged, printed or written anywhere.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import Config, hhmm_to_minutes
from resample import infer_bar_minutes, minute_of_day, resample_ohlcv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = HERE / "data"
HISTORIFY_PATH = REPO_ROOT / "db" / "historify.duckdb"
TZ = "Asia/Kolkata"
COLS = ["open", "high", "low", "close", "volume"]

# Calendar days per API request. Angel allows 100 days per 5m request and 30
# per 1m request (broker/angel/api/data.py); these stay well inside both.
API_CHUNK_DAYS = {"5m": 60, "1m": 25}
API_PAUSE_SECONDS = 0.6
API_TIMEOUT_SECONDS = 60

HOW_TO_PROVIDE_DATA = """\
No price data was found, so there is nothing to backtest yet.

Provide 5-minute or 1-minute NSE data in either of these ways:

  1. Historify (recommended). In OpenAlgo open Historify, add the symbols to the
     watchlist (exchange NSE) and download 1 minute data for {start} to {end}.
     The backtester reads db/historify.duckdb directly. If OpenAlgo is running
     and the file is busy, stop OpenAlgo for a moment and run again.

  2. OpenAlgo API. Start OpenAlgo, log in to your broker, copy your key from the
     API Key page and set it in this shell only:
         export OPENALGO_API_KEY=your_key_here
     (optional: export OPENALGO_HOST=http://127.0.0.1:5000)
     Then run this command again.
"""


class ApiUnavailable(Exception):
    """The API cannot serve any symbol (not running, or the key was refused)."""


@dataclass
class SymbolData:
    symbol: str
    bars: pd.DataFrame | None
    source: str
    message: str = ""


def data_range(today: date | None = None) -> tuple[date, date]:
    """12 full months ending at the last complete month before `today` (IST)."""
    if today is None:
        today = pd.Timestamp.now(tz=TZ).date()
    first_this_month = today.replace(day=1)
    end = first_this_month - timedelta(days=1)
    start = first_this_month.replace(year=first_this_month.year - 1)
    return start, end


def range_bounds(start: date, end: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start 00:00, day after end 00:00) in IST."""
    lo = pd.Timestamp(start).tz_localize(TZ)
    hi = pd.Timestamp(end + timedelta(days=1)).tz_localize(TZ)
    return lo, hi


def split_time(start: date, end: date, is_frac: float) -> pd.Timestamp:
    """One in-sample/out-of-sample boundary: is_frac of the way through the range."""
    lo, hi = range_bounds(start, end)
    return lo + (hi - lo) * is_frac


def safe_name(symbol: str) -> str:
    """Filename-safe symbol: M&M -> M_and_M."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", symbol.replace("&", "_and_"))


def cache_path(symbol: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{safe_name(symbol)}_5m.csv"


def to_ist_index(ts: pd.Series) -> pd.DatetimeIndex:
    """Timestamps (epoch seconds or milliseconds, or date strings) to an IST index.

    Numeric values are epoch seconds in UTC (what OpenAlgo returns); values above
    1e12 are taken as milliseconds. Strings without an offset are read as IST.
    """
    if pd.api.types.is_numeric_dtype(ts):
        values = ts.astype("int64")
        unit = "ms" if len(values) and values.abs().max() > 10**12 else "s"
        return pd.DatetimeIndex(pd.to_datetime(values, unit=unit, utc=True)).tz_convert(TZ)
    parsed = pd.to_datetime(ts, format="ISO8601")
    idx = pd.DatetimeIndex(parsed)
    if idx.tz is None:
        return idx.tz_localize(TZ)
    return idx.tz_convert(TZ)


def clean_bars(df: pd.DataFrame, start: date, end: date, cfg: Config | None = None) -> pd.DataFrame:
    """Sort, deduplicate, keep Mon-Fri session bars inside the range, float columns.

    Session filter: bar start at or after 09:15 and before 15:30 IST, and on the
    grid of its own bar size from 09:15.
    """
    cfg = cfg or Config()
    if df.empty:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz=TZ, name="timestamp"))
    out = df[COLS].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    lo, hi = range_bounds(start, end)
    out = out[(out.index >= lo) & (out.index < hi)]
    mod = minute_of_day(out.index)
    open_min = hhmm_to_minutes(cfg.session_open)
    close_min = hhmm_to_minutes(cfg.session_close)
    keep = (out.index.dayofweek < 5) & (mod >= open_min) & (mod < close_min)
    out = out[keep]
    base = infer_bar_minutes(out)
    if base:
        out = out[((minute_of_day(out.index) - open_min) % base) == 0]
    out = out.dropna()
    out.index.name = "timestamp"
    return out


def frame_from_records(records: list[dict]) -> pd.DataFrame:
    """OHLCV records with a 'timestamp' field into an IST-indexed frame."""
    raw = pd.DataFrame(records)
    if raw.empty or "timestamp" not in raw:
        return pd.DataFrame(columns=COLS)
    raw.index = to_ist_index(raw["timestamp"])
    if "volume" not in raw:
        raw["volume"] = 0.0
    return raw[COLS]


def to_5m(
    df: pd.DataFrame, interval: str, start: date, end: date, cfg: Config | None = None
) -> pd.DataFrame:
    """Clean `df` and, for 1m data, aggregate to 5m with the session rules."""
    cleaned = clean_bars(df, start, end, cfg)
    if interval == "1m" and not cleaned.empty:
        cleaned = clean_bars(resample_ohlcv(cleaned, "5m", cfg), start, end, cfg)
    return cleaned


# ---------------------------------------------------------------- Historify


def load_from_historify(
    symbol: str, start: date, end: date, db_path: Path = HISTORIFY_PATH, exchange: str = "NSE"
) -> tuple[pd.DataFrame, str]:
    """5m rows from Historify, else 1m rows aggregated to 5m. Opened read-only.

    market_data stores timestamp as epoch seconds (UTC) of the bar start, with
    interval strings such as "1m" and "D" (see database/historify_db.py).

    Returns:
        (bars, note). Empty bars when nothing is stored for the symbol.
    """
    if not Path(db_path).exists():
        return pd.DataFrame(columns=COLS), "Historify file not found"
    import duckdb

    lo, hi = range_bounds(start, end)
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return pd.DataFrame(columns=COLS), "Historify file is in use by another program"
    try:
        for interval in ("5m", "1m"):
            rows = conn.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND exchange = ? AND interval = ?
                  AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
                """,
                [symbol.upper(), exchange, interval, int(lo.timestamp()), int(hi.timestamp())],
            ).fetchdf()
            if not rows.empty:
                frame = frame_from_records(rows.to_dict(orient="records"))
                bars = to_5m(frame, interval, start, end)
                if not bars.empty:
                    return bars, f"historify {interval}"
    finally:
        conn.close()
    return pd.DataFrame(columns=COLS), "no rows for this symbol"


# ---------------------------------------------------------------- OpenAlgo API


def _date_chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    chunks = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        chunks.append((cur, stop))
        cur = stop + timedelta(days=1)
    return chunks


def _fetch_interval(
    symbol: str, interval: str, start: date, end: date, host: str, api_key: str, http, exchange: str
) -> tuple[pd.DataFrame, int, int]:
    """All chunks for one interval. Raises ApiUnavailable when the API itself is unusable.

    Returns:
        (bars, failed_requests, total_requests).
    """
    frames = []
    failed = 0
    url = host.rstrip("/") + "/api/v1/history"
    chunks = _date_chunks(start, end, API_CHUNK_DAYS[interval])
    for i, (a, b) in enumerate(chunks):
        if i:
            time.sleep(API_PAUSE_SECONDS)
        body = {
            "apikey": api_key,
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "start_date": a.isoformat(),
            "end_date": b.isoformat(),
        }
        try:
            resp = http.post(url, json=body, timeout=API_TIMEOUT_SECONDS)
        except Exception as exc:
            raise ApiUnavailable(
                f"OpenAlgo did not answer at {host}. Start OpenAlgo and log in to your broker."
            ) from exc
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if resp.status_code in (401, 403) or "apikey" in str(payload.get("message", "")).lower():
            raise ApiUnavailable(
                "OpenAlgo refused the request. Check that OPENALGO_API_KEY matches the key "
                "on the API Key page."
            )
        if payload.get("status") != "success":
            # One chunk failing (for example before a listing date) is not fatal,
            # but it is counted and reported in the source note.
            failed += 1
            continue
        frame = frame_from_records(payload.get("data") or [])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=COLS), failed, len(chunks)
    return pd.concat(frames), failed, len(chunks)


def load_from_api(
    symbol: str,
    start: date,
    end: date,
    host: str | None = None,
    api_key: str | None = None,
    http=None,
    exchange: str = "NSE",
) -> tuple[pd.DataFrame, str]:
    """History from the running OpenAlgo app: interval 5m, else 1m aggregated to 5m."""
    host = host or os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000")
    api_key = api_key if api_key is not None else os.environ.get("OPENALGO_API_KEY", "")
    if not api_key:
        return pd.DataFrame(columns=COLS), "OPENALGO_API_KEY not set"
    if http is None:
        import requests

        http = requests
    for interval in ("5m", "1m"):
        raw, failed, total = _fetch_interval(
            symbol, interval, start, end, host, api_key, http, exchange
        )
        bars = to_5m(raw, interval, start, end)
        if not bars.empty:
            gaps = f" ({failed} of {total} requests failed, data may have gaps)" if failed else ""
            return bars, f"openalgo api {interval}{gaps}"
    return pd.DataFrame(columns=COLS), "the API returned no bars"


# ---------------------------------------------------------------- cache


def read_sources(data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    path = data_dir / "source.txt"
    out: dict[str, list[str]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line and not line.startswith("#"):
                parts = line.split("\t")
                out[parts[0]] = parts
    return out


def write_sources(entries: dict[str, list[str]], data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# symbol\tsource\trequested_range\tfirst_bar\tlast_bar\tbars"]
    lines += ["\t".join(entries[k]) for k in sorted(entries)]
    (data_dir / "source.txt").write_text("\n".join(lines) + "\n")


def write_cache(bars: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars[COLS].copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.index.name = "timestamp"
    out.to_csv(path)


def read_cache(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    idx = pd.DatetimeIndex(pd.to_datetime(raw["timestamp"], format="ISO8601", utc=True)).tz_convert(
        TZ
    )
    out = raw[COLS].astype(float)
    out.index = idx
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_symbol(
    symbol: str,
    start: date,
    end: date,
    refresh: bool = False,
    data_dir: Path = DATA_DIR,
    db_path: Path = HISTORIFY_PATH,
    api_state: dict | None = None,
    http=None,
) -> SymbolData:
    """Cached 5m bars for one symbol, downloading when missing, stale or on refresh."""
    api_state = api_state if api_state is not None else {}
    path = cache_path(symbol, data_dir)
    requested = f"{start.isoformat()}..{end.isoformat()}"
    sources = read_sources(data_dir)
    entry = sources.get(symbol)
    if not refresh and path.exists() and entry and len(entry) > 2 and entry[2] == requested:
        bars = read_cache(path)
        if not bars.empty:
            return SymbolData(symbol, bars, entry[1], "cached")

    notes = []
    bars, note = load_from_historify(symbol, start, end, db_path)
    notes.append(f"Historify: {note}")
    source = note
    if bars.empty:
        if api_state.get("down"):
            notes.append(f"API: {api_state['down']}")
        else:
            try:
                bars, note = load_from_api(symbol, start, end, http=http)
                source = note
                notes.append(f"API: {note}")
            except ApiUnavailable as exc:
                api_state["down"] = str(exc)
                notes.append(f"API: {exc}")
    if bars.empty:
        return SymbolData(symbol, None, "none", "; ".join(notes))

    write_cache(bars, path)
    sources[symbol] = [
        symbol,
        source,
        requested,
        bars.index[0].isoformat(),
        bars.index[-1].isoformat(),
        str(len(bars)),
    ]
    write_sources(sources, data_dir)
    return SymbolData(symbol, bars, source, "downloaded")


def load_basket(
    symbols: list[str],
    refresh: bool = False,
    today: date | None = None,
    data_dir: Path = DATA_DIR,
    db_path: Path = HISTORIFY_PATH,
    log=print,
) -> tuple[dict[str, pd.DataFrame], dict[str, SymbolData], tuple[date, date]]:
    """Load every symbol; symbols without data are skipped with a message."""
    start, end = data_range(today)
    api_state: dict = {}
    loaded: dict[str, pd.DataFrame] = {}
    info: dict[str, SymbolData] = {}
    for sym in symbols:
        res = load_symbol(sym, start, end, refresh, data_dir, db_path, api_state)
        info[sym] = res
        if res.bars is None or res.bars.empty:
            log(f"  {sym}: no data, skipped ({res.message})")
            continue
        loaded[sym] = res.bars
        first, last = res.bars.index[0], res.bars.index[-1]
        short = (
            " (short history)"
            if first.normalize() > range_bounds(start, end)[0] + pd.Timedelta(days=30)
            else ""
        )
        log(
            f"  {sym}: {len(res.bars)} bars {first:%Y-%m-%d} to {last:%Y-%m-%d} from {res.source}{short}"
        )
    return loaded, info, (start, end)


def no_data_message(start: date, end: date) -> str:
    return HOW_TO_PROVIDE_DATA.format(start=start.isoformat(), end=end.isoformat())


__all__ = [
    "data_range",
    "split_time",
    "load_basket",
    "load_symbol",
    "load_from_historify",
    "load_from_api",
    "clean_bars",
    "safe_name",
    "no_data_message",
]
