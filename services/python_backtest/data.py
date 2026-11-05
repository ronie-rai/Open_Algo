"""Historical bars for the replay, and every view of them as of a moment.

The single rule this module exists to keep: **nothing after the simulated now
is ever visible.** A one-minute bar starting at ``t`` is complete at ``t + 60``.
At a moment ``now`` with ``t <= now < t + 60`` that bar is still forming, and
the only thing known about it is its open, the first trade of the minute.
Every price this module hands out (``price_at``, a forming bar, a quote) is
built from complete bars plus, at most, the open of the bar in progress.

Data comes from a provider:

- ``OpenAlgoProvider`` calls the running OpenAlgo app's history endpoint with
  the real SDK client, Historify (``source="db"``) first and the broker
  (``source="api"``) where Historify is short, in date chunks the brokers
  accept, and caches finished chunks on disk.
- ``FileProvider`` reads bars from files, which is how the tests run with no
  app and no broker.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from contextlib import contextmanager, nullcontext
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .sessions import (
    DAY,
    IST_OFFSET,
    Session,
    date_to_day,
    day_of,
    day_to_date,
    ist_midnight_epoch,
    parse_day,
    session_for,
)

logger = logging.getLogger("python_backtest")

BAR_COLUMNS = ["epoch", "open", "high", "low", "close", "volume", "oi"]
DAILY_COLUMNS = ["day", "open", "high", "low", "close", "volume", "oi"]

_INTRADAY_RE = re.compile(r"^(\d+)\s*(m|min|minute|h|hr|hour)$", re.IGNORECASE)


def interval_minutes(interval: str) -> int | None:
    """Minutes in an intraday interval ('1m', '5m', '1h'), 0 for daily, None if unsupported."""
    text = str(interval or "").strip()
    if text.upper() in ("D", "1D", "DAY", "1DAY"):
        return 0
    match = _INTRADAY_RE.match(text)
    if not match:
        return None
    n = int(match.group(1))
    unit = match.group(2).lower()
    minutes = n * 60 if unit.startswith("h") else n
    return minutes if 1 <= minutes <= 375 else None


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS})


def empty_daily() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in DAILY_COLUMNS})


def records_to_bars(records) -> pd.DataFrame:
    """History API records (epoch seconds) to the 1m bar frame."""
    if not records:
        return empty_bars()
    df = pd.DataFrame(records)
    if "timestamp" not in df.columns:
        return empty_bars()
    df = df.rename(columns={"timestamp": "epoch"})
    for col in BAR_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[BAR_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna(subset=["epoch", "close"])
    df["epoch"] = df["epoch"].astype("int64")
    return df.sort_values("epoch").drop_duplicates("epoch", keep="last").reset_index(drop=True)


def records_to_daily(records) -> pd.DataFrame:
    """Daily history records to a frame keyed by IST day number.

    Brokers stamp a daily bar at IST midnight or at UTC midnight; either way
    the IST calendar date of the stamp is the trading day.
    """
    bars = records_to_bars(records)
    if bars.empty:
        return empty_daily()
    bars["day"] = (bars["epoch"] + IST_OFFSET) // DAY
    out = bars[DAILY_COLUMNS].copy()
    return out.drop_duplicates("day", keep="last").reset_index(drop=True)


# =============================================================================
# PROVIDERS
# =============================================================================
class FileProvider:
    """Bars from files in a directory, for tests and offline runs.

    ``{EXCHANGE}_{SYMBOL}_1m.pkl`` (or ``.csv``) with columns epoch, open,
    high, low, close, volume[, oi]; optionally ``{EXCHANGE}_{SYMBOL}_D.pkl``
    with a ``day`` (IST day number) or ``date`` column. Without a daily file
    the daily bars are built from the one-minute bars.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _read(self, stem: str) -> pd.DataFrame | None:
        for suffix in (".pkl", ".csv"):
            path = self.directory / f"{stem}{suffix}"
            if path.exists():
                return pd.read_pickle(path) if suffix == ".pkl" else pd.read_csv(path)
        return None

    def load_1m(self, symbol: str, exchange: str, start: date, end: date) -> pd.DataFrame:
        df = self._read(f"{exchange}_{symbol}_1m".upper())
        if df is None or df.empty:
            return empty_bars()
        df = records_to_bars(df.rename(columns={"epoch": "timestamp"}).to_dict("records"))
        lo, hi = ist_midnight_epoch(start), ist_midnight_epoch(end) + DAY
        return df[(df["epoch"] >= lo) & (df["epoch"] < hi)].reset_index(drop=True)

    def load_daily(self, symbol: str, exchange: str, start: date, end: date) -> pd.DataFrame:
        df = self._read(f"{exchange}_{symbol}_D".upper())
        if df is None or df.empty:
            return empty_daily()
        df = df.copy()
        if "day" not in df.columns and "date" in df.columns:
            df["day"] = [date_to_day(parse_day(v)) for v in df["date"]]
        for col in DAILY_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        df = df[DAILY_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
        df["day"] = df["day"].astype("int64")
        return df[(df["day"] >= date_to_day(start)) & (df["day"] <= date_to_day(end))]

    def call_readonly(self, method: str, **kwargs):
        return None


class OpenAlgoProvider:
    """Bars from the running OpenAlgo app, through the real SDK client.

    The real client is used for exactly two things: the history endpoint and
    a handful of read-only metadata calls (symbol, search, expiry, intervals).
    It never places, modifies or cancels anything. A fresh client is made for
    each load and closed after it, inside ``real_time`` (the real clock, and the
    one window in which the network guard lets a connection reach the app), so
    no open connection to the app is left lying in the process for a script.
    """

    MINUTE_CHUNK_DAYS = 25  # inside every supported broker's one-minute limit
    DAILY_CHUNK_DAYS = 360
    API_PAUSE_SECONDS = 0.4  # the history endpoint allows about 3 a second

    def __init__(self, client_factory, cache_dir: str | Path | None, today: date, real_time=None):
        self._client_factory = client_factory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.today = today
        self.real_time = real_time or nullcontext
        self.sources_used: set[str] = set()
        self._last_api_call = 0.0

    @contextmanager
    def _client(self):
        with self.real_time():
            client = self._client_factory()
            try:
                yield client
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    # ---------- cache ----------
    def _cache_path(self, symbol, exchange, interval, start: date, end: date) -> Path | None:
        if not self.cache_dir:
            return None
        key = f"{exchange}|{symbol}|{interval}|{start}|{end}".upper()
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{exchange}_{symbol}_{interval}")
        return self.cache_dir / f"{safe}_{start:%Y%m%d}_{end:%Y%m%d}_{digest}.pkl"

    def _cache_get(self, path: Path | None):
        if path is None or not path.exists():
            return None
        try:
            return pd.read_pickle(path)
        except Exception:
            return None  # a damaged cache file is simply fetched again

    def _cache_put(self, path: Path | None, df: pd.DataFrame) -> None:
        if path is None or df.empty:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            df.to_pickle(tmp)
            os.replace(tmp, path)
        except Exception:
            pass  # without a cache the next run fetches again; nothing else depends on it

    # ---------- requests ----------
    def _request(self, client, symbol, exchange, interval, start: date, end: date, source: str):
        payload = {
            "apikey": client.api_key,
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "source": source,
        }
        if source == "api":
            wait = self.API_PAUSE_SECONDS - (time.monotonic() - self._last_api_call)
            if wait > 0:
                time.sleep(wait)
        result = None
        for attempt in range(3):
            result = client._make_request("history", payload)
            if source == "api":
                self._last_api_call = time.monotonic()
            message = str((result or {}).get("message", "")).lower()
            if (result or {}).get("status") == "error" and (
                "rate limit" in message or "too many" in message or "busy" in message
            ):
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        return result.get("data") or []

    def _covers(self, df: pd.DataFrame, start: date, end: date, daily: bool) -> bool:
        """Whether a chunk from Historify spans the dates asked for, near enough.

        A few days of slack at each edge, for holidays and weekends.
        """
        if df.empty:
            return False
        days = df["day"] if daily else (df["epoch"] + IST_OFFSET) // DAY
        first, last = day_to_date(int(days.min())), day_to_date(int(days.max()))
        last_wanted = min(end, self.today - timedelta(days=1))
        return first <= start + timedelta(days=4) and last >= last_wanted - timedelta(days=4)

    def _load(self, symbol, exchange, interval, start: date, end: date, chunk_days: int):
        daily = interval == "D"
        convert = records_to_daily if daily else records_to_bars
        frames, missing = [], []
        with self._client() as client:
            chunk_start = start
            while chunk_start <= end:
                chunk_end = min(end, chunk_start + timedelta(days=chunk_days - 1))
                frame = self._chunk(
                    client, symbol, exchange, interval, chunk_start, chunk_end, convert
                )
                if frame.empty:
                    missing.append((chunk_start, chunk_end))
                frames.append(frame)
                chunk_start = chunk_end + timedelta(days=1)
        for lo, hi in missing:
            label = "daily" if daily else "1-minute"
            logger.info(
                f"Could not load {exchange}:{symbol} {label} history for {lo} to {hi} from "
                "Historify or your broker. Download it in Historify, or check that the broker "
                "is logged in."
            )
        frames = [f for f in frames if not f.empty]
        if not frames:
            return empty_daily() if daily else empty_bars()
        key = "day" if daily else "epoch"
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(key).drop_duplicates(key, keep="last").reset_index(drop=True)

    def _chunk(self, client, symbol, exchange, interval, start, end, convert):
        daily = interval == "D"
        finished = end < self.today  # a chunk that ends today is still growing
        path = self._cache_path(symbol, exchange, interval, start, end) if finished else None
        cached = self._cache_get(path)
        if cached is not None:
            self.sources_used.add("cache")
            return cached

        db_rows = self._request(client, symbol, exchange, interval, start, end, "db")
        db_df = convert(db_rows or [])
        if self._covers(db_df, start, end, daily):
            self.sources_used.add("historify")
            self._cache_put(path, db_df)
            return db_df

        api_rows = self._request(client, symbol, exchange, interval, start, end, "api")
        api_df = convert(api_rows or [])
        best = api_df if len(api_df) >= len(db_df) else db_df
        if not best.empty:
            self.sources_used.add("broker" if best is api_df else "historify")
            self._cache_put(path, best)
        return best

    def load_1m(self, symbol: str, exchange: str, start: date, end: date) -> pd.DataFrame:
        return self._load(symbol, exchange, "1m", start, end, self.MINUTE_CHUNK_DAYS)

    def load_daily(self, symbol: str, exchange: str, start: date, end: date) -> pd.DataFrame:
        return self._load(symbol, exchange, "D", start, end, self.DAILY_CHUNK_DAYS)

    def call_readonly(self, method: str, **kwargs):
        """symbol, search, expiry, intervals: answered by the app itself."""
        if method not in ("symbol", "search", "expiry", "intervals"):
            return None
        try:
            with self._client() as client:
                return getattr(client, method)(**kwargs)
        except Exception:
            return None  # the caller falls back to a safe default


# =============================================================================
# ONE SYMBOL'S BARS
# =============================================================================
class SymbolSeries:
    """One symbol's one-minute bars and the daily bars, with as-of views."""

    def __init__(self, symbol: str, exchange: str, m1: pd.DataFrame, daily: pd.DataFrame):
        self.symbol = symbol
        self.exchange = exchange
        self.session: Session = session_for(exchange)

        m1 = m1.copy() if m1 is not None else empty_bars()
        if not m1.empty:
            sec = (m1["epoch"] + IST_OFFSET) % DAY
            m1 = m1[(sec >= self.session.open_sec) & (sec < self.session.close_sec)]
        self.t = m1["epoch"].to_numpy(dtype=np.int64)
        self.o = m1["open"].to_numpy(dtype=np.float64)
        self.h = m1["high"].to_numpy(dtype=np.float64)
        self.lo = m1["low"].to_numpy(dtype=np.float64)
        self.c = m1["close"].to_numpy(dtype=np.float64)
        self.v = m1["volume"].fillna(0).to_numpy(dtype=np.float64)
        self.oi = m1["oi"].fillna(0).to_numpy(dtype=np.float64)
        self.pos = {int(t): i for i, t in enumerate(self.t)}
        self.days = np.unique((self.t + IST_OFFSET) // DAY) if len(self.t) else np.array([], int)
        self._agg: dict[int, tuple] = {}
        self._build_daily(daily)

    @property
    def empty(self) -> bool:
        return len(self.t) == 0

    # ---------- daily ----------
    def _build_daily(self, daily: pd.DataFrame | None) -> None:
        """Daily bars from the provider, filled in from one-minute bars where missing."""
        rows: dict[int, tuple] = {}
        if daily is not None and not daily.empty:
            for r in daily.itertuples(index=False):
                rows[int(r.day)] = (r.open, r.high, r.low, r.close, r.volume, r.oi)
        b, o, h, lo, c, v, oi, _, _, _ = self._aggregate_days()
        for k in range(len(b)):
            day = int(day_of(b[k]))
            if day not in rows:
                rows[day] = (o[k], h[k], lo[k], c[k], v[k], oi[k])
        keys = sorted(rows)
        self.d_day = np.array(keys, dtype=np.int64)
        arr = np.array([rows[k] for k in keys], dtype=np.float64).reshape(-1, 6)
        self.d_o, self.d_h, self.d_l, self.d_c, self.d_v, self.d_oi = (arr[:, j] for j in range(6))

    def _aggregate_days(self):
        days = (self.t + IST_OFFSET) // DAY
        starts = days * DAY - IST_OFFSET + self.session.open_sec
        ends = days * DAY - IST_OFFSET + self.session.close_sec
        return self._reduce(starts, ends)

    # ---------- intraday aggregation ----------
    def _reduce(self, bucket_start: np.ndarray, bucket_end: np.ndarray):
        """Full bars per bucket: (start, open, high, low, close, volume, oi, end, i0, i1).

        i0 and i1 bound each bucket's one-minute bars, [i0, i1).
        """
        if len(self.t) == 0:
            z = np.array([], dtype=np.float64)
            n = np.array([], dtype=np.int64)
            return n, z, z, z, z, z, z, n, n, n
        change = np.flatnonzero(np.diff(bucket_start)) + 1
        i0 = np.concatenate(([0], change)).astype(np.int64)
        i1 = np.concatenate((change, [len(self.t)])).astype(np.int64)
        return (
            bucket_start[i0],
            self.o[i0],
            np.maximum.reduceat(self.h, i0),
            np.minimum.reduceat(self.lo, i0),
            self.c[i1 - 1],
            np.add.reduceat(self.v, i0),
            self.oi[i1 - 1],
            bucket_end[i0],
            i0,
            i1,
        )

    def _buckets(self, minutes: int):
        cached = self._agg.get(minutes)
        if cached is None:
            size = minutes * 60
            day_open = (self.t + IST_OFFSET) // DAY * DAY - IST_OFFSET + self.session.open_sec
            start = day_open + (self.t - day_open) // size * size
            close = day_open - self.session.open_sec + self.session.close_sec
            end = np.minimum(start + size, close)
            cached = self._reduce(start, end)
            self._agg[minutes] = cached
        return cached

    # ---------- as-of views ----------
    def last_index(self, now: float) -> int:
        """Index of the latest bar that has started at or before now, -1 if none."""
        return int(np.searchsorted(self.t, now, side="right")) - 1

    def price_at(self, now: float) -> float | None:
        """The last known price: the open of the bar in progress, else the last close."""
        i = self.last_index(now)
        if i < 0:
            return None
        return float(self.o[i]) if now < self.t[i] + 60 else float(self.c[i])

    def bar_at(self, start_epoch: int):
        """(open, high, low, close) of the one-minute bar starting at start_epoch."""
        i = self.pos.get(int(start_epoch))
        if i is None:
            return None
        return float(self.o[i]), float(self.h[i]), float(self.lo[i]), float(self.c[i])

    def _forming(self, i0: int, i1: int, now: float):
        """A bucket in progress, from sub-bars [i0, i1): complete ones plus the open of the current."""
        j = min(i1, int(np.searchsorted(self.t, now, side="right")))
        if j <= i0:
            return None
        last_complete = self.t[j - 1] + 60 <= now
        k = j if last_complete else j - 1  # sub-bars [i0, k) are complete
        o = self.o[i0]
        if k > i0:
            h, lo, c, v = self.h[i0:k].max(), self.lo[i0:k].min(), self.c[k - 1], self.v[i0:k].sum()
        else:
            h, lo, c, v = o, o, o, 0.0
        if not last_complete:
            cur = self.o[j - 1]
            h, lo, c = max(h, cur), min(lo, cur), cur
        return float(o), float(h), float(lo), float(c), float(v), float(self.oi[j - 1])

    def intraday(self, minutes: int, start: date, end: date, now: float) -> pd.DataFrame:
        b, o, h, lo, c, v, oi, be, i0, i1 = self._buckets(minutes)
        lo_e, hi_e = ist_midnight_epoch(start), ist_midnight_epoch(end) + DAY
        first = int(np.searchsorted(b, lo_e, side="left"))
        stop = int(np.searchsorted(b, hi_e, side="left"))
        complete = int(np.searchsorted(be, now, side="right"))  # buckets that have ended by now
        upto = max(first, min(stop, complete))
        parts = [arr[first:upto] for arr in (b, o, h, lo, c, v, oi)]
        k = complete  # the bucket in progress, if it has started and is in range
        if k < len(b) and first <= k < stop and b[k] <= now:
            forming = self._forming(int(i0[k]), int(i1[k]), now)
            if forming is not None:
                extra = (b[k],) + forming
                parts = [np.append(arr, value) for arr, value in zip(parts, extra, strict=True)]
        times, *cols, oi_col = parts
        index = pd.to_datetime(times.astype(np.int64), unit="s", utc=True)
        index = index.tz_convert("Asia/Kolkata").rename("timestamp")
        return _frame(index, cols, oi_col)

    def daily(self, start: date, end: date, now: float) -> pd.DataFrame:
        today = day_of(now)
        lo_d, hi_d = date_to_day(start), date_to_day(end)
        mask = (self.d_day >= lo_d) & (self.d_day <= hi_d) & (self.d_day < today)
        days = self.d_day[mask].tolist()
        cols = [arr[mask].tolist() for arr in (self.d_o, self.d_h, self.d_l, self.d_c, self.d_v)]
        oi = self.d_oi[mask].tolist()
        if lo_d <= today <= hi_d:
            forming = self.today_bar(now)
            if forming is not None:
                days.append(today)
                for col, value in zip(cols, forming[:5], strict=True):
                    col.append(value)
                oi.append(forming[5])
        index = pd.to_datetime([day_to_date(d) for d in days]).rename("timestamp")
        return _frame(index, cols, oi)

    def today_bar(self, now: float):
        """Today's daily bar so far, or None before the first trade."""
        today = day_of(now)
        lo_e = today * DAY - IST_OFFSET
        i0 = int(np.searchsorted(self.t, lo_e, side="left"))
        i1 = int(np.searchsorted(self.t, lo_e + DAY, side="left"))
        if i1 <= i0:
            return None
        return self._forming(i0, i1, now)

    def prev_close(self, now: float) -> float | None:
        today = day_of(now)
        k = int(np.searchsorted(self.d_day, today, side="left")) - 1
        return float(self.d_c[k]) if k >= 0 else None


def _frame(index, cols, oi) -> pd.DataFrame:
    o, h, lo, c, v = cols
    df = pd.DataFrame(
        {
            "open": np.asarray(o, dtype=np.float64),
            "high": np.asarray(h, dtype=np.float64),
            "low": np.asarray(lo, dtype=np.float64),
            "close": np.asarray(c, dtype=np.float64),
            "volume": np.asarray(v, dtype=np.float64).round().astype(np.int64),
            "oi": np.asarray(oi, dtype=np.float64).round().astype(np.int64),
        },
        index=index,
    )
    return df


# =============================================================================
# ALL SYMBOLS
# =============================================================================
class MarketData:
    """Every symbol the strategy has asked about, loaded on first use."""

    def __init__(
        self,
        provider,
        start: date,
        end: date,
        warmup_days: int = 15,
        daily_warmup_days: int = 120,
    ):
        self.provider = provider
        self.start = start
        self.end = end
        self.warmup_days = int(warmup_days)
        self.daily_warmup_days = int(daily_warmup_days)
        self.series: dict[tuple[str, str], SymbolSeries] = {}
        self.data_days: set[int] = set()
        self.load_seconds = 0.0
        self._history_cache: dict = {}

    @staticmethod
    def key(symbol, exchange) -> tuple[str, str]:
        return str(symbol or "").strip().upper(), str(exchange or "").strip().upper()

    def get(self, symbol, exchange) -> SymbolSeries:
        key = self.key(symbol, exchange)
        series = self.series.get(key)
        if series is None:
            began = time.perf_counter()
            sym, exch = key
            m1_start = self.start - timedelta(days=self.warmup_days)
            d_start = self.start - timedelta(days=self.daily_warmup_days)
            try:
                m1 = self.provider.load_1m(sym, exch, m1_start, self.end)
            except Exception:
                logger.info(f"Could not load {exch}:{sym} 1-minute history.")
                m1 = empty_bars()
            try:
                daily = self.provider.load_daily(sym, exch, d_start, self.end)
            except Exception:
                logger.info(f"Could not load {exch}:{sym} daily history.")
                daily = empty_daily()
            series = SymbolSeries(sym, exch, m1, daily)
            self.series[key] = series
            lo, hi = date_to_day(self.start), date_to_day(self.end)
            self.data_days.update(int(d) for d in series.days if lo <= d <= hi)
            self.load_seconds += time.perf_counter() - began
            in_range = sum(1 for d in series.days if lo <= d <= hi)
            logger.info(
                f"Loaded {sym} {exch}: {len(series.t)} one-minute bars, "
                f"{in_range} trading days in the test period, {len(series.d_day)} daily bars"
            )
        return series

    def has_any_data(self) -> bool:
        return bool(self.data_days)

    def history(self, symbol, exchange, interval, start_date, end_date, now: float):
        """A DataFrame shaped like the SDK's, or an error dict as the SDK returns."""
        minutes = interval_minutes(interval)
        if minutes is None:
            return {
                "status": "error",
                "message": f"Interval {interval} is not available in a backtest. "
                "Use 1m, 3m, 5m, 10m, 15m, 30m, 1h or D.",
                "error_type": "invalid_interval",
            }
        start, end = parse_day(start_date), parse_day(end_date)
        if start is None or end is None or start > end:
            return {
                "status": "error",
                "message": "start_date and end_date must be dates in YYYY-MM-DD form",
                "error_type": "invalid_dates",
            }
        series = self.get(symbol, exchange)
        cache_key = (series.symbol, series.exchange, minutes, start, end, int(now // 60))
        cached = self._history_cache.get(cache_key)
        if cached is None:
            df = (
                series.daily(start, end, now)
                if minutes == 0
                else series.intraday(minutes, start, end, now)
            )
            if len(self._history_cache) > 64:
                self._history_cache.clear()
            self._history_cache[cache_key] = df
            cached = df
        if cached.empty:
            return {
                "status": "error",
                "message": "No data available for the specified period",
                "error_type": "no_data",
            }
        return cached.copy()

    def summary(self) -> dict:
        lo, hi = date_to_day(self.start), date_to_day(self.end)
        return {
            f"{s.symbol}:{s.exchange}": {
                "minute_bars": int(len(s.t)),
                "trading_days_in_period": int(sum(1 for d in s.days if lo <= d <= hi)),
            }
            for s in self.series.values()
        }


def describe_sources(provider) -> list[str]:
    return sorted(getattr(provider, "sources_used", set()))
