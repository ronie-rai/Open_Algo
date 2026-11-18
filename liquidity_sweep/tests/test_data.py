"""Data loading from a Historify-shaped duckdb file and a mocked API. No network."""

from datetime import date

import duckdb
import pandas as pd
import pytest

import data as data_mod
from data import (
    ApiUnavailable,
    data_range,
    load_basket,
    load_from_api,
    load_from_historify,
    load_symbol,
    read_cache,
    safe_name,
    split_time,
)
from synthetic import session_index

TZ = "Asia/Kolkata"
START, END = date(2025, 11, 1), date(2026, 10, 31)

# Same columns as database/historify_db.py creates.
MARKET_DATA = """
CREATE TABLE market_data (
    symbol VARCHAR NOT NULL, exchange VARCHAR NOT NULL, interval VARCHAR NOT NULL,
    timestamp BIGINT NOT NULL, open DOUBLE NOT NULL, high DOUBLE NOT NULL,
    low DOUBLE NOT NULL, close DOUBLE NOT NULL, volume BIGINT NOT NULL,
    oi BIGINT DEFAULT 0, created_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, exchange, interval, timestamp))
"""
DATA_CATALOG = """
CREATE TABLE data_catalog (
    id INTEGER PRIMARY KEY, symbol VARCHAR NOT NULL, exchange VARCHAR NOT NULL,
    interval VARCHAR NOT NULL, first_timestamp BIGINT, last_timestamp BIGINT,
    record_count BIGINT DEFAULT 0, last_download_at TIMESTAMP,
    UNIQUE (symbol, exchange, interval))
"""


def minute_rows(days=("2026-01-05", "2026-01-06"), minutes=1):
    idx = session_index(list(days), minutes=minutes)
    # Rows outside the session that must be filtered: pre-open, after close, Saturday.
    extra = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-05 09:00", tz=TZ),
            pd.Timestamp("2026-01-05 15:30", tz=TZ),
            pd.Timestamp("2026-01-10 10:00", tz=TZ),
        ]
    )
    idx = idx.append(extra)
    epoch = idx.tz_convert("UTC").asi8 // 10**9
    k = pd.RangeIndex(len(idx))
    return pd.DataFrame(
        {
            "timestamp": epoch,
            "open": 100.0 + k * 0.01,
            "high": 100.5 + k * 0.01,
            "low": 99.5 + k * 0.01,
            "close": 100.2 + k * 0.01,
            "volume": 10,
        }
    )


def make_db(path, rows, symbol="SBIN", interval="1m"):
    con = duckdb.connect(str(path))
    con.execute(MARKET_DATA)
    con.execute(DATA_CATALOG)
    for r in rows.itertuples(index=False):
        con.execute(
            "INSERT INTO market_data (symbol, exchange, interval, timestamp, open, high, low, close, volume, oi)"
            " VALUES (?, 'NSE', ?, ?, ?, ?, ?, ?, ?, 0)",
            [symbol, interval, int(r.timestamp), r.open, r.high, r.low, r.close, int(r.volume)],
        )
    con.close()


def test_data_range_is_last_12_complete_months():
    assert data_range(date(2026, 11, 18)) == (date(2025, 11, 1), date(2026, 10, 31))
    assert data_range(date(2026, 1, 5)) == (date(2025, 1, 1), date(2025, 12, 31))


def test_split_time_is_70_percent_of_the_calendar_range():
    s = split_time(START, END, 0.7)
    lo, hi = pd.Timestamp("2025-11-01", tz=TZ), pd.Timestamp("2026-11-01", tz=TZ)
    assert s == lo + (hi - lo) * 0.7


def test_safe_name():
    assert safe_name("M&M") == "M_and_M"
    assert safe_name("SBIN") == "SBIN"


def test_historify_1m_aggregated_to_5m(tmp_path):
    db = tmp_path / "historify.duckdb"
    rows = minute_rows()
    make_db(db, rows)
    bars, note = load_from_historify("SBIN", START, END, db)
    assert note == "historify 1m"
    assert str(bars.index.tz) == TZ
    assert len(bars) == 150  # 75 five-minute bars per session, extras filtered out
    assert bars.index[0] == pd.Timestamp("2026-01-05 09:15", tz=TZ)
    assert bars.index[-1] == pd.Timestamp("2026-01-06 15:25", tz=TZ)
    first5 = rows.iloc[0:5]
    assert bars.iloc[0]["open"] == first5["open"].iloc[0]
    assert bars.iloc[0]["close"] == first5["close"].iloc[-1]
    assert bars.iloc[0]["high"] == first5["high"].max()
    assert bars.iloc[0]["volume"] == 50
    assert all(pd.api.types.is_float_dtype(t) for t in bars.dtypes)


def test_historify_prefers_5m_rows(tmp_path):
    db = tmp_path / "historify.duckdb"
    make_db(db, minute_rows(minutes=5), interval="5m")
    bars, note = load_from_historify("SBIN", START, END, db)
    assert note == "historify 5m" and len(bars) == 150


def test_historify_missing_symbol_and_missing_file(tmp_path):
    db = tmp_path / "historify.duckdb"
    make_db(db, minute_rows())
    bars, note = load_from_historify("TCS", START, END, db)
    assert bars.empty
    bars, note = load_from_historify("SBIN", START, END, tmp_path / "nope.duckdb")
    assert bars.empty and "not found" in note


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class FakeHttp:
    """Records requests; answers 5m with an error and 1m with session bars."""

    def __init__(self, five_minute_ok=False, fail=False):
        self.calls = []
        self.five_minute_ok = five_minute_ok
        self.fail = fail

    def post(self, url, json, timeout):
        self.calls.append((url, dict(json), timeout))
        if self.fail:
            raise ConnectionError("refused")
        a, b = pd.Timestamp(json["start_date"]), pd.Timestamp(json["end_date"])
        days = [
            d
            for d in pd.bdate_range(a, b)
            if pd.Timestamp("2026-01-05") <= d <= pd.Timestamp("2026-01-06")
        ]
        if json["interval"] == "5m" and not self.five_minute_ok:
            return FakeResponse({"status": "error", "message": "Interval not supported"}, 400)
        minutes = 5 if json["interval"] == "5m" else 1
        if not days:
            return FakeResponse({"status": "success", "data": []})
        rows = minute_rows(days=[d.strftime("%Y-%m-%d") for d in days], minutes=minutes)
        return FakeResponse(
            {"status": "success", "data": rows.assign(oi=0).to_dict(orient="records")}
        )


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(data_mod, "API_PAUSE_SECONDS", 0)


def test_api_falls_back_to_1m_and_chunks_requests():
    http = FakeHttp()
    bars, note = load_from_api("SBIN", START, END, host="http://h:1", api_key="k-test", http=http)
    assert note == "openalgo api 1m"
    assert len(bars) == 150
    urls = {c[0] for c in http.calls}
    assert urls == {"http://h:1/api/v1/history"}
    for _, body, timeout in http.calls:
        span = (pd.Timestamp(body["end_date"]) - pd.Timestamp(body["start_date"])).days + 1
        assert span <= data_mod.API_CHUNK_DAYS[body["interval"]]
        assert body["exchange"] == "NSE" and timeout
    intervals = [c[1]["interval"] for c in http.calls]
    assert "5m" in intervals and "1m" in intervals


def test_api_uses_5m_when_available():
    http = FakeHttp(five_minute_ok=True)
    bars, note = load_from_api("SBIN", START, END, host="http://h:1", api_key="k-test", http=http)
    assert note == "openalgo api 5m" and len(bars) == 150


def test_api_needs_key_from_env(monkeypatch):
    monkeypatch.delenv("OPENALGO_API_KEY", raising=False)
    http = FakeHttp()
    bars, note = load_from_api("SBIN", START, END, http=http)
    assert bars.empty and "OPENALGO_API_KEY" in note and not http.calls


def test_api_down_raises_trader_message():
    with pytest.raises(ApiUnavailable) as err:
        load_from_api(
            "SBIN", START, END, host="http://h:1", api_key="k-test", http=FakeHttp(fail=True)
        )
    assert "Start OpenAlgo" in str(err.value)


def test_load_symbol_caches_and_never_writes_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALGO_API_KEY", "secret-key-value")
    http = FakeHttp()
    res = load_symbol(
        "M&M", START, END, data_dir=tmp_path, db_path=tmp_path / "none.duckdb", http=http
    )
    assert res.source == "openalgo api 1m"
    cached = tmp_path / "M_and_M_5m.csv"
    assert cached.exists()
    back = read_cache(cached)
    pd.testing.assert_frame_equal(back, res.bars, check_freq=False)
    for f in tmp_path.iterdir():
        assert "secret-key-value" not in f.read_text()
    assert "M&M\topenalgo api 1m\t2025-11-01..2026-10-31" in (tmp_path / "source.txt").read_text()
    # Second call reads the cache without touching the API.
    calls = len(http.calls)
    again = load_symbol(
        "M&M", START, END, data_dir=tmp_path, db_path=tmp_path / "none.duckdb", http=http
    )
    assert again.message == "cached" and len(http.calls) == calls


def test_basket_skips_symbols_without_data(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENALGO_API_KEY", raising=False)
    db = tmp_path / "historify.duckdb"
    make_db(db, minute_rows())
    lines = []
    loaded, info, rng = load_basket(
        ["SBIN", "TCS"],
        today=date(2026, 11, 18),
        data_dir=tmp_path / "data",
        db_path=db,
        log=lines.append,
    )
    assert list(loaded) == ["SBIN"]
    assert any("TCS: no data, skipped" in line for line in lines)
    assert rng == (START, END)


def test_run_exits_cleanly_without_data(monkeypatch, capsys):
    import run

    monkeypatch.setattr(run, "load_basket", lambda symbols, refresh: ({}, {}, (START, END)))
    with pytest.raises(SystemExit) as exc:
        run.main(["--symbols", "SBIN"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "Historify" in out and "OPENALGO_API_KEY" in out and "Traceback" not in out


def test_api_failed_chunks_are_reported():
    class OneFails(FakeHttp):
        def post(self, url, json, timeout):
            if json["interval"] == "1m" and json["start_date"] == "2025-11-01":
                self.calls.append((url, dict(json), timeout))
                return FakeResponse({"status": "error", "message": "Broker busy"}, 500)
            return super().post(url, json, timeout)

    bars, note = load_from_api(
        "SBIN", START, END, host="http://h:1", api_key="k-test", http=OneFails()
    )
    assert len(bars) == 150
    assert "1 of" in note and "gaps" in note
