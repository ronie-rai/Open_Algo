"""Real strategies through the real runner, on synthetic bars.

Each test writes one-minute bars to files, points a run config at them
(``data_dir``, so no app and no broker are needed), and starts
``python -m services.python_backtest.runner`` as the /python page does. The
strategies are the shipped examples, copied unchanged into the run directory.

The SMC example is run with a stand-in ``litellm`` package first on the path
that records any call: in a backtest the strategy must not call the model
unless asked to.

Run:
    uv run pytest test/test_python_backtest_e2e.py -v
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from services.python_backtest.sessions import date_to_day, session_for

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "strategies" / "examples"


def write_bars(directory: Path, symbol: str, exchange: str, start: date, end: date, base, seed):
    rng = np.random.default_rng(seed)
    session = session_for(exchange)
    rows, price, d = [], float(base), start
    while d <= end:
        if d.weekday() < 5:
            t0 = session.open_epoch(date_to_day(d))
            for i in range(375):
                o = price
                c = max(1.0, o * (1 + rng.normal(0, 0.0015)))
                h = max(o, c) * (1 + abs(rng.normal(0, 0.0005)))
                lo = min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
                rows.append((t0 + 60 * i, o, h, lo, c, int(rng.integers(100, 1000)), 0))
                price = c
        d += timedelta(days=1)
    df = pd.DataFrame(rows, columns=["epoch", "open", "high", "low", "close", "volume", "oi"])
    directory.mkdir(parents=True, exist_ok=True)
    df.to_pickle(directory / f"{exchange}_{symbol}_1m.pkl".upper())


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bars")
    start, end = date(2026, 8, 24), date(2026, 9, 25)
    write_bars(directory, "NHPC", "NSE", start, end, 80, 1)
    write_bars(directory, "HDFCBANK", "NSE", start, end, 1600, 2)
    write_bars(directory, "NIFTY", "NSE_INDEX", start, end, 25000, 4)
    return directory


def run_backtest(
    tmp_path, script: Path, data_dir, start, end, extra_env=None, timeout=240, config_extra=None
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    copy = run_dir / script.name
    shutil.copy2(script, copy)
    config = {
        "run_dir": str(run_dir),
        "script": str(copy),
        "start_date": start,
        "end_date": end,
        "exchange": "NSE",
        "capital": 100000,
        "brokerage_per_order": 20,
        "slippage_bps": 2,
        "max_wall_seconds": timeout,
        **(config_extra or {}),
    }
    if data_dir is not None:
        config["data_dir"] = str(data_dir)
    (run_dir / "config.json").write_text(json.dumps(config))
    env = {**os.environ, "OPENALGO_API_KEY": "test-key", **(extra_env or {})}
    env.pop("OPENALGO_BACKTEST", None)
    with open(run_dir / "run.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "services.python_backtest.runner",
                "--config",
                str(run_dir / "config.json"),
            ],
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout + 60,
        )
    log_text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    assert proc.returncode == 0, log_text[-3000:]
    result = json.loads((run_dir / "result.json").read_text())
    return run_dir, result, log_text


@pytest.mark.timeout(400)
def test_simple_ema_strategy_runs_unchanged_and_trades(tmp_path, data_dir):
    original = (EXAMPLES / "simple_ema_strategy.py").read_bytes()
    run_dir, result, log = run_backtest(
        tmp_path, EXAMPLES / "simple_ema_strategy.py", data_dir, "2026-09-15", "2026-09-17"
    )
    assert result["status"] == "completed", result.get("message")
    m = result["metrics"]
    assert m["trades"] > 0
    assert m["trading_days"] == 3
    assert m["starting_capital"] == 100000
    assert m["net_pnl"] == pytest.approx(m["final_equity"] - 100000, abs=0.01)
    assert m["total_costs"] > 0
    assert len(result["equity_curve"]) > 10
    assert [d["date"] for d in result["daily"]] == ["2026-09-15", "2026-09-16", "2026-09-17"]
    trade = result["trades"][0]
    assert trade["symbol"] == "NHPC" and trade["side"] in ("LONG", "SHORT")
    assert trade["entry_time"] < trade["exit_time"]
    # Never anything open at the end: MIS squared off, the rest closed at the end.
    assert all(o["order_status"] != "open" for o in result["orders"])
    assert "Starting 5/10 EMA Crossover Strategy" in log
    assert json.loads((run_dir / "progress.json").read_text())["status"] == "finished"
    assert (EXAMPLES / "simple_ema_strategy.py").read_bytes() == original


@pytest.mark.timeout(400)
@pytest.mark.skipif(
    not (EXAMPLES / "smc_llm_levels_ob.py").exists(), reason="the SMC example is not present"
)
def test_smc_strategy_runs_without_calling_the_llm(tmp_path, data_dir):
    stub = tmp_path / "stub" / "litellm"
    stub.mkdir(parents=True)
    marker = tmp_path / "litellm_called"
    (stub / "__init__.py").write_text(
        "import os\n"
        "def completion(*args, **kwargs):\n"
        "    open(os.environ['LITELLM_MARKER'], 'w').write('called')\n"
        "    raise RuntimeError('the LLM must not be called in a backtest')\n"
    )
    run_dir, result, log = run_backtest(
        tmp_path,
        EXAMPLES / "smc_llm_levels_ob.py",
        data_dir,
        "2026-09-08",
        "2026-09-25",
        extra_env={
            "PYTHONPATH": str(tmp_path / "stub"),
            "LITELLM_MARKER": str(marker),
        },
    )
    assert result["status"] == "completed", result.get("message")
    assert not marker.exists(), "litellm.completion was called during a backtest"
    assert "the LLM is off" in log
    # The strategy's own state and journal land in the run directory, beside the copy.
    assert (run_dir / "smc_llm_data" / "state_HDFCBANK.json").exists()
    journal = run_dir / "smc_llm_data" / "journal_HDFCBANK.jsonl"
    plans = (
        [
            json.loads(line)
            for line in journal.read_text().splitlines()
            if json.loads(line)["event"] == "plan"
        ]
        if journal.exists()
        else []
    )
    assert plans, "no order block trigger reached the planner in the test period"
    assert all("mechanical (backtest)" in p["reply"] for p in plans)
    assert result["metrics"]["trades"] >= 1
    assert all(t["symbol"] == "HDFCBANK" for t in result["trades"])


@pytest.mark.timeout(200)
def test_no_data_finishes_with_a_message_a_trader_can_act_on(tmp_path, data_dir):
    script = tmp_path / "nodata.py"
    script.write_text(
        "import time\n"
        "from openalgo import api\n"
        "client = api(api_key='k', host='http://127.0.0.1:5000')\n"
        "while True:\n"
        "    try:\n"
        "        df = client.history(symbol='NOSUCH', exchange='NSE', interval='5m',\n"
        "                            start_date='2026-09-01', end_date='2026-09-10')\n"
        "    except Exception as e:\n"
        "        print('error', e)\n"
        "    time.sleep(5)\n"
    )
    _run_dir, result, _log = run_backtest(tmp_path, script, data_dir, "2026-09-07", "2026-09-10")
    assert result["status"] == "failed"
    assert "No historical data was found for NOSUCH" in result["message"]
    assert "Historify" in result["message"]


@pytest.mark.timeout(200)
def test_a_strategy_that_swallows_everything_still_ends(tmp_path, data_dir):
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import time\n"
        "from openalgo import api\n"
        "client = api(api_key='k')\n"
        "client.placeorder(symbol='NHPC', exchange='NSE', action='BUY', quantity=1, product='CNC')\n"
        "while True:\n"
        "    try:\n"
        "        time.sleep(60)\n"
        "    except BaseException:\n"
        "        pass\n"
    )
    _run_dir, result, log = run_backtest(tmp_path, script, data_dir, "2026-09-14", "2026-09-15")
    assert result["status"] == "completed"
    assert result["metrics"]["trades"] == 1
    assert result["trades"][0]["exit_reason"] == "end of backtest"
    assert "kept running after the test period ended" in log


# =============================================================================
# A BACKTEST CANNOT REACH THE APP
# =============================================================================
# A stand-in for the OpenAlgo app on a free local port: it answers the history
# endpoint with synthetic bars, as the real app does, and records every request
# that reaches it. The run is pointed at it as its host, so the data loader
# reads through it while a hostile strategy tries every way it can to reach it.


def _sink_bars(start: date, end: date) -> list[dict]:
    session = session_for("NSE")
    rows, price, d = [], 500.0, start
    while d <= end:
        if d.weekday() < 5:
            t0 = session.open_epoch(date_to_day(d))
            for i in range(375):
                o = price
                c = o + (0.2 if (i // 20) % 2 == 0 else -0.2)
                rows.append(
                    {
                        "timestamp": t0 + 60 * i,
                        "open": o,
                        "high": max(o, c) + 0.05,
                        "low": min(o, c) - 0.05,
                        "close": c,
                        "volume": 100,
                        "oi": 0,
                    }
                )
                price = c
        d += timedelta(days=1)
    return rows


@pytest.fixture
def app_sink():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _record(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                body = {}
            with lock:
                seen.append({"method": self.command, "path": self.path, "body": body})
            return body

        def do_POST(self):
            body = self._record()
            if self.path == "/api/v1/history" and body.get("interval") == "1m":
                start = date.fromisoformat(body["start_date"])
                end = date.fromisoformat(body["end_date"])
                payload = {"status": "success", "data": _sink_bars(start, end)}
            elif self.path == "/api/v1/history":
                payload = {"status": "error", "message": "No data found"}
            else:
                payload = {"status": "success", "orderid": "REAL-ORDER"}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="backtest-app-sink", daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


HOSTILE = r"""
import asyncio, os, socket, subprocess, sys, time
from openalgo import api

HOST, PORT = "127.0.0.1", int(os.environ["SINK_PORT"])
URL = f"http://{HOST}:{PORT}"
ORDER = b"POST /api/v1/placesmartorder HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}"

print("KEY", os.getenv("OPENALGO_API_KEY"))
client = api(api_key=os.getenv("OPENALGO_API_KEY"), host=URL)
df = client.history(symbol="SBIN", exchange="NSE", interval="5m",
                    start_date="2026-09-10", end_date="2026-09-15")
print("HISTORY", type(df).__name__, len(df))


def attempt(name, fn):
    try:
        fn()
        print(f"VECTOR {name}: SENT", flush=True)
    except Exception as e:
        print(f"VECTOR {name}: FAILED {type(e).__name__}", flush=True)


def v_http_client():
    import http.client
    c = http.client.HTTPConnection(HOST, PORT, timeout=3)
    c.request("POST", "/api/v1/placeorder", body="{}")
    c.getresponse().read()

def v_urllib3():
    import urllib3
    urllib3.PoolManager().request("POST", URL + "/api/v1/placeorder", body=b"{}",
                                  retries=False, timeout=3)

def v_aiohttp():
    import aiohttp
    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.post(URL + "/api/v1/placeorder", json={}) as r:
                await r.text()
    asyncio.run(go())  # the Proactor loop on Windows

def v_asyncio_stream():
    async def go():
        reader, writer = await asyncio.open_connection(HOST, PORT)
        writer.write(ORDER)
        await writer.drain()
        await reader.read(10)
        writer.close()
    asyncio.run(go())

def v_raw_socket():
    s = socket.create_connection((HOST, PORT), timeout=3)
    s.sendall(ORDER)
    s.recv(10)
    s.close()

def v_connect_ex():
    s = socket.socket()
    s.settimeout(3)
    rc = s.connect_ex((HOST, PORT))
    if rc:
        raise OSError(rc)
    s.sendall(ORDER)
    s.recv(10)

def v_localhost_name():
    s = socket.create_connection(("localhost", PORT), timeout=3)
    s.sendall(ORDER)
    s.recv(10)

def v_httpx_transport():
    import httpx
    httpx.HTTPTransport().handle_request(httpx.Request("POST", URL + "/api/v1/placeorder", json={}))

def v_encoded_slash():
    import httpx
    httpx.post(URL + "/api%2Fv1/placeorder", json={}, timeout=3)

def v_double_slash():
    import httpx
    httpx.post(URL + "/api//v1/placeorder", json={}, timeout=3)

def v_requests():
    import requests
    requests.post(URL + "/api/v1/placeorder", json={}, timeout=3)

def v_urllib():
    import urllib.request
    req = urllib.request.Request(URL + "/api/v1/placeorder", data=b"{}", method="POST")
    urllib.request.urlopen(req, timeout=3).read()

def v_subprocess_curl():
    subprocess.run(["curl", "-s", "-X", "POST", URL + "/api/v1/placeorder"], timeout=15, check=True)

def v_os_system_curl():
    os.system(f"curl -s -X POST {URL}/api/v1/placeorder")

def v_os_popen_curl():
    os.popen(f"curl -s -X POST {URL}/api/v1/placeorder").read()

def v_python_child():
    subprocess.Popen([sys.executable, "-c", "print(1)"]).wait()

def v_import_order_service():
    import services.place_order_service  # noqa: F401

def v_import_database():
    from database import auth_db  # noqa: F401

def v_import_broker():
    import broker  # noqa: F401

def v_import_restx():
    import restx_api  # noqa: F401

def v_import_sandbox():
    import sandbox  # noqa: F401


for name, fn in list(globals().items()):
    if name.startswith("v_"):
        attempt(name[2:], fn)

import json, pandas  # ordinary imports still work
print("ORDINARY IMPORTS OK", pandas.__name__)
time.sleep(10**7)
"""

VECTORS = [
    "http_client", "urllib3", "aiohttp", "asyncio_stream", "raw_socket", "connect_ex",
    "localhost_name", "httpx_transport", "encoded_slash", "double_slash", "requests", "urllib",
    "subprocess_curl", "os_system_curl", "os_popen_curl", "python_child",
    "import_order_service", "import_database", "import_broker", "import_restx", "import_sandbox",
]  # fmt: skip


@pytest.mark.timeout(300)
def test_nothing_but_history_reaches_the_app(tmp_path, app_sink):
    port, seen = app_sink
    url = f"http://127.0.0.1:{port}"
    script = tmp_path / "hostile.py"
    script.write_text(HOSTILE)
    env = {
        "SINK_PORT": str(port),
        "OPENALGO_API_KEY": "real-key-123",
        "HOST_SERVER": url,
        "MCP_LOOPBACK_URL": "",
        "WEBSOCKET_URL": "",
    }
    _run_dir, result, log = run_backtest(
        tmp_path,
        script,
        None,
        "2026-09-14",
        "2026-09-15",
        extra_env=env,
        config_extra={"host": url, "cache_dir": str(tmp_path / "cache")},
    )
    lines = {
        line.split(":")[0].removeprefix("VECTOR ").strip(): line
        for line in log.splitlines()
        if line.startswith("VECTOR ")
    }
    assert sorted(lines) == sorted(VECTORS), log[-3000:]
    for name, line in lines.items():
        assert "FAILED" in line, f"{name} got through: {line}"

    paths = {r["path"] for r in seen}
    assert paths == {"/api/v1/history"}, seen
    assert all(r["body"].get("apikey") == "real-key-123" for r in seen)  # the loader keeps the key

    assert "KEY backtest-no-real-key" in log
    assert "real-key-123" not in log
    assert "HISTORY DataFrame" in log
    assert "ORDINARY IMPORTS OK pandas" in log
    assert "tried to connect to OpenAlgo directly" in log
    assert "tried to start another program" in log
    assert "tried to import" in log
    assert result["status"] == "completed"


@pytest.mark.timeout(200)
def test_exit_handlers_run_and_open_files_are_flushed(tmp_path, data_dir):
    out = tmp_path / "journal.txt"
    script = tmp_path / "keeps_a_file.py"
    script.write_text(
        "import atexit, time\n"
        f"fh = open({str(out)!r}, 'w', buffering=1 << 20)\n"
        "fh.write('line written without a flush\\n')\n"
        "atexit.register(lambda: (fh.write('closed by atexit\\n'), fh.close()))\n"
        "time.sleep(10**7)\n"
    )
    _run_dir, result, _log = run_backtest(tmp_path, script, data_dir, "2026-09-14", "2026-09-14")
    assert result["status"] == "completed"
    assert out.read_text() == "line written without a flush\nclosed by atexit\n"


@pytest.mark.timeout(200)
def test_a_loop_of_client_calls_that_never_pauses_ends_with_a_reason(tmp_path, data_dir):
    script = tmp_path / "never_pauses.py"
    script.write_text(
        "from openalgo import api\n"
        "client = api(api_key='k')\n"
        "while True:\n"
        "    client.quotes(symbol='NHPC', exchange='NSE')\n"
    )
    _run_dir, result, _log = run_backtest(
        tmp_path, script, data_dir, "2026-09-14", "2026-09-18", timeout=5
    )
    assert result["status"] == "stopped"
    assert "never pauses" in result["message"]
