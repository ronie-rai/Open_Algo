"""The /python backtest endpoints: validation, ownership, one run at a time, paths.

The runner subprocess is replaced by a stand-in Popen, so these tests check
what the routes and the manager do with a run, not the replay itself (that is
test_python_backtest_engine.py and test_python_backtest_e2e.py).

Run:
    uv run pytest test/test_python_backtest_endpoints.py -v
"""

import json
import os
from datetime import date, datetime, timedelta

import pytest
from flask import Flask

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import utils.session  # noqa: E402
from blueprints import python_strategy as ps  # noqa: E402
from services.python_backtest import manager  # noqa: E402
from services.python_backtest.sessions import IST  # noqa: E402

TODAY = datetime.now(IST).date()


class FakePopen:
    instances: list["FakePopen"] = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = None
        self.terminated = False
        FakePopen.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def kill(self):
        self.returncode = -9


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.session, "is_session_valid", lambda: True)
    monkeypatch.setattr(manager, "BACKTESTS_DIR", tmp_path / "backtests")
    monkeypatch.setattr(manager, "CACHE_DIR", tmp_path / "backtests" / "_cache")
    monkeypatch.setattr(manager.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(manager, "_RUNS", {})
    monkeypatch.setattr(manager, "_shutting_down", False)
    monkeypatch.setattr(manager, "STOP_GRACE_SECONDS", 0.2)
    import database.auth_db as auth_db

    monkeypatch.setattr(auth_db, "get_api_key_for_tradingview", lambda user: "secret-key")
    FakePopen.instances.clear()

    script = tmp_path / "my_strategy.py"
    script.write_text("print('hello')\n")
    ps.STRATEGY_CONFIGS.clear()
    ps.STRATEGY_CONFIGS["mine_20260101"] = {
        "name": "Mine",
        "file_path": str(script),
        "file_name": script.name,
        "user_id": "alice",
        "exchange": "NSE",
    }
    ps.STRATEGY_CONFIGS["theirs_20260101"] = {
        "name": "Theirs",
        "file_path": str(script),
        "user_id": "mallory",
    }
    application = Flask(__name__)
    application.config["TESTING"] = True
    application.secret_key = "test-only"
    application.register_blueprint(ps.python_strategy_bp)
    yield application
    ps.STRATEGY_CONFIGS.clear()


@pytest.fixture
def client(app):
    c = app.test_client()
    with c.session_transaction() as sess:
        sess["user"] = "alice"
    return c


def body(start=None, end=None, **extra):
    end = end or TODAY - timedelta(days=1)
    start = start or end - timedelta(days=10)
    return {"start_date": str(start), "end_date": str(end), **extra}


def start_run(client, sid="mine_20260101", **kw):
    return client.post(f"/python/api/backtest/{sid}", json=body(**kw))


class TestValidation:
    @pytest.mark.parametrize(
        "payload,needle",
        [
            ({"start_date": "x", "end_date": "y"}, "start date and an end date"),
            (body(end=TODAY + timedelta(days=1)), "future"),
            (body(start=TODAY - timedelta(days=2), end=TODAY - timedelta(days=5)), "on or before"),
            (body(start=TODAY - timedelta(days=500)), "at most"),
            (body(capital="lots"), "Capital must be a number"),
            (body(capital=10), "Capital must be between"),
            (body(slippage_bps=-1), "Slippage"),
        ],
    )
    def test_refused_with_a_reason(self, client, payload, needle):
        resp = client.post("/python/api/backtest/mine_20260101", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error" and needle in data["message"]
        assert not FakePopen.instances

    def test_defaults_fill_in(self):
        params, err = manager.validate_request(
            {"start_date": "2026-01-01", "end_date": "2026-01-31"}, today=date(2026, 2, 1)
        )
        assert err == ""
        assert params["capital"] == 100000 and params["brokerage_per_order"] == 20
        assert params["slippage_bps"] == 2


class TestStart:
    def test_starts_a_runner_and_returns_the_run(self, client, tmp_path):
        resp = start_run(client, capital=50000, brokerage_per_order=10, slippage_bps=5)
        assert resp.status_code == 200, resp.get_json()
        run_id = resp.get_json()["data"]["run_id"]
        assert manager.valid_run_id(run_id)

        proc = FakePopen.instances[0]
        assert proc.cmd[1:5] == ["-u", "-m", "services.python_backtest.runner", "--config"]
        env = proc.kwargs["env"]
        assert env["OPENALGO_API_KEY"] == "secret-key"
        assert env["OPENALGO_BACKTEST"] == "1"
        assert env["STRATEGY_ID"] == "mine_20260101"
        assert env["OPENALGO_STRATEGY_EXCHANGE"] == "NSE"
        assert proc.kwargs["stdout"].closed  # the parent keeps no handle on run.log

        run_dir = tmp_path / "backtests" / "mine_20260101" / run_id
        config = json.loads((run_dir / "config.json").read_text())
        assert config["capital"] == 50000 and config["slippage_bps"] == 5
        assert "secret-key" not in (run_dir / "config.json").read_text()
        assert (run_dir / "my_strategy.py").read_text() == "print('hello')\n"

    def test_one_run_per_strategy(self, client):
        assert start_run(client).status_code == 200
        second = start_run(client)
        assert second.status_code == 409
        assert "already running" in second.get_json()["message"]

    def test_a_finished_run_frees_the_strategy(self, client):
        assert start_run(client).status_code == 200
        FakePopen.instances[0].returncode = 0
        assert start_run(client).status_code == 200

    def test_someone_elses_strategy_is_refused(self, client):
        resp = start_run(client, sid="theirs_20260101")
        assert resp.status_code == 403
        assert not FakePopen.instances

    def test_no_api_key_says_what_to_do(self, client, monkeypatch):
        import database.auth_db as auth_db

        monkeypatch.setattr(auth_db, "get_api_key_for_tradingview", lambda user: None)
        resp = start_run(client)
        assert resp.status_code == 400
        assert "API Key page" in resp.get_json()["message"]

    def test_does_not_touch_the_live_strategy(self, client):
        before = dict(ps.STRATEGY_CONFIGS["mine_20260101"])
        start_run(client)
        assert ps.STRATEGY_CONFIGS["mine_20260101"] == before
        assert "mine_20260101" not in ps.RUNNING_STRATEGIES


class TestStatusAndStop:
    def test_status_while_running_then_done(self, client, tmp_path):
        run_id = start_run(client).get_json()["data"]["run_id"]
        url = f"/python/api/backtest/mine_20260101/{run_id}"
        run_dir = tmp_path / "backtests" / "mine_20260101" / run_id
        (run_dir / "progress.json").write_text(json.dumps({"percent": 40.0}))
        (run_dir / "run.log").write_text("line 1\nline 2\n")

        data = client.get(url).get_json()["data"]
        assert data["status"] == "running"
        assert data["progress"]["percent"] == 40.0
        assert data["log_tail"].endswith("line 2")
        assert "result" not in data

        (run_dir / "result.json").write_text(
            json.dumps({"status": "completed", "message": "", "metrics": {"net_pnl": 12.5}})
        )
        FakePopen.instances[0].returncode = 0
        data = client.get(url).get_json()["data"]
        assert data["status"] == "completed"
        assert data["result"]["metrics"]["net_pnl"] == 12.5

        runs = client.get("/python/api/backtest/mine_20260101").get_json()["data"]["runs"]
        assert runs[0]["run_id"] == run_id and runs[0]["net_pnl"] == 12.5

    def test_a_run_with_no_process_and_no_result_reads_as_interrupted(self, client, tmp_path):
        run_id = start_run(client).get_json()["data"]["run_id"]
        manager._RUNS.clear()  # as after a server restart
        data = client.get(f"/python/api/backtest/mine_20260101/{run_id}").get_json()["data"]
        assert data["status"] == "stopped"
        assert "restarted" in data["message"]

    def test_stop_asks_then_terminates(self, client, tmp_path):
        run_id = start_run(client).get_json()["data"]["run_id"]
        resp = client.post(f"/python/api/backtest/mine_20260101/{run_id}/stop")
        assert resp.status_code == 200
        assert (tmp_path / "backtests" / "mine_20260101" / run_id / "stop").exists()
        assert FakePopen.instances[0].terminated  # the stand-in never exits on its own
        assert manager.active_run_id("mine_20260101") is None

    @pytest.mark.parametrize("run_id", ["..", "..%2F..%2Fetc", "20260101-000000-zzzzzz", "x"])
    def test_bad_run_ids_are_not_found(self, client, run_id):
        resp = client.get(f"/python/api/backtest/mine_20260101/{run_id}")
        assert resp.status_code == 404

    def test_strategy_id_traversal_is_refused(self, client):
        assert client.get("/python/api/backtest/..").status_code in (400, 404)
        assert manager._run_dir("../x", "20260101-000000-abcdef") is None

    def test_shutdown_stops_running_backtests(self, client):
        start_run(client)
        manager.stop_all(grace_seconds=0.1)
        assert FakePopen.instances[0].terminated
        resp = start_run(client)
        assert resp.status_code == 503


class TestHousekeeping:
    def test_twenty_runs_are_kept_including_the_new_one(self, client, tmp_path):
        base = tmp_path / "backtests" / "mine_20260101"
        for i in range(25):
            (base / f"20250101-0000{i:02d}-abcdef").mkdir(parents=True)
        assert start_run(client).status_code == 200
        runs = [p for p in base.iterdir() if manager.valid_run_id(p.name)]
        assert len(runs) == manager.MAX_RUNS_KEPT == 20
        assert manager.active_run_id("mine_20260101") in {p.name for p in runs}

    def test_shutdown_begun_during_a_start_launches_nothing(self, client, monkeypatch):
        def shutdown_meanwhile():
            manager._shutting_down = True

        monkeypatch.setattr(manager, "reap", shutdown_meanwhile)
        resp = start_run(client)
        assert resp.status_code == 503
        assert not FakePopen.instances

    def test_a_run_killed_at_the_time_limit_says_so(self, client, tmp_path, monkeypatch):
        run_id = start_run(client).get_json()["data"]["run_id"]
        run = manager._RUNS["mine_20260101"]
        run.started -= manager.MAX_WALL_SECONDS + 500
        manager.reap()
        assert FakePopen.instances[0].terminated
        data = client.get(f"/python/api/backtest/mine_20260101/{run_id}").get_json()["data"]
        assert data["status"] == "stopped"
        assert "never pauses" in data["message"] and "restarted" not in data["message"]

    def test_sandbox_square_off_times_reach_the_run(self, client, tmp_path, monkeypatch):
        import database.sandbox_db as sandbox_db

        monkeypatch.setattr(
            sandbox_db,
            "get_config",
            lambda key, default=None: {"nse_bse_square_off_time": "15:05"}.get(key),
        )
        run_id = start_run(client).get_json()["data"]["run_id"]
        config = json.loads(
            (tmp_path / "backtests" / "mine_20260101" / run_id / "config.json").read_text()
        )
        assert config["square_off_times"] == {"nse_bse_square_off_time": "15:05"}
