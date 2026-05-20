import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.status_store import get_status_store
from src.data.cache_manager import DuckDBCacheManager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.unit
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.unit
def test_status_endpoint(client):
    store = get_status_store()
    store.update(
        operator_mode="paper",
        symbol="ETHUSDC",
        regime="MEAN_REVERSION",
        kill_switch_active=False,
        portfolio={"long_qty": 0.1, "short_qty": 0.0, "equity": 1000.0},
    )
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["operator_mode"] == "paper"
    assert body["regime"] == "MEAN_REVERSION"


@pytest.mark.unit
def test_explain_latest_from_store(client):
    store = get_status_store()
    store.update(
        last_explanation={
            "decision": "LONG",
            "schema_version": 2,
            "conviction_score": 0.7,
        }
    )
    resp = client.get("/explain/latest")
    assert resp.status_code == 200
    assert resp.json()["decision"] == "LONG"


@pytest.mark.unit
def test_explain_latest_from_journal(client, tmp_path, monkeypatch):
    journal = tmp_path / "trade_journal.jsonl"
    entry = {"decision": "SHORT", "schema_version": 2}
    journal.write_text(json.dumps(entry) + "\n")

    import src.api.server as server_module

    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "explainability": {"journal_path": str(journal)},
            "data": {"storage": {"db_path": str(tmp_path / "x.duckdb")}},
            "paper": {},
            "live": {},
        },
    )
    store = get_status_store()
    with store._lock:
        store.last_explanation = None

    resp = client.get("/explain/latest")
    assert resp.status_code == 200
    assert resp.json()["decision"] == "SHORT"


@pytest.mark.unit
def test_explain_latest_returns_404_when_journal_is_empty(
    client, tmp_path, monkeypatch
):
    import src.api.server as server_module

    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "explainability": {"journal_path": str(tmp_path / "missing.jsonl")},
            "data": {"storage": {"db_path": str(tmp_path / "x.duckdb")}},
            "paper": {},
            "live": {},
        },
    )
    store = get_status_store()
    with store._lock:
        store.last_explanation = None

    resp = client.get("/explain/latest")

    assert resp.status_code == 404


@pytest.mark.unit
def test_portfolio_endpoint(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    db_path = str(tmp_path / "api.duckdb")
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {"storage": {"db_path": db_path}},
            "paper": {},
            "live": {},
        },
    )
    store = get_status_store()
    store.update(
        operator_mode="paper",
        portfolio={"long_qty": 0.2, "short_qty": 0.0, "equity": 1050.0},
    )

    resp = client.get("/portfolio")
    assert resp.status_code == 200
    body = resp.json()
    assert body["runtime"]["long_qty"] == 0.2


@pytest.mark.unit
def test_paper_metrics_endpoint(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    db_path = str(tmp_path / "metrics.duckdb")
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {"storage": {"db_path": db_path}},
            "paper": {},
            "live": {},
        },
    )
    resp = client.get("/metrics/paper")
    assert resp.status_code == 200
    body = resp.json()
    assert body["book_id"] == "primary"
    assert "sharpe" in body


@pytest.mark.unit
def test_positions_alias_and_aggregate_metrics(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    db_path = str(tmp_path / "aggregate.duckdb")
    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {"storage": {"db_path": db_path}},
            "paper": {},
            "live": {},
        },
    )

    positions_resp = client.get("/positions")
    metrics_resp = client.get("/metrics")

    assert positions_resp.status_code == 200
    assert positions_resp.json()["book_id"] == "primary"
    assert metrics_resp.status_code == 200
    assert "status" in metrics_resp.json()
    assert "paper" in metrics_resp.json()


@pytest.mark.unit
def test_ws_status_stream(client):
    store = get_status_store()
    store.update(operator_mode="paper", symbol="ETHUSDC", regime="MEAN_REVERSION")
    with client.websocket_connect("/ws/status") as ws:
        message = ws.receive_json()
    assert message["operator_mode"] == "paper"
    assert message["symbol"] == "ETHUSDC"


@pytest.mark.unit
def test_main_uses_configurable_bind_host(monkeypatch):
    import src.api.server as server_module

    calls = []

    class FakeUvicorn:
        @staticmethod
        def run(*args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setitem(sys.modules, "uvicorn", FakeUvicorn)
    monkeypatch.setenv("APEX_API_HOST", "0.0.0.0")

    server_module.main()

    assert calls[0][0] == ("src.api.server:app",)
    assert calls[0][1]["host"] == "0.0.0.0"
    assert calls[0][1]["port"] == 8080


@pytest.mark.unit
def test_history_reports_models_and_gate_endpoints(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    journal = tmp_path / "journal.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    journal.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-20T00:00:00+00:00",
                "decision": "LONG",
                "conviction_score": 0.7,
                "execution": {"mode": "paper"},
                "hedge": {
                    "enabled": True,
                    "selected": "protective_hedge",
                    "candidates": {"protective_hedge": 0.8},
                },
            }
        )
        + "\n"
    )
    decisions.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-20T00:00:01+00:00",
                "book": {"id": "shadow-v1"},
                "action": 2,
                "equity": 1000,
                "hedge": {
                    "enabled": True,
                    "selected": "maker_grid_hedge",
                    "candidates": {"maker_grid_hedge": 0.6},
                },
            }
        )
        + "\n"
    )

    class FakeRegistry:
        registry_data = {
            "models": {
                "shadow-v1": {
                    "type": "GBM",
                    "status": "SHADOW",
                    "created_at": "2026-05-20T00:00:00+00:00",
                    "metrics": {"sharpe": 1.2},
                }
            },
            "active_prod": None,
            "active_shadow": "shadow-v1",
        }

    server_module._config = None
    monkeypatch.setattr(server_module, "ModelRegistry", lambda: FakeRegistry())
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "explainability": {"journal_path": str(journal)},
            "shadow": {"decision_log_path": str(decisions)},
            "data": {"storage": {"db_path": str(tmp_path / "missing.duckdb")}},
            "paper": {"enabled": True, "min_days": 7, "min_trades": 2},
            "live": {"enabled": False},
            "promotion": {"min_shadow_trades": 1, "min_sharpe_delta": 0.1},
        },
    )

    history = client.get("/history/decisions?decision=LONG").json()
    assert history["total"] == 1
    assert history["items"][0]["decision"] == "LONG"

    hedge = client.get("/reports/hedge").json()
    assert hedge["total_selected"] == 2
    assert "protective_hedge" in hedge["strategies"]

    gate = client.get("/live/gate").json()
    assert gate["passed"] is False
    assert any("paper run" in reason for reason in gate["reasons"])

    models = client.get("/models").json()
    assert models["active_shadow"] == "shadow-v1"
    assert client.get("/models/shadow-v1").json()["status"] == "SHADOW"

    promotion = client.get("/models/promotion/status").json()
    assert promotion["active_shadow"] == "shadow-v1"
    assert "decision" in promotion


@pytest.mark.unit
def test_history_equity_market_logs_audit_and_controls(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    db_path = str(tmp_path / "history.duckdb")
    cache = DuckDBCacheManager(db_path=db_path)
    try:
        cache.insert_paper_equity_snapshot(
            "primary", 1000.0, 0.1, 0.0, 2120.0, "MEAN_REVERSION"
        )
        cache.insert_market_snapshot(
            "ETHUSDC",
            pd.Timestamp("2026-05-20T00:00:00Z"),
            funding_rate=0.0001,
            open_interest=100.0,
            mark_price=2120.0,
        )
        cache.insert_ohlcv(
            pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp("2026-05-20T00:00:00Z"),
                        "symbol": "ETHUSDC",
                        "timeframe": "3m",
                        "open": 2110.0,
                        "high": 2130.0,
                        "low": 2100.0,
                        "close": 2120.0,
                        "volume": 10.0,
                    }
                ]
            )
        )
    finally:
        cache.close()

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "runtime.log").write_text("one\ntwo\n", encoding="utf-8")
    audit_path = tmp_path / "audit.jsonl"
    control_path = tmp_path / "controls.json"

    server_module._config = None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server_module, "AUDIT_PATH", audit_path)
    monkeypatch.setattr(server_module, "CONTROL_STATE_PATH", control_path)
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {
                "target_symbol": "ETHUSDC",
                "target_interval": "3m",
                "storage": {"db_path": db_path},
            },
            "paper": {},
            "live": {},
        },
    )

    assert client.get("/history/equity").json()["total"] == 1
    market = client.get("/history/market").json()
    assert market["ohlcv"][0]["close"] == 2120.0
    assert market["market"][0]["mark_price"] == 2120.0
    assert client.get("/logs/runtime").json()["files"][0]["lines"] == ["one", "two"]

    rejected = client.post("/control/pause", json={"confirm": False})
    assert rejected.status_code == 400
    accepted = client.post(
        "/control/set-mode",
        json={"confirm": True, "reason": "test", "mode": "paper"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["state_after"]["mode_request"] == "paper"
    assert client.get("/control/state").json()["mode_request"] == "paper"
    assert client.get("/audit").json()["total"] == 1


@pytest.mark.unit
def test_control_validation_and_all_command_transitions(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    audit_path = tmp_path / "audit.jsonl"
    control_path = tmp_path / "controls.json"
    monkeypatch.setattr(server_module, "AUDIT_PATH", audit_path)
    monkeypatch.setattr(server_module, "CONTROL_STATE_PATH", control_path)

    assert client.post("/control/unknown", json={"confirm": True}).status_code == 404
    assert (
        client.post(
            "/control/set-mode", json={"confirm": True, "mode": "invalid"}
        ).status_code
        == 400
    )
    shadow_mode = client.post(
        "/control/set-mode", json={"confirm": True, "mode": "shadow"}
    )
    assert shadow_mode.status_code == 400
    assert shadow_mode.json()["detail"] == "mode must be paper/live"
    assert (
        client.post("/control/set-risk-profile", json={"confirm": True}).status_code
        == 400
    )

    control_path.write_text("{not-json", encoding="utf-8")
    assert client.get("/control/state").json() == {"error": "control state unreadable"}

    pause = client.post(
        "/control/pause", json={"confirm": True, "reason": "maintenance"}
    )
    assert pause.status_code == 200
    assert pause.json()["state_after"]["paused"] is True

    resume = client.post("/control/resume", json={"confirm": True})
    assert resume.json()["state_after"]["paused"] is False

    kill = client.post("/control/kill-switch", json={"confirm": True})
    assert kill.json()["state_after"]["kill_switch_requested"] is True
    assert get_status_store().kill_switch_active is True

    clear = client.post("/control/clear-kill-switch", json={"confirm": True})
    assert clear.json()["state_after"]["kill_switch_requested"] is False
    assert get_status_store().kill_switch_active is False

    flatten = client.post("/control/flatten", json={"confirm": True})
    assert flatten.json()["state_after"]["flatten_requested_at"]

    profile = client.post(
        "/control/set-risk-profile",
        json={"confirm": True, "profile": "defensive"},
    )
    assert profile.json()["state_after"]["risk_profile_request"] == "defensive"
    assert client.get("/audit").json()["total"] == 6


@pytest.mark.unit
def test_decision_history_ignores_invalid_json_and_filters_mode(
    client, tmp_path, monkeypatch
):
    import src.api.server as server_module

    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        "\n".join(
            [
                json.dumps({"decision": "LONG", "execution": {"mode": "paper"}}),
                "",
                "{bad-json",
                json.dumps({"decision": "SHORT", "execution": {"mode": "live"}}),
            ]
        ),
        encoding="utf-8",
    )
    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "explainability": {"journal_path": str(journal)},
            "data": {"storage": {"db_path": str(tmp_path / "missing.duckdb")}},
            "paper": {},
            "live": {},
        },
    )

    resp = client.get("/history/decisions?mode=paper&limit=0&offset=-10")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["decision"] == "LONG"
    assert body["limit"] == 1
    assert body["offset"] == 0


@pytest.mark.unit
def test_explain_latest_from_persisted_runtime_snapshot(client, tmp_path, monkeypatch):
    import src.api.status_store as status_store_module

    runtime_path = tmp_path / "runtime_status.json"
    runtime_path.write_text(
        json.dumps(
            {
                "updated_at": "2999-01-01T00:00:00+00:00",
                "last_explanation": {"decision": "HOLD", "schema_version": 2},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status_store_module, "RUNTIME_STATUS_PATH", runtime_path)
    store = get_status_store()
    with store._lock:
        store.last_explanation = None
        store.updated_at = None

    resp = client.get("/explain/latest")

    assert resp.status_code == 200
    assert resp.json()["decision"] == "HOLD"


@pytest.mark.unit
def test_portfolio_reads_latest_equity_from_db(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    db_path = str(tmp_path / "portfolio.duckdb")
    cache = DuckDBCacheManager(db_path=db_path)
    try:
        cache.insert_paper_equity_snapshot(
            "primary",
            1200.0,
            0.3,
            0.0,
            2125.0,
            "TRENDING",
            timestamp=pd.Timestamp("2026-05-20T00:00:00Z"),
        )
    finally:
        cache.close()

    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {"storage": {"db_path": db_path}},
            "paper": {},
            "live": {},
        },
    )

    body = client.get("/portfolio").json()

    assert body["paper_snapshots"] == 1
    assert body["latest_equity_from_db"] == 1200.0


@pytest.mark.unit
def test_history_model_and_dataframe_empty_paths(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    server_module._config = None
    missing_db = tmp_path / "missing.duckdb"
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {
                "target_symbol": "ETHUSDC",
                "target_interval": "3m",
                "storage": {"db_path": str(missing_db)},
            },
            "paper": {},
            "live": {},
        },
    )

    assert client.get("/history/equity").json()["items"] == []
    assert client.get("/history/market").json()["ohlcv"] == []
    assert server_module._records_from_df(pd.DataFrame()) == []

    class EmptyRegistry:
        registry_data = {"models": {}, "active_shadow": None}

    monkeypatch.setattr(server_module, "ModelRegistry", lambda: EmptyRegistry())
    monkeypatch.setattr(
        server_module,
        "generate_paper_report",
        lambda config, book_id="primary", journal_path=None: {"sharpe": 0.0},
    )

    promotion = client.get("/models/promotion/status").json()
    assert promotion["decision"]["reason"] == "no_active_shadow_model"
    assert client.get("/models/missing-model").status_code == 404


@pytest.mark.unit
def test_history_database_error_paths(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    db_path = tmp_path / "broken.duckdb"
    db_path.write_text("not-a-real-duckdb", encoding="utf-8")
    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {
                "target_symbol": "ETHUSDC",
                "target_interval": "3m",
                "storage": {"db_path": str(db_path)},
            },
            "paper": {},
            "live": {},
        },
    )

    class FailingCache:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("equity unavailable")

    monkeypatch.setattr(server_module, "DuckDBCacheManager", FailingCache)
    equity = client.get("/history/equity").json()
    assert equity["items"] == []
    assert "equity unavailable" in equity["error"]

    monkeypatch.setattr(
        server_module.duckdb,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("market unavailable")
        ),
    )
    market = client.get("/history/market").json()
    assert market["ohlcv"] == []
    assert "market unavailable" in market["error"]


@pytest.mark.unit
def test_read_jsonl_missing_and_portfolio_db_error(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    assert server_module._read_jsonl(tmp_path / "missing.jsonl") == []

    db_path = tmp_path / "exists.duckdb"
    db_path.write_text("placeholder", encoding="utf-8")
    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "data": {"storage": {"db_path": str(db_path)}},
            "paper": {},
            "live": {},
        },
    )

    class FailingCache:
        def __init__(self, *args, **kwargs):
            pass

        def load_paper_equity_snapshots(self, book_id):
            raise RuntimeError("portfolio db unavailable")

        def close(self):
            self.closed = True

    monkeypatch.setattr(server_module, "DuckDBCacheManager", FailingCache)

    body = client.get("/portfolio").json()

    assert body["paper_snapshots"] == 0
    assert body["latest_equity_from_db"] is None


@pytest.mark.unit
def test_model_lifecycle_manifest_and_readiness_paths(client, tmp_path, monkeypatch):
    import src.api.server as server_module

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model_id": "model-v1"}), encoding="utf-8")

    class FakeRegistry:
        registry_data = {
            "models": {
                "model-v1": {
                    "status": "PROD",
                    "manifest_path": str(manifest),
                },
                "model-missing-manifest": {"status": "PROD"},
            },
            "active_prod": "model-v1",
            "active_shadow": "shadow-v1",
            "events": [{"event": "promoted"}],
        }

        def production_readiness(self, model_id=None):
            return {"model_id": model_id or "model-v1", "ready": True}

    def registry_factory(*args, **kwargs):
        if kwargs:
            raise TypeError("legacy registry")
        return FakeRegistry()

    fake_tracker = SimpleNamespace(
        from_config=lambda config: SimpleNamespace(
            list_runs=lambda limit=25: [{"run_id": "run-1", "limit": limit}]
        )
    )
    server_module._config = None
    monkeypatch.setattr(
        server_module,
        "load_config",
        lambda *a, **k: {
            "mlops": {"registry_dir": str(tmp_path / "models")},
            "paper": {},
            "live": {},
        },
    )
    monkeypatch.setattr(server_module, "ModelRegistry", registry_factory)
    monkeypatch.setattr(server_module, "ExperimentTracker", fake_tracker)

    lifecycle = client.get("/models/lifecycle?limit=0").json()
    manifest_body = client.get("/models/model-v1/manifest").json()

    assert lifecycle["production_readiness"]["ready"] is True
    assert lifecycle["runs"][0]["limit"] == 1
    assert lifecycle["registry_events"] == [{"event": "promoted"}]
    assert manifest_body["model_id"] == "model-v1"
    assert client.get("/models/missing/manifest").status_code == 404
    assert client.get("/models/model-missing-manifest/manifest").status_code == 404
    assert server_module._production_readiness(FakeRegistry())["ready"] is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_ws_status_logs_disconnect():
    import src.api.server as server_module

    class DisconnectingWebSocket:
        def __init__(self):
            self.accepted = False

        async def accept(self):
            self.accepted = True

        async def send_json(self, payload):
            raise server_module.WebSocketDisconnect()

    websocket = DisconnectingWebSocket()

    await server_module.ws_status(websocket)

    assert websocket.accepted is True
