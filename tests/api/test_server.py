import json

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.status_store import get_status_store


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
def test_ws_status_stream(client):
    store = get_status_store()
    store.update(operator_mode="paper", symbol="ETHUSDC", regime="MEAN_REVERSION")
    with client.websocket_connect("/ws/status") as ws:
        message = ws.receive_json()
    assert message["operator_mode"] == "paper"
    assert message["symbol"] == "ETHUSDC"
