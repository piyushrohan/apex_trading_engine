"""Read-only FastAPI server for operator status and explainability."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.status_store import get_status_store
from src.core.config_loader import load_config
from src.data.cache_manager import DuckDBCacheManager
from src.execution.live_gate import evaluate_paper_gate
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.explainability import ExplainabilityEngine
from src.mlops.promotion_service import PromotionService
from src.mlops.registry import ModelRegistry
from src.reports.hedge_report import generate_hedge_report
from src.reports.paper_report import generate_paper_report

logger = logging.getLogger(__name__)
AUDIT_PATH = Path(os.getenv("APEX_AUDIT_PATH", "data_lake/audit_events.jsonl"))
CONTROL_STATE_PATH = Path(
    os.getenv("APEX_CONTROL_STATE_PATH", "data_lake/operator_controls.json")
)

app = FastAPI(
    title="APEX Trading Engine API",
    description="Read-only status and explainability for paper/live operator modes.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _journal_path(config: Dict[str, Any]) -> str:
    return config.get("explainability", {}).get(
        "journal_path", "data_lake/trade_journal.jsonl"
    )


def _decision_path(config: Dict[str, Any]) -> str:
    return config.get("shadow", {}).get(
        "decision_log_path", "data_lake/hedge_bandit/training/decisions.jsonl"
    )


def _db_path(config: Dict[str, Any]) -> str:
    return (
        config.get("data", {})
        .get("storage", {})
        .get("db_path", "data_lake/apex_market_data.duckdb")
    )


def _read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping invalid JSONL row in %s", source)
    return rows


def _page_rows(rows: list[Dict[str, Any]], limit: int, offset: int) -> Dict[str, Any]:
    safe_limit = max(1, min(limit, 500))
    safe_offset = max(0, offset)
    return {
        "items": rows[safe_offset : safe_offset + safe_limit],
        "total": len(rows),
        "limit": safe_limit,
        "offset": safe_offset,
    }


def _records_from_df(df) -> list[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    normalized = df.copy()
    for col in normalized.columns:
        if "time" in col:
            normalized[col] = normalized[col].astype(str)
    return normalized.to_dict(orient="records")


def _append_audit(event: Dict[str, Any]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _load_control_state() -> Dict[str, Any]:
    if not CONTROL_STATE_PATH.exists():
        return {
            "paused": False,
            "kill_switch_requested": False,
            "flatten_requested_at": None,
            "mode_request": None,
            "risk_profile_request": None,
            "last_command": None,
        }
    try:
        return json.loads(CONTROL_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": "control state unreadable"}


def _save_control_state(state: Dict[str, Any]) -> None:
    CONTROL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CONTROL_STATE_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(CONTROL_STATE_PATH)


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "service": "apex-trading-engine"}


@app.get("/status")
def status():
    """Runtime operator snapshot (mode, regime, kill switch, portfolio summary)."""
    store = get_status_store()
    snap = store.snapshot()
    config = get_config()
    snap["live_enabled"] = config.get("live", {}).get("enabled", False)
    snap["paper_gate"] = {
        "enabled": config.get("paper", {}).get("enabled", True),
        "min_days": config.get("paper", {}).get("min_days", 7),
    }
    return snap


@app.get("/control/state")
def control_state():
    """Recorded operator command state for the terminal control deck."""
    return _load_control_state()


@app.post("/control/{command}")
def record_control_command(command: str, payload: Optional[Dict[str, Any]] = None):
    """
    Record a guarded operator command.

    These endpoints intentionally persist auditable operator intent. Trading
    pipelines can consume the command ledger, but the API does not directly
    place orders or mutate exchange state.
    """
    body = payload or {}
    if not body.get("confirm"):
        raise HTTPException(status_code=400, detail="confirm=true is required")

    allowed = {
        "pause",
        "resume",
        "kill-switch",
        "clear-kill-switch",
        "flatten",
        "set-mode",
        "set-risk-profile",
    }
    if command not in allowed:
        raise HTTPException(status_code=404, detail="unknown control command")

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    state = _load_control_state()
    if "error" in state:
        state = {}
    state.setdefault("paused", False)
    state.setdefault("kill_switch_requested", False)

    if command == "pause":
        state["paused"] = True
    elif command == "resume":
        state["paused"] = False
    elif command == "kill-switch":
        state["kill_switch_requested"] = True
        get_status_store().update(kill_switch_active=True)
    elif command == "clear-kill-switch":
        state["kill_switch_requested"] = False
        get_status_store().update(kill_switch_active=False)
    elif command == "flatten":
        state["flatten_requested_at"] = now
    elif command == "set-mode":
        mode = body.get("mode")
        if mode not in ("paper", "live"):
            raise HTTPException(status_code=400, detail="mode must be paper/live")
        state["mode_request"] = mode
    elif command == "set-risk-profile":
        profile = body.get("profile")
        if not profile:
            raise HTTPException(status_code=400, detail="profile is required")
        state["risk_profile_request"] = profile

    state_after = dict(state)
    state_after.pop("last_command", None)
    event = {
        "timestamp": now,
        "command": command,
        "reason": body.get("reason", ""),
        "payload": {k: v for k, v in body.items() if k not in {"confirm", "reason"}},
        "state_after": state_after,
    }
    state["last_command"] = {k: v for k, v in event.items() if k != "state_after"}
    _save_control_state(state)
    _append_audit(event)
    return {"accepted": True, "effect": "recorded_for_operator_pipeline", **event}


@app.get("/explain/latest")
def explain_latest():
    """Most recent decision explanation from runtime store or trade journal."""
    store = get_status_store()
    if store.last_explanation:
        return store.last_explanation
    persisted = store.snapshot().get("last_explanation")
    if persisted:
        return persisted

    config = get_config()
    engine = ExplainabilityEngine(config)
    entry = engine.read_latest_journal_entry()
    if entry is None:
        raise HTTPException(status_code=404, detail="No explanations in journal yet")
    return entry


@app.get("/portfolio")
def portfolio(book_id: str = "primary"):
    """Primary book positions and equity (runtime store + optional DB snapshots)."""
    store = get_status_store()
    snap = store.snapshot()
    runtime = snap.get("portfolio", {})

    config = get_config()
    db_path = _db_path(config)
    equity_df = None
    if Path(db_path).exists():
        cache = None
        try:
            cache = DuckDBCacheManager(db_path=db_path, read_only=True)
            equity_df = cache.load_paper_equity_snapshots(book_id)
        except Exception as exc:
            logger.debug("Paper equity DB read unavailable: %s", exc)
            equity_df = None
        finally:
            if cache is not None:
                cache.close()

    latest_equity = None
    snapshot_count = 0
    if equity_df is not None and not equity_df.empty:
        snapshot_count = len(equity_df)
        latest_equity = float(equity_df["equity"].iloc[-1])

    return {
        "book_id": book_id,
        "symbol": snap.get("symbol"),
        "operator_mode": snap.get("operator_mode"),
        "runtime": runtime,
        "paper_snapshots": snapshot_count,
        "latest_equity_from_db": latest_equity,
        "updated_at": snap.get("updated_at"),
    }


@app.get("/positions")
def positions(book_id: str = "primary"):
    """Alias for terminal position panels."""
    return portfolio(book_id=book_id)


@app.get("/metrics")
def metrics(book_id: str = "primary"):
    """Dashboard-friendly aggregate metrics."""
    return {
        "status": status(),
        "paper": paper_metrics(book_id=book_id),
    }


@app.get("/metrics/paper")
def paper_metrics(book_id: str = "primary"):
    """Paper performance summary for terminal/dashboard surfaces."""
    config = get_config()
    return generate_paper_report(config=config, book_id=book_id)


@app.get("/reports/hedge")
def hedge_metrics(days: int = 7):
    """Hedge strategy attribution for the selected historical window."""
    config = get_config()
    return generate_hedge_report(_journal_path(config), _decision_path(config), days)


@app.get("/live/gate")
def live_gate(book_id: str = "primary"):
    """Paper-to-live readiness gate with concrete failing reasons."""
    config = get_config()
    result = evaluate_paper_gate(
        config, book_id=book_id, journal_path=_journal_path(config)
    )
    return {
        "passed": result.passed,
        "reasons": result.reasons,
        "metrics": result.metrics,
        "live_enabled": config.get("live", {}).get("enabled", False),
        "skip_paper_gate": config.get("live", {}).get("skip_paper_gate", False),
    }


@app.get("/history/decisions")
def decision_history(
    limit: int = 100,
    offset: int = 0,
    mode: Optional[str] = None,
    decision: Optional[str] = None,
):
    """Paginated decision journal, newest first."""
    rows = list(reversed(_read_jsonl(_journal_path(get_config()))))
    if mode:
        rows = [row for row in rows if row.get("execution", {}).get("mode") == mode]
    if decision:
        rows = [row for row in rows if row.get("decision") == decision]
    return _page_rows(rows, limit, offset)


@app.get("/history/equity")
def equity_history(book_id: str = "primary", limit: int = 500):
    """Paper/live equity snapshots from DuckDB."""
    config = get_config()
    db_path = _db_path(config)
    if not Path(db_path).exists():
        return {"book_id": book_id, "items": [], "total": 0}
    cache = None
    try:
        cache = DuckDBCacheManager(db_path=db_path, read_only=True)
        df = cache.load_paper_equity_snapshots(book_id)
    except Exception as exc:
        logger.debug("Equity history unavailable: %s", exc)
        return {"book_id": book_id, "items": [], "total": 0, "error": str(exc)}
    finally:
        if cache is not None:
            cache.close()
    total = len(df)
    return {
        "book_id": book_id,
        "items": _records_from_df(df.tail(max(1, min(limit, 2000)))),
        "total": total,
    }


@app.get("/history/market")
def market_history(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: int = 200,
):
    """Recent OHLCV and market snapshots for replay-oriented views."""
    config = get_config()
    db_path = _db_path(config)
    symbol = symbol or config.get("data", {}).get("target_symbol", "ETHUSDC")
    timeframe = timeframe or config.get("data", {}).get("target_interval", "3m")
    if not Path(db_path).exists():
        return {"symbol": symbol, "timeframe": timeframe, "ohlcv": [], "market": []}
    conn = None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        safe_limit = max(1, min(limit, 2000))
        ohlcv = conn.execute(
            """
            SELECT * FROM (
                SELECT * FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ) ORDER BY timestamp ASC
            """,
            [symbol, timeframe, safe_limit],
        ).df()
        market = conn.execute(
            """
            SELECT * FROM (
                SELECT * FROM market_snapshots
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ) ORDER BY timestamp ASC
            """,
            [symbol, safe_limit],
        ).df()
    except Exception as exc:
        logger.debug("Market history unavailable: %s", exc)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "ohlcv": [],
            "market": [],
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ohlcv": _records_from_df(ohlcv),
        "market": _records_from_df(market),
    }


@app.get("/models")
def models():
    """Model registry summary for observability and promotion review."""
    registry = ModelRegistry()
    payload = dict(registry.registry_data)
    payload["production_readiness"] = _production_readiness(registry)
    return payload


@app.get("/models/lifecycle")
def model_lifecycle(limit: int = 25):
    """Auditable model lifecycle, training run history, and live blockers."""
    config = get_config()
    try:
        registry = ModelRegistry(
            registry_dir=config.get("mlops", {}).get("registry_dir", "data_lake/models")
        )
    except TypeError:
        registry = ModelRegistry()
    tracker = ExperimentTracker.from_config(config)
    return {
        "production_readiness": _production_readiness(registry),
        "active_prod": registry.registry_data.get("active_prod"),
        "active_shadow": registry.registry_data.get("active_shadow"),
        "runs": tracker.list_runs(limit=max(1, min(limit, 100))),
        "registry_events": registry.registry_data.get("events", [])[-100:],
        "discipline": {
            "live_requires_prod": True,
            "prod_requires_status": "PROD",
            "prod_requires_manifest": True,
            "prod_requires_artifact": True,
            "promotion_ladder": [
                "CANDIDATE",
                "EVALUATING",
                "SHADOW",
                "APPROVED",
                "PROD",
            ],
        },
    }


@app.get("/models/promotion/status")
def promotion_status(model_id: Optional[str] = None):
    """Current shadow promotion decision without applying it."""
    config = get_config()
    registry = ModelRegistry()
    selected = model_id or registry.registry_data.get("active_shadow")
    primary_metrics = generate_paper_report(config=config)
    if not selected:
        return {
            "active_shadow": None,
            "decision": {
                "action": "hold",
                "reason": "no_active_shadow_model",
            },
            "primary_metrics": primary_metrics,
            "shadow_metrics": None,
        }
    service = PromotionService(config, registry=registry)
    shadow_metrics = service.metrics_from_decision_log(
        decision_path=_decision_path(config), book_id=selected
    )
    decision = service.evaluate(selected, primary_metrics, shadow_metrics)
    return {
        "active_shadow": selected,
        "decision": decision.__dict__,
        "primary_metrics": primary_metrics,
        "shadow_metrics": shadow_metrics,
        "thresholds": {
            "min_shadow_trades": service.min_shadow_trades,
            "min_sharpe_delta": service.min_sharpe_delta,
            "max_shadow_drawdown": service.max_shadow_drawdown,
        },
    }


@app.get("/models/{model_id}")
def model_detail(model_id: str):
    """Single model registry entry."""
    registry = ModelRegistry()
    model = registry.registry_data.get("models", {}).get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="model not found")
    return {
        "model_id": model_id,
        **model,
        "production_readiness": _production_readiness(registry, model_id),
    }


@app.get("/models/{model_id}/manifest")
def model_manifest(model_id: str):
    """Return the immutable manifest for a registered model."""
    registry = ModelRegistry()
    model = registry.registry_data.get("models", {}).get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="model not found")
    manifest_path = model.get("manifest_path")
    if not manifest_path or not Path(manifest_path).exists():
        raise HTTPException(status_code=404, detail="manifest not found")
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _production_readiness(
    registry: ModelRegistry, model_id: Optional[str] = None
) -> Dict[str, Any]:
    readiness_fn = getattr(registry, "production_readiness", None)
    if readiness_fn:
        return readiness_fn(model_id)
    selected = model_id or registry.registry_data.get("active_prod")
    return {
        "model_id": selected,
        "ready": False,
        "status": None,
        "manifest_exists": False,
        "artifact_exists": False,
        "blockers": ["production_readiness_unavailable"],
    }


@app.get("/logs/runtime")
def runtime_logs(limit: int = 200):
    """Tail local runtime logs for operator diagnostics."""
    safe_limit = max(1, min(limit, 1000))
    files = []
    for path in sorted(Path("logs").glob("*.log")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        files.append({"path": str(path), "lines": lines[-safe_limit:]})
    return {"files": files, "limit": safe_limit}


@app.get("/audit")
def audit_events(limit: int = 100, offset: int = 0):
    """Operator command audit log, newest first."""
    return _page_rows(list(reversed(_read_jsonl(AUDIT_PATH))), limit, offset)


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """Push operator status snapshots for the terminal."""
    await websocket.accept()
    store = get_status_store()
    try:
        while True:
            payload = store.snapshot()
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        logger.info("Status websocket disconnected")


def main():
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("APEX_API_HOST", "127.0.0.1"),
        port=8080,
        reload=False,
    )


if __name__ == "__main__":
    main()
