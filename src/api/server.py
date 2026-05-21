"""FastAPI server for operator status, controls, and explainability."""

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.api.status_store import get_status_store
from src.core.config_loader import load_config
from src.data.cache_manager import DuckDBCacheManager
from src.execution.kill_switch import (
    active_kill_switch_lanes,
    kill_switch_active,
    normalize_kill_switch_lanes,
    set_kill_switch_lane,
)
from src.execution.live_gate import evaluate_paper_gate
from src.execution.order_lifecycle import summarize_order_lifecycle
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.explainability import ExplainabilityEngine
from src.mlops.feature_drift import (
    compare_feature_drift,
    latest_feature_frame_from_ohlcv,
)
from src.mlops.promotion_service import PromotionService
from src.mlops.registry import ModelRegistry
from src.ops.process_manager import PROCESS_MANAGER
from src.reports.hedge_report import generate_hedge_report
from src.reports.paper_report import generate_paper_report

logger = logging.getLogger(__name__)
AUDIT_PATH = Path(os.getenv("APEX_AUDIT_PATH", "data_lake/audit_events.jsonl"))
CONTROL_STATE_PATH = Path(
    os.getenv("APEX_CONTROL_STATE_PATH", "data_lake/operator_controls.json")
)
DIAGNOSTIC_TABLE_QUERIES = {
    "features": (
        "SELECT COUNT(*) FROM features",
        "SELECT MAX(timestamp) FROM features",
    ),
    "market_snapshots": (
        "SELECT COUNT(*) FROM market_snapshots",
        "SELECT MAX(timestamp) FROM market_snapshots",
    ),
    "ohlcv": (
        "SELECT COUNT(*) FROM ohlcv",
        "SELECT MAX(timestamp) FROM ohlcv",
    ),
    "order_lifecycle_events": (
        "SELECT COUNT(*) FROM order_lifecycle_events",
        "SELECT MAX(timestamp) FROM order_lifecycle_events",
    ),
    "paper_equity_snapshots": (
        "SELECT COUNT(*) FROM paper_equity_snapshots",
        "SELECT MAX(timestamp) FROM paper_equity_snapshots",
    ),
    "ticks": (
        "SELECT COUNT(*) FROM ticks",
        "SELECT MAX(timestamp) FROM ticks",
    ),
}

app = FastAPI(
    title="APEX Trading Engine API",
    description="Status, controls, and explainability for paper/live operator modes.",
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


def _order_lifecycle_path(config: Dict[str, Any]) -> str:
    return config.get("execution", {}).get(
        "order_lifecycle_path", "data_lake/order_lifecycle.jsonl"
    )


def _db_path(config: Dict[str, Any]) -> str:
    return (
        config.get("data", {})
        .get("storage", {})
        .get("db_path", "data_lake/apex_market_data.duckdb")
    )


def _api_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("api", {})


def _market_stream_url(config: Dict[str, Any], symbol: str) -> str:
    ws_url = (
        config.get("data", {})
        .get("urls", {})
        .get("ws_stream", "wss://fstream.binance.com/stream")
    )
    safe_symbol = symbol.lower()
    streams = [
        f"{safe_symbol}@aggTrade",
        f"{safe_symbol}@markPrice@1s",
        f"{safe_symbol}@depth5@100ms",
    ]
    return f"{ws_url}?streams={'/'.join(streams)}"


def _normalize_market_ws_event(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize Binance combined stream events for the browser market tape."""
    stream = str(payload.get("stream", ""))
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return None

    received_at = datetime.now(timezone.utc)
    event_time_ms = data.get("E") or data.get("T")
    event_time = None
    latency_ms = None
    if event_time_ms is not None:
        try:
            event_dt = datetime.fromtimestamp(
                int(event_time_ms) / 1000, tz=timezone.utc
            )
            event_time = event_dt.isoformat()
            latency_ms = max(0.0, (received_at - event_dt).total_seconds() * 1000.0)
        except (TypeError, ValueError, OSError):
            event_time = None

    if "@markPrice" in stream or data.get("e") == "markPriceUpdate":
        price = data.get("p") or data.get("markPrice")
        if price is None:
            return None
        return {
            "type": "mark",
            "symbol": str(data.get("s", "")).upper(),
            "price": float(price),
            "mark_price": float(price),
            "event_time": event_time,
            "received_at": received_at.isoformat(),
            "latency_ms": latency_ms,
            "source": "binance_ws",
        }

    if "@aggTrade" in stream or data.get("e") == "aggTrade":
        price = data.get("p")
        qty = data.get("q")
        if price is None:
            return None
        return {
            "type": "trade",
            "symbol": str(data.get("s", "")).upper(),
            "price": float(price),
            "quantity": float(qty or 0.0),
            "is_buyer_maker": bool(data.get("m", False)),
            "event_time": event_time,
            "received_at": received_at.isoformat(),
            "latency_ms": latency_ms,
            "source": "binance_ws",
        }

    if "@depth" in stream or data.get("e") == "depthUpdate":
        bids = data.get("b") or data.get("bids") or []
        asks = data.get("a") or data.get("asks") or []
        best_bid = float(bids[0][0]) if bids else None
        best_ask = float(asks[0][0]) if asks else None
        mid = (
            (best_bid + best_ask) / 2.0
            if best_bid is not None and best_ask is not None
            else None
        )
        return {
            "type": "depth",
            "symbol": str(data.get("s", "")).upper(),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": (
                ((best_ask - best_bid) / mid) * 10000.0
                if mid and best_bid is not None and best_ask is not None
                else None
            ),
            "event_time": event_time,
            "received_at": received_at.isoformat(),
            "latency_ms": latency_ms,
            "source": "binance_ws",
        }
    return None


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

    def clean_cell(value: Any) -> Any:
        if value is None:
            return None
        try:
            if isinstance(value, Real) and not math.isfinite(float(value)):
                return None
        except (TypeError, ValueError):
            pass
        try:
            if value != value:
                return None
        except (TypeError, ValueError):
            pass
        return value

    records = df.copy().astype(object).to_dict(orient="records")
    cleaned: list[Dict[str, Any]] = []
    for row in records:
        cleaned_row = {}
        for key, value in row.items():
            safe_value = clean_cell(value)
            if safe_value is not None and "time" in key:
                safe_value = str(safe_value)
            cleaned_row[key] = safe_value
        cleaned.append(cleaned_row)
    return cleaned


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: Any) -> Optional[float]:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return (datetime.now(timezone.utc) - parsed).total_seconds()


def _add_check(
    checks: list[Dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    **metadata: Any,
) -> None:
    checks.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            **metadata,
        }
    )


def _latest_journal_decision(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rows = _read_jsonl(_journal_path(config))
    return rows[-1] if rows else None


def _load_order_lifecycle_rows(
    config: Dict[str, Any], book_id: str = "primary", limit: int = 500
) -> list[Dict[str, Any]]:
    db_path = _db_path(config)
    if Path(db_path).exists():
        cache = None
        try:
            cache = DuckDBCacheManager(db_path=db_path, read_only=True)
            df = cache.load_order_lifecycle_events(book_id=book_id, limit=limit)
            return _records_from_df(df)
        except Exception as exc:
            logger.debug("Order lifecycle DB read unavailable: %s", exc)
        finally:
            if cache is not None:
                cache.close()
    rows = list(reversed(_read_jsonl(_order_lifecycle_path(config))))
    rows = [row for row in rows if row.get("book_id") == book_id]
    return rows[: max(1, min(limit, 2000))]


def _model_feature_reference(
    registry: ModelRegistry, model_id: Optional[str]
) -> Dict[str, Any]:
    models = registry.registry_data.get("models", {})
    model = models.get(model_id or "") if model_id else None
    if not model:
        return {}
    metrics = model.get("metrics", {}) or {}
    metadata = model.get("metadata", {}) or {}
    return metrics.get("feature_reference") or metadata.get("feature_reference") or {}


def _feature_drift_snapshot(
    config: Dict[str, Any], registry: ModelRegistry, model_id: Optional[str] = None
) -> Dict[str, Any]:
    selected = (
        model_id
        or registry.registry_data.get("active_prod")
        or registry.registry_data.get("active_shadow")
    )
    reference = _model_feature_reference(registry, selected)
    symbol = config.get("data", {}).get("target_symbol", "ETHUSDC")
    timeframe = config.get("data", {}).get("target_interval", "3m")
    db_path = _db_path(config)
    if not Path(db_path).exists():
        return {
            "model_id": selected,
            "status": "unavailable",
            "reason": "duckdb_missing",
            "symbol": symbol,
            "timeframe": timeframe,
        }
    cache = None
    try:
        cache = DuckDBCacheManager(db_path=db_path, read_only=True)
        ohlcv = cache.load_ohlcv(symbol, timeframe)
    except Exception as exc:
        return {
            "model_id": selected,
            "status": "unavailable",
            "reason": "duckdb_read_failed",
            "error": str(exc),
            "symbol": symbol,
            "timeframe": timeframe,
        }
    finally:
        if cache is not None:
            cache.close()
    current = latest_feature_frame_from_ohlcv(ohlcv.tail(500))
    report = compare_feature_drift(reference, current)
    return {
        "model_id": selected,
        "symbol": symbol,
        "timeframe": timeframe,
        **report,
    }


def _duckdb_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    db_path = _db_path(config)
    if not Path(db_path).exists():
        return {"db_path": db_path, "exists": False, "tables": {}, "latest": {}}
    conn = None
    try:
        conn = duckdb.connect(db_path, read_only=True)
        table_names = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        tables = {}
        latest = {}
        for table, (count_sql, latest_sql) in DIAGNOSTIC_TABLE_QUERIES.items():
            if table not in table_names:
                continue
            tables[table] = int(conn.execute(count_sql).fetchone()[0])
            try:
                latest_value = conn.execute(latest_sql).fetchone()[0]
            except Exception:
                latest_value = None
            latest[table] = str(latest_value) if latest_value else None
        return {
            "db_path": db_path,
            "exists": True,
            "tables": tables,
            "latest": latest,
        }
    except Exception as exc:
        return {
            "db_path": db_path,
            "exists": True,
            "tables": {},
            "latest": {},
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()


def _append_audit(event: Dict[str, Any]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _load_control_state() -> Dict[str, Any]:
    if not CONTROL_STATE_PATH.exists():
        return {
            "paused": False,
            "kill_switch_requested": False,
            "kill_switch_lanes": normalize_kill_switch_lanes({}),
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


@app.get("/ops/workflow")
def ops_workflow():
    """Guided operator workflow so startup is visible from one cockpit view."""
    config = get_config()
    runtime = status()
    registry = ModelRegistry()
    processes = PROCESS_MANAGER.list_processes()
    readiness = _production_readiness(registry)
    active_shadow = registry.registry_data.get("active_shadow")
    active_prod = registry.registry_data.get("active_prod")
    paper_running = processes.get("paper", {}).get("running", False)
    live_running = processes.get("live", {}).get("running", False)
    training_running = processes.get("training", {}).get("running", False)
    governance_returncode = processes.get("model_governance", {}).get("returncode")
    steps = [
        {
            "id": "cockpit",
            "label": "Open cockpit",
            "status": "ready",
            "command": "make start",
            "action": None,
        },
        {
            "id": "paper",
            "label": "Run paper trading",
            "status": "running" if paper_running else "idle",
            "action": "paper",
        },
        {
            "id": "train",
            "label": "Train or retrain model",
            "status": "running" if training_running else "idle",
            "action": "training",
        },
        {
            "id": "evaluate",
            "label": "Evaluate governance and paper evidence",
            "status": "complete" if governance_returncode == 0 else "idle",
            "action": "model_governance",
        },
        {
            "id": "shadow",
            "label": "Collect shadow evidence",
            "status": "ready" if active_shadow else "blocked",
            "model_id": active_shadow,
            "action": "shadow",
        },
        {
            "id": "prod",
            "label": "Promote only reviewed PROD model",
            "status": "ready" if readiness.get("ready") else "blocked",
            "model_id": active_prod,
            "blockers": readiness.get("blockers", []),
        },
        {
            "id": "live",
            "label": "Start live trading only after gates clear",
            "status": "running" if live_running else "blocked",
            "action": "live",
            "blockers": [] if readiness.get("ready") else readiness.get("blockers", []),
        },
    ]
    api_host = os.getenv("APEX_API_HOST", "127.0.0.1")
    api_port = int(os.getenv("APEX_API_PORT", "8080"))
    frontend_port = int(os.getenv("APEX_FRONTEND_PORT", "5173"))
    frontend_url = (
        f"http://{api_host}:{frontend_port}/?api=http://{api_host}:{api_port}"
    )
    return {
        "title": "APEX Guided Operator Workflow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_command": "make start",
        "frontend_url": frontend_url,
        "operator_mode": runtime.get("operator_mode"),
        "symbol": runtime.get("symbol")
        or config.get("data", {}).get("target_symbol", "ETHUSDC"),
        "api": {
            "host": api_host,
            "port": api_port,
            "status_ws_interval_sec": float(
                _api_cfg(config).get("status_ws_interval_sec", 0.5)
            ),
        },
        "frontend": {"port": frontend_port, "url": frontend_url},
        "processes": processes,
        "capabilities": PROCESS_MANAGER.capabilities(),
        "registry": {
            "active_prod": active_prod,
            "active_shadow": active_shadow,
            "production_ready": readiness.get("ready", False),
            "blockers": readiness.get("blockers", []),
        },
        "steps": steps,
        "recommended_next": [
            "Run make start once, then operate from this browser cockpit.",
            "Start paper from the control center if it is idle.",
            "Train only after DuckDB has enough OHLCV history.",
            (
                "Run model governance, paper health, shadow sanity, and data "
                "checks after training."
            ),
            "Keep candidate models in shadow until promotion gates pass.",
            "Do not request live mode until PROD readiness and paper gate clear.",
        ],
    }


@app.get("/ops/processes")
def ops_processes():
    """Local allow-listed paper/training subprocess status."""
    return {
        "processes": PROCESS_MANAGER.list_processes(),
        "capabilities": PROCESS_MANAGER.capabilities(),
        "allowed": sorted(PROCESS_MANAGER.SPECS),
    }


@app.post("/ops/processes/{process_name}")
def ops_process_action(process_name: str, payload: Optional[Dict[str, Any]] = None):
    """Start or stop allow-listed local operator processes."""
    body = payload or {}
    if not body.get("confirm"):
        raise HTTPException(status_code=400, detail="confirm=true is required")
    action = body.get("action")
    dry_run = bool(body.get("dry_run", False))
    try:
        spec = PROCESS_MANAGER.SPECS[process_name]
        if (
            process_name == "live"
            and action in {"start", "restart"}
            and body.get("confirm_phrase") != "START LIVE"
        ):
            raise HTTPException(
                status_code=400,
                detail="confirm_phrase=START LIVE is required for live trading",
            )
        if action == "start":
            result = PROCESS_MANAGER.start(process_name, dry_run=dry_run)
        elif action == "stop":
            result = PROCESS_MANAGER.stop(process_name, dry_run=dry_run)
        elif action == "restart":
            result = PROCESS_MANAGER.restart(process_name, dry_run=dry_run)
        else:
            raise HTTPException(
                status_code=400, detail="action must be start/stop/restart"
            )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown process") from None

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": f"process-{action}",
        "reason": body.get("reason", ""),
        "payload": {
            "process": process_name,
            "label": spec.label or process_name,
            "category": spec.category,
            "danger_level": spec.danger_level,
            "dry_run": dry_run,
        },
        "state_after": result,
    }
    _append_audit(event)
    return {"accepted": True, "effect": "local_process_control", **event}


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
    state["kill_switch_lanes"] = normalize_kill_switch_lanes(
        state.get("kill_switch_lanes")
    )

    if command == "pause":
        state["paused"] = True
    elif command == "resume":
        state["paused"] = False
    elif command == "kill-switch":
        lane = body.get("lane", "manual")
        state["kill_switch_lanes"] = set_kill_switch_lane(
            state["kill_switch_lanes"],
            lane,
            active=True,
            reason=body.get("reason"),
        )
        state["kill_switch_requested"] = kill_switch_active(state["kill_switch_lanes"])
        get_status_store().update(
            kill_switch_active=state["kill_switch_requested"],
            kill_switch_lanes=state["kill_switch_lanes"],
        )
    elif command == "clear-kill-switch":
        lane = body.get("lane")
        if lane:
            state["kill_switch_lanes"] = set_kill_switch_lane(
                state["kill_switch_lanes"],
                lane,
                active=False,
                reason=body.get("reason"),
            )
        else:
            state["kill_switch_lanes"] = normalize_kill_switch_lanes({})
        state["kill_switch_requested"] = kill_switch_active(state["kill_switch_lanes"])
        get_status_store().update(
            kill_switch_active=state["kill_switch_requested"],
            kill_switch_lanes=state["kill_switch_lanes"],
        )
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


@app.get("/ops/readiness")
def ops_readiness(book_id: str = "primary"):
    """Trader-facing live readiness checklist and operational guardrails."""
    config = get_config()
    store = get_status_store()
    runtime = store.snapshot()
    registry = ModelRegistry()
    readiness = _production_readiness(registry)
    drift = _feature_drift_snapshot(config, registry)
    paper = generate_paper_report(config=config, book_id=book_id)
    gate_result = evaluate_paper_gate(
        config, book_id=book_id, journal_path=_journal_path(config)
    )
    latest_decision = _latest_journal_decision(config)
    db_snapshot = _duckdb_snapshot(config)
    lifecycle_rows = _load_order_lifecycle_rows(config, book_id=book_id, limit=500)
    lifecycle_summary = summarize_order_lifecycle(lifecycle_rows)
    explain = runtime.get("last_explanation") or {}
    checks: list[Dict[str, Any]] = []

    runtime_age = _age_seconds(runtime.get("updated_at"))
    if runtime_age is None:
        _add_check(
            checks,
            "critical",
            "runtime_status_missing",
            "Runtime status has not been published yet.",
        )
    elif runtime_age > 60:
        _add_check(
            checks,
            "critical",
            "runtime_status_stale",
            "Runtime status is stale for live supervision.",
            age_seconds=round(runtime_age, 2),
        )
    elif runtime_age > 15:
        _add_check(
            checks,
            "warning",
            "runtime_status_lagging",
            "Runtime status is lagging behind the expected operator heartbeat.",
            age_seconds=round(runtime_age, 2),
        )

    if runtime.get("kill_switch_active"):
        _add_check(
            checks,
            "critical",
            "kill_switch_active",
            "Kill switch is active; live trading must remain blocked.",
            active_lanes=list(
                active_kill_switch_lanes(
                    normalize_kill_switch_lanes(runtime.get("kill_switch_lanes"))
                )
            ),
        )

    if not readiness.get("ready"):
        _add_check(
            checks,
            "critical",
            "prod_model_not_ready",
            "No production-ready model is active.",
            blockers=readiness.get("blockers", []),
        )

    if not gate_result.passed:
        _add_check(
            checks,
            "critical",
            "paper_to_live_gate_blocked",
            "Paper-to-live gate has not passed.",
            reasons=gate_result.reasons,
        )

    if int(paper.get("filled_orders") or 0) <= 0:
        _add_check(
            checks,
            "warning",
            "fill_evidence_missing",
            "No maker fill evidence is available for the primary paper book.",
        )
    elif float(paper.get("fill_rate") or 0.0) <= 0.05:
        _add_check(
            checks,
            "warning",
            "fill_rate_low",
            "Paper fill rate is too low to trust execution assumptions.",
            fill_rate=paper.get("fill_rate"),
        )
    if lifecycle_summary["submitted"] and not lifecycle_summary["fills"]:
        _add_check(
            checks,
            "warning",
            "order_lifecycle_missing_fills",
            "Order lifecycle telemetry has submissions but no fills.",
            submitted=lifecycle_summary["submitted"],
        )
    if lifecycle_summary["rejects"]:
        _add_check(
            checks,
            "warning",
            "order_rejections_present",
            "Order lifecycle contains rejected orders that need execution review.",
            rejects=lifecycle_summary["rejects"],
        )

    conviction = explain.get("conviction_score")
    min_conviction = float(config.get("risk", {}).get("min_live_conviction", 0.55))
    if conviction is not None and float(conviction) < min_conviction:
        _add_check(
            checks,
            "warning",
            "model_conviction_low",
            "Latest model conviction is below the live review threshold.",
            conviction=conviction,
            min_conviction=min_conviction,
        )

    decision_age = _age_seconds((latest_decision or {}).get("timestamp"))
    if decision_age is None:
        _add_check(
            checks,
            "warning",
            "decision_journal_missing",
            "No persisted decision journal entry is available.",
        )
    elif decision_age > 300:
        _add_check(
            checks,
            "warning",
            "decision_journal_stale",
            "Persisted decision journal is stale relative to runtime status.",
            age_seconds=round(decision_age, 2),
        )

    if db_snapshot.get("error"):
        _add_check(
            checks,
            "warning",
            "duckdb_read_unavailable",
            "DuckDB could not be read for operator diagnostics.",
            error=db_snapshot["error"],
        )
    if db_snapshot.get("exists") and not db_snapshot.get("tables", {}).get("ticks", 0):
        _add_check(
            checks,
            "warning",
            "tick_history_missing",
            "Tick history is empty; fill and microstructure analytics are limited.",
        )
    if drift.get("status") in {"warning", "critical"}:
        _add_check(
            checks,
            "warning" if drift.get("status") == "warning" else "critical",
            "feature_drift_detected",
            "Current feature distribution has drifted from the active model reference.",
            max_abs_z=drift.get("max_abs_z"),
        )

    severity_order = {"critical": 2, "warning": 1, "info": 0}
    critical_count = sum(1 for check in checks if check["severity"] == "critical")
    warning_count = sum(1 for check in checks if check["severity"] == "warning")
    live_ready = critical_count == 0
    return {
        "title": "APEX Trader Production Readiness",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if live_ready and warning_count == 0 else "blocked",
        "summary": {
            "live_ready": live_ready,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "operator_mode": runtime.get("operator_mode"),
            "symbol": runtime.get("symbol"),
            "mark_price": runtime.get("mark_price"),
            "active_kill_switch_lanes": list(
                active_kill_switch_lanes(
                    normalize_kill_switch_lanes(runtime.get("kill_switch_lanes"))
                )
            ),
            "active_prod": registry.registry_data.get("active_prod"),
            "active_shadow": registry.registry_data.get("active_shadow"),
            "paper_sharpe": paper.get("sharpe"),
            "fill_rate": paper.get("fill_rate"),
        },
        "checks": sorted(
            checks,
            key=lambda item: (-severity_order.get(item["severity"], 0), item["code"]),
        ),
        "runtime": runtime,
        "paper": paper,
        "order_lifecycle": {
            "summary": lifecycle_summary,
            "recent": lifecycle_rows[:20],
        },
        "live_gate": {
            "passed": gate_result.passed,
            "reasons": gate_result.reasons,
            "metrics": gate_result.metrics,
        },
        "production_readiness": readiness,
        "feature_drift": drift,
        "data": db_snapshot,
        "next_actions": [
            (
                "Repair unsafe shadow artifacts and keep shadow lanes running "
                "continuously."
            ),
            "Collect forward paper evidence with real maker fills before live review.",
            "Promote only a manifest-backed PROD model after shadow evidence clears.",
            (
                "Keep data freshness, user stream, and kill-switch health visible "
                "in cockpit."
            ),
        ],
    }


@app.get("/orders/lifecycle")
def order_lifecycle(book_id: str = "primary", limit: int = 500):
    """Order lifecycle telemetry and execution-quality summary."""
    config = get_config()
    rows = _load_order_lifecycle_rows(config, book_id=book_id, limit=limit)
    return {
        "book_id": book_id,
        "items": rows,
        "total": len(rows),
        "summary": summarize_order_lifecycle(rows),
    }


@app.get("/models/drift")
def model_drift(model_id: Optional[str] = None):
    """Compare current feature distributions with active model training evidence."""
    config = get_config()
    registry = ModelRegistry()
    return _feature_drift_snapshot(config, registry, model_id=model_id)


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
    config = get_config()
    interval = float(_api_cfg(config).get("status_ws_interval_sec", 0.5))
    try:
        while True:
            payload = store.snapshot()
            await websocket.send_json(payload)
            await asyncio.sleep(max(0.1, interval))
    except WebSocketDisconnect:
        logger.info("Status websocket disconnected")


@app.websocket("/ws/market")
async def ws_market(websocket: WebSocket):
    """Direct low-latency market stream for the cockpit live price chart."""
    await websocket.accept()
    config = get_config()
    symbol = (
        websocket.query_params.get("symbol")
        or config.get("data", {}).get("target_symbol", "ETHUSDC")
    ).upper()
    timeout = float(_api_cfg(config).get("market_ws_timeout_sec", 15.0))
    stream_url = _market_stream_url(config, symbol)
    try:
        async with websockets.connect(
            stream_url,
            ping_interval=20,
            ping_timeout=10,
            max_queue=32,
        ) as upstream:
            await websocket.send_json(
                {
                    "type": "connected",
                    "symbol": symbol,
                    "source": "binance_ws",
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            while True:
                raw = await asyncio.wait_for(upstream.recv(), timeout=timeout)
                payload = _normalize_market_ws_event(json.loads(raw))
                if payload and payload.get("symbol") in {"", symbol}:
                    await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info("Market websocket disconnected")
    except Exception as exc:
        await websocket.send_json(
            {
                "type": "error",
                "symbol": symbol,
                "source": "binance_ws",
                "message": str(exc),
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        await websocket.close()


def main():
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("APEX_API_HOST", "127.0.0.1"),
        port=int(os.getenv("APEX_API_PORT", "8080")),
        reload=False,
    )


if __name__ == "__main__":
    main()
