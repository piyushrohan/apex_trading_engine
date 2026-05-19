"""Read-only FastAPI server for operator status and explainability."""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from src.api.status_store import get_status_store
from src.core.config_loader import load_config
from src.data.cache_manager import DuckDBCacheManager
from src.mlops.explainability import ExplainabilityEngine
from src.reports.paper_report import generate_paper_report

logger = logging.getLogger(__name__)

app = FastAPI(
    title="APEX Trading Engine API",
    description="Read-only status and explainability for paper/live operator modes.",
    version="1.0.0",
)

_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


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


@app.get("/explain/latest")
def explain_latest():
    """Most recent decision explanation from runtime store or trade journal."""
    store = get_status_store()
    if store.last_explanation:
        return store.last_explanation

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
    runtime = store.snapshot().get("portfolio", {})

    config = get_config()
    db_path = (
        config.get("data", {})
        .get("storage", {})
        .get("db_path", "data_lake/apex_market_data.duckdb")
    )
    cache = DuckDBCacheManager(db_path=db_path)
    try:
        equity_df = cache.load_paper_equity_snapshots(book_id)
    finally:
        cache.close()

    latest_equity = None
    snapshot_count = 0
    if not equity_df.empty:
        snapshot_count = len(equity_df)
        latest_equity = float(equity_df["equity"].iloc[-1])

    return {
        "book_id": book_id,
        "symbol": store.symbol,
        "operator_mode": store.operator_mode,
        "runtime": runtime,
        "paper_snapshots": snapshot_count,
        "latest_equity_from_db": latest_equity,
        "updated_at": store.updated_at,
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


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """Push operator status snapshots for the terminal."""
    await websocket.accept()
    store = get_status_store()
    try:
        while True:
            payload = store.snapshot()
            payload["last_explanation"] = store.last_explanation
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
        host="0.0.0.0",
        port=8080,
        reload=False,
    )


if __name__ == "__main__":
    main()
