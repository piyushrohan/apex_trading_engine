"""Read-only FastAPI server for operator status and explainability."""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from src.api.status_store import get_status_store
from src.core.config_loader import load_config
from src.data.cache_manager import DuckDBCacheManager
from src.mlops.explainability import ExplainabilityEngine

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
