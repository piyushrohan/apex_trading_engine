"""Operator paper→live gate: refuse live start until paper criteria are met."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.cache_manager import DuckDBCacheManager
from src.reports.paper_report import generate_paper_report

logger = logging.getLogger(__name__)


@dataclass
class PaperGateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def evaluate_paper_gate(
    config: Dict[str, Any],
    *,
    book_id: str = "primary",
    journal_path: str = "data_lake/trade_journal.jsonl",
    skip_gate: bool = False,
) -> PaperGateResult:
    """
    Check paper trading history against config thresholds.
    Returns passed=False with human-readable reasons when criteria fail.
    """
    paper_cfg = config.get("paper", {})
    if skip_gate or not paper_cfg.get("enabled", True):
        return PaperGateResult(passed=True, metrics={"gate": "skipped"})

    min_days = float(paper_cfg.get("min_days", 7))
    min_trades = int(paper_cfg.get("min_trades", 100))
    min_sharpe = float(paper_cfg.get("min_sharpe", 1.0))
    max_drawdown = float(paper_cfg.get("max_drawdown", 0.08))

    report = generate_paper_report(config, book_id=book_id, journal_path=journal_path)
    paper_days = _paper_run_days(config, book_id)
    metrics = {
        **report,
        "paper_days": paper_days,
        "min_days": min_days,
        "min_trades": min_trades,
        "min_sharpe": min_sharpe,
        "max_drawdown_limit": max_drawdown,
    }

    reasons: List[str] = []
    if paper_days < min_days:
        reasons.append(
            f"paper run {paper_days:.1f}d < required {min_days:.0f}d "
            "(equity snapshot span)"
        )
    directional = int(report.get("directional_decisions", 0))
    if directional < min_trades:
        reasons.append(f"directional decisions {directional} < required {min_trades}")
    sharpe = float(report.get("sharpe") or 0.0)
    if sharpe < min_sharpe:
        reasons.append(f"Sharpe {sharpe:.2f} < required {min_sharpe:.2f}")
    dd = float(report.get("max_drawdown") or 0.0)
    if dd > max_drawdown:
        reasons.append(f"max drawdown {dd:.2%} exceeds limit {max_drawdown:.2%}")
    if int(report.get("snapshots", 0)) == 0:
        reasons.append("no paper equity snapshots in data lake")

    passed = len(reasons) == 0
    if passed:
        logger.info("Paper gate PASSED — live startup allowed.")
    else:
        logger.warning("Paper gate FAILED: %s", "; ".join(reasons))

    return PaperGateResult(passed=passed, reasons=reasons, metrics=metrics)


def _paper_run_days(config: Dict[str, Any], book_id: str) -> float:
    db_path = (
        config.get("data", {})
        .get("storage", {})
        .get("db_path", "data_lake/apex_market_data.duckdb")
    )
    cache = DuckDBCacheManager(db_path=db_path)
    try:
        df = cache.load_paper_equity_snapshots(book_id)
    finally:
        cache.close()
    if df.empty or "timestamp" not in df.columns:
        return 0.0
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if len(ts) < 2:
        return 0.0
    return max((ts.max() - ts.min()).total_seconds() / 86400.0, 0.0)


def validate_live_startup(config: Dict[str, Any]) -> None:
    """
    Raise RuntimeError if live mode cannot start safely.
    Checks live.enabled and paper gate (unless live.skip_paper_gate).
    """
    live_cfg = config.get("live", {})
    if not live_cfg.get("enabled", False):
        raise RuntimeError(
            "Live operator mode blocked: set live.enabled: true in config"
        )

    if live_cfg.get("skip_paper_gate", False):
        logger.warning("live.skip_paper_gate=true — paper criteria not enforced")
        return

    result = evaluate_paper_gate(config)
    if not result.passed:
        raise RuntimeError(
            "Live startup blocked — paper gate failed: " + "; ".join(result.reasons)
        )


def check_api_credentials(config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Ensure signed REST can run (keys from env or config)."""
    import os

    key = os.getenv("BINANCE_API_KEY") or config.get("live", {}).get("api_key")
    secret = os.getenv("BINANCE_API_SECRET") or config.get("live", {}).get("api_secret")
    if not key or not secret:
        return False, "BINANCE_API_KEY and BINANCE_API_SECRET required for live mode"
    return True, None
