import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.mlops.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class PromotionDecision:
    action: str
    model_id: str
    reason: str
    shadow_score: float
    primary_score: float


class PromotionService:
    """
    Gatekeeper for MLOps shadow-to-prod promotion.

    This is intentionally separate from the operator paper-to-live gate.
    """

    def __init__(self, config: Dict[str, Any], registry: ModelRegistry | None = None):
        self.config = config
        self.registry = registry or ModelRegistry()
        cfg = config.get("promotion", {})
        self.min_shadow_trades = cfg.get("min_shadow_trades", 50)
        self.min_sharpe_delta = cfg.get("min_sharpe_delta", 0.15)
        self.max_shadow_drawdown = cfg.get("max_shadow_drawdown", 0.10)

    def evaluate(
        self,
        model_id: str,
        primary_metrics: Dict[str, float],
        shadow_metrics: Dict[str, float],
    ) -> PromotionDecision:
        shadow_trades = shadow_metrics.get("total_trades", 0)
        if shadow_trades < self.min_shadow_trades:
            return PromotionDecision(
                "hold",
                model_id,
                "insufficient_shadow_trades",
                shadow_metrics.get("sharpe", 0.0),
                primary_metrics.get("sharpe", 0.0),
            )

        shadow_dd = shadow_metrics.get("max_drawdown", 1.0)
        if shadow_dd > self.max_shadow_drawdown:
            return PromotionDecision(
                "discard",
                model_id,
                "shadow_drawdown_breach",
                shadow_metrics.get("sharpe", 0.0),
                primary_metrics.get("sharpe", 0.0),
            )

        primary_sharpe = primary_metrics.get("sharpe", 0.0)
        shadow_sharpe = shadow_metrics.get("sharpe", 0.0)
        if shadow_sharpe < primary_sharpe + self.min_sharpe_delta:
            return PromotionDecision(
                "hold",
                model_id,
                "shadow_edge_not_material",
                shadow_sharpe,
                primary_sharpe,
            )

        return PromotionDecision(
            "promote",
            model_id,
            "shadow_outperformed_primary",
            shadow_sharpe,
            primary_sharpe,
        )

    def evaluate_and_apply(
        self,
        model_id: str,
        primary_metrics: Dict[str, float],
        shadow_metrics: Dict[str, float],
    ) -> PromotionDecision:
        decision = self.evaluate(model_id, primary_metrics, shadow_metrics)
        if decision.action == "promote":
            self.registry.promote_to_prod(model_id)
        elif decision.action == "discard":
            self.registry.set_model_status(model_id, "REJECTED")
        logger.info("Promotion decision for %s: %s", model_id, decision.reason)
        return decision

    def metrics_from_decision_log(
        self,
        *,
        decision_path: str = "data_lake/hedge_bandit/training/decisions.jsonl",
        book_id: str,
    ) -> Dict[str, float]:
        """Build promotion metrics from persisted primary/shadow decision rows."""
        path = Path(decision_path)
        if not path.exists():
            return {"total_trades": 0, "sharpe": 0.0, "max_drawdown": 1.0}

        equities = []
        total_trades = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("book", {}).get("id") != book_id:
                continue
            if row.get("action") in (0, 2) or row.get("approved_fraction", 0) > 0:
                total_trades += 1
            if row.get("equity") is not None:
                equities.append(float(row["equity"]))

        if len(equities) < 3:
            return {"total_trades": total_trades, "sharpe": 0.0, "max_drawdown": 1.0}

        values = np.asarray(equities, dtype=float)
        returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
        sharpe = 0.0
        if returns.std() > 0:
            sharpe = float((returns.mean() / returns.std()) * np.sqrt(365 * 24 * 20))
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / np.maximum(running_max, 1e-9)
        return {
            "total_trades": total_trades,
            "sharpe": sharpe,
            "max_drawdown": float(abs(drawdown.min())),
        }

    def rollback_if_live_breach(
        self,
        live_metrics: Dict[str, float],
        *,
        max_live_drawdown: float | None = None,
        min_live_sharpe: float | None = None,
    ) -> str | None:
        """Rollback production when live metrics breach configured safety limits."""
        cfg = self.config.get("promotion", {})
        dd_limit = max_live_drawdown or cfg.get("max_live_drawdown", 0.10)
        sharpe_floor = min_live_sharpe or cfg.get("min_live_sharpe", -0.25)
        if (
            live_metrics.get("max_drawdown", 0.0) > dd_limit
            or live_metrics.get("sharpe", 0.0) < sharpe_floor
        ):
            return self.rollback_active_prod()
        return None

    def rollback_active_prod(self) -> str | None:
        """Rollback live production to the previous registry prod model."""
        restored = self.registry.rollback_prod()
        if restored:
            logger.critical("Rolled production model back to %s", restored)
        return restored
