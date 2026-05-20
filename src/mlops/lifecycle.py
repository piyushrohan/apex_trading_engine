import logging
from typing import Any, Dict, Optional

from src.mlops.auto_retrain import AutoRetrainPipeline
from src.mlops.experiment_tracker import ExperimentTracker, stable_hash
from src.mlops.promotion_service import PromotionService
from src.mlops.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelLifecycleOrchestrator:
    """Governed train/evaluate/shadow/promote workflow for model evolution."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        registry: Optional[ModelRegistry] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.config = config
        self.registry = registry or ModelRegistry(
            registry_dir=config.get("mlops", {}).get("registry_dir", "data_lake/models")
        )
        self.tracker = tracker or ExperimentTracker.from_config(config)

    def run_candidate_cycle(self) -> Dict[str, Any]:
        """Train, validate, backtest, stress, register, and shadow a candidate."""
        pipeline = AutoRetrainPipeline(
            self.config,
            registry=self.registry,
            tracker=self.tracker,
        )
        return pipeline.execute_nightly_retrain()

    def review_shadow_promotion(
        self,
        *,
        model_id: Optional[str] = None,
        primary_metrics: Dict[str, float],
        shadow_metrics: Dict[str, float],
        apply: bool = False,
        reviewer: str = "promotion_service",
    ) -> Dict[str, Any]:
        selected = model_id or self.registry.registry_data.get("active_shadow")
        if not selected:
            return {
                "decision": {
                    "action": "hold",
                    "reason": "no_active_shadow_model",
                    "model_id": None,
                }
            }

        run = self.tracker.start_run(
            "promotion_review",
            metadata={
                "model_id": selected,
                "config_hash": stable_hash(self.config),
                "apply": apply,
            },
        )
        service = PromotionService(self.config, registry=self.registry)
        try:
            decision = (
                service.evaluate_and_apply(
                    selected,
                    primary_metrics,
                    shadow_metrics,
                    reviewer=reviewer,
                )
                if apply
                else service.evaluate(selected, primary_metrics, shadow_metrics)
            )
            payload = decision.__dict__
            self.tracker.complete_run(
                run["run_id"],
                "COMPLETED",
                model_id=selected,
                metrics={
                    "shadow_score": decision.shadow_score,
                    "primary_score": decision.primary_score,
                },
                metadata={"decision": payload},
            )
            return {"decision": payload}
        except Exception as exc:
            logger.exception("Promotion review failed")
            self.tracker.complete_run(
                run["run_id"],
                "FAILED",
                model_id=selected,
                metadata={"error": str(exc)},
            )
            raise
