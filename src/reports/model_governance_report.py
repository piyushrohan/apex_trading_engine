"""Daily model governance report for registry, retraining, and promotion state."""

import argparse
import sys
from collections import Counter
from typing import Any, Dict, Optional

from src.core.config_loader import load_config
from src.mlops.experiment_tracker import ExperimentTracker
from src.mlops.promotion_service import PromotionDecision, PromotionService
from src.mlops.registry import ModelRegistry
from src.reports.ops_common import (
    add_finding,
    should_exit_nonzero,
    status_from_findings,
    utc_now,
    write_report,
)
from src.reports.paper_report import generate_paper_report


def _model_summary(
    model_id: Optional[str], model: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not model_id or not model:
        return {"model_id": model_id, "present": False}
    return {
        "model_id": model_id,
        "present": True,
        "type": model.get("type"),
        "status": model.get("status"),
        "created_at": model.get("created_at"),
        "metrics": model.get("metrics", {}),
        "manifest_path": model.get("manifest_path"),
        "artifact_path": model.get("artifact_path"),
        "metadata": model.get("metadata", {}),
    }


def _shadow_metrics(
    service: PromotionService,
    *,
    decision_path: str,
    active_shadow: str,
) -> Dict[str, Any]:
    primary_book_id = active_shadow
    metrics = service.metrics_from_decision_log(
        decision_path=decision_path, book_id=primary_book_id
    )
    metrics["book_id"] = primary_book_id
    if metrics.get("total_trades", 0) > 0:
        return metrics

    shadow_book_id = f"shadow_{active_shadow}"
    fallback = service.metrics_from_decision_log(
        decision_path=decision_path, book_id=shadow_book_id
    )
    fallback["book_id"] = shadow_book_id
    return fallback if fallback.get("total_trades", 0) > 0 else metrics


def _decision_payload(
    decision: Optional[PromotionDecision],
) -> Optional[Dict[str, Any]]:
    return None if decision is None else decision.__dict__


def generate_model_governance_report(
    config: Dict[str, Any],
    *,
    registry: Optional[ModelRegistry] = None,
    tracker: Optional[ExperimentTracker] = None,
    decision_path: Optional[str] = None,
    journal_path: Optional[str] = None,
    run_limit: int = 10,
) -> Dict[str, Any]:
    """Summarize model registry discipline and the current promotion posture."""
    registry = registry or ModelRegistry(
        registry_dir=config.get("mlops", {}).get("registry_dir", "data_lake/models")
    )
    tracker = tracker or ExperimentTracker.from_config(config)
    registry_data = registry.registry_data
    models = registry_data.get("models", {})
    active_prod = registry_data.get("active_prod")
    active_shadow = registry_data.get("active_shadow")
    decision_path = decision_path or config.get("shadow", {}).get(
        "decision_log_path", "data_lake/hedge_bandit/training/decisions.jsonl"
    )
    journal_path = journal_path or config.get("explainability", {}).get(
        "journal_path", "data_lake/trade_journal.jsonl"
    )

    findings: list[Dict[str, Any]] = []
    readiness = registry.production_readiness(active_prod)
    if not active_prod:
        add_finding(
            findings,
            "warning",
            "active_prod_missing",
            "No active production model is registered.",
        )
    elif not readiness.get("ready"):
        add_finding(
            findings,
            "warning",
            "active_prod_not_ready",
            "Active production model is blocked from live inference.",
            blockers=readiness.get("blockers", []),
        )

    if not active_shadow:
        add_finding(
            findings,
            "warning",
            "active_shadow_missing",
            "No active shadow model is available for promotion review.",
        )
    elif active_shadow not in models:
        add_finding(
            findings,
            "error",
            "active_shadow_registry_missing",
            "active_shadow points to a missing registry entry.",
            active_shadow=active_shadow,
        )

    runs = tracker.list_runs(limit=max(1, run_limit))
    if not runs:
        add_finding(
            findings,
            "warning",
            "experiment_runs_missing",
            "No experiment tracker runs were found.",
        )
    elif runs[0].get("status") not in {"PASSED", "SUCCESS", "COMPLETED"}:
        add_finding(
            findings,
            "warning",
            "latest_run_not_successful",
            "Latest experiment run did not complete successfully.",
            status=runs[0].get("status"),
        )

    primary_metrics = generate_paper_report(
        config=config, book_id="primary", journal_path=journal_path
    )
    promotion_decision = None
    shadow_metrics = None
    if active_shadow and active_shadow in models:
        service = PromotionService(config, registry=registry)
        shadow_metrics = _shadow_metrics(
            service, decision_path=decision_path, active_shadow=active_shadow
        )
        promotion_decision = service.evaluate(
            active_shadow, primary_metrics, shadow_metrics
        )
        if promotion_decision.action == "promote":
            add_finding(
                findings,
                "warning",
                "shadow_ready_for_review",
                "Shadow model outperformed primary and needs human promotion review.",
                model_id=active_shadow,
            )
        elif promotion_decision.reason == "insufficient_shadow_trades":
            add_finding(
                findings,
                "warning",
                "shadow_needs_more_evidence",
                "Shadow model has insufficient decision history.",
                model_id=active_shadow,
            )

    status_counts = Counter(model.get("status", "UNKNOWN") for model in models.values())
    recommendation = "hold"
    if not active_prod:
        recommendation = "train_and_promote_prod_candidate"
    elif not readiness.get("ready"):
        recommendation = "fix_active_prod_readiness"
    elif promotion_decision and promotion_decision.action == "promote":
        recommendation = "review_shadow_for_prod_promotion"
    elif (
        active_shadow and shadow_metrics and shadow_metrics.get("total_trades", 0) == 0
    ):
        recommendation = "continue_shadow_data_collection"

    summary = {
        "active_prod": active_prod,
        "active_shadow": active_shadow,
        "model_count": len(models),
        "status_counts": dict(status_counts),
        "latest_run_id": runs[0]["run_id"] if runs else None,
        "latest_run_status": runs[0]["status"] if runs else None,
        "promotion_action": (
            promotion_decision.action if promotion_decision is not None else None
        ),
        "promotion_reason": (
            promotion_decision.reason if promotion_decision is not None else None
        ),
        "recommendation": recommendation,
    }
    return {
        "title": "APEX Daily Model Governance Report",
        "generated_at": utc_now().isoformat(),
        "status": status_from_findings(findings),
        "summary": summary,
        "production_readiness": readiness,
        "active_prod": _model_summary(active_prod, models.get(active_prod)),
        "active_shadow": _model_summary(active_shadow, models.get(active_shadow)),
        "primary_metrics": primary_metrics,
        "shadow_metrics": shadow_metrics,
        "promotion_decision": _decision_payload(promotion_decision),
        "recent_runs": runs,
        "registry_events": registry_data.get("events", [])[-100:],
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX model governance report")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--decision-path")
    parser.add_argument("--journal-path")
    parser.add_argument("--run-limit", type=int, default=10)
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    report = generate_model_governance_report(
        config,
        decision_path=args.decision_path,
        journal_path=args.journal_path,
        run_limit=args.run_limit,
    )
    rendered = write_report(report, output=args.output, fmt=args.format)
    if not args.output:
        print(rendered)
    if should_exit_nonzero(report, fail_on_warning=args.fail_on_warning):
        sys.exit(1)


if __name__ == "__main__":
    main()
