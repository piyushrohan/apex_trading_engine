"""Shadow lane sanity monitor for candidate model evaluation."""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.config_loader import load_config
from src.mlops.registry import ModelRegistry
from src.reports.ops_common import (
    add_finding,
    age_minutes,
    parse_timestamp,
    read_jsonl,
    should_exit_nonzero,
    status_from_findings,
    utc_now,
    write_report,
)


def _artifact_exists(
    registry: ModelRegistry, model_id: str, model: Dict[str, Any]
) -> bool:
    model_path = model.get("artifact_path") or registry.get_model_path(model_id)
    if not model_path or not os.path.isdir(model_path):
        return False
    expected = {
        "PPO": "ppo_actor_critic.pt",
        "GBM": "gbm_model.pkl",
        "LIGHTGBM": "gbm_model.pkl",
    }.get(str(model.get("type", "")).upper())
    if expected:
        return os.path.exists(os.path.join(model_path, expected))
    return any(os.scandir(model_path))


def _latest_row(rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    dated = [(parse_timestamp(row.get("timestamp")), row) for row in rows]
    dated = [(ts, row) for ts, row in dated if ts is not None]
    if dated:
        return max(dated, key=lambda item: item[0])[1]
    return rows[-1] if rows else None


def generate_shadow_sanity_report(
    config: Dict[str, Any],
    *,
    registry: Optional[ModelRegistry] = None,
    decision_path: Optional[str] = None,
    now: Optional[datetime] = None,
    max_decision_age_minutes: float = 60.0,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate that shadow lanes remain virtual and produce usable evidence."""
    generated_at = (now or utc_now()).isoformat()
    findings: list[Dict[str, Any]] = []
    shadow_cfg = config.get("shadow", {})
    decision_path = decision_path or shadow_cfg.get(
        "decision_log_path", "data_lake/hedge_bandit/training/decisions.jsonl"
    )
    registry = registry or ModelRegistry(
        registry_dir=config.get("mlops", {}).get("registry_dir", "data_lake/models")
    )
    registry_data = registry.registry_data
    models = registry_data.get("models", {})
    active_shadow = registry_data.get("active_shadow")
    candidates = {
        model_id: model
        for model_id, model in models.items()
        if model.get("status") in {"CANDIDATE", "EVALUATING", "SHADOW"}
    }

    if not shadow_cfg.get("enabled", False):
        add_finding(
            findings,
            "warning",
            "shadow_disabled",
            "Shadow lane config is disabled.",
        )

    if not active_shadow:
        add_finding(
            findings,
            "error" if strict else "warning",
            "active_shadow_missing",
            "No active shadow model is registered.",
        )
    elif active_shadow not in models:
        add_finding(
            findings,
            "error",
            "active_shadow_registry_missing",
            "active_shadow points to a missing registry entry.",
            active_shadow=active_shadow,
        )
    else:
        model = models[active_shadow]
        if model.get("status") != "SHADOW":
            add_finding(
                findings,
                "error" if strict else "warning",
                "active_shadow_status_unexpected",
                "Active shadow model is not in SHADOW status.",
                status=model.get("status"),
            )
        if not _artifact_exists(registry, active_shadow, model):
            add_finding(
                findings,
                "error" if strict else "warning",
                "active_shadow_artifact_missing",
                "Active shadow model artifact is missing.",
                active_shadow=active_shadow,
            )
        manifest_path = model.get("manifest_path")
        if not manifest_path or not os.path.exists(manifest_path):
            add_finding(
                findings,
                "warning",
                "active_shadow_manifest_missing",
                "Active shadow model has no reproducibility manifest.",
                active_shadow=active_shadow,
            )

    rows = read_jsonl(decision_path)
    invalid_rows = [row for row in rows if row.get("_invalid_json")]
    shadow_rows = [
        row
        for row in rows
        if row.get("book", {}).get("role") == "shadow"
        or str(row.get("book", {}).get("id", "")).startswith("shadow_")
    ]
    active_rows = [
        row
        for row in shadow_rows
        if active_shadow and row.get("model_id") == active_shadow
    ]
    latest_shadow = _latest_row(shadow_rows)
    hedge_enabled_rows = [
        row for row in shadow_rows if (row.get("hedge") or {}).get("enabled")
    ]
    hedge_rows_missing_candidates = [
        row
        for row in hedge_enabled_rows
        if not (row.get("hedge") or {}).get("candidates")
    ]
    malformed_shadow_book_rows = [
        row
        for row in shadow_rows
        if row.get("book", {}).get("role") != "shadow"
        or not str(row.get("book", {}).get("id", "")).startswith("shadow_")
    ]

    if invalid_rows:
        add_finding(
            findings,
            "warning",
            "decision_log_invalid_rows",
            "Shadow decision log contains invalid JSON rows.",
            count=len(invalid_rows),
        )
    if not shadow_rows:
        add_finding(
            findings,
            "error" if strict else "warning",
            "shadow_decisions_missing",
            "No shadow decision rows were found.",
            decision_path=decision_path,
        )
    elif latest_shadow:
        decision_age = age_minutes(latest_shadow.get("timestamp"), now=now)
        if decision_age is not None and decision_age > max_decision_age_minutes:
            add_finding(
                findings,
                "error" if strict else "warning",
                "shadow_decision_stale",
                "Latest shadow decision is stale.",
                age_minutes=round(decision_age, 2),
                max_age_minutes=max_decision_age_minutes,
            )
    if active_shadow and not active_rows:
        add_finding(
            findings,
            "error" if strict else "warning",
            "active_shadow_decisions_missing",
            "No decision evidence exists for the active shadow model.",
            active_shadow=active_shadow,
        )
    if malformed_shadow_book_rows:
        add_finding(
            findings,
            "error",
            "shadow_book_tagging_invalid",
            "Shadow decisions must use book.role=shadow and book.id=shadow_<model>.",
            count=len(malformed_shadow_book_rows),
        )
    if hedge_rows_missing_candidates:
        add_finding(
            findings,
            "warning",
            "hedge_candidate_scores_missing",
            "Some shadow hedge rows lack candidate score maps.",
            count=len(hedge_rows_missing_candidates),
        )

    model_ids = sorted(
        {row.get("model_id") for row in shadow_rows if row.get("model_id")}
    )
    summary = {
        "shadow_enabled": bool(shadow_cfg.get("enabled", False)),
        "active_shadow": active_shadow,
        "candidate_models": len(candidates),
        "decision_path": decision_path,
        "shadow_decision_rows": len(shadow_rows),
        "active_shadow_decision_rows": len(active_rows),
        "shadow_model_ids_seen": model_ids,
        "latest_shadow_decision_at": (latest_shadow or {}).get("timestamp"),
        "hedge_enabled_rows": len(hedge_enabled_rows),
    }
    return {
        "title": "APEX Shadow Lane Sanity Monitor",
        "generated_at": generated_at,
        "status": status_from_findings(findings),
        "summary": summary,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX shadow lane sanity monitor")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--decision-path")
    parser.add_argument("--max-decision-age-minutes", type=float, default=60.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    report = generate_shadow_sanity_report(
        config,
        decision_path=args.decision_path,
        max_decision_age_minutes=args.max_decision_age_minutes,
        strict=args.strict,
    )
    rendered = write_report(report, output=args.output, fmt=args.format)
    if not args.output:
        print(rendered)
    if should_exit_nonzero(report, fail_on_warning=args.fail_on_warning):
        sys.exit(1)


if __name__ == "__main__":
    main()
