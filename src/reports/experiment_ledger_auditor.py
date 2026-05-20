"""Audit experiment ledger discipline and registry linkage."""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.core.config_loader import load_config
from src.mlops.experiment_tracker import ExperimentTracker
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

SUCCESS_STATUSES = {"PASSED", "SUCCESS", "COMPLETED", "SKIPPED"}
FAILURE_STATUSES = {"FAILED", "ERROR", "FAIL", "REJECTED"}
TERMINAL_STATUSES = SUCCESS_STATUSES | FAILURE_STATUSES
REQUIRED_CANDIDATE_STEPS = {"data_snapshot", "train", "oos_backtest", "stress"}


def _artifact_exists(model: Dict[str, Any]) -> bool:
    artifact_path = model.get("artifact_path")
    if not artifact_path or not os.path.isdir(artifact_path):
        return False
    expected = {
        "PPO": "ppo_actor_critic.pt",
        "GBM": "gbm_model.pkl",
        "LIGHTGBM": "gbm_model.pkl",
    }.get(str(model.get("type", "")).upper())
    if expected:
        return os.path.exists(os.path.join(artifact_path, expected))
    return any(os.scandir(artifact_path))


def _group_runs(events: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    runs: Dict[str, Dict[str, Any]] = {}
    for idx, event in enumerate(events):
        run_id = event.get("run_id")
        if not run_id:
            continue
        run = runs.setdefault(
            run_id,
            {
                "run_id": run_id,
                "run_type": None,
                "status": "UNKNOWN",
                "started_at": None,
                "completed_at": None,
                "model_id": None,
                "metadata": {},
                "metrics": {},
                "steps": [],
                "_start_count": 0,
                "_complete_count": 0,
                "_events": [],
            },
        )
        run["_events"].append({"idx": idx, "event": event.get("event")})
        event_type = event.get("event")
        if event_type == "run_started":
            run["_start_count"] += 1
            run["run_type"] = event.get("run_type")
            run["status"] = event.get("status", "RUNNING")
            run["started_at"] = event.get("timestamp")
            run["metadata"].update(event.get("metadata", {}))
        elif event_type == "step":
            run["steps"].append(
                {
                    "step": event.get("step"),
                    "status": event.get("status"),
                    "timestamp": event.get("timestamp"),
                    "metrics": event.get("metrics", {}),
                    "metadata": event.get("metadata", {}),
                }
            )
        elif event_type == "run_completed":
            run["_complete_count"] += 1
            run["status"] = event.get("status", run["status"])
            run["completed_at"] = event.get("timestamp")
            run["model_id"] = event.get("model_id")
            run["metrics"].update(event.get("metrics", {}))
            run["metadata"].update(event.get("metadata", {}))
    return runs


def _ordered_runs(runs: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    return sorted(
        runs.values(),
        key=lambda row: row.get("completed_at") or row.get("started_at") or "",
        reverse=True,
    )


def _public_run(run: Dict[str, Any]) -> Dict[str, Any]:
    visible = {key: value for key, value in run.items() if not key.startswith("_")}
    steps = list(visible.get("steps", []))
    visible["step_count"] = len(steps)
    visible["steps"] = steps[-20:]
    return visible


def _is_completed_successfully(status: Optional[str]) -> bool:
    return str(status or "").upper() in {"PASSED", "SUCCESS", "COMPLETED"}


def _add_lifecycle_findings(
    findings: list[Dict[str, Any]],
    run: Dict[str, Any],
    *,
    now: Optional[datetime],
    strict: bool,
    max_running_age_minutes: float,
) -> None:
    run_id = run["run_id"]
    if run["_start_count"] == 0:
        add_finding(
            findings,
            "error",
            "run_start_missing",
            "Run has ledger events but no run_started event.",
            run_id=run_id,
        )
    elif run["_start_count"] > 1:
        add_finding(
            findings,
            "warning",
            "duplicate_run_start",
            "Run has multiple run_started events.",
            run_id=run_id,
            count=run["_start_count"],
        )

    if run["_complete_count"] > 1:
        add_finding(
            findings,
            "warning",
            "duplicate_run_completion",
            "Run has multiple run_completed events.",
            run_id=run_id,
            count=run["_complete_count"],
        )

    completed_at = parse_timestamp(run.get("completed_at"))
    started_at = parse_timestamp(run.get("started_at"))
    if completed_at and started_at and completed_at < started_at:
        add_finding(
            findings,
            "error",
            "completion_before_start",
            "Run completed before its recorded start timestamp.",
            run_id=run_id,
        )

    status = str(run.get("status") or "UNKNOWN").upper()
    if status == "RUNNING":
        running_age = age_minutes(run.get("started_at"), now=now)
        if running_age is not None and running_age > max_running_age_minutes:
            add_finding(
                findings,
                "error" if strict else "warning",
                "stale_running_run",
                "Run is still RUNNING beyond the expected window.",
                run_id=run_id,
                age_minutes=round(running_age, 2),
                max_age_minutes=max_running_age_minutes,
            )
    elif status not in TERMINAL_STATUSES:
        add_finding(
            findings,
            "warning",
            "unknown_run_status",
            "Run has an unrecognized terminal status.",
            run_id=run_id,
            status=status,
        )
    elif status in FAILURE_STATUSES:
        add_finding(
            findings,
            "error" if strict else "warning",
            "run_unsuccessful",
            "Run finished with a non-success status.",
            run_id=run_id,
            status=status,
        )


def _add_step_findings(
    findings: list[Dict[str, Any]], run: Dict[str, Any], *, strict: bool
) -> None:
    failed_steps = [
        step
        for step in run["steps"]
        if str(step.get("status") or "").upper() in FAILURE_STATUSES
    ]
    if failed_steps:
        add_finding(
            findings,
            "error" if strict else "warning",
            "run_failed_steps",
            "Run contains one or more failed steps.",
            run_id=run["run_id"],
            steps=sorted({step.get("step") for step in failed_steps}),
        )

    if run.get("run_type") != "candidate_retrain":
        return
    if not _is_completed_successfully(run.get("status")):
        return
    step_names = {step.get("step") for step in run["steps"]}
    missing = sorted(REQUIRED_CANDIDATE_STEPS - step_names)
    if missing:
        add_finding(
            findings,
            "warning",
            "candidate_retrain_steps_missing",
            "Successful candidate retrain is missing expected audit steps.",
            run_id=run["run_id"],
            missing_steps=missing,
        )


def _add_model_findings(
    findings: list[Dict[str, Any]],
    run: Dict[str, Any],
    *,
    registry: ModelRegistry,
    strict: bool,
) -> None:
    model_id = run.get("model_id")
    if not model_id or not _is_completed_successfully(run.get("status")):
        return

    models = registry.registry_data.get("models", {})
    model = models.get(model_id)
    if model is None:
        add_finding(
            findings,
            "error",
            "run_model_missing_registry",
            "Successful run references a model missing from the registry.",
            run_id=run["run_id"],
            model_id=model_id,
        )
        return

    manifest_path = model.get("manifest_path")
    if not manifest_path or not os.path.exists(manifest_path):
        add_finding(
            findings,
            "error" if strict else "warning",
            "run_model_manifest_missing",
            "Successful run model lacks an immutable manifest.",
            run_id=run["run_id"],
            model_id=model_id,
        )
    if not _artifact_exists(model):
        add_finding(
            findings,
            "error" if strict else "warning",
            "run_model_artifact_missing",
            "Successful run model lacks a saved artifact.",
            run_id=run["run_id"],
            model_id=model_id,
        )
    metadata = model.get("metadata", {})
    if not metadata.get("data_snapshot_id"):
        add_finding(
            findings,
            "warning",
            "run_model_snapshot_missing",
            "Successful run model lacks a data snapshot id.",
            run_id=run["run_id"],
            model_id=model_id,
        )


def generate_experiment_ledger_audit(
    config: Dict[str, Any],
    *,
    tracker: Optional[ExperimentTracker] = None,
    registry: Optional[ModelRegistry] = None,
    ledger_path: Optional[str] = None,
    now: Optional[datetime] = None,
    max_running_age_minutes: float = 240.0,
    strict: bool = False,
    run_limit: int = 50,
) -> Dict[str, Any]:
    """Validate the append-only experiment ledger and model handoff evidence."""
    tracker = tracker or ExperimentTracker.from_config(config)
    ledger_path = ledger_path or str(tracker.path)
    registry = registry or ModelRegistry(
        registry_dir=config.get("mlops", {}).get("registry_dir", "data_lake/models")
    )

    findings: list[Dict[str, Any]] = []
    source = Path(ledger_path)
    rows = read_jsonl(source)
    invalid_rows = [row for row in rows if row.get("_invalid_json")]
    events = [row for row in rows if not row.get("_invalid_json")]
    runs = _group_runs(events)
    ordered = _ordered_runs(runs)
    run_ids = {run["run_id"] for run in ordered}

    if not source.exists():
        add_finding(
            findings,
            "error" if strict else "warning",
            "ledger_missing",
            f"No experiment ledger exists at {ledger_path}.",
        )
    if invalid_rows:
        add_finding(
            findings,
            "error" if strict else "warning",
            "ledger_invalid_rows",
            "Experiment ledger contains malformed JSON rows.",
            count=len(invalid_rows),
        )

    unknown_events = [
        row.get("event")
        for row in events
        if row.get("event") not in {"run_started", "step", "run_completed"}
    ]
    if unknown_events:
        add_finding(
            findings,
            "warning",
            "ledger_unknown_events",
            "Experiment ledger contains unknown event types.",
            event_counts=dict(Counter(unknown_events)),
        )

    orphan_events = [row for row in events if row.get("run_id") not in run_ids]
    if orphan_events:
        add_finding(
            findings,
            "warning",
            "ledger_events_without_run_id",
            "Experiment ledger contains events without a run_id.",
            count=len(orphan_events),
        )

    if source.exists() and not ordered:
        add_finding(
            findings,
            "warning",
            "experiment_runs_missing",
            "Experiment ledger exists but contains no reconstructable runs.",
        )

    for run in ordered:
        _add_lifecycle_findings(
            findings,
            run,
            now=now,
            strict=strict,
            max_running_age_minutes=max_running_age_minutes,
        )
        _add_step_findings(findings, run, strict=strict)
        _add_model_findings(findings, run, registry=registry, strict=strict)

    status_counts = Counter(str(run.get("status") or "UNKNOWN") for run in ordered)
    latest = ordered[0] if ordered else {}
    summary = {
        "ledger_path": ledger_path,
        "registry_dir": registry.registry_dir,
        "events": len(events),
        "invalid_rows": len(invalid_rows),
        "runs": len(ordered),
        "status_counts": dict(status_counts),
        "latest_run_id": latest.get("run_id"),
        "latest_run_status": latest.get("status"),
        "latest_run_type": latest.get("run_type"),
    }
    return {
        "title": "APEX Experiment Ledger Auditor",
        "generated_at": (now or utc_now()).isoformat(),
        "status": status_from_findings(findings),
        "summary": summary,
        "recent_runs": [_public_run(run) for run in ordered[: max(1, run_limit)]],
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX experiment ledger auditor")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ledger-path")
    parser.add_argument("--run-limit", type=int, default=50)
    parser.add_argument("--max-running-age-minutes", type=float, default=240.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    report = generate_experiment_ledger_audit(
        config,
        ledger_path=args.ledger_path,
        run_limit=args.run_limit,
        max_running_age_minutes=args.max_running_age_minutes,
        strict=args.strict,
    )
    rendered = write_report(report, output=args.output, fmt=args.format)
    if not args.output:
        print(rendered)
    if should_exit_nonzero(report, fail_on_warning=args.fail_on_warning):
        sys.exit(1)


if __name__ == "__main__":
    main()
