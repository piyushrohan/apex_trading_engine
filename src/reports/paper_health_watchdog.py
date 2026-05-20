"""Paper trading health watchdog for runtime and journal freshness."""

import argparse
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.config_loader import load_config
from src.reports.ops_common import (
    add_finding,
    age_minutes,
    parse_timestamp,
    read_json_file,
    read_jsonl,
    should_exit_nonzero,
    status_from_findings,
    utc_now,
    write_report,
)
from src.reports.paper_report import generate_paper_report


def _latest_row(rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    dated = [(parse_timestamp(row.get("timestamp")), row) for row in rows]
    dated = [(ts, row) for ts, row in dated if ts is not None]
    if dated:
        return max(dated, key=lambda item: item[0])[1]
    return rows[-1] if rows else None


def generate_paper_health_report(
    config: Dict[str, Any],
    *,
    runtime_status_path: str = "data_lake/runtime_status.json",
    journal_path: Optional[str] = None,
    now: Optional[datetime] = None,
    max_status_age_minutes: float = 15.0,
    max_decision_age_minutes: float = 30.0,
    strict: bool = False,
) -> Dict[str, Any]:
    """Check whether the paper operator loop is alive, explained, and journaled."""
    generated_at = (now or utc_now()).isoformat()
    findings: list[Dict[str, Any]] = []
    journal_path = journal_path or config.get("explainability", {}).get(
        "journal_path", "data_lake/trade_journal.jsonl"
    )
    runtime = read_json_file(runtime_status_path)
    paper = generate_paper_report(config=config, journal_path=journal_path)
    rows = read_jsonl(journal_path)
    invalid_rows = [row for row in rows if row.get("_invalid_json")]
    paper_rows = [
        row
        for row in rows
        if row.get("execution", {}).get("mode") == "paper"
        and row.get("book", {}).get("role", "primary") == "primary"
    ]
    decisions = [row for row in paper_rows if row.get("decision") in {"LONG", "SHORT"}]
    fills = [row for row in paper_rows if row.get("event") == "paper_fill"]
    latest_decision = _latest_row(decisions)
    latest_fill = _latest_row(fills)

    if runtime is None:
        add_finding(
            findings,
            "error" if strict else "warning",
            "runtime_status_missing",
            f"No runtime status file found at {runtime_status_path}.",
        )
    else:
        if runtime.get("operator_mode") != "paper":
            add_finding(
                findings,
                "warning",
                "operator_not_paper",
                "Runtime status is not currently in paper mode.",
                operator_mode=runtime.get("operator_mode"),
            )
        status_age = age_minutes(runtime.get("updated_at"), now=now)
        if status_age is None:
            add_finding(
                findings,
                "error" if strict else "warning",
                "runtime_timestamp_missing",
                "Runtime status does not include updated_at.",
            )
        elif status_age > max_status_age_minutes:
            add_finding(
                findings,
                "error" if strict else "warning",
                "runtime_status_stale",
                "Runtime status is stale.",
                age_minutes=round(status_age, 2),
                max_age_minutes=max_status_age_minutes,
            )
        if runtime.get("kill_switch_active"):
            add_finding(
                findings,
                "error",
                "kill_switch_active",
                "Paper loop reports an active kill switch.",
            )
        if not runtime.get("last_explanation"):
            add_finding(
                findings,
                "error" if strict else "warning",
                "last_explanation_missing",
                "Runtime status has no latest explanation payload.",
            )
        if not runtime.get("portfolio"):
            add_finding(
                findings,
                "warning",
                "portfolio_missing",
                "Runtime status has no portfolio snapshot.",
            )

    if invalid_rows:
        add_finding(
            findings,
            "warning",
            "journal_invalid_rows",
            "Trade journal contains invalid JSON rows.",
            count=len(invalid_rows),
        )

    if not decisions:
        add_finding(
            findings,
            "error" if strict else "warning",
            "paper_decisions_missing",
            "No primary paper LONG/SHORT decisions found in the journal.",
        )
    else:
        decision_age = age_minutes(latest_decision.get("timestamp"), now=now)
        if decision_age is not None and decision_age > max_decision_age_minutes:
            add_finding(
                findings,
                "error" if strict else "warning",
                "paper_decision_stale",
                "Latest paper decision is stale.",
                age_minutes=round(decision_age, 2),
                max_age_minutes=max_decision_age_minutes,
            )

    if int(paper.get("snapshots", 0)) == 0:
        add_finding(
            findings,
            "error" if strict else "warning",
            "paper_equity_snapshots_missing",
            "No paper equity snapshots found in DuckDB.",
        )

    fill_rate = paper.get("fill_rate")
    if fill_rate == 0 and int(paper.get("directional_decisions", 0)) > 0:
        add_finding(
            findings,
            "warning",
            "paper_fill_rate_zero",
            "Paper decisions exist but no simulated maker fills were recorded.",
        )

    summary = {
        "runtime_status_path": runtime_status_path,
        "journal_path": journal_path,
        "operator_mode": (runtime or {}).get("operator_mode"),
        "runtime_updated_at": (runtime or {}).get("updated_at"),
        "snapshots": paper.get("snapshots", 0),
        "directional_decisions": paper.get("directional_decisions", 0),
        "filled_orders": paper.get("filled_orders", 0),
        "fill_rate": fill_rate,
        "latest_decision_at": (latest_decision or {}).get("timestamp"),
        "latest_fill_at": (latest_fill or {}).get("timestamp"),
        "kill_switch_active": bool((runtime or {}).get("kill_switch_active", False)),
    }
    return {
        "title": "APEX Paper Trading Health Watchdog",
        "generated_at": generated_at,
        "status": status_from_findings(findings),
        "summary": summary,
        "paper_report": paper,
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX paper health watchdog")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--runtime-status-path", default="data_lake/runtime_status.json"
    )
    parser.add_argument("--journal-path")
    parser.add_argument("--max-status-age-minutes", type=float, default=15.0)
    parser.add_argument("--max-decision-age-minutes", type=float, default=30.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    report = generate_paper_health_report(
        config,
        runtime_status_path=args.runtime_status_path,
        journal_path=args.journal_path,
        max_status_age_minutes=args.max_status_age_minutes,
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
