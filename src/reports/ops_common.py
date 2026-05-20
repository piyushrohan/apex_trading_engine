"""Shared helpers for operational watchdog reports."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def age_minutes(value: Any, *, now: Optional[datetime] = None) -> Optional[float]:
    parsed = parse_timestamp(value)
    if parsed is None:
        return None
    reference = now or utc_now()
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return max((reference - parsed).total_seconds() / 60.0, 0.0)


def read_json_file(path: str | Path) -> Optional[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return None
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_invalid_json": line})
    return rows


def add_finding(
    findings: List[Dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    **context: Any,
) -> None:
    finding = {"severity": severity, "code": code, "message": message}
    if context:
        finding["context"] = context
    findings.append(finding)


def status_from_findings(findings: Iterable[Dict[str, Any]]) -> str:
    severities = {row.get("severity") for row in findings}
    if "error" in severities:
        return "fail"
    if "warning" in severities:
        return "warn"
    return "pass"


def should_exit_nonzero(
    payload: Dict[str, Any], *, fail_on_warning: bool = False
) -> bool:
    if payload.get("status") == "fail":
        return True
    return bool(fail_on_warning and payload.get("status") == "warn")


def write_report(
    payload: Dict[str, Any],
    *,
    output: Optional[str] = None,
    fmt: str = "json",
) -> str:
    if fmt == "markdown":
        rendered = render_markdown(payload)
    else:
        rendered = json.dumps(payload, indent=2, sort_keys=True, default=str)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(rendered + "\n", encoding="utf-8")
    return rendered


def render_markdown(payload: Dict[str, Any]) -> str:
    title = payload.get("title", "APEX Operational Report")
    lines = [f"# {title}", ""]
    lines.append(f"- Status: `{payload.get('status', 'unknown')}`")
    if payload.get("generated_at"):
        lines.append(f"- Generated at: `{payload['generated_at']}`")
    lines.append("")

    summary = payload.get("summary") or {}
    if summary:
        lines.extend(["## Summary", ""])
        for key, value in summary.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    findings = payload.get("findings") or []
    lines.extend(["## Findings", ""])
    if not findings:
        lines.append("No findings.")
    else:
        lines.append("| Severity | Code | Message |")
        lines.append("| --- | --- | --- |")
        for finding in findings:
            lines.append(
                "| `{}` | `{}` | {} |".format(
                    finding.get("severity"),
                    finding.get("code"),
                    str(finding.get("message", "")).replace("|", "\\|"),
                )
            )
    return "\n".join(lines)
