import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ExperimentTracker:
    """Append-only model experiment ledger for reproducibility and audit review."""

    def __init__(self, path: str = "data_lake/mlops/experiments.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ExperimentTracker":
        mlops_cfg = config.get("mlops", {})
        return cls(
            mlops_cfg.get("experiment_log_path", "data_lake/mlops/experiments.jsonl")
        )

    def start_run(
        self,
        run_type: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        event = {
            "event": "run_started",
            "run_id": run_id,
            "run_type": run_type,
            "status": "RUNNING",
            "timestamp": utc_now(),
            "metadata": metadata or {},
        }
        self._append(event)
        return event

    def log_step(
        self,
        run_id: str,
        step: str,
        status: str,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            "event": "step",
            "run_id": run_id,
            "step": step,
            "status": status,
            "timestamp": utc_now(),
            "metrics": metrics or {},
            "metadata": metadata or {},
        }
        self._append(event)
        return event

    def complete_run(
        self,
        run_id: str,
        status: str,
        *,
        model_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = {
            "event": "run_completed",
            "run_id": run_id,
            "status": status,
            "model_id": model_id,
            "timestamp": utc_now(),
            "metrics": metrics or {},
            "metadata": metadata or {},
        }
        self._append(event)
        return event

    def events(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        runs: Dict[str, Dict[str, Any]] = {}
        for event in self.events():
            run_id = event.get("run_id")
            if not run_id:
                continue
            run = runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "run_type": event.get("run_type"),
                    "status": "UNKNOWN",
                    "started_at": event.get("timestamp"),
                    "completed_at": None,
                    "model_id": None,
                    "metadata": {},
                    "metrics": {},
                    "steps": [],
                },
            )
            if event["event"] == "run_started":
                run["run_type"] = event.get("run_type")
                run["status"] = event.get("status", "RUNNING")
                run["started_at"] = event.get("timestamp")
                run["metadata"].update(event.get("metadata", {}))
            elif event["event"] == "step":
                run["steps"].append(
                    {
                        "step": event.get("step"),
                        "status": event.get("status"),
                        "timestamp": event.get("timestamp"),
                        "metrics": event.get("metrics", {}),
                        "metadata": event.get("metadata", {}),
                    }
                )
            elif event["event"] == "run_completed":
                run["status"] = event.get("status", run["status"])
                run["completed_at"] = event.get("timestamp")
                run["model_id"] = event.get("model_id")
                run["metrics"].update(event.get("metrics", {}))
                run["metadata"].update(event.get("metadata", {}))

        ordered = sorted(
            runs.values(),
            key=lambda row: row.get("completed_at") or row.get("started_at") or "",
            reverse=True,
        )
        return ordered[: max(1, limit)]

    def _append(self, event: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, default=str))
            f.write(os.linesep)
