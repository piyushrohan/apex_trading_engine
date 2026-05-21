"""Allow-listed local process controls for the operator API."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ManagedProcessSpec:
    """Static command specification for a process the cockpit may supervise."""

    name: str
    module: str
    description: str
    log_name: str
    label: str | None = None
    args: tuple[str, ...] = ()
    env: Dict[str, str] = field(default_factory=dict)
    category: str = "runtime"
    danger_level: str = "normal"
    long_running: bool = True
    requires_confirmation: bool = True
    operator_note: str = ""


class LocalProcessManager:
    """Start and stop a small allow-list of local APEX subprocesses."""

    SPECS: Dict[str, ManagedProcessSpec] = {
        "paper": ManagedProcessSpec(
            name="paper",
            module="src.pipelines.paper_trade",
            label="Paper trading",
            description="Primary paper trading loop with live market data.",
            log_name="paper_trade.log",
            env={"APEX_EXECUTION_MODE": "paper"},
            category="trading",
            danger_level="safe",
        ),
        "live": ManagedProcessSpec(
            name="live",
            module="src.pipelines.live_trade",
            label="Live trading",
            description=(
                "Real signed Binance futures trading loop. The live pipeline still "
                "enforces config, paper-gate, risk, and production-model guards."
            ),
            log_name="live_trade.log",
            env={"APEX_EXECUTION_MODE": "live"},
            category="trading",
            danger_level="critical",
            operator_note=(
                "Requires live config, credentials, paper evidence, and PROD "
                "model readiness."
            ),
        ),
        "shadow": ManagedProcessSpec(
            name="shadow",
            module="src.pipelines.shadow_trade",
            label="Shadow lane",
            description=(
                "Standalone virtual shadow lane loop for candidate-model "
                "observation."
            ),
            log_name="shadow_trade.log",
            env={"APEX_EXECUTION_MODE": "shadow"},
            category="trading",
            danger_level="safe",
        ),
        "training": ManagedProcessSpec(
            name="training",
            module="src.mlops.auto_retrain",
            label="Train / retrain",
            description="Governed model retraining and shadow registration.",
            log_name="auto_retrain.log",
            category="mlops",
            danger_level="moderate",
            long_running=False,
        ),
        "model_governance": ManagedProcessSpec(
            name="model_governance",
            module="src.reports.model_governance_report",
            label="Evaluate model governance",
            description=(
                "Model registry, promotion, and live-readiness evaluation " "report."
            ),
            log_name="model_governance_report.log",
            args=("--format", "markdown"),
            category="evaluation",
            danger_level="safe",
            long_running=False,
        ),
        "paper_health": ManagedProcessSpec(
            name="paper_health",
            module="src.reports.paper_health_watchdog",
            label="Evaluate paper health",
            description=(
                "Paper trading health, decision freshness, fill evidence, and "
                "runtime watchdog."
            ),
            log_name="paper_health_watchdog.log",
            args=("--format", "markdown"),
            category="evaluation",
            danger_level="safe",
            long_running=False,
        ),
        "shadow_sanity": ManagedProcessSpec(
            name="shadow_sanity",
            module="src.reports.shadow_sanity_monitor",
            label="Evaluate shadow sanity",
            description=(
                "Shadow candidate evidence, book tagging, and hedge attribution "
                "sanity check."
            ),
            log_name="shadow_sanity_monitor.log",
            args=("--format", "markdown"),
            category="evaluation",
            danger_level="safe",
            long_running=False,
        ),
        "data_freshness": ManagedProcessSpec(
            name="data_freshness",
            module="src.reports.data_freshness_check",
            label="Check data freshness",
            description=(
                "DuckDB integrity, freshness, duplicate-key, and OHLCV gap " "check."
            ),
            log_name="data_freshness_check.log",
            args=("--format", "markdown"),
            category="data",
            danger_level="safe",
            long_running=False,
        ),
        "ledger_audit": ManagedProcessSpec(
            name="ledger_audit",
            module="src.reports.experiment_ledger_auditor",
            label="Audit experiment ledger",
            description="Append-only experiment ledger and registry consistency audit.",
            log_name="experiment_ledger_auditor.log",
            args=("--format", "markdown"),
            category="mlops",
            danger_level="safe",
            long_running=False,
        ),
        "frontend_contract": ManagedProcessSpec(
            name="frontend_contract",
            module="src.reports.frontend_api_contract_smoke",
            label="Smoke-test frontend/API contract",
            description=(
                "Static plus live API contract smoke test for the browser " "cockpit."
            ),
            log_name="frontend_api_contract_smoke.log",
            args=("--live-api", "--format", "markdown"),
            category="validation",
            danger_level="safe",
            long_running=False,
        ),
        "paper_report": ManagedProcessSpec(
            name="paper_report",
            module="src.reports.paper_report",
            label="Generate paper report",
            description=(
                "Paper equity, decision, Sharpe, drawdown, and fill-rate " "report."
            ),
            log_name="paper_report.log",
            category="evaluation",
            danger_level="safe",
            long_running=False,
        ),
        "hedge_report": ManagedProcessSpec(
            name="hedge_report",
            module="src.reports.hedge_report",
            label="Generate hedge report",
            description="Hedge strategy selection and score attribution report.",
            log_name="hedge_report.log",
            category="evaluation",
            danger_level="safe",
            long_running=False,
        ),
    }

    def __init__(
        self,
        *,
        cwd: str | Path = ".",
        log_dir: str | Path = "logs",
        python_executable: str = sys.executable,
    ) -> None:
        self.cwd = Path(cwd)
        self.log_dir = Path(log_dir)
        self.python_executable = python_executable
        self._processes: Dict[str, subprocess.Popen] = {}
        self._started_at: Dict[str, str] = {}

    def list_processes(self) -> Dict[str, Any]:
        """Return status for every allow-listed process."""
        return {name: self.status(name) for name in sorted(self.SPECS)}

    def capabilities(self) -> list[Dict[str, Any]]:
        """Return the static supervised capability catalog."""
        return [self._spec_payload(spec) for spec in self.SPECS.values()]

    def status(self, name: str) -> Dict[str, Any]:
        """Return status for one process, including exited process codes."""
        spec = self._spec(name)
        process = self._processes.get(name)
        running = bool(process and process.poll() is None)
        log_path = self.log_dir / spec.log_name
        return {
            **self._spec_payload(spec),
            "name": name,
            "running": running,
            "pid": process.pid if process and running else None,
            "returncode": None if running or process is None else process.returncode,
            "started_at": self._started_at.get(name),
            "log_path": str(log_path),
            "last_log_lines": self._tail(log_path, limit=12),
        }

    def start(self, name: str, *, dry_run: bool = False) -> Dict[str, Any]:
        """Start an allow-listed process unless it is already running."""
        spec = self._spec(name)
        command = self.command_for(spec)
        if dry_run:
            payload = self.status(name)
            payload.update({"would_run": command, "dry_run": True})
            return payload

        existing = self._processes.get(name)
        if existing and existing.poll() is None:
            payload = self.status(name)
            payload["already_running"] = True
            return payload

        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / spec.log_name
        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update(spec.env)
        process = subprocess.Popen(
            command,
            cwd=str(self.cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        self._processes[name] = process
        self._started_at[name] = datetime.now(timezone.utc).isoformat()
        payload = self.status(name)
        payload["started"] = True
        payload["command"] = command
        return payload

    def restart(self, name: str, *, dry_run: bool = False) -> Dict[str, Any]:
        """Restart a supervised process, preserving the same allow-listed command."""
        if dry_run:
            payload = self.status(name)
            payload["dry_run"] = True
            payload["would_restart"] = True
            return payload
        stop_result = self.stop(name)
        start_result = self.start(name)
        start_result["restarted"] = True
        start_result["stop_result"] = stop_result
        return start_result

    def stop(self, name: str, *, dry_run: bool = False) -> Dict[str, Any]:
        """Terminate a managed process if it is running."""
        self._spec(name)
        process = self._processes.get(name)
        if dry_run:
            payload = self.status(name)
            payload["dry_run"] = True
            return payload
        if not process or process.poll() is not None:
            payload = self.status(name)
            payload["already_stopped"] = True
            return payload

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        payload = self.status(name)
        payload["stopped"] = True
        return payload

    def _spec(self, name: str) -> ManagedProcessSpec:
        if name not in self.SPECS:
            raise KeyError(name)
        return self.SPECS[name]

    def command_for(self, spec_or_name: ManagedProcessSpec | str) -> list[str]:
        """Build the exact allow-listed command for a spec or process name."""
        spec = (
            self._spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
        )
        return [self.python_executable, "-m", spec.module, *spec.args]

    def _spec_payload(self, spec: ManagedProcessSpec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "label": spec.label or spec.name.replace("_", " ").title(),
            "description": spec.description,
            "module": spec.module,
            "args": list(spec.args),
            "command": self.command_for(spec),
            "category": spec.category,
            "danger_level": spec.danger_level,
            "long_running": spec.long_running,
            "requires_confirmation": spec.requires_confirmation,
            "operator_note": spec.operator_note,
        }

    @staticmethod
    def _tail(path: Path, *, limit: int) -> list[str]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-max(1, limit) :]


PROCESS_MANAGER = LocalProcessManager()
