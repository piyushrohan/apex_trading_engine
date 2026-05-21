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
    env: Dict[str, str] = field(default_factory=dict)


class LocalProcessManager:
    """Start and stop a small allow-list of local APEX subprocesses."""

    SPECS: Dict[str, ManagedProcessSpec] = {
        "paper": ManagedProcessSpec(
            name="paper",
            module="src.pipelines.paper_trade",
            description="Primary paper trading loop with live market data.",
            log_name="paper_trade.log",
            env={"APEX_EXECUTION_MODE": "paper"},
        ),
        "training": ManagedProcessSpec(
            name="training",
            module="src.mlops.auto_retrain",
            description="Governed model retraining and shadow registration.",
            log_name="auto_retrain.log",
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

    def status(self, name: str) -> Dict[str, Any]:
        """Return status for one process, including exited process codes."""
        spec = self._spec(name)
        process = self._processes.get(name)
        running = bool(process and process.poll() is None)
        return {
            "name": name,
            "description": spec.description,
            "module": spec.module,
            "running": running,
            "pid": process.pid if process and running else None,
            "returncode": None if running or process is None else process.returncode,
            "started_at": self._started_at.get(name),
            "log_path": str(self.log_dir / spec.log_name),
        }

    def start(self, name: str, *, dry_run: bool = False) -> Dict[str, Any]:
        """Start an allow-listed process unless it is already running."""
        spec = self._spec(name)
        command = [self.python_executable, "-m", spec.module]
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


PROCESS_MANAGER = LocalProcessManager()
