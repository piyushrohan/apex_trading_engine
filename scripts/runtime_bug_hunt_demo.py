#!/usr/bin/env python3
"""Run a safe end-to-end APEX runtime demo focused on finding bugs.

The default path starts a temporary API and frontend on non-standard local ports,
checks critical HTTP endpoints, verifies control endpoints with dry-run intent,
opens the status websocket, and runs the frontend/API contract smoke test.

Network-dependent checks are opt-in:
- --live-market checks the Binance-backed /ws/market stream.
- --start-paper starts the paper trading subprocess briefly through the API
  process manager, then stops it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import websockets

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "PASS"


class RuntimeDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.api_base = f"http://127.0.0.1:{args.api_port}"
        self.frontend_base = f"http://127.0.0.1:{args.frontend_port}"
        self.processes: list[subprocess.Popen] = []
        self.results: list[CheckResult] = []
        self.tempdir = tempfile.TemporaryDirectory(prefix="apex-runtime-demo-")
        self.temp_path = Path(self.tempdir.name)

    def run(self) -> int:
        try:
            self._start_api()
            self._start_frontend()
            self._http_checks()
            self._frontend_checks()
            self._control_checks()
            self._websocket_checks()
            self._contract_smoke()
            if self.args.start_paper:
                self._paper_process_check()
            if self.args.live_market:
                self._market_websocket_check()
        except Exception as exc:  # noqa: BLE001 - keep demo diagnostics visible
            self._record("runtime harness", "FAIL", str(exc))
        finally:
            self._stop_processes()
        self._print_summary()
        if all(result.ok for result in self.results):
            self.tempdir.cleanup()
        return 0 if all(result.ok for result in self.results) else 1

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "APEX_API_HOST": "127.0.0.1",
                "APEX_API_PORT": str(self.args.api_port),
                "APEX_CONTROL_STATE_PATH": str(self.temp_path / "controls.json"),
                "APEX_AUDIT_PATH": str(self.temp_path / "audit.jsonl"),
                "APEX_RUNTIME_STATUS_PATH": str(self.temp_path / "runtime.json"),
            }
        )
        return env

    def _start_api(self) -> None:
        log_path = self.temp_path / "api.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-m", "src.api.server"],
            cwd=ROOT,
            env=self._env(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        self.processes.append(process)
        self._wait_for_http("/health", "api health", timeout=self.args.timeout)

    def _start_frontend(self) -> None:
        log_path = self.temp_path / "frontend.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                str(self.args.frontend_port),
                "--directory",
                "frontend",
            ],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        self.processes.append(process)
        self._wait_for_url(
            self.frontend_base, "frontend root", timeout=self.args.timeout
        )

    def _wait_for_http(self, path: str, name: str, timeout: float) -> None:
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            try:
                payload = self._get(path)
                self._record(name, "PASS", f"{path}: keys={sorted(payload)[:8]}")
                return
            except Exception as exc:  # noqa: BLE001 - diagnostic capture
                last_error = str(exc)
                time.sleep(0.25)
        self._record(name, "FAIL", last_error or "timed out")
        raise RuntimeError(f"{name} failed: {last_error}")

    def _wait_for_url(self, url: str, name: str, timeout: float) -> None:
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            try:
                with urlopen(url, timeout=1.5) as response:
                    body = response.read().decode("utf-8", errors="replace")
                if response.status == 200 and "root" in body:
                    self._record(name, "PASS", url)
                    return
                last_error = f"unexpected status/body: {response.status}"
            except Exception as exc:  # noqa: BLE001 - diagnostic capture
                last_error = str(exc)
                time.sleep(0.25)
        self._record(name, "FAIL", last_error or "timed out")
        raise RuntimeError(f"{name} failed: {last_error}")

    def _http_checks(self) -> None:
        endpoints = [
            "/status",
            "/portfolio",
            "/ops/workflow",
            "/ops/processes",
            "/ops/readiness",
            "/models",
            "/models/lifecycle?limit=2",
            "/models/promotion/status",
            "/models/drift",
            "/orders/lifecycle?limit=2",
            "/control/state",
        ]
        for endpoint in endpoints:
            try:
                payload = self._get(endpoint)
                keys = (
                    sorted(payload)[:8]
                    if isinstance(payload, dict)
                    else type(payload).__name__
                )
                self._record(
                    f"GET {endpoint}",
                    "PASS",
                    f"keys={keys}",
                )
            except Exception as exc:  # noqa: BLE001 - demo diagnostics
                self._record(f"GET {endpoint}", "FAIL", str(exc))

    def _frontend_checks(self) -> None:
        try:
            app = self._fetch_text(f"{self.frontend_base}/app.js?v=operator-runbook")
            missing = [
                token
                for token in (
                    "Runbook",
                    "Live Price Tape",
                    "/ws/market",
                    "/ops/workflow",
                )
                if token not in app
            ]
            if missing:
                self._record("frontend app contract", "FAIL", f"missing={missing}")
            else:
                self._record(
                    "frontend app contract", "PASS", "runbook + live chart found"
                )
        except Exception as exc:  # noqa: BLE001
            self._record("frontend app contract", "FAIL", str(exc))

    def _control_checks(self) -> None:
        checks = [
            (
                "/control/pause",
                {"confirm": True, "reason": "runtime demo pause"},
                "control pause",
            ),
            (
                "/control/resume",
                {"confirm": True, "reason": "runtime demo resume"},
                "control resume",
            ),
            (
                "/ops/processes/paper",
                {
                    "confirm": True,
                    "action": "start",
                    "dry_run": True,
                    "reason": "runtime demo dry run",
                },
                "process dry-run start paper",
            ),
        ]
        for path, body, label in checks:
            try:
                payload = self._post(path, body)
                self._record(label, "PASS", f"accepted={payload.get('accepted')}")
            except Exception as exc:  # noqa: BLE001
                self._record(label, "FAIL", str(exc))

    def _websocket_checks(self) -> None:
        async def check_status_ws() -> str:
            url = f"ws://127.0.0.1:{self.args.api_port}/ws/status"
            async with websockets.connect(url, open_timeout=3) as websocket:
                message = json.loads(
                    await asyncio.wait_for(websocket.recv(), timeout=3)
                )
                mode = message.get("operator_mode")
                symbol = message.get("symbol")
                return f"mode={mode} symbol={symbol}"

        try:
            detail = asyncio.run(check_status_ws())
            self._record("WS /ws/status", "PASS", detail)
        except Exception as exc:  # noqa: BLE001
            self._record("WS /ws/status", "FAIL", str(exc))

    def _market_websocket_check(self) -> None:
        async def check_market_ws() -> str:
            url = f"ws://127.0.0.1:{self.args.api_port}/ws/market" "?symbol=ETHUSDC"
            async with websockets.connect(url, open_timeout=5) as websocket:
                message = json.loads(
                    await asyncio.wait_for(websocket.recv(), timeout=self.args.timeout)
                )
                if message.get("type") == "error":
                    raise RuntimeError(message.get("message", "market ws error"))
                return f"type={message.get('type')} source={message.get('source')}"

        try:
            detail = asyncio.run(check_market_ws())
            self._record("WS /ws/market live", "PASS", detail)
        except Exception as exc:  # noqa: BLE001
            self._record("WS /ws/market live", "FAIL", str(exc))

    def _contract_smoke(self) -> None:
        command = [
            sys.executable,
            "-m",
            "src.reports.frontend_api_contract_smoke",
            "--api-base-url",
            self.api_base,
            "--live-api",
            "--format",
            "json",
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=self.args.timeout,
        )
        if result.returncode == 0:
            self._record("frontend/API live contract smoke", "PASS", "exit=0")
        else:
            detail = (result.stdout + result.stderr).strip()[-500:]
            self._record("frontend/API live contract smoke", "FAIL", detail)

    def _paper_process_check(self) -> None:
        try:
            started = self._post(
                "/ops/processes/paper",
                {
                    "confirm": True,
                    "action": "start",
                    "reason": "runtime demo bounded paper start",
                },
            )
            self._record(
                "process start paper",
                "PASS",
                f"pid={started.get('state_after', {}).get('pid')}",
            )
            time.sleep(self.args.paper_seconds)
            status = self._get("/ops/processes")["processes"]["paper"]
            if status.get("running"):
                self._record("paper runtime alive", "PASS", f"pid={status.get('pid')}")
            else:
                returncode = status.get("returncode")
                outcome = "PASS" if returncode in (0, None) else "FAIL"
                self._record(
                    "paper runtime alive",
                    outcome,
                    f"exited early returncode={returncode}",
                )
        except Exception as exc:  # noqa: BLE001
            self._record("process start paper", "FAIL", str(exc))
        finally:
            try:
                stopped = self._post(
                    "/ops/processes/paper",
                    {
                        "confirm": True,
                        "action": "stop",
                        "reason": "runtime demo bounded paper stop",
                    },
                )
                state_after = stopped.get("state_after", {})
                returncode = state_after.get("returncode")
                already_stopped = state_after.get("already_stopped", False)
                ok = (not already_stopped) or returncode in (0, None)
                self._record(
                    "process stop paper",
                    "PASS" if ok else "FAIL",
                    f"returncode={returncode} already_stopped={already_stopped}",
                )
            except Exception as exc:  # noqa: BLE001
                self._record("process stop paper", "FAIL", str(exc))

    def _get(self, path: str) -> Any:
        return json.loads(self._fetch_text(f"{self.api_base}{path}"))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        request = Request(
            f"{self.api_base}{path}",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            payload = response.read().decode("utf-8", errors="replace")
        return json.loads(payload or "{}")

    @staticmethod
    def _fetch_text(url: str) -> str:
        with urlopen(url, timeout=5) as response:
            if response.status >= 400:
                raise URLError(f"{url} returned {response.status}")
            return response.read().decode("utf-8", errors="replace")

    def _record(self, name: str, status: str, detail: str) -> None:
        self.results.append(CheckResult(name=name, status=status, detail=detail))

    def _stop_processes(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in reversed(self.processes):
            if process.poll() is None:
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)

    def _print_summary(self) -> None:
        print("\nAPEX runtime bug-hunt demo")
        print("=" * 28)
        for result in self.results:
            print(f"{result.status:4}  {result.name}: {result.detail}")
        failed = [result for result in self.results if not result.ok]
        print("-" * 28)
        if failed:
            print(f"FAILED checks: {len(failed)}")
            print(f"Temporary logs were under: {self.temp_path}")
        else:
            print("All runtime demo checks passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="APEX runtime bug-hunt demo")
    parser.add_argument("--api-port", type=int, default=18080)
    parser.add_argument("--frontend-port", type=int, default=15173)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--live-market",
        action="store_true",
        help="Also test the Binance-backed /ws/market stream.",
    )
    parser.add_argument(
        "--start-paper",
        action="store_true",
        help="Start paper trading briefly through the process manager.",
    )
    parser.add_argument("--paper-seconds", type=float, default=12.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return RuntimeDemo(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
