"""One-command local launcher for the APEX operator cockpit."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class LaunchedProcess:
    """Bookkeeping for a child service launched by the cockpit supervisor."""

    name: str
    process: subprocess.Popen
    log_path: Path


def _open_log(log_dir: Path, name: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    return (log_dir / name).open("a", encoding="utf-8")


def _launch(
    *,
    name: str,
    args: list[str],
    log_dir: Path,
    log_name: str,
    env: dict[str, str] | None = None,
) -> LaunchedProcess:
    log_path = log_dir / log_name
    log_handle = _open_log(log_dir, log_name)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    process = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=child_env,
        start_new_session=True,
    )
    log_handle.close()
    return LaunchedProcess(name=name, process=process, log_path=log_path)


def _stop_all(processes: Iterable[LaunchedProcess]) -> None:
    for launched in processes:
        if launched.process.poll() is None:
            try:
                os.killpg(launched.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for launched in processes:
        if launched.process.poll() is None:
            try:
                launched.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(launched.process.pid, signal.SIGKILL)
                launched.process.wait(timeout=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the APEX API, frontend, and optional operator jobs."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8080)
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--no-frontend", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--once", action="store_true", help="Print plan and exit.")
    return parser


def launch_from_args(args: argparse.Namespace) -> list[LaunchedProcess]:
    log_dir = Path(args.log_dir)
    python = sys.executable
    processes: list[LaunchedProcess] = []
    api_env = {
        "APEX_API_HOST": args.host,
        "APEX_API_PORT": str(args.api_port),
        "APEX_FRONTEND_PORT": str(args.frontend_port),
    }
    if not args.no_frontend:
        processes.append(
            _launch(
                name="frontend",
                args=[
                    python,
                    "-m",
                    "http.server",
                    str(args.frontend_port),
                    "--directory",
                    "frontend",
                ],
                log_dir=log_dir,
                log_name="cockpit_frontend.log",
                env={"APEX_FRONTEND_PORT": str(args.frontend_port)},
            )
        )
    processes.append(
        _launch(
            name="api",
            args=[python, "-m", "src.api.server"],
            log_dir=log_dir,
            log_name="cockpit_api.log",
            env=api_env,
        )
    )
    if args.paper:
        processes.append(
            _launch(
                name="paper",
                args=[python, "-m", "src.pipelines.paper_trade"],
                log_dir=log_dir,
                log_name="cockpit_paper.log",
                env={"APEX_EXECUTION_MODE": "paper"},
            )
        )
    if args.train:
        processes.append(
            _launch(
                name="training",
                args=[python, "-m", "src.mlops.auto_retrain"],
                log_dir=log_dir,
                log_name="cockpit_training.log",
            )
        )
    if args.live:
        processes.append(
            _launch(
                name="live",
                args=[python, "-m", "src.pipelines.live_trade"],
                log_dir=log_dir,
                log_name="cockpit_live.log",
                env={"APEX_EXECUTION_MODE": "live"},
            )
        )
    return processes


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    url = (
        f"http://{args.host}:{args.frontend_port}/"
        f"?api=http://{args.host}:{args.api_port}"
    )
    if args.once:
        print(f"API: http://{args.host}:{args.api_port}")
        if not args.no_frontend:
            print(f"Frontend: {url}")
        print(f"Paper: {'on' if args.paper else 'off'}")
        print(f"Training: {'on' if args.train else 'off'}")
        print(f"Live: {'on' if args.live else 'off'}")
        return 0

    processes = launch_from_args(args)
    print("APEX cockpit is starting.")
    if not args.no_frontend:
        print("Frontend starts first; use the browser control center after it loads.")
    print(f"API: http://{args.host}:{args.api_port}")
    if not args.no_frontend:
        print(f"Frontend: {url}")
    for launched in processes:
        print(f"{launched.name}: pid={launched.process.pid} log={launched.log_path}")
    print("Press Ctrl+C to stop supervised child processes.")
    try:
        while True:
            failed = [
                launched
                for launched in processes
                if launched.process.poll() not in (None, 0)
            ]
            if failed:
                for launched in failed:
                    print(
                        f"{launched.name} exited with "
                        f"{launched.process.returncode}; see {launched.log_path}"
                    )
                return 1
            time.sleep(1)
    except KeyboardInterrupt:
        _stop_all(processes)
        print("APEX cockpit stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
