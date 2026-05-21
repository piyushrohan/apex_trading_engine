import subprocess

import pytest

from src.ops import cockpit
from src.ops.cockpit import LaunchedProcess, build_parser, launch_from_args, main
from src.ops.process_manager import LocalProcessManager


@pytest.mark.unit
def test_process_manager_dry_run_and_unknown_process(tmp_path):
    manager = LocalProcessManager(cwd=tmp_path, log_dir=tmp_path / "logs")

    dry = manager.start("paper", dry_run=True)

    assert dry["dry_run"] is True
    assert dry["would_run"][-1] == "src.pipelines.paper_trade"
    with pytest.raises(KeyError):
        manager.status("unknown")


@pytest.mark.unit
def test_process_manager_start_stop_with_fake_popen(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 1234
        returncode = None

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            self._running = True

        def poll(self):
            return None if self._running else 0

        def wait(self, timeout=None):
            self._running = False
            self.returncode = 0
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    monkeypatch.setattr("src.ops.process_manager.os.killpg", lambda *a, **k: None)

    manager = LocalProcessManager(cwd=tmp_path, log_dir=tmp_path / "logs")
    started = manager.start("training")
    stopped = manager.stop("training")

    assert started["running"] is True
    assert started["pid"] == 1234
    assert stopped["running"] is False
    assert calls[0][0][0][-1] == "src.mlops.auto_retrain"


@pytest.mark.unit
def test_process_manager_already_running_dry_stop_and_already_stopped(
    tmp_path, monkeypatch
):
    class FakeProcess:
        pid = 2345
        returncode = 0

        def __init__(self):
            self.running = True

        def poll(self):
            return None if self.running else self.returncode

        def wait(self, timeout=None):
            self.running = False
            return self.returncode

    fake = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr("src.ops.process_manager.os.killpg", lambda *a, **k: None)

    manager = LocalProcessManager(cwd=tmp_path, log_dir=tmp_path / "logs")
    manager.start("paper")

    assert manager.start("paper")["already_running"] is True
    assert manager.stop("paper", dry_run=True)["dry_run"] is True
    stopped = manager.stop("paper")
    assert stopped["stopped"] is True
    assert manager.stop("paper")["already_stopped"] is True
    assert set(manager.list_processes()) == {"paper", "training"}


@pytest.mark.unit
def test_process_manager_stop_handles_missing_and_stubborn_process(
    tmp_path, monkeypatch
):
    kill_calls = []

    class FakeProcess:
        pid = 3456
        returncode = None

        def __init__(self):
            self.waits = 0

        def poll(self):
            return None if self.returncode is None else self.returncode

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(cmd="paper", timeout=timeout)
            self.returncode = -9
            return self.returncode

    fake = FakeProcess()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake)

    def fake_killpg(pid, sig):
        kill_calls.append(sig)
        if len(kill_calls) == 1:
            raise ProcessLookupError

    monkeypatch.setattr("src.ops.process_manager.os.killpg", fake_killpg)

    manager = LocalProcessManager(cwd=tmp_path, log_dir=tmp_path / "logs")
    manager.start("paper")
    stopped = manager.stop("paper")

    assert stopped["stopped"] is True
    assert kill_calls == [15, 9]


@pytest.mark.unit
def test_cockpit_launch_from_args_builds_all_processes(tmp_path, monkeypatch):
    launched = []

    def fake_launch(**kwargs):
        launched.append(kwargs)
        return SimpleLaunched(kwargs["name"])

    class SimpleLaunched:
        def __init__(self, name):
            self.name = name
            self.process = type("P", (), {"pid": 1, "poll": lambda self: None})()
            self.log_path = tmp_path / f"{name}.log"

    monkeypatch.setattr(cockpit, "_launch", fake_launch)
    args = build_parser().parse_args(
        ["--paper", "--train", "--log-dir", str(tmp_path / "logs")]
    )

    processes = launch_from_args(args)

    assert [item.name for item in processes] == [
        "api",
        "frontend",
        "paper",
        "training",
    ]
    assert launched[0]["env"]["APEX_API_PORT"] == "8080"
    assert launched[2]["env"]["APEX_EXECUTION_MODE"] == "paper"


@pytest.mark.unit
def test_cockpit_launch_opens_log_and_merges_env(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 909

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def poll(self):
            return None

    monkeypatch.setattr(cockpit.subprocess, "Popen", FakeProcess)
    launched = cockpit._launch(
        name="api",
        args=["python", "-m", "src.api.server"],
        log_dir=tmp_path / "logs",
        log_name="api.log",
        env={"APEX_API_PORT": "9999"},
    )

    assert launched.name == "api"
    assert launched.process.pid == 909
    assert launched.log_path.exists()
    assert calls[0][1]["env"]["APEX_API_PORT"] == "9999"


@pytest.mark.unit
def test_cockpit_stop_all_ignores_missing_process(monkeypatch):
    class Process:
        pid = 888
        returncode = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        cockpit.os,
        "killpg",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError),
    )

    cockpit._stop_all([LaunchedProcess("api", Process(), cockpit.Path("logs/api.log"))])


@pytest.mark.unit
def test_cockpit_main_supervisor_failure_and_keyboard_stop(monkeypatch, capsys):
    class FakeProcess:
        pid = 11
        returncode = 2

        def poll(self):
            return 2

    failed = LaunchedProcess("api", FakeProcess(), cockpit.Path("logs/api.log"))
    monkeypatch.setattr(cockpit, "launch_from_args", lambda args: [failed])

    assert main([]) == 1
    assert "api exited with 2" in capsys.readouterr().out

    class RunningProcess:
        pid = 12
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    stopped = {"called": False}
    running = LaunchedProcess("api", RunningProcess(), cockpit.Path("logs/api.log"))
    monkeypatch.setattr(cockpit, "launch_from_args", lambda args: [running])
    monkeypatch.setattr(
        cockpit.time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt)
    )
    monkeypatch.setattr(
        cockpit, "_stop_all", lambda processes: stopped.update(called=True)
    )

    assert main([]) == 0
    assert stopped["called"] is True


@pytest.mark.unit
def test_cockpit_stop_all_terminates_and_kills_stubborn_process(monkeypatch):
    kill_calls = []

    class Process:
        pid = 777
        returncode = None

        def __init__(self):
            self.waits = 0

        def poll(self):
            return None if self.returncode is None else self.returncode

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired(cmd="api", timeout=timeout)
            self.returncode = -9
            return self.returncode

    process = Process()
    monkeypatch.setattr(cockpit.os, "killpg", lambda pid, sig: kill_calls.append(sig))

    cockpit._stop_all([LaunchedProcess("api", process, cockpit.Path("logs/api.log"))])

    assert kill_calls == [15, 9]


@pytest.mark.unit
def test_cockpit_once_prints_plan(capsys):
    code = main(["--once", "--paper", "--train"])

    output = capsys.readouterr().out
    assert code == 0
    assert "API: http://127.0.0.1:8080" in output
    assert "Frontend: http://127.0.0.1:5173/" in output
    assert "Paper: on" in output
    assert "Training: on" in output


@pytest.mark.unit
def test_cockpit_parser_defaults():
    args = build_parser().parse_args([])

    assert args.api_port == 8080
    assert args.frontend_port == 5173
    assert args.paper is False
