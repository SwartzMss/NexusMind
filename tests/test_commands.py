from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
import stat
import sys
import tempfile
import time

import pytest

import nexusmind.commands as command_module
from nexusmind.commands import CommandConfigError, CommandLimitError, ProcessExecutionBudget, ProcessExecutionLimits, RunCommandTool, command_profile_summary, load_command_config
from nexusmind.runtime.chat import AgentLoopLimits, ChatRuntime
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.policy import ToolPolicyDecision
from nexusmind.tools import (
    ToolCall,
    ToolErrorCode,
    ToolExecutor,
    ToolRegistry,
    ToolResultBudget,
    ToolResultBudgetError,
    ToolRiskLevel,
)
from nexusmind.workspace import Workspace


def _write_config(path: Path, commands: dict) -> None:
    path.write_text(json.dumps({"schema_version": 1, "commands": commands}), encoding="utf-8")


def _registry(tool: RunCommandTool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def test_command_config_rejects_duplicate_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    config_path.write_text(
        (
            '{"schema_version":1,"commands":{"tests":'
            '{"argv":["python"],"argv":["python","other.py"],"cwd":".","timeout_seconds":5}}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(CommandConfigError, match="duplicate fields"):
        load_command_config(config_path, Workspace(tmp_path))


@pytest.mark.parametrize("bad_arg", ["\x00", "\ud800"])
def test_command_config_rejects_unsafe_argv_strings(tmp_path: Path, bad_arg: str) -> None:
    config_path = tmp_path / "commands.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "commands": {
                    "bad": {
                        "argv": [sys.executable, bad_arg],
                        "cwd": ".",
                        "timeout_seconds": 5,
                    }
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(CommandConfigError, match="argv item is invalid"):
        load_command_config(config_path, Workspace(tmp_path))


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object capability preflight")
def test_windows_job_preflight_reports_stable_config_error(monkeypatch) -> None:
    def fail_job_creation():
        raise OSError("injected job failure")

    monkeypatch.setattr(command_module, "_create_kill_on_close_job", fail_job_creation)

    with pytest.raises(CommandConfigError, match="Windows job support"):
        command_module._validate_windows_job_support()


def test_command_config_loads_profiles_and_dynamic_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {"tests": {"argv": [sys.executable, "-c", "print('ok')"], "cwd": ".", "timeout_seconds": 5}},
    )

    config = load_command_config(config_path, Workspace(tmp_path))
    tool = RunCommandTool(config)

    assert sorted(config.profiles) == ["tests"]
    assert tool.definition.risk_level is ToolRiskLevel.LOCAL_EXEC
    assert tool.definition.input_schema["properties"]["profile"]["enum"] == ["tests"]


def test_command_config_rejects_invalid_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"bad": {"argv": [], "cwd": ".", "timeout_seconds": 5}})

    try:
        load_command_config(config_path, Workspace(tmp_path))
    except CommandConfigError as exc:
        assert "argv" in str(exc)
    else:
        raise AssertionError("expected CommandConfigError")


def test_command_config_rejects_workspace_escape_and_symlink_cwd(tmp_path: Path) -> None:
    outside_config = tmp_path / "outside.json"
    _write_config(outside_config, {"x": {"argv": [sys.executable, "-V"], "cwd": "..", "timeout_seconds": 5}})
    try:
        load_command_config(outside_config, Workspace(tmp_path))
    except CommandConfigError:
        pass
    else:
        raise AssertionError("expected CommandConfigError")

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        return
    symlink_config = tmp_path / "symlink.json"
    _write_config(symlink_config, {"x": {"argv": [sys.executable, "-V"], "cwd": "link", "timeout_seconds": 5}})
    try:
        load_command_config(symlink_config, Workspace(tmp_path))
    except CommandConfigError:
        pass
    else:
        raise AssertionError("expected CommandConfigError")


def test_run_command_success_and_nonzero_exit(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {
            "ok": {"argv": [sys.executable, "-c", "print('ok')"], "cwd": ".", "timeout_seconds": 5},
            "fail": {"argv": [sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"], "cwd": ".", "timeout_seconds": 5},
        },
    )
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    ok = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "ok"})))
    failed = asyncio.run(executor.execute(ToolCall(id="2", name="run_command", arguments={"profile": "fail"})))

    assert ok.error is None
    assert ok.output["exit_code"] == 0
    assert ok.output["stdout"].splitlines() == ["ok"]
    assert failed.error is None
    assert failed.output["exit_code"] == 7


def test_run_command_rejects_model_supplied_args(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"ok": {"argv": [sys.executable, "-V"], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(
        executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "ok", "args": ["evil"]}))
    )

    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS


def test_run_command_timeout_and_budget(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {"slow": {"argv": [sys.executable, "-c", "import time; time.sleep(5)"], "cwd": ".", "timeout_seconds": 1}},
    )
    budget = ProcessExecutionBudget(ProcessExecutionLimits(max_process_starts=1, max_total_duration_seconds=300))
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)), budget)))

    timed_out = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "slow"})))
    blocked = asyncio.run(executor.execute(ToolCall(id="2", name="run_command", arguments={"profile": "slow"})))

    assert timed_out.error is None
    assert timed_out.output["timed_out"] is True
    assert timed_out.output["exit_code"] is None
    assert budget.process_starts == 1
    assert blocked.error is not None
    assert "budget exceeded" in blocked.error.message


def test_run_command_preserves_legitimate_exit_code_125(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"exit125": {"argv": [sys.executable, "-c", "import sys; sys.exit(125)"], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "exit125"})))

    assert result.error is None
    assert result.output["exit_code"] == 125


def test_run_command_preserves_legitimate_exit_code_124(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"exit124": {"argv": [sys.executable, "-c", "import sys; sys.exit(124)"], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "exit124"})))

    assert result.error is None
    assert result.output["exit_code"] == 124


def test_run_command_reports_start_error_when_executable_disappears_after_load(tmp_path: Path) -> None:
    tool = tmp_path / ("tool.cmd" if os.name == "nt" else "tool")
    tool.write_text("@echo off\nexit /b 0\n" if os.name == "nt" else "#!/bin/sh\nexit 0\n", encoding="utf-8")
    if os.name != "nt":
        tool.chmod(0o755)
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"gone": {"argv": [str(tool)], "cwd": ".", "timeout_seconds": 5}})
    config = load_command_config(config_path, Workspace(tmp_path))
    tool.unlink()
    executor = ToolExecutor(_registry(RunCommandTool(config)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "gone"})))

    assert result.error is not None
    assert "could not be started" in result.error.message


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable permission check")
def test_command_profile_rejects_non_executable_file(tmp_path: Path) -> None:
    tool = tmp_path / "tool"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o644)
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"bad": {"argv": [str(tool)], "cwd": ".", "timeout_seconds": 5}})

    with pytest.raises(CommandConfigError, match="not executable"):
        load_command_config(config_path, Workspace(tmp_path))


@pytest.mark.skipif(os.name == "nt", reason="POSIX supervisor pipe protocol")
def test_posix_target_cannot_forge_cleanup_status_file(tmp_path: Path) -> None:
    marker = tmp_path / "forged-marker.txt"
    pid_path = tmp_path / "detached.pid"
    config_path = tmp_path / "commands.json"
    child_code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        f"time.sleep(10); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    code = (
        "import json,os,pathlib,signal,subprocess,sys,tempfile; "
        "fake=pathlib.Path(tempfile.gettempdir())/'nexusmind-command-forged'/'status.json'; "
        "fake.parent.mkdir(exist_ok=True); "
        "fake.write_text(json.dumps({'target_started':True,'start_succeeded':True,'root_exit_code':0,'cleanup_succeeded':True}), encoding='utf-8'); "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True,"
        "env={'PATH': os.environ.get('PATH','')},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "os.kill(os.getppid(), signal.SIGKILL)"
    )
    _write_config(config_path, {"forge": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    try:
        result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "forge"})))
    finally:
        deadline = time.monotonic() + 2
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text(encoding="ascii")), signal.SIGKILL)
            except ProcessLookupError:
                pass

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX supervisor signal behavior")
@pytest.mark.parametrize("supervisor_signal", [signal.SIGTERM, signal.SIGINT])
def test_posix_target_cannot_signal_supervisor_into_normal_result(
    tmp_path: Path,
    supervisor_signal: signal.Signals,
) -> None:
    config_path = tmp_path / "commands.json"
    code = f"import os,signal; os.kill(os.getppid(), {int(supervisor_signal)})"
    _write_config(config_path, {"signal": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "signal"})))

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED
    assert "status could not be verified" in result.error.message


def test_process_budget_reserves_start_slot_atomically() -> None:
    budget = ProcessExecutionBudget(ProcessExecutionLimits(max_process_starts=1, max_total_duration_seconds=300))

    reserved = budget.reserve_start_and_duration(1000)
    with pytest.raises(CommandLimitError, match="budget exceeded"):
        budget.reserve_start_and_duration(1000)

    budget.release_reservation(reserved)
    next_reserved = budget.reserve_start_and_duration(1000)
    budget.commit_start()
    budget.commit_actual_duration(next_reserved, 1500)

    assert budget.process_starts == 1
    assert budget.total_duration_ms == 1500


def test_run_command_reports_supervisor_cleanup_failure(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"ok": {"argv": [sys.executable, "-c", "print('ok')"], "cwd": ".", "timeout_seconds": 5}})
    nonces = iter(["execution-token", "status-nonce"])

    class FakeUuid:
        @property
        def hex(self) -> str:
            return next(nonces)

    monkeypatch.setattr(command_module.uuid, "uuid4", lambda: FakeUuid())
    monkeypatch.setattr(
        command_module,
        "_read_supervisor_status",
        lambda _fd: command_module._SupervisorStatus(
            nonce="status-nonce",
            target_started=True,
            start_succeeded=True,
            root_exit_code=0,
            cleanup_succeeded=False,
        ),
    )
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "ok"})))

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED


def _supervisor_status_from_payload(payload: bytes):
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, payload)
        os.close(write_fd)
        write_fd = None
        return command_module._read_supervisor_status(read_fd)
    finally:
        if write_fd is not None:
            os.close(write_fd)


def test_supervisor_status_reader_accepts_cleanup_failure() -> None:
    status = _supervisor_status_from_payload(
        b'{"nonce":"n","target_started":true,"start_succeeded":true,"root_exit_code":0,"cleanup_succeeded":false}'
    )

    assert status.nonce == "n"
    assert status.cleanup_succeeded is False


def test_supervisor_status_reader_accepts_start_failure() -> None:
    status = _supervisor_status_from_payload(
        b'{"nonce":"n","target_started":false,"start_succeeded":false,"root_exit_code":null,"cleanup_succeeded":true}'
    )

    assert status.start_succeeded is False


def test_supervisor_status_reader_rejects_invalid_json() -> None:
    with pytest.raises(command_module.CommandCleanupError):
        _supervisor_status_from_payload(b"{")


def test_supervisor_status_nonce_mismatch_is_cleanup_error(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"ok": {"argv": [sys.executable, "-c", "print('ok')"], "cwd": ".", "timeout_seconds": 5}})
    monkeypatch.setattr(
        command_module,
        "_read_supervisor_status",
        lambda _fd: command_module._SupervisorStatus(
            nonce="wrong",
            target_started=True,
            start_succeeded=True,
            root_exit_code=0,
            cleanup_succeeded=True,
        ),
    )
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "ok"})))

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED


def test_tool_executor_cancellation_cleans_up_process(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    config_path = tmp_path / "commands.json"
    code = f"import pathlib,time; time.sleep(2); pathlib.Path({str(marker)!r}).write_text('alive')"
    _write_config(config_path, {"slow": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 10}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=20)

    async def run_and_cancel() -> None:
        task = asyncio.create_task(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "slow"})))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        raise AssertionError("expected cancellation")

    asyncio.run(run_and_cancel())
    time.sleep(2.5)

    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor initialization")
def test_immediate_cancellation_after_supervisor_spawn_preserves_cancellation(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {"slow": {"argv": [sys.executable, "-c", "import time; time.sleep(2)"], "cwd": ".", "timeout_seconds": 10}},
    )
    tool = RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))
    executor = ToolExecutor(_registry(tool), timeout=20)
    real_create_subprocess_exec = command_module.asyncio.create_subprocess_exec

    async def create_and_cancel(*args, **kwargs):
        process = await real_create_subprocess_exec(*args, **kwargs)
        task = asyncio.current_task()
        assert task is not None
        asyncio.get_running_loop().call_soon(task.cancel)
        return process

    monkeypatch.setattr(command_module.asyncio, "create_subprocess_exec", create_and_cancel)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "slow"})))


def test_cancellation_reports_cleanup_failure(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"slow": {"argv": [sys.executable, "-c", "import time; time.sleep(2)"], "cwd": ".", "timeout_seconds": 10}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=20)

    async def failed_cleanup(*args, **kwargs):
        return command_module._CleanupResult((b"", False), (b"", False), False, False)

    monkeypatch.setattr(command_module, "_cleanup_process", failed_cleanup)

    async def run_and_cancel():
        task = asyncio.create_task(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "slow"})))
        await asyncio.sleep(0.2)
        task.cancel()
        return await task

    result = asyncio.run(run_and_cancel())

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED


@pytest.mark.skipif(os.name == "nt", reason="POSIX process group cleanup")
def test_cancellation_cleans_process_group_once(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "marker.txt"
    config_path = tmp_path / "commands.json"
    code = f"import pathlib,time; time.sleep(2); pathlib.Path({str(marker)!r}).write_text('alive')"
    _write_config(config_path, {"slow": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 10}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=20)
    real_killpg = os.killpg
    calls: list[int] = []

    def counting_killpg(pid: int, sig: int) -> None:
        calls.append(sig)
        real_killpg(pid, sig)

    monkeypatch.setattr(command_module.os, "killpg", counting_killpg)

    async def run_and_cancel() -> None:
        task = asyncio.create_task(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "slow"})))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())
    time.sleep(2.5)

    assert calls.count(signal.SIGTERM) <= 1
    assert calls.count(signal.SIGKILL) <= 1
    assert not marker.exists()


def test_profile_timeout_cleans_up_child_process(tmp_path: Path) -> None:
    marker = tmp_path / "child-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = f"import pathlib,time; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(10)"
    _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 1}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=10)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
    time.sleep(3.5)

    assert result.error is None
    assert result.output["timed_out"] is True
    assert result.output["exit_code"] is None
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")
def test_profile_timeout_kills_child_that_ignores_sigterm(tmp_path: Path) -> None:
    marker = tmp_path / "child-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = (
        "import pathlib,signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "sys.stdout.close(); sys.stderr.close(); "
        f"time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    parent_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(10)"
    _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 1}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=10)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
    time.sleep(3.5)

    assert result.error is None
    assert result.output["timed_out"] is True
    assert result.output["exit_code"] is None
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX detached process cleanup")
def test_profile_timeout_kills_detached_setsid_child(tmp_path: Path) -> None:
    marker = tmp_path / "detached-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = f"import pathlib,time; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent_code = (
        f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(10)"
    )
    _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 1}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=20)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
    time.sleep(3.5)

    assert result.error is None
    assert result.output["timed_out"] is True
    assert result.output["exit_code"] is None
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX detached process cleanup")
def test_profile_timeout_kills_detached_child_with_clean_environment(tmp_path: Path) -> None:
    marker = tmp_path / "clean-env-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = f"import pathlib,time; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent_code = (
        "import os,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "start_new_session=True,env={'PATH': os.environ.get('PATH','')},"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(10)"
    )
    _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 1}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=20)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
    time.sleep(3.5)

    assert result.error is None
    assert result.output["timed_out"] is True
    assert result.output["exit_code"] is None
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX supervisor isolation")
def test_run_command_cleanup_does_not_kill_unrelated_child_process(tmp_path: Path) -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    marker = tmp_path / "clean-env-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = f"import pathlib,time; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent_code = (
        "import os,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "start_new_session=True,env={'PATH': os.environ.get('PATH','')},"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); time.sleep(10)"
    )
    try:
        _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 1}})
        executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=20)

        result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
        time.sleep(3.5)

        assert result.error is None
        assert result.output["timed_out"] is True
        assert result.output["exit_code"] is None
        assert not marker.exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        try:
            unrelated.wait(timeout=2)
        except subprocess.TimeoutExpired:
            unrelated.kill()
            unrelated.wait(timeout=2)


def test_run_command_cleanup_timeout_budget_matches_platform() -> None:
    if os.name == "nt":
        assert command_module._command_cleanup_timeout_budget() == command_module.COMMAND_CLEANUP_GRACE_SECONDS * 4
    else:
        assert command_module._command_cleanup_timeout_budget() == (
            command_module.COMMAND_POSIX_SUPERVISOR_CLEANUP_SECONDS
            + (command_module.COMMAND_CLEANUP_GRACE_SECONDS * 3)
        )


def test_forced_cleanup_skips_hard_kill_after_soft_reap() -> None:
    class FakeProcess:
        returncode = None

    class FakeGuard:
        killed = False

        def terminate(self) -> bool:
            process.returncode = -15
            return True

        def kill(self) -> bool:
            self.killed = True
            return True

        def close(self) -> bool:
            return True

    process = FakeProcess()
    guard = FakeGuard()

    result = asyncio.run(command_module._cleanup_process(process, guard, None, None, force=True))

    assert result.root_reaped is True
    assert result.tree_terminated is True
    assert guard.killed is False


def test_profile_timeout_cleans_up_child_after_parent_exits_with_inherited_pipe(tmp_path: Path) -> None:
    marker = tmp_path / "child-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = f"import pathlib,time; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent_code = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child_code!r}])"
    _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 1}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=10)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
    time.sleep(3.5)

    assert result.error is None
    assert result.output["timed_out"] in {True, False}
    assert not marker.exists()


def test_successful_parent_exit_cleans_up_child_with_closed_pipe(tmp_path: Path) -> None:
    marker = tmp_path / "child-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = f"import pathlib,time; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    parent_code = (
        f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )
    _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=10)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
    time.sleep(3.5)

    assert result.error is None
    assert result.output["timed_out"] is False
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX supervisor cleanup")
def test_successful_parent_exit_cleans_up_detached_child_that_ignores_sigterm(tmp_path: Path) -> None:
    marker = tmp_path / "ignored-detached-marker.txt"
    config_path = tmp_path / "commands.json"
    child_code = (
        "import pathlib,signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "sys.stdout.close(); sys.stderr.close(); "
        f"time.sleep(3); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    parent_code = (
        f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child_code!r}],"
        "start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )
    _write_config(config_path, {"tree": {"argv": [sys.executable, "-c", parent_code], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=20)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "tree"})))
    time.sleep(3.5)

    assert result.error is None
    assert result.output["timed_out"] is False
    assert not marker.exists()


def test_run_command_uses_remaining_duration_budget_as_timeout(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"slow": {"argv": [sys.executable, "-c", "import time; time.sleep(5)"], "cwd": ".", "timeout_seconds": 5}})
    budget = ProcessExecutionBudget(ProcessExecutionLimits(max_process_starts=2, max_total_duration_seconds=1))
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)), budget)), timeout=10)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "slow"})))

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED
    assert "budget exceeded" in result.error.message
    assert budget.process_starts == 0
    assert budget.total_duration_ms == 0


def test_run_command_profile_timeout_extends_executor_timeout(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {"slow": {"argv": [sys.executable, "-c", "import time; time.sleep(1); print('done')"], "cwd": ".", "timeout_seconds": 2}},
    )
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=0.1)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "slow"})))

    assert result.error is None
    assert result.output["timed_out"] is False
    assert result.output["stdout"].splitlines() == ["done"]


@pytest.mark.skipif(os.name != "nt", reason="Windows gate behavior")
def test_windows_gate_does_not_require_workspace_write_permission(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"ok": {"argv": [sys.executable, "-c", "print('ok')"], "cwd": ".", "timeout_seconds": 5}})
    tmp_path.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))), timeout=10)
        result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "ok"})))
    finally:
        tmp_path.chmod(stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)

    assert result.error is None
    assert result.output["stdout"].splitlines() == ["ok"]


@pytest.mark.skipif(os.name != "nt", reason="Windows gate behavior")
def test_windows_gate_publish_failure_consumes_budget_and_cleans_temp_dir(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "marker.txt"
    config_path = tmp_path / "commands.json"
    code = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('started')"
    _write_config(config_path, {"ok": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    before = {path for path in Path(tempfile.gettempdir()).glob("nexusmind-command-*") if path.is_dir()}

    def fail_publish(gate_path: Path) -> None:
        gate_path.write_text("", encoding="ascii")
        time.sleep(0.02)
        raise OSError("injected gate failure")

    monkeypatch.setattr(command_module, "_publish_windows_gate", fail_publish)
    budget = ProcessExecutionBudget()
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)), budget)), timeout=10)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "ok"})))
    time.sleep(0.5)
    after = {path for path in Path(tempfile.gettempdir()).glob("nexusmind-command-*") if path.is_dir()}

    assert result.error is not None
    assert "could not be started" in result.error.message
    assert budget.process_starts == 1
    assert budget.total_duration_ms > 0
    assert not marker.exists()
    assert after == before


def test_run_command_revalidates_cwd_before_spawn(tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"pwd": {"argv": [sys.executable, "-V"], "cwd": "work", "timeout_seconds": 5}})
    config = load_command_config(config_path, Workspace(tmp_path))
    cwd.rmdir()
    outside = tmp_path.parent
    try:
        cwd.symlink_to(outside, target_is_directory=True)
    except OSError:
        return
    budget = ProcessExecutionBudget()
    executor = ToolExecutor(_registry(RunCommandTool(config, budget)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "pwd"})))

    assert result.error is not None
    assert "cwd is invalid" in result.error.message
    assert budget.process_starts == 0
    assert budget.total_duration_ms == 0


def test_run_command_create_subprocess_failure_does_not_consume_start_budget(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"ok": {"argv": [sys.executable, "-c", "print('ok')"], "cwd": ".", "timeout_seconds": 5}})
    budget = ProcessExecutionBudget()
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)), budget)))

    async def fail_create(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(command_module.asyncio, "create_subprocess_exec", fail_create)

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "ok"})))

    assert result.error is not None
    assert "could not be started" in result.error.message
    assert budget.process_starts == 0


def test_command_profile_relative_executable_resolves_from_profile_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "work"
    cwd.mkdir()
    tool = cwd / ("tool.cmd" if os.name == "nt" else "tool")
    tool.write_text("@echo off\n" if os.name == "nt" else "#!/bin/sh\n", encoding="utf-8")
    if os.name != "nt":
        tool.chmod(0o755)
    config_path = tmp_path / "commands.json"
    _write_config(config_path, {"local": {"argv": [f".{os.sep}{tool.name}"], "cwd": "work", "timeout_seconds": 5}})

    profile = load_command_config(config_path, Workspace(tmp_path)).profiles["local"]

    assert profile.executable == str(tool.resolve(strict=True))


def test_command_profile_summary_preserves_identity_fields(tmp_path: Path) -> None:
    argv_tail = ["same-tail"] * 20
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {"ProfileA": {"argv": [sys.executable, *argv_tail], "cwd": ".", "timeout_seconds": 30}},
    )
    profile = load_command_config(config_path, Workspace(tmp_path)).profiles["ProfileA"]

    summary = command_profile_summary(profile, max_length=80)

    assert "profile=ProfileA" in summary
    assert 'cwd="."' in summary
    assert "timeout=30s" in summary


def test_command_profile_summary_preserves_timeout_and_distinct_argv_with_long_cwd(tmp_path: Path) -> None:
    cwd = tmp_path
    for part in ["very", "long", "nested", "path", "to", "workspace", "commands"]:
        cwd /= part
    cwd.mkdir(parents=True)
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {
            "tests": {
                "argv": [sys.executable, "-c", "print('a b')", "a b"],
                "cwd": "very/long/nested/path/to/workspace/commands",
                "timeout_seconds": 120,
            }
        },
    )
    profile = load_command_config(config_path, Workspace(tmp_path)).profiles["tests"]

    summary = command_profile_summary(profile, max_length=160)

    assert "profile=tests" in summary
    assert "timeout=120s" in summary
    assert "commands" in summary
    assert '["' in summary
    assert "argv=[" in summary
    assert str(tmp_path) not in summary


def test_command_profile_summary_preserves_argv_tail(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {
            "tests": {
                "argv": [sys.executable, "-m", "pytest", "--very-long-option-name=" + ("x" * 80), "--config", "release.toml"],
                "cwd": ".",
                "timeout_seconds": 120,
            }
        },
    )
    profile = load_command_config(config_path, Workspace(tmp_path)).profiles["tests"]

    summary = command_profile_summary(profile, max_length=120)

    assert "..." in summary
    assert "release.toml" in summary


def test_command_profile_summary_escapes_bidi_controls_in_cwd(tmp_path: Path) -> None:
    cwd = tmp_path / "safe\u202eevil"
    cwd.mkdir()
    config_path = tmp_path / "commands.json"
    _write_config(
        config_path,
        {"tests": {"argv": [sys.executable, "-V"], "cwd": cwd.name, "timeout_seconds": 30}},
    )
    profile = load_command_config(config_path, Workspace(tmp_path)).profiles["tests"]

    summary = command_profile_summary(profile)

    assert "\u202e" not in summary
    assert "\\u202e" in summary


def test_run_command_does_not_inherit_sensitive_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "secret-value")
    config_path = tmp_path / "commands.json"
    code = "import os; print(os.environ.get('SECRET_TOKEN', 'missing'))"
    _write_config(config_path, {"env": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "env"})))

    assert result.error is None
    assert result.output["stdout"].splitlines() == ["missing"]


def test_run_command_replaces_invalid_utf8_output(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    code = "import sys; sys.stdout.buffer.write(b'\\xff')"
    _write_config(config_path, {"bad": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "bad"})))

    assert result.error is None
    assert result.output["stdout"] == "\ufffd"
    assert result.output["encoding_replaced"] is True


def test_run_command_drains_and_truncates_stdout_and_stderr(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    code = "import sys; sys.stdout.write('x'*140000); sys.stderr.write('y'*140000)"
    _write_config(config_path, {"loud": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    executor = ToolExecutor(_registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="run_command", arguments={"profile": "loud"})))

    assert result.error is None
    assert result.output["exit_code"] == 0
    assert len(result.output["stdout"].encode("utf-8")) == 128 * 1024
    assert len(result.output["stderr"].encode("utf-8")) == 128 * 1024
    assert result.output["stdout_truncated"] is True
    assert result.output["stderr_truncated"] is True


def test_run_command_nul_output_fits_real_chat_runtime_budget(tmp_path: Path) -> None:
    config_path = tmp_path / "commands.json"
    code = (
        "import sys; "
        "sys.stdout.buffer.write(b'\\x00'*(128*1024)); "
        "sys.stderr.buffer.write(b'\\x00'*(128*1024))"
    )
    _write_config(config_path, {"loud": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    tool = RunCommandTool(load_command_config(config_path, Workspace(tmp_path)))
    executor = ToolExecutor(_registry(tool))

    class AllowPolicy:
        async def evaluate(self, _call, _context):
            return ToolPolicyDecision.ALLOW

    class CommandModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_1", name="run_command", arguments={"profile": "loud"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model):
        runtime = ChatRuntime(
            model,
            tool_executor=executor,
            tool_policy=AllowPolicy(),
            limits=AgentLoopLimits(
                max_tool_result_bytes_per_call=300_000,
                max_tool_result_bytes_total=300_000,
            ),
        )
        return [
            event
            async for event in runtime.stream_user_message(
                "run",
                tools=[tool.definition],
            )
        ]

    model = CommandModel()
    events = asyncio.run(collect(model))

    assert events[-1].type is RuntimeEventType.RUN_COMPLETED
    assert len(model.messages_by_turn) == 2
    assert len(model.messages_by_turn[1][-1].content.encode("utf-8")) <= 300_000


@pytest.mark.parametrize("limited_dimension", ["bytes", "nodes", "depth"])
def test_run_command_rejects_insufficient_result_budget_before_execution(
    tmp_path: Path,
    limited_dimension: str,
) -> None:
    marker = tmp_path / "started.txt"
    config_path = tmp_path / "commands.json"
    code = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('started')"
    _write_config(config_path, {"write": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    budget = ProcessExecutionBudget()
    tool = RunCommandTool(load_command_config(config_path, Workspace(tmp_path)), budget)
    executor = ToolExecutor(_registry(tool))
    requirements = tool.result_requirements({"profile": "write"})
    policy_calls = 0

    class CountingPolicy:
        async def evaluate(self, _call, _context):
            nonlocal policy_calls
            policy_calls += 1
            return ToolPolicyDecision.ALLOW

    class CommandModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="run_command", arguments={"profile": "write"}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    limit_values = {
        "max_tool_result_bytes_per_call": requirements.min_bytes,
        "max_tool_result_bytes_total": requirements.min_bytes,
        "max_json_nodes_per_payload": requirements.min_nodes,
        "max_json_depth": requirements.min_depth,
    }
    if limited_dimension == "bytes":
        limit_values["max_tool_result_bytes_per_call"] -= 1
        limit_values["max_tool_result_bytes_total"] -= 1
    elif limited_dimension == "nodes":
        limit_values["max_json_nodes_per_payload"] -= 1
    else:
        limit_values["max_json_depth"] -= 1

    async def collect():
        runtime = ChatRuntime(
            CommandModel(),
            tool_executor=executor,
            tool_policy=CountingPolicy(),
            limits=AgentLoopLimits(**limit_values),
        )
        return [event async for event in runtime.stream_user_message("run", tools=[tool.definition])]

    events = asyncio.run(collect())

    assert events[-1].type is RuntimeEventType.RUN_FAILED
    assert events[-1].error == "Agent loop limit exceeded"
    expected_policy_calls = 0 if limited_dimension == "depth" else 1
    assert policy_calls == expected_policy_calls
    assert budget.process_starts == 0
    assert not marker.exists()


def test_run_command_policy_deny_fits_without_success_result_budget(tmp_path: Path) -> None:
    marker = tmp_path / "started.txt"
    config_path = tmp_path / "commands.json"
    code = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('started')"
    _write_config(config_path, {"write": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    process_budget = ProcessExecutionBudget()
    tool = RunCommandTool(load_command_config(config_path, Workspace(tmp_path)), process_budget)
    executor = ToolExecutor(_registry(tool))

    class DenyPolicy:
        async def evaluate(self, _call, _context):
            return ToolPolicyDecision.DENY

    class CommandModel:
        def __init__(self) -> None:
            self.turn = 0

        async def stream(self, messages, tools=None):
            self.turn += 1
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if self.turn == 1:
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_1", name="run_command", arguments={"profile": "write"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect():
        runtime = ChatRuntime(
            CommandModel(),
            tool_executor=executor,
            tool_policy=DenyPolicy(),
            limits=AgentLoopLimits(
                max_tool_result_bytes_per_call=105,
                max_tool_result_bytes_total=105,
                max_json_nodes_per_payload=6,
                max_json_depth=2,
            ),
        )
        return [event async for event in runtime.stream_user_message("run", tools=[tool.definition])]

    events = asyncio.run(collect())

    assert events[-1].type is RuntimeEventType.RUN_COMPLETED
    tool_result = next(event.tool_result for event in events if event.type is RuntimeEventType.TOOL_RESULT)
    assert tool_result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert process_budget.process_starts == 0
    assert not marker.exists()


def test_tool_executor_rejects_small_command_budget_before_spawn(tmp_path: Path) -> None:
    marker = tmp_path / "started.txt"
    config_path = tmp_path / "commands.json"
    code = f"import pathlib; pathlib.Path({str(marker)!r}).write_text('started')"
    _write_config(config_path, {"write": {"argv": [sys.executable, "-c", code], "cwd": ".", "timeout_seconds": 5}})
    process_budget = ProcessExecutionBudget()
    executor = ToolExecutor(
        _registry(RunCommandTool(load_command_config(config_path, Workspace(tmp_path)), process_budget))
    )
    call = ToolCall(id="1", name="run_command", arguments={"profile": "write"})
    requirements = executor.result_requirements(call)

    with pytest.raises(ToolResultBudgetError, match="budget is too small"):
        asyncio.run(
            executor.execute_with_result_budget(
                call,
                result_budget=ToolResultBudget(
                    max_bytes=requirements.min_bytes - 1,
                    max_nodes=requirements.min_nodes,
                    max_depth=requirements.min_depth,
                ),
            )
        )

    assert process_budget.process_starts == 0
    assert not marker.exists()
