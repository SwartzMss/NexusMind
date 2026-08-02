from __future__ import annotations

import asyncio
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any

from nexusmind.command_errors import CommandCleanupError, CommandConfigError, CommandLimitError, CommandProfileError, CommandStartError
from nexusmind.tools.contracts import ToolDefinition, ToolResultBudget, ToolResultRequirements, ToolRiskLevel, json_result_requirements
from nexusmind.workspace import Workspace, WorkspaceError, resolve_workspace_path, workspace_relative_path

MAX_COMMAND_PROFILES = 32
MAX_ARGV_ITEMS = 32
MAX_ARG_CHARS = 1024
MAX_ARGV_BYTES = 8192
MAX_TIMEOUT_SECONDS = 300
MAX_OUTPUT_BYTES = 128 * 1024
MAX_REPORTED_STREAM_BYTES = 2**63 - 1
DEFAULT_TOOL_RESULT_BYTES = 1024 * 1024
COMMAND_CLEANUP_GRACE_SECONDS = 2.0
COMMAND_POSIX_SUPERVISOR_CLEANUP_SECONDS = COMMAND_CLEANUP_GRACE_SECONDS * 3
COMMAND_STARTUP_GRACE_SECONDS = 10.0
COMMAND_DISPATCH_GRACE_SECONDS = 2.0
PR_SET_CHILD_SUBREAPER = 36
PR_SET_DUMPABLE = 4
PROFILE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
ALLOWED_ENV_KEYS = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "HOME",
        "USERPROFILE",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
    }
)


@dataclass(frozen=True, slots=True)
class CommandProfile:
    profile_id: str
    argv: tuple[str, ...]
    executable: str
    workspace: Workspace
    cwd_config: str
    cwd: Path
    cwd_relative: str
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class CommandConfig:
    profiles: dict[str, CommandProfile]


@dataclass(frozen=True, slots=True)
class _SupervisorStatus:
    nonce: str
    target_started: bool
    start_succeeded: bool
    root_exit_code: int | None
    cleanup_succeeded: bool


@dataclass(frozen=True, slots=True)
class ProcessExecutionLimits:
    max_process_starts: int = 8
    max_total_duration_seconds: int = 300


class ProcessExecutionBudget:
    def __init__(self, limits: ProcessExecutionLimits | None = None) -> None:
        self._limits = limits or ProcessExecutionLimits()
        self.process_starts = 0
        self.total_duration_ms = 0
        self._reserved_starts = 0
        self._reserved_duration_ms = 0
        self._lock = threading.Lock()

    def reserve_start_and_duration(self, requested_ms: int) -> int:
        with self._lock:
            self._check_locked()
            remaining = self._remaining_ms_locked()
            reserved = min(max(requested_ms, 1), remaining)
            if reserved <= 0:
                raise CommandLimitError("Command execution budget exceeded")
            self._reserved_starts += 1
            self._reserved_duration_ms += reserved
            return reserved

    def commit_start(self) -> None:
        with self._lock:
            if self._reserved_starts <= 0:
                raise RuntimeError("Command process start was not reserved")
            self._reserved_starts -= 1
            self.process_starts += 1

    def commit_actual_duration(self, reserved_ms: int, actual_ms: int) -> None:
        with self._lock:
            self._reserved_duration_ms = max(0, self._reserved_duration_ms - max(reserved_ms, 0))
            self.total_duration_ms += max(actual_ms, 0)

    def release_reservation(self, reserved_ms: int) -> None:
        with self._lock:
            self._reserved_starts = max(0, self._reserved_starts - 1)
            self._reserved_duration_ms = max(0, self._reserved_duration_ms - max(reserved_ms, 0))

    def _check_locked(self) -> None:
        if self.process_starts + self._reserved_starts >= self._limits.max_process_starts:
            raise CommandLimitError("Command execution budget exceeded")
        if self._remaining_ms_locked() <= 0:
            raise CommandLimitError("Command execution budget exceeded")

    def _remaining_ms_locked(self) -> int:
        return self._limits.max_total_duration_seconds * 1000 - self.total_duration_ms - self._reserved_duration_ms


@dataclass(frozen=True, slots=True)
class _CleanupResult:
    stdout: "CapturedStream"
    stderr: "CapturedStream"
    root_reaped: bool
    tree_terminated: bool


class RunCommandTool:
    def __init__(self, config: CommandConfig, budget: ProcessExecutionBudget | None = None) -> None:
        self._config = config
        self._budget = budget or ProcessExecutionBudget()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_command",
            description="Run a host-configured command profile in the workspace.",
            input_schema={
                "type": "object",
                "properties": {"profile": {"type": "string", "enum": sorted(self._config.profiles)}},
                "required": ["profile"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.LOCAL_EXEC,
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.invoke_with_result_budget(
            arguments,
            result_budget=ToolResultBudget(
                max_bytes=DEFAULT_TOOL_RESULT_BYTES,
                max_nodes=100_000,
                max_depth=100,
            ),
        )

    async def invoke_with_result_budget(
        self,
        arguments: dict[str, Any],
        *,
        result_budget: ToolResultBudget,
    ) -> dict[str, Any]:
        profile_id = arguments["profile"]
        try:
            profile = self._config.profiles[profile_id]
        except KeyError as exc:
            raise CommandProfileError("Command profile not found") from exc
        return await _run_profile(profile, self._budget, max_result_bytes=result_budget.max_bytes)

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        profile_id = arguments.get("profile")
        profile = self._config.profiles.get(profile_id) if type(profile_id) is str else None
        profiles = [profile] if profile is not None else list(self._config.profiles.values())
        requirements = [
            json_result_requirements({"ok": True, "output": _minimum_command_output(item)})
            for item in profiles
        ]
        return ToolResultRequirements(
            min_bytes=max(item.min_bytes for item in requirements),
            min_nodes=max(item.min_nodes for item in requirements),
            min_depth=max(item.min_depth for item in requirements),
        )

    def timeout_for_call(self, arguments: dict[str, Any]) -> float:
        profile_id = arguments.get("profile")
        if type(profile_id) is not str:
            return 30.0
        try:
            profile = self._config.profiles[profile_id]
        except KeyError:
            return 30.0
        return (
            profile.timeout_seconds
            + COMMAND_STARTUP_GRACE_SECONDS
            + _command_cleanup_timeout_budget()
            + COMMAND_DISPATCH_GRACE_SECONDS
        )


def _command_cleanup_timeout_budget() -> float:
    if os.name == "nt":
        return COMMAND_CLEANUP_GRACE_SECONDS * 4
    return COMMAND_POSIX_SUPERVISOR_CLEANUP_SECONDS + (COMMAND_CLEANUP_GRACE_SECONDS * 3)


def load_command_config(path: str | Path, workspace: Workspace) -> CommandConfig:
    validate_command_execution_platform()
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise CommandConfigError("Command config could not be read") from exc
    if len(raw) > 128 * 1024:
        raise CommandConfigError("Command config exceeds the size limit")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_command_fields)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommandConfigError("Command config is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != {"schema_version", "commands"}:
        raise CommandConfigError("Command config has invalid fields")
    if data["schema_version"] != 1:
        raise CommandConfigError("Command config schema_version is unsupported")
    commands = data["commands"]
    if not isinstance(commands, dict) or not commands:
        raise CommandConfigError("Command config commands must be an object")
    if len(commands) > MAX_COMMAND_PROFILES:
        raise CommandConfigError("Command config contains too many profiles")
    profiles: dict[str, CommandProfile] = {}
    for profile_id, profile_data in commands.items():
        if type(profile_id) is not str or not PROFILE_ID_RE.fullmatch(profile_id):
            raise CommandConfigError("Command profile id is invalid")
        if not isinstance(profile_data, dict) or set(profile_data) != {"argv", "cwd", "timeout_seconds"}:
            raise CommandConfigError("Command profile has invalid fields")
        profiles[profile_id] = _load_profile(profile_id, profile_data, workspace)
    return CommandConfig(profiles=dict(sorted(profiles.items())))


def _reject_duplicate_command_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CommandConfigError("Command config contains duplicate fields")
        result[key] = value
    return result


def command_profile_summary(profile: CommandProfile, *, max_length: int = 120) -> str:
    argv = json.dumps(list(profile.argv), ensure_ascii=True, separators=(",", ":"))
    cwd_json = json.dumps(profile.cwd_relative, ensure_ascii=True)
    cwd = _abbreviate_middle(cwd_json, max_length=max(16, max_length // 3))
    prefix = f"profile={profile.profile_id}; cwd={cwd}; "
    suffix = f"; timeout={profile.timeout_seconds}s"
    available_argv = max_length - len(prefix) - len("argv=") - len(suffix)
    if available_argv < 16:
        available_argv = 16
    if len(argv) > available_argv:
        argv = _abbreviate_middle(argv, max_length=available_argv)
    text = f"{prefix}argv={argv}{suffix}"
    if len(text) <= max_length:
        return text
    # Keep profile and timeout intact even for very small render budgets.
    minimum = f"profile={profile.profile_id}; cwd=...; argv=...; timeout={profile.timeout_seconds}s"
    return minimum if len(minimum) > max_length else minimum[:max_length]


async def _run_profile(
    profile: CommandProfile,
    budget: ProcessExecutionBudget,
    *,
    max_result_bytes: int = DEFAULT_TOOL_RESULT_BYTES,
) -> dict[str, Any]:
    start = time.monotonic()
    process: asyncio.subprocess.Process | None = None
    stdout_task: asyncio.Task[CapturedStream] | None = None
    stderr_task: asyncio.Task[CapturedStream] | None = None
    process_guard: _ProcessTreeGuard | None = None
    gate_path: Path | None = None
    gate_dir: Path | None = None
    status_read_fd: int | None = None
    status_write_fd: int | None = None
    cleanup_result: _CleanupResult | None = None
    cleanup_task: asyncio.Task[_CleanupResult] | None = None
    lifecycle: asyncio.Future | None = None
    supervisor_status: _SupervisorStatus | None = None
    supervisor_status_error: CommandCleanupError | None = None
    original_exception: BaseException | None = None
    reserved_duration_ms = 0
    status_nonce = uuid.uuid4().hex
    cleanup_budget_seconds = _command_cleanup_timeout_budget()
    cwd = _resolve_profile_cwd(profile)
    minimum_overhead_seconds = COMMAND_STARTUP_GRACE_SECONDS + cleanup_budget_seconds
    requested_budget_ms = int((profile.timeout_seconds + minimum_overhead_seconds) * 1000)
    reserved_duration_ms = budget.reserve_start_and_duration(requested_budget_ms)
    target_budget_seconds = (reserved_duration_ms / 1000) - minimum_overhead_seconds
    if target_budget_seconds <= 0:
        budget.release_reservation(reserved_duration_ms)
        reserved_duration_ms = 0
        raise CommandLimitError("Command execution budget exceeded")
    effective_timeout = min(profile.timeout_seconds, target_budget_seconds)
    completed_normally = False
    timed_out = False

    async def ensure_cleanup_once(*, force: bool) -> _CleanupResult:
        nonlocal cleanup_result, cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(
                _cleanup_process(process, process_guard, stdout_task, stderr_task, force=force)
            )
        if cleanup_result is None:
            cleanup_result = await asyncio.shield(cleanup_task)
        return cleanup_result

    try:
        executable = profile.executable
        argv_tail = profile.argv[1:]
        status_read_fd, status_write_fd = os.pipe()
        os.set_inheritable(status_read_fd, False)
        os.set_inheritable(status_write_fd, True)
        if os.name == "nt":
            import msvcrt

            gate_dir = Path(tempfile.mkdtemp(prefix="nexusmind-command-"))
            gate_path = gate_dir / "gate"
            status_write_handle = msvcrt.get_osfhandle(status_write_fd)
            os.set_handle_inheritable(status_write_handle, True)
            executable = sys.executable
            argv_tail = (
                "-c",
                _WINDOWS_COMMAND_BOOTSTRAP,
                json.dumps([profile.executable, *profile.argv[1:]], ensure_ascii=True),
                str(gate_path),
                str(status_write_handle),
                status_nonce,
            )
        else:
            executable = sys.executable
            argv_tail = (
                "-c",
                _POSIX_COMMAND_SUPERVISOR,
                json.dumps([profile.executable, *profile.argv[1:]], ensure_ascii=True),
                str(status_write_fd),
                status_nonce,
            )
        env = _minimal_environment()
        process_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdin": subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": env,
        }
        if os.name == "nt":
            process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.lpAttributeList = {"handle_list": [status_write_handle]}
            process_kwargs["startupinfo"] = startupinfo
            process_kwargs["close_fds"] = True
        else:
            process_kwargs["start_new_session"] = True
            process_kwargs["pass_fds"] = (status_write_fd,)
        process = await asyncio.create_subprocess_exec(
            executable,
            *argv_tail,
            **process_kwargs,
        )
        budget.commit_start()
        if status_write_fd is not None:
            os.close(status_write_fd)
            status_write_fd = None
        stdout_task = asyncio.create_task(_read_stream_limited(process.stdout))
        stderr_task = asyncio.create_task(_read_stream_limited(process.stderr))
        process_guard = _ProcessTreeGuard(process)
        if gate_path is not None:
            _publish_windows_gate(gate_path)

        lifecycle = asyncio.gather(process.wait(), stdout_task, stderr_task)
        try:
            await asyncio.wait_for(asyncio.shield(lifecycle), timeout=effective_timeout)
            completed_normally = True
        except asyncio.TimeoutError:
            timed_out = True
            await ensure_cleanup_once(force=True)
    except asyncio.CancelledError as exc:
        original_exception = exc
        if process is not None:
            await ensure_cleanup_once(force=True)
    except OSError as exc:
        original_exception = CommandStartError("Command process could not be started")
        original_exception.__cause__ = exc
        if process is not None:
            await ensure_cleanup_once(force=True)
    finally:
        try:
            if process is not None and cleanup_result is None:
                await ensure_cleanup_once(force=not completed_normally)
        finally:
            if lifecycle is not None and not lifecycle.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(lifecycle),
                        timeout=COMMAND_CLEANUP_GRACE_SECONDS,
                    )
                except asyncio.TimeoutError:
                    lifecycle.cancel()
                    await asyncio.gather(lifecycle, return_exceptions=True)
            _cleanup_private_temp(gate_path, gate_dir)
            try:
                if process is not None:
                    supervisor_status = _read_supervisor_status(status_read_fd)
                    status_read_fd = None
            except CommandCleanupError as exc:
                supervisor_status_error = exc
            finally:
                _close_fd(status_write_fd)
                _close_fd(status_read_fd)
            duration_ms = int((time.monotonic() - start) * 1000)
            if process is not None:
                budget.commit_actual_duration(reserved_duration_ms, duration_ms)
            else:
                budget.release_reservation(reserved_duration_ms)
    cancelled = isinstance(original_exception, asyncio.CancelledError)
    if process is None and original_exception is not None:
        raise original_exception
    if cleanup_result is None:
        empty = CapturedStream(b"", 0, False)
        cleanup_result = _CleanupResult(empty, empty, process.returncode is not None, True)
    # Cancellation is the primary outcome even when best-effort cleanup did
    # not fully reap the process tree.
    if cancelled:
        raise original_exception
    if not cleanup_result.root_reaped or not cleanup_result.tree_terminated:
        raise CommandCleanupError("Command process cleanup failed")
    status_optional = cancelled or (os.name == "nt" and (timed_out or original_exception is not None))
    if supervisor_status_error is not None and not status_optional:
        raise supervisor_status_error
    if supervisor_status is None and not status_optional:
        raise CommandCleanupError("Command process cleanup status could not be verified")
    if supervisor_status is not None and supervisor_status.nonce != status_nonce:
        raise CommandCleanupError("Command process cleanup status is invalid")
    if supervisor_status is not None and not supervisor_status.cleanup_succeeded:
        raise CommandCleanupError("Command process cleanup failed")
    if (
        supervisor_status is not None
        and not supervisor_status.start_succeeded
        and not timed_out
        and original_exception is None
    ):
        raise CommandStartError("Command process could not be started")
    if (
        supervisor_status is not None
        and supervisor_status.start_succeeded
        and supervisor_status.root_exit_code is None
        and not timed_out
        and original_exception is None
    ):
        raise CommandCleanupError("Command execution status could not be verified")
    if original_exception is not None:
        raise original_exception
    stdout_raw = cleanup_result.stdout.content
    stderr_raw = cleanup_result.stderr.content
    stdout_bytes = cleanup_result.stdout.total_bytes
    stderr_bytes = cleanup_result.stderr.total_bytes
    stdout_truncated = cleanup_result.stdout.truncated
    stderr_truncated = cleanup_result.stderr.truncated
    stdout, stdout_replaced = _decode_output(stdout_raw)
    stderr, stderr_replaced = _decode_output(stderr_raw)
    exit_code = process.returncode
    if timed_out:
        exit_code = None
    elif supervisor_status is not None:
        exit_code = supervisor_status.root_exit_code
    output = {
        "profile": profile.profile_id,
        "cwd": profile.cwd_relative,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "encoding_replaced": stdout_replaced or stderr_replaced,
    }
    return _compact_command_output(output, max_result_bytes=max_result_bytes)


def _load_profile(profile_id: str, data: dict[str, Any], workspace: Workspace) -> CommandProfile:
    argv = data["argv"]
    if not isinstance(argv, list) or not argv:
        raise CommandConfigError("Command profile argv must be a non-empty array")
    if len(argv) > MAX_ARGV_ITEMS:
        raise CommandConfigError("Command profile argv contains too many items")
    if any(type(item) is not str or item == "" or len(item) > MAX_ARG_CHARS for item in argv):
        raise CommandConfigError("Command profile argv item is invalid")
    argv_tuple = tuple(argv)
    try:
        if any("\x00" in item for item in argv_tuple):
            raise UnicodeError
        argv_bytes = sum(len(item.encode("utf-8")) for item in argv_tuple)
    except UnicodeError as exc:
        raise CommandConfigError("Command profile argv item is invalid") from exc
    if argv_bytes > MAX_ARGV_BYTES:
        raise CommandConfigError("Command profile argv exceeds the size limit")
    cwd_value = data["cwd"]
    if type(cwd_value) is not str:
        raise CommandConfigError("Command profile cwd must be a string")
    try:
        if "\x00" in cwd_value:
            raise UnicodeError
        cwd_value.encode("utf-8")
    except UnicodeError as exc:
        raise CommandConfigError("Command profile cwd is invalid") from exc
    try:
        cwd = resolve_workspace_path(workspace, cwd_value, expected_type="directory")
    except WorkspaceError as exc:
        raise CommandConfigError("Command profile cwd is invalid") from exc
    executable = _resolve_command_executable(argv_tuple[0], cwd)
    timeout = data["timeout_seconds"]
    if type(timeout) is not int or isinstance(timeout, bool) or timeout < 1 or timeout > MAX_TIMEOUT_SECONDS:
        raise CommandConfigError("Command profile timeout is invalid")
    return CommandProfile(
        profile_id=profile_id,
        argv=argv_tuple,
        executable=executable,
        cwd=cwd,
        workspace=workspace,
        cwd_config=cwd_value,
        cwd_relative=workspace_relative_path(workspace, cwd),
        timeout_seconds=timeout,
    )


def _resolve_command_executable(value: str, cwd: Path) -> str:
    try:
        executable_path = Path(value)
    except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
        raise CommandConfigError("Command executable could not be resolved") from exc
    if executable_path.is_absolute():
        try:
            resolved = executable_path.resolve(strict=True)
        except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
            raise CommandConfigError("Command executable could not be resolved") from exc
        _validate_executable_file(resolved)
        return str(resolved)
    if any(separator in value for separator in ("/", "\\")):
        try:
            resolved = (cwd / executable_path).resolve(strict=True)
        except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
            raise CommandConfigError("Command executable could not be resolved") from exc
        _validate_executable_file(resolved)
        return str(resolved)
    try:
        executable = shutil.which(value)
    except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
        raise CommandConfigError("Command executable could not be resolved") from exc
    if executable is None:
        raise CommandConfigError("Command executable could not be resolved")
    try:
        resolved = Path(executable).resolve(strict=True)
        _validate_executable_file(resolved)
        return str(resolved)
    except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
        raise CommandConfigError("Command executable could not be resolved") from exc


def _validate_executable_file(path: Path) -> None:
    if not path.is_file():
        raise CommandConfigError("Command executable must be a file")
    if os.name != "nt" and not os.access(path, os.X_OK):
        raise CommandConfigError("Command executable is not executable")


def _abbreviate_middle(value: str, *, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return "." * max_length
    keep = max_length - 3
    head = max(1, keep // 2)
    tail = max(1, keep - head)
    return f"{value[:head]}...{value[-tail:]}"


def validate_command_execution_platform() -> None:
    if os.name == "nt":
        _validate_windows_job_support()
        return
    if sys.platform != "linux":
        raise CommandConfigError("Command profiles require Linux or Windows process containment")
    if not Path("/proc/self/stat").is_file():
        raise CommandConfigError("Command profiles require procfs")
    code = (
        "import ctypes,sys; "
        f"libc=ctypes.CDLL(None,use_errno=True); "
        f"dumpable=libc.prctl({PR_SET_DUMPABLE},0,0,0,0); "
        f"subreaper=libc.prctl({PR_SET_CHILD_SUBREAPER},1,0,0,0); "
        "sys.exit(0 if dumpable==0 and subreaper==0 else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommandConfigError("Command profiles require Linux subreaper support") from exc
    if result.returncode != 0:
        raise CommandConfigError("Command profiles require Linux subreaper support")


def _validate_windows_job_support() -> None:
    process: subprocess.Popen[bytes] | None = None
    job_handle: Any = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=0x00000004,
        )
        job_handle = _create_kill_on_close_job()
        if _KERNEL32 is None or not _KERNEL32.AssignProcessToJobObject(
            job_handle,
            wintypes.HANDLE(process._handle),  # type: ignore[attr-defined]
        ):
            raise OSError("Could not assign validation process to Windows job")
        if not _KERNEL32.CloseHandle(job_handle):
            raise OSError("Could not close validation Windows job")
        job_handle = None
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommandConfigError("Command profiles require Windows job support") from exc
    finally:
        if job_handle is not None and _KERNEL32 is not None:
            _KERNEL32.CloseHandle(job_handle)
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def _minimal_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.upper() in ALLOWED_ENV_KEYS}


_WINDOWS_COMMAND_BOOTSTRAP = r"""
import json
import os
import pathlib
import subprocess
import sys
import time
if sys.platform == "win32":
    import msvcrt

argv = json.loads(sys.argv[1])
gate = pathlib.Path(sys.argv[2])
if sys.platform == "win32":
    status_fd = msvcrt.open_osfhandle(int(sys.argv[3]), os.O_WRONLY)
else:
    status_fd = int(sys.argv[3])
nonce = sys.argv[4]

def write_status(target_started, start_succeeded, root_exit_code, cleanup_succeeded):
    try:
        status = {
            "nonce": nonce,
            "target_started": target_started,
            "start_succeeded": start_succeeded,
            "root_exit_code": root_exit_code,
            "cleanup_succeeded": cleanup_succeeded,
        }
        os.write(status_fd, json.dumps(status, separators=(",", ":")).encode("utf-8"))
        os.close(status_fd)
    except (OSError, TypeError, ValueError):
        return False
    return True

deadline = time.monotonic() + 10
marker = "go\n"
while True:
    try:
        if gate.read_text(encoding="ascii") == marker:
            break
    except OSError:
        pass
    if time.monotonic() > deadline:
        ok = write_status(False, False, None, True)
        sys.exit(125 if ok else 126)
    time.sleep(0.01)
try:
    completed = subprocess.run(argv, stdin=subprocess.DEVNULL)
except OSError:
    ok = write_status(False, False, None, True)
    sys.exit(124 if ok else 126)
ok = write_status(True, True, completed.returncode, True)
if not ok:
    sys.exit(126)
sys.exit(completed.returncode)
"""


_POSIX_COMMAND_SUPERVISOR = r"""
import ctypes
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
PR_SET_DUMPABLE = 4
blocked_signals = {signal.SIGTERM, signal.SIGINT}
signal.pthread_sigmask(signal.SIG_BLOCK, blocked_signals)
child = None
cleaning = False
status_fd = None
nonce = ""

def enable_subreaper():
    if sys.platform != "linux":
        raise RuntimeError("POSIX command supervisor requires Linux")
    if not pathlib.Path("/proc/self/stat").is_file():
        raise RuntimeError("POSIX command supervisor requires procfs")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE) failed")
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")

def parse_stat(text):
    close = text.rfind(")")
    if close < 0:
        raise ValueError("invalid stat")
    pid = int(text[: text.find("(")].strip())
    rest = text[close + 2 :].split()
    if len(rest) < 2:
        raise ValueError("invalid stat")
    return pid, int(rest[1])

def descendants(root):
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        return None
    parents = {}
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            pid, ppid = parse_stat((entry / "stat").read_text(encoding="ascii"))
        except FileNotFoundError:
            continue
        except OSError:
            return None
        except ValueError:
            return None
        parents[pid] = ppid
    found = []
    pending = [root]
    seen = {root}
    while pending:
        parent = pending.pop()
        for pid, ppid in parents.items():
            if ppid == parent and pid not in seen and pid != os.getpid():
                seen.add(pid)
                found.append(pid)
                pending.append(pid)
    return found

def signal_pid(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return True
    except (OSError, TypeError, ValueError):
        return False
    return True

def wait_child(timeout):
    if child is None:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return True
        time.sleep(0.02)
    return child.poll() is not None

def reap_available():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return

def cleanup_descendants():
    ok = True
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        found = descendants(os.getpid())
        if found is None:
            return False
        if not found:
            reap_available()
            return ok
        for pid in found:
            if not signal_pid(pid, signal.SIGTERM):
                ok = False
        time.sleep(0.05)
        reap_available()
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        found = descendants(os.getpid())
        if found is None:
            return False
        if not found:
            reap_available()
            return ok
        for pid in found:
            if not signal_pid(pid, signal.SIGKILL):
                ok = False
        time.sleep(0.05)
        reap_available()
    found = descendants(os.getpid())
    return found == [] and ok

def cleanup(signum=None):
    global cleaning
    if cleaning:
        return True
    cleaning = True
    ok = True
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            ok = False
        wait_child(1.0)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                ok = False
            ok = wait_child(1.0) and ok
    return cleanup_descendants() and ok

def write_status(target_started, start_succeeded, root_exit_code, cleanup_succeeded):
    if status_fd is None:
        return False
    try:
        status = {
            "nonce": nonce,
            "target_started": target_started,
            "start_succeeded": start_succeeded,
            "root_exit_code": root_exit_code,
            "cleanup_succeeded": cleanup_succeeded,
        }
        payload = json.dumps(status, separators=(",", ":")).encode("utf-8")
        os.write(status_fd, payload)
        os.close(status_fd)
    except OSError:
        return False
    return True

def cleanup_and_exit(signum=None, frame=None):
    if cleaning:
        return
    ok = cleanup(signum)
    ok = write_status(child is not None, child is not None, None, ok) and ok
    sys.exit(128 + (signum or 0) if ok else 125)

signal.signal(signal.SIGTERM, cleanup_and_exit)
signal.signal(signal.SIGINT, cleanup_and_exit)
argv = json.loads(sys.argv[1])
status_fd = int(sys.argv[2])
nonce = sys.argv[3]
try:
    enable_subreaper()
except Exception:
    ok = write_status(False, False, None, True)
    sys.exit(124 if ok else 125)
def unblock_command_signals():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, blocked_signals)
try:
    child = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        preexec_fn=unblock_command_signals,
    )
except OSError:
    ok = write_status(False, False, None, True)
    sys.exit(124 if ok else 125)
signal.pthread_sigmask(signal.SIG_UNBLOCK, blocked_signals)
while child.poll() is None:
    time.sleep(0.05)
exit_code = child.returncode
ok = cleanup(0)
ok = write_status(True, True, exit_code, ok) and ok
sys.exit(0 if ok else 125)
"""


def _publish_windows_gate(gate_path: Path) -> None:
    temp_path = gate_path.with_name(f"{gate_path.name}.tmp")
    with temp_path.open("x", encoding="ascii") as gate:
        gate.write("go\n")
        gate.flush()
        os.fsync(gate.fileno())
    temp_path.replace(gate_path)


def _read_supervisor_status(status_fd: int | None) -> _SupervisorStatus | None:
    if status_fd is None:
        return None
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(status_fd, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > 4096:
                raise CommandCleanupError("Command process cleanup status exceeds the size limit")
            chunks.append(chunk)
        if not chunks:
            return None
        data = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommandCleanupError("Command process cleanup status could not be read") from exc
    finally:
        _close_fd(status_fd)
    nonce = data.get("nonce")
    target_started = data.get("target_started")
    start_succeeded = data.get("start_succeeded")
    root_exit_code = data.get("root_exit_code")
    cleanup_succeeded = data.get("cleanup_succeeded")
    if (
        type(nonce) is not str
        or type(target_started) is not bool
        or type(start_succeeded) is not bool
        or (root_exit_code is not None and type(root_exit_code) is not int)
        or type(cleanup_succeeded) is not bool
    ):
        raise CommandCleanupError("Command process cleanup status is invalid")
    return _SupervisorStatus(
        nonce=nonce,
        target_started=target_started,
        start_succeeded=start_succeeded,
        root_exit_code=root_exit_code,
        cleanup_succeeded=cleanup_succeeded,
    )


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _cleanup_private_temp(path: Path | None, directory: Path | None) -> None:
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass
        try:
            path.with_name(f"{path.name}.tmp").unlink()
        except OSError:
            pass
    if directory is not None:
        shutil.rmtree(directory, ignore_errors=True)


async def _read_stream_limited(stream: asyncio.StreamReader | None) -> CapturedStream:
    if stream is None:
        return CapturedStream(b"", 0, False)
    chunks: list[bytes] = []
    size = 0
    total_bytes = 0
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        total_bytes = min(total_bytes + len(chunk), MAX_REPORTED_STREAM_BYTES)
        previous_size = size
        if size < MAX_OUTPUT_BYTES:
            remaining = MAX_OUTPUT_BYTES - size
            chunks.append(chunk[:remaining])
            size += min(len(chunk), remaining)
        if previous_size + len(chunk) > MAX_OUTPUT_BYTES:
            truncated = True
    return CapturedStream(b"".join(chunks), total_bytes, truncated)


async def _finish_reader(task: asyncio.Task[CapturedStream] | None) -> CapturedStream:
    if task is None:
        return CapturedStream(b"", 0, False)
    try:
        return await asyncio.wait_for(task, timeout=COMMAND_CLEANUP_GRACE_SECONDS)
    except asyncio.TimeoutError:
        task.cancel()
        return CapturedStream(b"", 0, True)
    except asyncio.CancelledError:
        raise
    except Exception:
        return CapturedStream(b"", 0, True)


def _resolve_profile_cwd(profile: CommandProfile) -> Path:
    try:
        cwd = resolve_workspace_path(profile.workspace, profile.cwd_config, expected_type="directory")
    except WorkspaceError as exc:
        raise CommandConfigError("Command profile cwd is invalid") from exc
    return cwd


def _decode_output(raw: bytes) -> tuple[str, bool]:
    text = raw.decode("utf-8", errors="replace")
    replaced = "\ufffd" in text
    return text, replaced


def _minimum_command_output(profile: CommandProfile) -> dict[str, Any]:
    return {
        "profile": profile.profile_id,
        "cwd": profile.cwd_relative,
        "exit_code": -2_147_483_648,
        "timed_out": False,
        "duration_ms": 999_999_999,
        "stdout": "",
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_bytes": MAX_REPORTED_STREAM_BYTES,
        "stderr_bytes": MAX_REPORTED_STREAM_BYTES,
        "encoding_replaced": False,
    }


def _command_result_size(output: dict[str, Any]) -> int:
    payload = {"ok": True, "output": output}
    return len(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _compact_command_output(output: dict[str, Any], *, max_result_bytes: int) -> dict[str, Any]:
    if _command_result_size(output) <= max_result_bytes:
        return output
    stdout = output["stdout"]
    stderr = output["stderr"]
    total_chars = len(stdout) + len(stderr)

    def candidate(keep_chars: int) -> dict[str, Any]:
        stdout_keep = 0 if total_chars == 0 else (keep_chars * len(stdout)) // total_chars
        stderr_keep = keep_chars - stdout_keep
        compacted = dict(output)
        compacted["stdout"] = stdout[:stdout_keep]
        compacted["stderr"] = stderr[:stderr_keep]
        compacted["stdout_truncated"] = output["stdout_truncated"] or stdout_keep < len(stdout)
        compacted["stderr_truncated"] = output["stderr_truncated"] or stderr_keep < len(stderr)
        return compacted

    smallest = candidate(0)
    if _command_result_size(smallest) > max_result_bytes:
        raise CommandLimitError("Command result budget is too small")
    low = 0
    high = total_chars
    best = smallest
    while low <= high:
        middle = (low + high) // 2
        current = candidate(middle)
        if _command_result_size(current) <= max_result_bytes:
            best = current
            low = middle + 1
        else:
            high = middle - 1
    return best


async def _cleanup_process(
    process: asyncio.subprocess.Process,
    guard: _ProcessTreeGuard | None,
    stdout_task: asyncio.Task[tuple[bytes, bool]] | None,
    stderr_task: asyncio.Task[tuple[bytes, bool]] | None,
    *,
    force: bool,
) -> _CleanupResult:
    root_reaped = True
    tree_terminated = True
    if force:
        tree_terminated = _terminate_process_tree(process, guard, soft=True) and tree_terminated
        soft_wait = COMMAND_CLEANUP_GRACE_SECONDS
        if os.name != "nt":
            soft_wait = COMMAND_POSIX_SUPERVISOR_CLEANUP_SECONDS
        await _wait_for_process(process, timeout=soft_wait)
        if process.returncode is None:
            tree_terminated = _terminate_process_tree(process, guard, soft=False) and tree_terminated
            root_reaped = await _wait_for_process(process)
    stdout = await _finish_reader(stdout_task)
    stderr = await _finish_reader(stderr_task)
    if guard is not None:
        tree_terminated = guard.close() and tree_terminated
    return _CleanupResult(stdout, stderr, root_reaped, tree_terminated)


async def _wait_for_process(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = COMMAND_CLEANUP_GRACE_SECONDS,
) -> bool:
    if process.returncode is not None:
        return True
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


def _terminate_process_tree(process: asyncio.subprocess.Process, guard: _ProcessTreeGuard | None, *, soft: bool) -> bool:
    try:
        if guard is not None:
            if soft:
                return guard.terminate()
            else:
                return guard.kill()
        elif os.name == "nt":
            _kill_windows_process_tree(process.pid)
        else:
            os.killpg(process.pid, signal.SIGTERM if soft else signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except OSError:
                return False
        return False
    return True


def _kill_windows_process_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _signal_pid(pid: int, sig: signal.Signals) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return True
    except OSError:
        return False


class _ProcessTreeGuard:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._job: _WindowsJob | None = None
        self._closed = False
        if os.name == "nt":
            self._job = _WindowsJob(process)

    def terminate(self) -> bool:
        if self._closed:
            return True
        if self._job is not None:
            return self.close()
        return _signal_pid(self._process.pid, signal.SIGTERM)

    def kill(self) -> bool:
        if self._closed:
            return True
        if self._job is not None:
            return self.close()
        return _signal_pid(self._process.pid, signal.SIGKILL)

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        if self._job is None:
            return True
        return self._job.close()

class _WindowsJob:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._handle = _create_kill_on_close_job()
        popen = process._transport.get_extra_info("subprocess")  # type: ignore[attr-defined]
        if not _KERNEL32.AssignProcessToJobObject(self._handle, wintypes.HANDLE(popen._handle)):  # type: ignore[attr-defined]
            _KERNEL32.CloseHandle(self._handle)
            self._handle = None
            raise OSError("Could not assign process to Windows job")

    def close(self) -> bool:
        if self._handle is not None:
            if not _KERNEL32.CloseHandle(self._handle):
                self._handle = None
                return False
            self._handle = None
        return True


if os.name == "nt":
    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
    _KERNEL32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
else:
    _KERNEL32 = None


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _create_kill_on_close_job() -> Any:
    if _KERNEL32 is None:
        raise OSError("Windows jobs are not available")
    handle = _KERNEL32.CreateJobObjectW(None, None)
    if not handle:
        raise OSError("Could not create Windows job")
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00002000
    if not _KERNEL32.SetInformationJobObject(
        handle,
        9,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        _KERNEL32.CloseHandle(handle)
        raise OSError("Could not configure Windows job")
    return handle
@dataclass(frozen=True, slots=True)
class CapturedStream:
    content: bytes
    total_bytes: int
    truncated: bool
