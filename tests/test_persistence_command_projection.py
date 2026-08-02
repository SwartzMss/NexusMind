from nexusmind.cli import project_runtime_event
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.tools.contracts import ToolResult

def test_project_nonzero_command_result():
    result = ToolResult(call_id="call-1", name="run_command", output={"profile":"tests", "cwd":".", "exit_code":1, "timed_out":False, "duration_ms":1200, "stdout":"1 failed", "stderr":"", "stdout_truncated":False, "stderr_truncated":False})
    payload = project_runtime_event(RuntimeEvent(RuntimeEventType.TOOL_RESULT, tool_result=result))
    assert payload["exit_code"] == 1
    assert payload["timed_out"] is False
    assert payload["stdout_bytes"] == len("1 failed".encode())

def test_project_command_uses_host_total_byte_counts():
    result = ToolResult(call_id="call-1", name="run_command", output={
        "profile": "tests", "cwd": ".", "exit_code": 1, "timed_out": False,
        "duration_ms": 1200, "stdout": "kept", "stderr": "",
        "stdout_bytes": 5 * 1024 * 1024, "stderr_bytes": 17,
        "stdout_truncated": True, "stderr_truncated": False,
    })
    payload = project_runtime_event(RuntimeEvent(RuntimeEventType.TOOL_RESULT, tool_result=result))
    assert payload["stdout_bytes"] == 5 * 1024 * 1024
    assert payload["stderr_bytes"] == 17
