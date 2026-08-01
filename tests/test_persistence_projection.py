from nexusmind.cli import project_runtime_event
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.tools.contracts import ToolResult

def test_project_tool_result_event():
    result = ToolResult(call_id="call-1", name="echo", output={"text": "hello"})
    payload = project_runtime_event(RuntimeEvent(RuntimeEventType.TOOL_RESULT, tool_result=result))
    assert payload["call_id"] == "call-1"
    assert payload["tool_name"] == "echo"
    assert payload["ok"] is True
