import asyncio
from io import StringIO

from nexusmind import cli
from nexusmind.config import ConfigError, ModelConfig
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.policy import ApprovalDecision, ApprovalRequest
from nexusmind.tools import ToolCall, ToolDefinition, ToolRegistry, ToolRiskLevel


def test_cli_outputs_streaming_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))

    class StubModel:
        def __init__(self, config):
            self.config = config

    class StubRuntime:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        async def stream_user_message(self, message, tools=None):
            yield RuntimeEvent(RuntimeEventType.RUN_STARTED)
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="hello")
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text=" world")
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", StubModel)
    monkeypatch.setattr(cli, "ChatRuntime", StubRuntime)

    assert cli.main(["chat", "hi"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "hello world\n"
    assert captured.err == ""


def test_cli_returns_nonzero_on_model_failure(monkeypatch, capsys) -> None:
    secret = "sk-test-secret"
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", secret, "fake"))

    class StubModel:
        def __init__(self, config):
            self.config = config

    class StubRuntime:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        async def stream_user_message(self, message, tools=None):
            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error="provider rejected request")

    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", StubModel)
    monkeypatch.setattr(cli, "ChatRuntime", StubRuntime)

    assert cli.main(["chat", "hi"]) == 1

    captured = capsys.readouterr()
    assert secret not in captured.err
    assert "provider rejected request" in captured.err


def test_cli_returns_nonzero_on_missing_config(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: (_ for _ in ()).throw(ConfigError("missing config")))
    monkeypatch.delenv("NEXUSMIND_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("NEXUSMIND_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("NEXUSMIND_MODEL_NAME", raising=False)

    assert cli.main(["chat", "hi"]) == 2

    captured = capsys.readouterr()
    assert "Configuration error" in captured.err


class _RecordingWriteTool:
    def __init__(self) -> None:
        self.calls = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            risk_level=ToolRiskLevel.LOCAL_WRITE,
        )

    async def invoke(self, arguments):
        self.calls.append(arguments)
        return {"written": True}


class _ToolCallThenStopModel:
    instances = []

    def __init__(self, config):
        self.config = config
        self.messages_by_turn = []
        self.tools_by_turn = []
        _ToolCallThenStopModel.instances.append(self)

    async def stream(self, messages, tools=None):
        self.messages_by_turn.append(list(messages))
        self.tools_by_turn.append(tools)
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        if len(self.messages_by_turn) == 1:
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="write_file", arguments={}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
        else:
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")


class _ApprovalDemoModel:
    instances = []

    def __init__(self, config):
        self.config = config
        self.messages_by_turn = []
        self.tools_by_turn = []
        _ApprovalDemoModel.instances.append(self)

    async def stream(self, messages, tools=None):
        self.messages_by_turn.append(list(messages))
        self.tools_by_turn.append(tools)
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        if len(self.messages_by_turn) == 1:
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="approval_demo", arguments={"message": "ok"}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
        else:
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")


def test_cli_chat_approval_allow_executes_tool_and_feeds_result(monkeypatch, capsys) -> None:
    tool = _RecordingWriteTool()

    def registry_factory():
        registry = ToolRegistry()
        registry.register(tool)
        return registry

    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", _ToolCallThenStopModel)
    monkeypatch.setattr(cli, "_build_builtin_tool_registry", registry_factory)
    monkeypatch.setattr(cli.sys, "stdin", StringIO("a\n"))
    _ToolCallThenStopModel.instances = []

    assert cli.main(["chat", "write"]) == 0

    captured = capsys.readouterr()
    model = _ToolCallThenStopModel.instances[0]
    assert tool.calls == [{}]
    assert model.tools_by_turn[0][0].name == "write_file"
    assert model.messages_by_turn[1][-1].content == '{"ok":true,"output":{"written":true}}'
    assert "Tool approval required" in captured.err
    assert captured.out == "done\n"


def test_cli_chat_default_registry_can_trigger_real_approval_demo(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", _ApprovalDemoModel)
    monkeypatch.setattr(cli.sys, "stdin", StringIO("a\n"))
    _ApprovalDemoModel.instances = []

    assert cli.main(["chat", "approval demo"]) == 0

    captured = capsys.readouterr()
    model = _ApprovalDemoModel.instances[0]
    assert [tool.name for tool in model.tools_by_turn[0]] == ["approval_demo", "echo"]
    assert model.messages_by_turn[1][-1].content == '{"ok":true,"output":{"message":"ok"}}'
    assert "Tool approval required" in captured.err
    assert captured.out == "done\n"


def test_cli_chat_approval_deny_does_not_execute_tool(monkeypatch, capsys) -> None:
    tool = _RecordingWriteTool()

    def registry_factory():
        registry = ToolRegistry()
        registry.register(tool)
        return registry

    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", _ToolCallThenStopModel)
    monkeypatch.setattr(cli, "_build_builtin_tool_registry", registry_factory)
    monkeypatch.setattr(cli.sys, "stdin", StringIO("d\n"))
    _ToolCallThenStopModel.instances = []

    assert cli.main(["chat", "write"]) == 0

    captured = capsys.readouterr()
    model = _ToolCallThenStopModel.instances[0]
    assert tool.calls == []
    assert '"code":"PERMISSION_DENIED"' in model.messages_by_turn[1][-1].content
    assert "Tool approval required" in captured.err
    assert captured.out == "done\n"


def test_cli_approval_provider_allows_once_for_a() -> None:
    provider = cli.CLIApprovalProvider(input_stream=StringIO("a\n"), output_stream=StringIO())
    request = ApprovalRequest("req_1", "call_1", "write_file", ToolRiskLevel.LOCAL_WRITE, "Write file ./report.md")

    decision = asyncio.run(provider.request(request))

    assert decision == ApprovalDecision.ALLOW_ONCE


def test_cli_approval_provider_denies_by_default_for_invalid_or_eof() -> None:
    request = ApprovalRequest("req_1", "call_1", "write_file", ToolRiskLevel.LOCAL_WRITE, "Write file ./report.md")

    invalid = asyncio.run(cli.CLIApprovalProvider(input_stream=StringIO("x\n"), output_stream=StringIO()).request(request))
    eof = asyncio.run(cli.CLIApprovalProvider(input_stream=StringIO(""), output_stream=StringIO()).request(request))

    assert invalid == ApprovalDecision.DENY
    assert eof == ApprovalDecision.DENY


def test_cli_approval_provider_does_not_print_arguments() -> None:
    output = StringIO()
    provider = cli.CLIApprovalProvider(input_stream=StringIO("d\n"), output_stream=output)
    request = ApprovalRequest("req_1", "call_1", "write_file", ToolRiskLevel.LOCAL_WRITE, "Write file ./safe.md")

    decision = asyncio.run(provider.request(request))

    assert decision == ApprovalDecision.DENY
    assert "Write file ./safe.md" in output.getvalue()

