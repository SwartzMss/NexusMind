import asyncio
from io import StringIO

from nexusmind import cli
from nexusmind.config import ConfigError, ModelConfig
from nexusmind.runtime.policy import ApprovalDecision, ApprovalRequest
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.tools import ToolRiskLevel


def test_cli_outputs_streaming_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))

    class StubModel:
        def __init__(self, config):
            self.config = config

    class StubRuntime:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

        async def stream_user_message(self, message):
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

        async def stream_user_message(self, message):
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

