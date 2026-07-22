import asyncio

from nexusmind import cli
from nexusmind.config import ConfigError, ModelConfig
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType


def test_cli_outputs_streaming_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))

    class StubModel:
        def __init__(self, config):
            self.config = config

    class StubRuntime:
        def __init__(self, model):
            self.model = model

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
        def __init__(self, model):
            self.model = model

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

