from pathlib import Path
import asyncio

from nexusmind import cli
from nexusmind.config import ConfigError, ModelConfig
from nexusmind.mcp import MCPStdioServerConfig
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType


def _write_skill(root: Path, name: str = "review", allowed_tools: str = '["builtin:echo"]') -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.toml").write_text(
        f"""
schema_version = 1
name = "{name}"
description = "Review code"
instructions_file = "instructions.md"
allowed_tools = {allowed_tools}
""".strip(),
        encoding="utf-8",
    )
    (skill_dir / "instructions.md").write_text("System instructions", encoding="utf-8")
    return skill_dir


class StubModel:
    instances = []

    def __init__(self, config):
        self.config = config
        self.messages = None
        self.tools = None
        StubModel.instances.append(self)

    async def stream(self, messages, tools=None):
        self.messages = list(messages)
        self.tools = tools
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="ok")
        yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")


class CancelModel:
    def __init__(self, config):
        self.config = config

    async def stream(self, messages, tools=None):
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        raise asyncio.CancelledError()


def test_skill_list_and_show_do_not_print_instructions(tmp_path, capsys) -> None:
    _write_skill(tmp_path, "zeta")
    _write_skill(tmp_path, "alpha", allowed_tools="[]")

    assert cli.main(["skill", "list", "--skills-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0].startswith("alpha\t")
    assert "System instructions" not in captured.out

    assert cli.main(["skill", "show", "zeta", "--skills-dir", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "allowed_tools\tbuiltin:echo" in captured.out
    assert "System instructions" not in captured.out


def test_skill_run_sends_system_prompt_and_allowlisted_tools(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path)
    StubModel.instances = []
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", StubModel)

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "hello"]) == 0

    captured = capsys.readouterr()
    model = StubModel.instances[0]
    assert captured.out == "ok\n"
    assert [message.role.value for message in model.messages[:2]] == ["system", "user"]
    assert model.messages[0].content == "System instructions"
    assert model.messages[1].content == "hello"
    assert [tool.name for tool in model.tools] == ["echo"]


def test_skill_run_config_error_does_not_resolve_mcp(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path, allowed_tools='["mcp:demo:remote_echo"]')
    calls = {"load_mcp": 0, "client": 0}
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: (_ for _ in ()).throw(ConfigError("missing config")))
    monkeypatch.setattr(cli, "load_mcp_server_config", lambda path, server: calls.__setitem__("load_mcp", 1))
    monkeypatch.setattr(cli, "MCPStdioClient", lambda config: calls.__setitem__("client", 1))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "--mcp-server", "demo", "hi"]) == 2

    captured = capsys.readouterr()
    assert "Configuration error" in captured.err
    assert calls == {"load_mcp": 0, "client": 0}


def test_skill_run_rejects_unknown_option(capsys, tmp_path) -> None:
    _write_skill(tmp_path)

    try:
        cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--unkown-option", "value"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err


def test_skill_run_rejects_unknown_option_before_separator(capsys, tmp_path) -> None:
    _write_skill(tmp_path)

    try:
        cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--unknown", "value", "--"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err


def test_skill_run_accepts_dash_prefixed_message_after_separator(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path)
    StubModel.instances = []
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", StubModel)

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--", "--special message"]) == 0

    model = StubModel.instances[0]
    assert model.messages[1].content == "--special message"


def test_skill_run_mcp_server_mismatch_does_not_start_client(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path, allowed_tools='["mcp:other:remote_echo"]')
    calls = {"client": 0}
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(
        cli,
        "load_mcp_server_config",
        lambda path, server: MCPStdioServerConfig(server_id="demo", command="python"),
    )
    monkeypatch.setattr(cli, "MCPStdioClient", lambda config: calls.__setitem__("client", 1))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "--mcp-server", "demo", "hi"]) == 2

    captured = capsys.readouterr()
    assert "different server_id" in captured.err
    assert calls == {"client": 0}


def test_skill_run_missing_builtin_reference_does_not_start_mcp(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path, allowed_tools='["builtin:missing", "mcp:demo:remote_echo"]')
    calls = {"client": 0}
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(
        cli,
        "load_mcp_server_config",
        lambda path, server: MCPStdioServerConfig(server_id="demo", command="python"),
    )
    monkeypatch.setattr(cli, "MCPStdioClient", lambda config: calls.__setitem__("client", 1))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "--mcp-server", "demo", "hi"]) == 2

    captured = capsys.readouterr()
    assert "builtin:missing" in captured.err
    assert calls == {"client": 0}


def test_skill_run_cleanup_failure_does_not_cover_cancel(monkeypatch, tmp_path) -> None:
    _write_skill(tmp_path, allowed_tools='["mcp:demo:remote_echo"]')

    class ExitFailingClient:
        entered = False

        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            type(self).entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            raise RuntimeError("cleanup failed")

        async def list_tools(self):
            from nexusmind.mcp import MCPRemoteTool

            return [
                MCPRemoteTool(
                    name="remote_echo",
                    description="Remote echo",
                    input_schema={"type": "object", "properties": {}},
                )
            ]

    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(
        cli,
        "load_mcp_server_config",
        lambda path, server: MCPStdioServerConfig(server_id="demo", command="python"),
    )
    monkeypatch.setattr(cli, "MCPStdioClient", ExitFailingClient)
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", CancelModel)

    try:
        cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "--mcp-server", "demo", "hi"])
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert ExitFailingClient.entered is True
