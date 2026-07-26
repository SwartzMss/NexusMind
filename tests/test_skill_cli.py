from pathlib import Path
import asyncio
import json
from io import StringIO
import sys

from nexusmind import cli
from nexusmind.config import ConfigError, ModelConfig
from nexusmind.mcp import MCPRemoteTool, MCPStdioServerConfig, mcp_tool_local_name
from nexusmind.mcp.tool_adapter import register_mcp_tools
from nexusmind.mcp import MAX_MCP_CLIENTS_PER_GROUP
from nexusmind.tools import ToolCall
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


class TwoMCPToolCallsModel:
    instances = []

    def __init__(self, config):
        self.messages_by_turn = []
        self.tools_by_turn = []
        TwoMCPToolCallsModel.instances.append(self)

    async def stream(self, messages, tools=None):
        self.messages_by_turn.append(list(messages))
        self.tools_by_turn.append(tools)
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        if len(self.messages_by_turn) == 1:
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(
                    id="call_1",
                    name=mcp_tool_local_name("alpha", "echo"),
                    arguments={"text": "one"},
                ),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
        elif len(self.messages_by_turn) == 2:
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(
                    id="call_2",
                    name=mcp_tool_local_name("zeta", "echo"),
                    arguments={"text": "two"},
                ),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
        else:
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")


class RecordingRuntime:
    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.messages = []
        self.tools = []
        RecordingRuntime.instances.append(self)

    async def stream_user_message(self, message, system_prompt=None, tools=None):
        self.message = message
        self.system_prompt = system_prompt
        self.tools = tools
        yield RuntimeEvent(RuntimeEventType.RUN_STARTED)
        yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="ok")
        yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)


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
    monkeypatch.setattr(cli, "load_mcp_server_configs", lambda path, server_ids: calls.__setitem__("load_mcp", 1))
    monkeypatch.setattr(cli, "MCPStdioClient", lambda config: calls.__setitem__("client", 1))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "hi"]) == 2

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


def test_skill_run_missing_mcp_config_for_mcp_skill_returns_2(tmp_path, capsys) -> None:
    _write_skill(tmp_path, allowed_tools='["mcp:demo:remote_echo"]')

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "hi"]) == 2

    captured = capsys.readouterr()
    assert "require --mcp-config" in captured.err


def test_skill_run_missing_mcp_server_config_does_not_start_client(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path, allowed_tools='["mcp:other:remote_echo"]')
    calls = {"client": 0}
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(
        cli,
        "load_mcp_server_configs",
        lambda path, server_ids: (_ for _ in ()).throw(cli.MCPError("MCP server not found: other")),
    )
    monkeypatch.setattr(cli, "MCPClientGroup", lambda configs: calls.__setitem__("client", 1))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "hi"]) == 2

    captured = capsys.readouterr()
    assert "MCP server not found" in captured.err
    assert calls == {"client": 0}


def test_skill_run_too_many_mcp_servers_does_not_start_client(monkeypatch, tmp_path, capsys) -> None:
    references = ", ".join(f'"mcp:server{i}:echo"' for i in range(MAX_MCP_CLIENTS_PER_GROUP + 1))
    _write_skill(tmp_path, allowed_tools=f"[{references}]")
    calls = {"client": 0, "config": 0}
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(cli, "load_mcp_server_configs", lambda path, server_ids: calls.__setitem__("config", 1))
    monkeypatch.setattr(cli, "MCPClientGroup", lambda configs: calls.__setitem__("client", 1))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "hi"]) == 2

    captured = capsys.readouterr()
    assert "too many MCP servers" in captured.err
    assert calls == {"client": 0, "config": 0}


def test_skill_run_missing_builtin_reference_does_not_start_mcp(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path, allowed_tools='["builtin:missing", "mcp:demo:remote_echo"]')
    calls = {"client": 0}
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(
        cli,
        "load_mcp_server_config",
        lambda path, server: MCPStdioServerConfig(server_id="demo", command="python"),
    )
    monkeypatch.setattr(cli, "MCPClientGroup", lambda configs: calls.__setitem__("client", 1))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "hi"]) == 2

    captured = capsys.readouterr()
    assert "builtin:missing" in captured.err
    assert calls == {"client": 0}


def test_skill_run_cleanup_failure_does_not_cover_cancel(monkeypatch, tmp_path) -> None:
    _write_skill(tmp_path, allowed_tools='["mcp:demo:remote_echo"]')

    class ExitFailingGroup:
        entered = False

        def __init__(self, configs):
            self.configs = configs

        async def __aenter__(self):
            type(self).entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            raise RuntimeError("cleanup failed")

        async def register_tools(self, registry):
            from nexusmind.mcp import MCPRemoteTool
            from nexusmind.mcp.tool_adapter import register_mcp_tools

            class Client:
                async def list_tools(self):
                    return [
                        MCPRemoteTool(
                            name="remote_echo",
                            description="Remote echo",
                            input_schema={"type": "object", "properties": {}},
                        )
                    ]

                async def call_tool(self, name, arguments):
                    return None

            await register_mcp_tools(Client(), "demo", registry)

    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(
        cli,
        "load_mcp_server_configs",
        lambda path, server_ids: {"demo": MCPStdioServerConfig(server_id="demo", command="python")},
    )
    monkeypatch.setattr(cli, "MCPClientGroup", ExitFailingGroup)
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", CancelModel)

    try:
        cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "hi"])
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert ExitFailingGroup.entered is True


def test_skill_run_loads_multiple_mcp_servers_and_only_advertises_allowlist(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(
        tmp_path,
        allowed_tools='["mcp:filesystem:read_file", "mcp:github:get_pull_request"]',
    )

    class MultiGroup:
        seen_configs = None
        entered = False
        exited = False

        def __init__(self, configs):
            type(self).seen_configs = configs

        async def __aenter__(self):
            type(self).entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            type(self).exited = True

        async def register_tools(self, registry):
            class Client:
                def __init__(self, server_id):
                    self.server_id = server_id

                async def list_tools(self):
                    if self.server_id == "filesystem":
                        return [
                            MCPRemoteTool("read_file", "Read file", {"type": "object", "properties": {}}),
                            MCPRemoteTool("write_file", "Write file", {"type": "object", "properties": {}}),
                        ]
                    return [
                        MCPRemoteTool("get_pull_request", "Get PR", {"type": "object", "properties": {}}),
                        MCPRemoteTool("delete_repo", "Delete repo", {"type": "object", "properties": {}}),
                    ]

                async def call_tool(self, name, arguments):
                    return None

            await register_mcp_tools(Client("filesystem"), "filesystem", registry)
            await register_mcp_tools(Client("github"), "github", registry)

    RecordingRuntime.instances = []
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(
        cli,
        "load_mcp_server_configs",
        lambda path, server_ids: {
            server_id: MCPStdioServerConfig(server_id=server_id, command="python", request_timeout=1)
            for server_id in server_ids
        },
    )
    monkeypatch.setattr(cli, "MCPClientGroup", MultiGroup)
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", StubModel)
    monkeypatch.setattr(cli, "ChatRuntime", RecordingRuntime)

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", "mcp.json", "hi"]) == 0

    runtime = RecordingRuntime.instances[0]
    assert list(MultiGroup.seen_configs) == ["filesystem", "github"]
    assert MultiGroup.entered is True
    assert MultiGroup.exited is True
    assert sorted(tool.name for tool in runtime.tools) == [
        mcp_tool_local_name("filesystem", "read_file"),
        mcp_tool_local_name("github", "get_pull_request"),
    ]
    assert runtime.kwargs["tool_executor"]._timeout == 30.0


def test_skill_run_real_multi_mcp_stdio_tool_loop(monkeypatch, tmp_path, capsys) -> None:
    _write_skill(tmp_path, allowed_tools='["mcp:alpha:echo", "mcp:zeta:echo"]')
    config_path = tmp_path / "mcp.json"
    server_path = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "alpha": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_path)],
                        "request_timeout": 5,
                    },
                    "zeta": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_path)],
                        "request_timeout": 5,
                    },
                    "unused": {
                        "transport": "stdio",
                        "command": "missing-command-that-must-not-start",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    TwoMCPToolCallsModel.instances = []
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "secret", "fake"))
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", TwoMCPToolCallsModel)
    monkeypatch.setattr(cli.sys, "stdin", StringIO("a\na\n"))

    assert cli.main(["skill", "run", "review", "--skills-dir", str(tmp_path), "--mcp-config", str(config_path), "hi"]) == 0

    captured = capsys.readouterr()
    model = TwoMCPToolCallsModel.instances[0]
    first_tool_result = model.messages_by_turn[1][-1].content
    second_tool_result = model.messages_by_turn[2][-1].content
    assert '"text":"one"' in first_tool_result
    assert '"text":"two"' in second_tool_result
    assert sorted(tool.name for tool in model.tools_by_turn[0]) == [
        mcp_tool_local_name("alpha", "echo"),
        mcp_tool_local_name("zeta", "echo"),
    ]
    assert captured.out == "done\n"
