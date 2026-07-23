import json
from dataclasses import dataclass

from nexusmind import cli
from nexusmind.mcp.client import MCPRemoteTool


@dataclass
class RemoteTool:
    name: str
    inputSchema: dict
    description: str | None = None


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class Result:
    content: list
    structuredContent: object = None
    isError: bool = False


class FakeClient:
    entered = 0

    def __init__(self, config) -> None:
        self.config = config

    async def __aenter__(self):
        type(self).entered += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def list_tools(self):
        return [
            MCPRemoteTool(
                name="echo",
                description="Echo text",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            )
        ]

    async def call_tool(self, name, arguments):
        return Result(content=[TextBlock(arguments["text"])], structuredContent={"text": arguments["text"]})


def _config_file(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"servers": {"demo": {"transport": "stdio", "command": "python"}}}),
        encoding="utf-8",
    )
    return path


def test_mcp_tools_outputs_discovered_tools(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "MCPStdioClient", FakeClient)
    path = _config_file(tmp_path)

    assert cli.main(["mcp", "tools", "--config", str(path), "--server", "demo"]) == 0

    captured = capsys.readouterr()
    line = json.loads(captured.out)
    assert line["name"].startswith("demo__echo_")
    assert line["remote_name"] == "echo"
    assert captured.err == ""


def test_mcp_tools_sanitizes_untrusted_terminal_output(monkeypatch, tmp_path, capsys) -> None:
    class UnsafeClient(FakeClient):
        async def list_tools(self):
            return [
                MCPRemoteTool(
                    name="bad\nname\t\x1b]0;fake title\x07",
                    description="line1\nFAKE_TOOL\tadmin\t\x1b[2J" + ("x" * 200),
                    input_schema={"type": "object", "properties": {}},
                )
            ]

    monkeypatch.setattr(cli, "MCPStdioClient", UnsafeClient)
    path = _config_file(tmp_path)

    assert cli.main(["mcp", "tools", "--config", str(path), "--server", "demo"]) == 0

    captured = capsys.readouterr()
    assert "\nFAKE_TOOL" not in captured.out
    assert "\t" not in captured.out
    assert "\x1b" not in captured.out
    assert "\x07" not in captured.out
    line = json.loads(captured.out)
    assert len(line["description"]) <= 160


def test_mcp_call_outputs_result_through_executor(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "MCPStdioClient", FakeClient)
    path = _config_file(tmp_path)

    assert (
        cli.main(
            [
                "mcp",
                "call",
                "--config",
                str(path),
                "--server",
                "demo",
                "--tool",
                "demo__echo_290c9db7d5",
                "--arguments",
                '{"text":"hello"}',
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert '"structured_content": {"text": "hello"}' in captured.out
    assert captured.err == ""


def test_mcp_bad_config_returns_nonzero(tmp_path, capsys) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"demo": {"transport": "stdio"}}}), encoding="utf-8")

    assert cli.main(["mcp", "tools", "--config", str(path), "--server", "demo"]) == 1

    captured = capsys.readouterr()
    assert "MCP error" in captured.err


def test_mcp_call_invalid_json_returns_nonzero(monkeypatch, tmp_path, capsys) -> None:
    FakeClient.entered = 0
    monkeypatch.setattr(cli, "MCPStdioClient", FakeClient)
    path = _config_file(tmp_path)

    assert cli.main(["mcp", "call", "--config", str(path), "--server", "demo", "--tool", "echo", "--arguments", "{bad"]) == 2

    captured = capsys.readouterr()
    assert "Invalid JSON arguments" in captured.err
    assert FakeClient.entered == 0


def test_mcp_call_non_object_arguments_do_not_start_client(monkeypatch, tmp_path, capsys) -> None:
    FakeClient.entered = 0
    monkeypatch.setattr(cli, "MCPStdioClient", FakeClient)
    path = _config_file(tmp_path)

    assert cli.main(["mcp", "call", "--config", str(path), "--server", "demo", "--tool", "echo", "--arguments", "[]"]) == 2

    captured = capsys.readouterr()
    assert "Tool arguments must be a JSON object" in captured.err
    assert FakeClient.entered == 0
