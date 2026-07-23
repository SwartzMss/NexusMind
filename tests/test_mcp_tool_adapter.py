import asyncio
from dataclasses import dataclass

import pytest

from nexusmind.mcp.errors import MCPDiscoveryError, MCPToolCallError
from nexusmind.mcp.tool_adapter import MCPToolAdapter


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
class ImageBlock:
    data: str
    mimeType: str
    type: str = "image"


@dataclass
class Result:
    content: list
    structuredContent: object = None
    isError: bool = False


class FakeMCPClient:
    def __init__(self, result):
        self.result = result
        self.called_with = None

    async def call_tool(self, name, arguments):
        self.called_with = (name, arguments)
        return self.result


def test_adapter_definition_has_no_mcp_sdk_type_and_invokes_remote_name() -> None:
    client = FakeMCPClient(Result(content=[TextBlock("hello")], structuredContent={"ok": True}))
    adapter = MCPToolAdapter(
        client,
        "demo",
        RemoteTool(name="admin.tools/list", description="Echo", inputSchema={"type": "object", "properties": {}}),
    )

    result = asyncio.run(adapter.invoke({"text": "hello"}))

    assert adapter.definition.name.startswith("demo__admin_tools_list_")
    assert client.called_with == ("admin.tools/list", {"text": "hello"})
    assert result == {"structured_content": {"ok": True}, "content": [{"type": "text", "text": "hello"}]}


def test_adapter_summarizes_binary_content() -> None:
    client = FakeMCPClient(Result(content=[ImageBlock(data="abcd", mimeType="image/png")]))
    adapter = MCPToolAdapter(client, "demo", RemoteTool(name="image", inputSchema={"type": "object"}))

    result = asyncio.run(adapter.invoke({}))

    assert result["content"] == [{"type": "image", "mime_type": "image/png", "size": 4}]


def test_adapter_rejects_invalid_remote_schema() -> None:
    with pytest.raises(MCPDiscoveryError):
        MCPToolAdapter(object(), "demo", RemoteTool(name="bad", inputSchema={"type": "array"}))


def test_adapter_converts_remote_tool_error_to_controlled_exception() -> None:
    client = FakeMCPClient(Result(content=[TextBlock("secret sk-live-secret")], isError=True))
    adapter = MCPToolAdapter(client, "demo", RemoteTool(name="echo", inputSchema={"type": "object"}))

    with pytest.raises(MCPToolCallError) as exc:
        asyncio.run(adapter.invoke({"token": "sk-live-secret"}))

    assert "sk-live-secret" not in str(exc.value)

