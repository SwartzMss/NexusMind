import asyncio
from dataclasses import dataclass

import pytest

from nexusmind.mcp.errors import MCPDiscoveryError, MCPToolCallError
from nexusmind.mcp.client import MCPRemoteTool, _mcp_tool_to_remote_tool
from nexusmind.mcp.tool_adapter import MCPToolAdapter
from nexusmind.tools import (
    ToolCall,
    ToolExecutor,
    ToolRegistry,
    ToolResultBudget,
    ToolResultBudgetError,
    ToolRiskLevel,
)


@dataclass
class RemoteTool:
    name: str
    inputSchema: dict
    description: str | None = None
    annotations: object | None = None


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
        MCPRemoteTool(name="admin.tools/list", description="Echo", input_schema={"type": "object", "properties": {}}),
    )

    result = asyncio.run(adapter.invoke({"text": "hello"}))

    assert adapter.definition.name.startswith("demo__admin_tools_list_")
    assert client.called_with == ("admin.tools/list", {"text": "hello"})
    assert result == {
        "structured_content": {"ok": True},
        "content": [{"type": "text", "text": "hello"}],
        "truncated": False,
    }


def test_mcp_tool_is_not_called_when_result_budget_is_too_small() -> None:
    client = FakeMCPClient(Result(content=[TextBlock("created")]))
    adapter = MCPToolAdapter(
        client,
        "demo",
        MCPRemoteTool(name="create_issue", description=None, input_schema={"type": "object"}),
    )
    registry = ToolRegistry()
    registry.register(adapter)
    executor = ToolExecutor(registry)
    call = ToolCall(id="1", name=adapter.definition.name, arguments={})
    requirements = executor.result_requirements(call)

    with pytest.raises(ToolResultBudgetError, match="budget is too small"):
        asyncio.run(
            executor.execute_with_result_budget(
                call,
                result_budget=ToolResultBudget(
                    max_bytes=requirements.min_bytes - 1,
                    max_nodes=requirements.min_nodes,
                    max_depth=requirements.min_depth,
                ),
            )
        )

    assert adapter.definition.risk_level is ToolRiskLevel.UNSPECIFIED
    assert client.called_with is None


def test_mcp_tool_compacts_large_result_to_runtime_budget() -> None:
    client = FakeMCPClient(Result(content=[TextBlock("x" * 10000)], structuredContent={"items": ["y" * 10000]}))
    adapter = MCPToolAdapter(
        client,
        "demo",
        MCPRemoteTool(name="create_issue", description=None, input_schema={"type": "object"}),
    )
    registry = ToolRegistry()
    registry.register(adapter)
    executor = ToolExecutor(registry)
    call = ToolCall(id="1", name=adapter.definition.name, arguments={})
    requirements = executor.result_requirements(call)
    budget = ToolResultBudget(
        max_bytes=requirements.min_bytes + 128,
        max_nodes=requirements.min_nodes + 4,
        max_depth=requirements.min_depth + 2,
    )

    result = asyncio.run(executor.execute_with_result_budget(call, result_budget=budget))

    assert result.error is None
    assert result.output["truncated"] is True
    assert result.output["structured_content"] is None
    assert client.called_with == ("create_issue", {})


def test_remote_read_only_hint_does_not_change_security_risk() -> None:
    annotations = type("Annotations", (), {"readOnlyHint": True})()
    remote_tool = _mcp_tool_to_remote_tool(
        RemoteTool(name="read", inputSchema={"type": "object"}, annotations=annotations)
    )
    adapter = MCPToolAdapter(
        FakeMCPClient(Result(content=[])),
        "demo",
        remote_tool,
    )

    assert adapter.definition.risk_level is ToolRiskLevel.UNSPECIFIED


def test_adapter_summarizes_binary_content() -> None:
    client = FakeMCPClient(Result(content=[ImageBlock(data="abcd", mimeType="image/png")]))
    adapter = MCPToolAdapter(client, "demo", MCPRemoteTool(name="image", description=None, input_schema={"type": "object"}))

    result = asyncio.run(adapter.invoke({}))

    assert result["content"] == [{"type": "image", "mime_type": "image/png", "size": 4}]


def test_adapter_rejects_invalid_remote_schema() -> None:
    with pytest.raises(MCPDiscoveryError):
        MCPToolAdapter(object(), "demo", MCPRemoteTool(name="bad", description=None, input_schema={"type": "array"}))


def test_adapter_converts_remote_tool_error_to_controlled_exception() -> None:
    client = FakeMCPClient(Result(content=[TextBlock("secret sk-live-secret")], isError=True))
    adapter = MCPToolAdapter(client, "demo", MCPRemoteTool(name="echo", description=None, input_schema={"type": "object"}))

    with pytest.raises(MCPToolCallError) as exc:
        asyncio.run(adapter.invoke({"token": "sk-live-secret"}))

    assert "sk-live-secret" not in str(exc.value)
