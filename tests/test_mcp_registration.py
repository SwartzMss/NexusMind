import asyncio
from dataclasses import dataclass

import pytest

from nexusmind.mcp.tool_adapter import register_mcp_tools
from nexusmind.mcp.naming import mcp_tool_local_name
from nexusmind.tools import ToolDefinition, ToolRegistry, ToolRegistryError
from nexusmind.tools.builtin import EchoTool


@dataclass
class RemoteTool:
    name: str
    inputSchema: dict
    description: str | None = None


class FakeClient:
    def __init__(self, tools):
        self._tools = tools

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        return None


def test_register_mcp_tools_registers_all_adapters() -> None:
    registry = ToolRegistry()
    definitions = asyncio.run(
        register_mcp_tools(
            FakeClient([RemoteTool("echo", {"type": "object", "properties": {}})]),
            "demo",
            registry,
        )
    )

    assert len(definitions) == 1
    assert registry.contains(definitions[0].name)


def test_register_mcp_tools_rolls_back_on_conflict() -> None:
    registry = ToolRegistry()

    class ConflictingTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name=mcp_tool_local_name("demo", "echo"))

        async def invoke(self, arguments):
            return None

    registry.register(ConflictingTool())

    with pytest.raises(ToolRegistryError):
        asyncio.run(register_mcp_tools(FakeClient([RemoteTool("echo", {"type": "object"})]), "demo", registry))

    assert len(registry.list_definitions()) == 1


def test_mcp_tool_cannot_override_builtin_tool() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    asyncio.run(register_mcp_tools(FakeClient([RemoteTool("echo", {"type": "object"})]), "demo", registry))

    assert registry.contains("echo")
    assert len(registry.list_definitions()) == 2
