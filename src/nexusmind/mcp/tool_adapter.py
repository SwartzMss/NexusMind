from __future__ import annotations

from copy import deepcopy
from typing import Any

from nexusmind.mcp.client import MCPClient, is_mcp_tool_error, mcp_tool_to_definition, normalize_call_tool_result
from nexusmind.mcp.errors import MCPDiscoveryError, MCPToolCallError
from nexusmind.mcp.naming import mcp_tool_local_name
from nexusmind.tools.base import Tool
from nexusmind.tools.contracts import ToolDefinition
from nexusmind.tools.registry import ToolRegistry


class MCPToolAdapter:
    def __init__(self, client: MCPClient, server_id: str, remote_tool: Any) -> None:
        remote_name = getattr(remote_tool, "name", None)
        if not isinstance(remote_name, str) or not remote_name:
            raise MCPDiscoveryError("MCP tool is missing a valid name")
        self.server_id = server_id
        self.remote_name = remote_name
        self.local_name = mcp_tool_local_name(server_id, remote_name)
        self._client = client
        self._definition = mcp_tool_to_definition(self.local_name, remote_tool)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._definition.name,
            description=self._definition.description,
            input_schema=deepcopy(self._definition.input_schema),
        )

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        result = await self._client.call_tool(self.remote_name, arguments)
        if is_mcp_tool_error(result):
            raise MCPToolCallError("MCP tool returned an error")
        return normalize_call_tool_result(result)


async def register_mcp_tools(client: MCPClient, server_id: str, registry: ToolRegistry) -> list[ToolDefinition]:
    remote_tools = await client.list_tools()
    adapters: list[Tool] = [MCPToolAdapter(client, server_id, remote_tool) for remote_tool in remote_tools]
    registry.register_many(adapters)
    return [adapter.definition for adapter in adapters]

