from __future__ import annotations

from copy import deepcopy
from typing import Any

from nexusmind.mcp.client import MCPClient, MCPRemoteTool, is_mcp_tool_error, mcp_tool_to_definition, normalize_call_tool_result
from nexusmind.mcp.errors import MCPToolCallError
from nexusmind.mcp.naming import mcp_tool_local_name
from nexusmind.tools.base import Tool
from nexusmind.tools.contracts import (
    ToolDefinition,
    ToolResultBudget,
    ToolResultRequirements,
    json_result_requirements,
)
from nexusmind.tools.registry import ToolRegistry


class MCPToolAdapter:
    def __init__(self, client: MCPClient, server_id: str, remote_tool: MCPRemoteTool) -> None:
        remote_name = remote_tool.name
        self.server_id = server_id
        self.remote_name = remote_name
        self.local_name = mcp_tool_local_name(server_id, remote_name)
        self._client = client
        self._definition = mcp_tool_to_definition(self.local_name, remote_tool)

    @property
    def definition(self) -> ToolDefinition:
        return deepcopy(self._definition)

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        result = await self._client.call_tool(self.remote_name, arguments)
        if is_mcp_tool_error(result):
            raise MCPToolCallError("MCP tool returned an error")
        return normalize_call_tool_result(result)

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        return _mcp_output_requirements(
            {"structured_content": None, "content": [], "truncated": True}
        )

    async def invoke_with_result_budget(
        self,
        arguments: dict[str, Any],
        *,
        result_budget: ToolResultBudget,
    ) -> Any:
        result = await self._client.call_tool(self.remote_name, arguments)
        if is_mcp_tool_error(result):
            raise MCPToolCallError("MCP tool returned an error")
        output = normalize_call_tool_result(result)
        if _mcp_output_fits(output, result_budget):
            return output
        output["structured_content"] = None
        output["truncated"] = True
        while output["content"]:
            if _mcp_output_fits(output, result_budget):
                return output
            block = output["content"][-1]
            text = block.get("text") if isinstance(block, dict) else None
            if isinstance(text, str) and text:
                low, high = 0, len(text)
                while low < high:
                    middle = (low + high + 1) // 2
                    block["text"] = text[:middle]
                    if _mcp_output_fits(output, result_budget):
                        low = middle
                    else:
                        high = middle - 1
                block["text"] = text[:low]
                if _mcp_output_fits(output, result_budget):
                    return output
            output["content"].pop()
        if not _mcp_output_fits(output, result_budget):
            raise MCPToolCallError("MCP tool result budget is too small")
        return output


async def register_mcp_tools(client: MCPClient, server_id: str, registry: ToolRegistry) -> list[ToolDefinition]:
    remote_tools = await client.list_tools()
    adapters: list[Tool] = [MCPToolAdapter(client, server_id, remote_tool) for remote_tool in remote_tools]
    registry.register_many(adapters)
    return [adapter.definition for adapter in adapters]


def _mcp_output_requirements(output: dict[str, Any]) -> ToolResultRequirements:
    return json_result_requirements({"ok": True, "output": output})


def _mcp_output_fits(output: dict[str, Any], budget: ToolResultBudget) -> bool:
    return budget.satisfies(_mcp_output_requirements(output))
