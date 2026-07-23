from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import Draft202012Validator

from nexusmind.mcp.errors import MCPConnectionError, MCPDiscoveryError, MCPToolCallError
from nexusmind.tools.contracts import ToolDefinition, ToolErrorCode

_MAX_TOOL_LIST_PAGES = 100
_MAX_DISCOVERED_TOOLS = 1000


class MCPClient:
    async def list_tools(self) -> list[MCPRemoteTool]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MCPRemoteTool:
    name: str
    description: str | None
    input_schema: dict[str, Any]


async def list_all_mcp_tools(session: Any, request_timeout: float) -> list[MCPRemoteTool]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    tools: list[MCPRemoteTool] = []
    for _ in range(_MAX_TOOL_LIST_PAGES):
        try:
            result = await asyncio.wait_for(_call_list_tools(session, cursor), timeout=request_timeout)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as exc:
            raise MCPConnectionError("MCP tools/list timed out") from exc
        except Exception as exc:
            raise MCPConnectionError("MCP tools/list failed") from exc
        page_tools = getattr(result, "tools", None)
        if not isinstance(page_tools, Sequence):
            raise MCPDiscoveryError("MCP tools/list returned invalid tools")
        tools.extend(_mcp_tool_to_remote_tool(tool) for tool in page_tools)
        if len(tools) > _MAX_DISCOVERED_TOOLS:
            raise MCPDiscoveryError("MCP tools/list returned too many tools")
        cursor = getattr(result, "nextCursor", None) or getattr(result, "next_cursor", None)
        if not cursor:
            return sorted(tools, key=lambda tool: tool.name)
        if cursor in seen_cursors:
            raise MCPDiscoveryError("MCP tools/list returned a repeated cursor")
        seen_cursors.add(cursor)
    raise MCPDiscoveryError("MCP tools/list exceeded the maximum page count")


async def call_mcp_tool(session: Any, name: str, arguments: dict[str, Any], request_timeout: float) -> Any:
    try:
        return await asyncio.wait_for(session.call_tool(name, arguments), timeout=request_timeout)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError as exc:
        raise MCPToolCallError("MCP tool call timed out") from exc
    except Exception as exc:
        raise MCPToolCallError("MCP tool call failed") from exc


def mcp_tool_to_definition(local_name: str, remote_tool: MCPRemoteTool) -> ToolDefinition:
    try:
        Draft202012Validator.check_schema(remote_tool.input_schema)
    except SchemaError as exc:
        raise MCPDiscoveryError(f"MCP tool {remote_tool.name} has an invalid input schema") from exc
    if remote_tool.input_schema.get("type") != "object":
        raise MCPDiscoveryError(f"MCP tool {remote_tool.name} input schema must describe an object")
    return ToolDefinition(
        name=local_name,
        description=remote_tool.description,
        input_schema=deepcopy(remote_tool.input_schema),
    )


def _mcp_tool_to_remote_tool(remote_tool: Any) -> MCPRemoteTool:
    remote_name = getattr(remote_tool, "name", None)
    if not isinstance(remote_name, str) or not remote_name:
        raise MCPDiscoveryError("MCP tool is missing a valid name")
    schema = deepcopy(getattr(remote_tool, "inputSchema", None) or getattr(remote_tool, "input_schema", None))
    if not isinstance(schema, dict):
        raise MCPDiscoveryError(f"MCP tool {remote_name} is missing a JSON input schema")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise MCPDiscoveryError(f"MCP tool {remote_name} has an invalid input schema") from exc
    if schema.get("type") != "object":
        raise MCPDiscoveryError(f"MCP tool {remote_name} input schema must describe an object")
    description = getattr(remote_tool, "description", None)
    return MCPRemoteTool(
        name=remote_name,
        description=description if isinstance(description, str) else None,
        input_schema=schema,
    )


def normalize_call_tool_result(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    return {
        "structured_content": _json_safe(structured),
        "content": [_normalize_content_block(block) for block in getattr(result, "content", [])],
    }


def is_mcp_tool_error(result: Any) -> bool:
    return bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


def mcp_tool_error_code(result: Any) -> ToolErrorCode:
    return ToolErrorCode.EXECUTION_FAILED if is_mcp_tool_error(result) else ToolErrorCode.EXECUTION_FAILED


async def _call_list_tools(session: Any, cursor: str | None) -> Any:
    method = session.list_tools
    signature = inspect.signature(method)
    if cursor and "cursor" in signature.parameters:
        return await method(cursor=cursor)
    return await method()


def _normalize_content_block(block: Any) -> dict[str, Any]:
    block_type = getattr(block, "type", "unknown")
    if block_type == "text":
        return {"type": "text", "text": str(getattr(block, "text", ""))}
    if block_type == "image":
        data = getattr(block, "data", "") or ""
        return {"type": "image", "mime_type": str(getattr(block, "mimeType", "")), "size": len(data)}
    if block_type == "audio":
        data = getattr(block, "data", "") or ""
        return {"type": "audio", "mime_type": str(getattr(block, "mimeType", "")), "size": len(data)}
    if block_type == "resource":
        resource = getattr(block, "resource", None)
        return {
            "type": "resource",
            "uri": str(getattr(resource, "uri", "")),
            "mime_type": str(getattr(resource, "mimeType", "")),
        }
    return {"type": str(block_type)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(type(value).__name__)
