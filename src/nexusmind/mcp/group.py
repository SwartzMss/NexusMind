from __future__ import annotations

import asyncio
from collections.abc import Mapping

from nexusmind.mcp.config import MCPStdioServerConfig
from nexusmind.mcp.errors import MCPConnectionError, MCPError
from nexusmind.mcp.limits import MAX_GROUP_DISCOVERED_TOOLS, MAX_MCP_CLIENTS_PER_GROUP
from nexusmind.mcp.stdio import MCPStdioClient
from nexusmind.mcp.tool_adapter import MCPToolAdapter
from nexusmind.tools.base import Tool
from nexusmind.tools.contracts import ToolDefinition
from nexusmind.tools.registry import ToolRegistry

class MCPClientGroup:
    def __init__(self, configs: Mapping[str, MCPStdioServerConfig], client_class=MCPStdioClient) -> None:
        if len(configs) > MAX_MCP_CLIENTS_PER_GROUP:
            raise MCPConnectionError("MCP client group contains too many servers")
        for server_id, config in configs.items():
            if server_id != config.server_id:
                raise MCPConnectionError("MCP client group config identity mismatch")
        self._configs = dict(sorted(configs.items()))
        self._client_class = client_class
        self._clients: dict[str, object] = {}
        self._entered_order: list[str] = []
        self._entered = False
        self._active = False

    async def __aenter__(self) -> MCPClientGroup:
        if self._entered:
            raise MCPConnectionError("MCP client group is already connected")
        self._entered = True
        try:
            for server_id, config in self._configs.items():
                client = self._client_class(config)
                await client.__aenter__()
                self._clients[server_id] = client
                self._entered_order.append(server_id)
        except BaseException as exc:
            await self._cleanup(type(exc), exc, exc.__traceback__, raise_without_original=False)
            raise
        self._active = True
        return self

    async def register_tools(self, registry: ToolRegistry) -> list[ToolDefinition]:
        if not self._active:
            raise MCPConnectionError("MCP client group is not connected")
        adapters: list[Tool] = []
        for server_id in sorted(self._clients):
            client = self._clients[server_id]
            remote_tools = await client.list_tools()
            if len(adapters) + len(remote_tools) > MAX_GROUP_DISCOVERED_TOOLS:
                raise MCPConnectionError("MCP client group discovered too many tools")
            adapters.extend(MCPToolAdapter(client, server_id, remote_tool) for remote_tool in remote_tools)
        registry.register_many(adapters)
        return [adapter.definition for adapter in adapters]

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self._cleanup(exc_type, exc, traceback, raise_without_original=exc_type is None)

    async def _cleanup(self, exc_type, exc, traceback, *, raise_without_original: bool) -> None:
        self._active = False
        base_error: BaseException | None = None
        cleanup_error: Exception | None = None
        for server_id in reversed(self._entered_order):
            client = self._clients.get(server_id)
            if client is None:
                continue
            try:
                await client.__aexit__(exc_type, exc, traceback)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit) as cleanup_exc:
                base_error = base_error or cleanup_exc
            except BaseException as cleanup_exc:
                if isinstance(cleanup_exc, Exception):
                    cleanup_error = cleanup_error or cleanup_exc
                else:
                    base_error = base_error or cleanup_exc
        self._clients.clear()
        self._entered_order.clear()
        if not raise_without_original:
            return
        if base_error is not None:
            raise base_error
        if cleanup_error is not None:
            if isinstance(cleanup_error, MCPError):
                raise cleanup_error
            raise MCPConnectionError("MCP client group cleanup failed") from cleanup_error
