from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from nexusmind.mcp.client import call_mcp_tool, list_all_mcp_tools
from nexusmind.mcp.config import MCPStdioServerConfig
from nexusmind.mcp.errors import MCPConnectionError


class MCPStdioClient:
    def __init__(self, config: MCPStdioServerConfig) -> None:
        self._config = config
        self._stdio_context: Any | None = None
        self._session_context: Any | None = None
        self._session: ClientSession | None = None
        self._errlog: Any | None = None
        self._entered = False

    async def __aenter__(self) -> MCPStdioClient:
        if self._entered:
            raise MCPConnectionError("MCP stdio client is already connected")
        self._entered = True
        params = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            cwd=self._config.cwd,
            env=self._config.env or None,
        )
        try:
            self._errlog = _open_errlog()
            self._stdio_context = stdio_client(params, errlog=self._errlog)
            read_stream, write_stream = await asyncio.wait_for(
                self._stdio_context.__aenter__(), timeout=self._config.connect_timeout
            )
            self._session_context = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self._config.request_timeout),
            )
            self._session = await asyncio.wait_for(
                self._session_context.__aenter__(), timeout=self._config.connect_timeout
            )
            await asyncio.wait_for(self._session.initialize(), timeout=self._config.connect_timeout)
        except asyncio.CancelledError:
            await self._cleanup()
            raise
        except asyncio.TimeoutError as exc:
            await self._cleanup()
            raise MCPConnectionError("MCP stdio client initialization timed out") from exc
        except Exception as exc:
            await self._cleanup()
            raise MCPConnectionError("MCP stdio client initialization failed") from exc
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self._cleanup()

    async def list_tools(self) -> list[Any]:
        session = self._require_session()
        return await list_all_mcp_tools(session, self._config.request_timeout)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        session = self._require_session()
        return await call_mcp_tool(session, name, arguments, self._config.request_timeout)

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPConnectionError("MCP stdio client is not connected")
        return self._session

    async def _cleanup(self) -> None:
        session_context = self._session_context
        stdio_context = self._stdio_context
        errlog = self._errlog
        self._session = None
        self._session_context = None
        self._stdio_context = None
        self._errlog = None
        if session_context is not None:
            with contextlib.suppress(Exception):
                await session_context.__aexit__(None, None, None)
        if stdio_context is not None:
            with contextlib.suppress(Exception):
                await stdio_context.__aexit__(None, None, None)
        if errlog is not None:
            with contextlib.suppress(Exception):
                errlog.close()


def _open_errlog() -> Any:
    return open(os.devnull, "w", encoding="utf-8")
