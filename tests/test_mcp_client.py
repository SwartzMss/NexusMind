import asyncio
from dataclasses import dataclass

import pytest

from nexusmind.mcp.client import call_mcp_tool, list_all_mcp_tools
from nexusmind.mcp.errors import MCPConnectionError, MCPDiscoveryError, MCPToolCallError


@dataclass
class Page:
    tools: list
    nextCursor: str | None = None


@dataclass
class Tool:
    name: str


class PagedSession:
    def __init__(self) -> None:
        self.calls = 0

    async def list_tools(self, cursor=None):
        self.calls += 1
        if cursor is None:
            return Page([Tool("zeta")], nextCursor="next")
        return Page([Tool("alpha")])


def test_list_all_mcp_tools_reads_pages_and_sorts() -> None:
    async def run():
        return await list_all_mcp_tools(PagedSession(), 1)

    tools = asyncio.run(run())

    assert [tool.name for tool in tools] == ["alpha", "zeta"]


def test_list_all_mcp_tools_rejects_repeated_cursor() -> None:
    class RepeatingCursorSession:
        async def list_tools(self, cursor=None):
            return Page([Tool("echo")], nextCursor="same-cursor")

    async def run():
        return await list_all_mcp_tools(RepeatingCursorSession(), 1)

    with pytest.raises(MCPDiscoveryError, match="repeated cursor"):
        asyncio.run(run())


def test_list_all_mcp_tools_rejects_too_many_pages() -> None:
    class ManyPagesSession:
        def __init__(self) -> None:
            self.index = 0

        async def list_tools(self, cursor=None):
            self.index += 1
            return Page([], nextCursor=f"cursor-{self.index}")

    async def run():
        return await list_all_mcp_tools(ManyPagesSession(), 1)

    with pytest.raises(MCPDiscoveryError, match="maximum page count"):
        asyncio.run(run())


def test_list_all_mcp_tools_times_out() -> None:
    class SlowSession:
        async def list_tools(self):
            await asyncio.sleep(10)

    async def run():
        return await list_all_mcp_tools(SlowSession(), 0.01)

    with pytest.raises(MCPConnectionError):
        asyncio.run(run())


def test_call_mcp_tool_times_out() -> None:
    class SlowSession:
        async def call_tool(self, name, arguments):
            await asyncio.sleep(10)

    async def run():
        return await call_mcp_tool(SlowSession(), "echo", {}, 0.01)

    with pytest.raises(MCPToolCallError):
        asyncio.run(run())


def test_call_mcp_tool_propagates_cancelled_error() -> None:
    class CancelSession:
        async def call_tool(self, name, arguments):
            raise asyncio.CancelledError()

    async def run():
        return await call_mcp_tool(CancelSession(), "echo", {}, 1)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
