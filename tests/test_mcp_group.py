import asyncio

import pytest

from nexusmind.mcp import (
    MAX_GROUP_DISCOVERED_TOOLS,
    MAX_MCP_CLIENTS_PER_GROUP,
    MCPClientGroup,
    MCPConnectionError,
    MCPRemoteTool,
    MCPStdioServerConfig,
    mcp_tool_local_name,
)
from nexusmind.tools.registry import ToolRegistry, ToolRegistryError


class FakeClient:
    events: list[str] = []
    fail_enter: set[str] = set()
    fail_list: set[str] = set()
    fail_exit: set[str] = set()
    invalid_schema: set[str] = set()
    remote_name_by_server: dict[str, str] = {}
    exit_cancel: set[str] = set()
    tool_count_by_server: dict[str, int] = {}

    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        self.events.append(f"enter:{self.config.server_id}")
        if self.config.server_id in self.fail_enter:
            raise MCPConnectionError("connect failed")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.events.append(f"exit:{self.config.server_id}")
        if self.config.server_id in self.exit_cancel:
            raise asyncio.CancelledError()
        if self.config.server_id in self.fail_exit:
            raise MCPConnectionError("cleanup failed")

    async def list_tools(self):
        self.events.append(f"list:{self.config.server_id}")
        if self.config.server_id in self.fail_list:
            raise MCPConnectionError("list failed")
        tool_count = self.tool_count_by_server.get(self.config.server_id, 1)
        return [
            MCPRemoteTool(
                name=(
                    f"echo_{index}"
                    if tool_count != 1
                    else self.remote_name_by_server.get(self.config.server_id, "echo")
                ),
                description=f"{self.config.server_id} echo",
                input_schema=(
                    {"type": "array"}
                    if self.config.server_id in self.invalid_schema
                    else {"type": "object", "properties": {}}
                ),
            )
            for index in range(tool_count)
        ]

    async def call_tool(self, name, arguments):
        self.events.append(f"call:{self.config.server_id}:{name}")
        return {"ok": True}


def _configs(*server_ids):
    return {server_id: MCPStdioServerConfig(server_id=server_id, command="python") for server_id in server_ids}


def _reset_fake_client():
    FakeClient.events = []
    FakeClient.fail_enter = set()
    FakeClient.fail_list = set()
    FakeClient.fail_exit = set()
    FakeClient.invalid_schema = set()
    FakeClient.remote_name_by_server = {}
    FakeClient.exit_cancel = set()
    FakeClient.tool_count_by_server = {}


def test_mcp_client_group_connects_registers_and_closes_in_reverse_order() -> None:
    _reset_fake_client()

    async def run():
        registry = ToolRegistry()
        async with MCPClientGroup(_configs("zeta", "alpha"), client_class=FakeClient) as group:
            definitions = await group.register_tools(registry)
            assert sorted(definition.name for definition in definitions) == [
                mcp_tool_local_name("alpha", "echo"),
                mcp_tool_local_name("zeta", "echo"),
            ]

    asyncio.run(run())

    assert FakeClient.events == [
        "enter:alpha",
        "enter:zeta",
        "list:alpha",
        "list:zeta",
        "exit:zeta",
        "exit:alpha",
    ]


def test_mcp_client_group_rejects_config_identity_mismatch_before_starting_clients() -> None:
    _reset_fake_client()
    registry = ToolRegistry()
    configs = {"github": MCPStdioServerConfig(server_id="filesystem", command="python")}

    with pytest.raises(MCPConnectionError, match="identity mismatch"):
        MCPClientGroup(configs, client_class=FakeClient)

    assert FakeClient.events == []
    assert registry.list_definitions() == []


def test_mcp_client_group_rejects_too_many_servers_before_starting_clients() -> None:
    _reset_fake_client()
    registry = ToolRegistry()
    configs = _configs(*(f"server{i}" for i in range(MAX_MCP_CLIENTS_PER_GROUP + 1)))

    with pytest.raises(MCPConnectionError, match="too many servers"):
        MCPClientGroup(configs, client_class=FakeClient)

    assert FakeClient.events == []
    assert registry.list_definitions() == []


def test_mcp_client_group_register_tools_requires_active_connection() -> None:
    _reset_fake_client()

    async def run():
        registry = ToolRegistry()
        group = MCPClientGroup(_configs("alpha"), client_class=FakeClient)
        with pytest.raises(MCPConnectionError, match="not connected"):
            await group.register_tools(registry)
        await group.__aenter__()
        await group.__aexit__(None, None, None)
        with pytest.raises(MCPConnectionError, match="not connected"):
            await group.register_tools(registry)

    asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "exit:alpha"]


def test_mcp_client_group_second_connect_failure_closes_first_and_does_not_register() -> None:
    _reset_fake_client()
    FakeClient.fail_enter = {"zeta"}

    async def run():
        registry = ToolRegistry()
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient) as group:
            await group.register_tools(registry)

    with pytest.raises(MCPConnectionError):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "exit:alpha"]


def test_mcp_client_group_tools_list_failure_closes_all_clients() -> None:
    _reset_fake_client()
    FakeClient.fail_list = {"zeta"}

    registry = ToolRegistry()

    async def run():
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient) as group:
            await group.register_tools(registry)

    with pytest.raises(MCPConnectionError):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "list:alpha", "list:zeta", "exit:zeta", "exit:alpha"]
    assert registry.list_definitions() == []


def test_mcp_client_group_definition_failure_keeps_registry_unchanged() -> None:
    _reset_fake_client()
    FakeClient.invalid_schema = {"zeta"}
    registry = ToolRegistry()

    async def run():
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient) as group:
            await group.register_tools(registry)

    with pytest.raises(Exception):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "list:alpha", "list:zeta", "exit:zeta", "exit:alpha"]
    assert registry.list_definitions() == []


def test_mcp_client_group_cross_server_registration_conflict_is_atomic() -> None:
    _reset_fake_client()

    from nexusmind.tools.contracts import ToolDefinition, ToolRiskLevel

    class PreexistingTool:
        @property
        def definition(self):
            return ToolDefinition(
                name=mcp_tool_local_name("alpha", "echo"),
                input_schema={"type": "object", "properties": {}},
                risk_level=ToolRiskLevel.READ_ONLY,
            )

        async def invoke(self, arguments):
            return {}

    registry = ToolRegistry()
    registry.register(PreexistingTool())

    async def run():
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient) as group:
            await group.register_tools(registry)

    with pytest.raises(ToolRegistryError):
        asyncio.run(run())

    names = [definition.name for definition in registry.list_definitions()]
    assert names == [mcp_tool_local_name("alpha", "echo")]


def test_mcp_client_group_registry_failure_closes_all_clients() -> None:
    _reset_fake_client()

    class BadRegistry(ToolRegistry):
        def register_many(self, tools):
            raise ToolRegistryError("duplicate")

    registry = BadRegistry()

    async def run():
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient) as group:
            await group.register_tools(registry)

    with pytest.raises(ToolRegistryError):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "list:alpha", "list:zeta", "exit:zeta", "exit:alpha"]
    assert registry.list_definitions() == []


def test_mcp_client_group_discovered_tool_limit_closes_clients_and_keeps_registry_unchanged() -> None:
    _reset_fake_client()
    FakeClient.tool_count_by_server = {
        "alpha": MAX_GROUP_DISCOVERED_TOOLS,
        "zeta": 1,
    }
    registry = ToolRegistry()

    async def run():
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient) as group:
            await group.register_tools(registry)

    with pytest.raises(MCPConnectionError, match="too many tools"):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "list:alpha", "list:zeta", "exit:zeta", "exit:alpha"]
    assert registry.list_definitions() == []


def test_mcp_client_group_cleanup_failure_does_not_cover_cancel_and_attempts_all() -> None:
    _reset_fake_client()
    FakeClient.fail_exit = {"zeta"}

    async def run():
        group = await MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient).__aenter__()
        try:
            raise asyncio.CancelledError()
        except asyncio.CancelledError as exc:
            await group.__aexit__(type(exc), exc, exc.__traceback__)
            raise

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "exit:zeta", "exit:alpha"]


def test_mcp_client_group_normal_cleanup_failure_reports_mcp_error() -> None:
    _reset_fake_client()
    FakeClient.fail_exit = {"zeta"}

    async def run():
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient):
            pass

    with pytest.raises(MCPConnectionError):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "exit:zeta", "exit:alpha"]


def test_mcp_client_group_normal_cleanup_cancel_propagates_after_attempting_all() -> None:
    _reset_fake_client()
    FakeClient.exit_cancel = {"zeta"}

    async def run():
        async with MCPClientGroup(_configs("alpha", "zeta"), client_class=FakeClient):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "enter:zeta", "exit:zeta", "exit:alpha"]


def test_mcp_client_group_cannot_be_entered_twice_or_reused_after_cleanup() -> None:
    _reset_fake_client()

    async def run():
        group = MCPClientGroup(_configs("alpha"), client_class=FakeClient)
        await group.__aenter__()
        with pytest.raises(MCPConnectionError):
            await group.__aenter__()
        await group.__aexit__(None, None, None)
        with pytest.raises(MCPConnectionError):
            await group.__aenter__()

    asyncio.run(run())

    assert FakeClient.events == ["enter:alpha", "exit:alpha"]
