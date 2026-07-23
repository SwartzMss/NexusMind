import asyncio
import json
import sys

from nexusmind.mcp import MCPStdioClient, MCPStdioServerConfig, register_mcp_tools
from nexusmind.tools import ToolCall, ToolExecutor, ToolRegistry


def test_stdio_mcp_echo_server_end_to_end() -> None:
    async def run():
        config = MCPStdioServerConfig(
            server_id="demo",
            command=sys.executable,
            args=("tests/fixtures/mcp_echo_server.py",),
            connect_timeout=5,
            request_timeout=5,
        )
        async with MCPStdioClient(config) as client:
            registry = ToolRegistry()
            definitions = await register_mcp_tools(client, "demo", registry)
            echo = next(definition for definition in definitions if "echo" in definition.name)
            result = await ToolExecutor(registry, timeout=5).execute(
                ToolCall(id="call-1", name=echo.name, arguments={"text": "hello"})
            )
            return result

    result = asyncio.run(asyncio.wait_for(run(), timeout=15))

    assert result.error is None
    assert json.dumps(result.output, sort_keys=True)
    assert "hello" in json.dumps(result.output)
