import asyncio

import pytest

from nexusmind.tools import ToolCall, ToolDefinition, ToolErrorCode, ToolExecutor, ToolRegistry
from nexusmind.tools.builtin import EchoTool


class FailingTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="fail", input_schema={"type": "object", "properties": {}})

    async def invoke(self, arguments):
        raise RuntimeError("implementation detail")


class SlowTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="slow", input_schema={"type": "object", "properties": {}})

    async def invoke(self, arguments):
        await asyncio.sleep(10)


class CancelledTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(name="cancel", input_schema={"type": "object", "properties": {}})

    async def invoke(self, arguments):
        raise asyncio.CancelledError()


def _registry_with(*tools) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_executor_runs_tool_successfully() -> None:
    async def run():
        executor = ToolExecutor(_registry_with(EchoTool()))
        return await executor.execute(ToolCall(id="call-1", name="echo", arguments={"text": "hello"}))

    result = asyncio.run(run())

    assert result.error is None
    assert result.output == {"text": "hello"}


def test_executor_returns_tool_not_found() -> None:
    async def run():
        return await ToolExecutor(ToolRegistry()).execute(ToolCall(id="call-1", name="missing", arguments={}))

    result = asyncio.run(run())

    assert result.error is not None
    assert result.error.code == ToolErrorCode.TOOL_NOT_FOUND


@pytest.mark.parametrize("arguments", [{}, {"text": 42}])
def test_executor_validates_arguments(arguments: dict) -> None:
    async def run():
        executor = ToolExecutor(_registry_with(EchoTool()))
        return await executor.execute(ToolCall(id="call-1", name="echo", arguments=arguments))

    result = asyncio.run(run())

    assert result.error is not None
    assert result.error.code == ToolErrorCode.INVALID_ARGUMENTS


def test_executor_converts_ordinary_exception_to_failed_result() -> None:
    async def run():
        executor = ToolExecutor(_registry_with(FailingTool()))
        return await executor.execute(ToolCall(id="call-1", name="fail", arguments={}))

    result = asyncio.run(run())

    assert result.error is not None
    assert result.error.code == ToolErrorCode.EXECUTION_FAILED
    assert "implementation detail" not in result.error.message


def test_executor_converts_timeout_to_failed_result() -> None:
    async def run():
        executor = ToolExecutor(_registry_with(SlowTool()), timeout=0.01)
        return await executor.execute(ToolCall(id="call-1", name="slow", arguments={}))

    result = asyncio.run(run())

    assert result.error is not None
    assert result.error.code == ToolErrorCode.EXECUTION_TIMEOUT


def test_executor_propagates_cancelled_error() -> None:
    async def run():
        executor = ToolExecutor(_registry_with(CancelledTool()))
        return await executor.execute(ToolCall(id="call-1", name="cancel", arguments={}))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

