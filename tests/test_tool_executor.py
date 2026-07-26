import asyncio

import pytest

from nexusmind.tools import (
    ToolCall,
    ToolDefinition,
    ToolErrorCode,
    ToolExecutor,
    ToolRegistry,
    ToolResultBudget,
    ToolResultRequirements,
    ToolRiskLevel,
)
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


class MutableSchemaTool:
    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="mutable",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def invoke(self, arguments):
        return {"text": arguments["text"]}


class SecretArgumentTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="secret_arg",
            input_schema={
                "type": "object",
                "properties": {"token": {"type": "integer"}},
                "required": ["token"],
                "additionalProperties": False,
            },
        )

    async def invoke(self, arguments):
        return arguments


class StrictRequirementsTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="strict_requirements",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    def result_requirements(self, arguments):
        return ToolResultRequirements(min_bytes=22 + len(arguments["text"]), min_nodes=3, min_depth=1)

    async def invoke(self, arguments):
        return arguments


class UnbudgetedWriteTool:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="unbudgeted_write",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            risk_level=ToolRiskLevel.LOCAL_WRITE,
        )

    async def invoke(self, arguments):
        self.calls += 1
        return {"committed": True}


class UnbudgetedUnspecifiedTool(UnbudgetedWriteTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="unbudgeted_unspecified",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )


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


def test_executor_uses_registered_schema_snapshot() -> None:
    tool = MutableSchemaTool()
    registry = _registry_with(tool)
    tool.definition.input_schema["type"] = "invalid-type"

    async def run():
        executor = ToolExecutor(registry)
        return await executor.execute(ToolCall(id="call-1", name="mutable", arguments={"text": "hello"}))

    result = asyncio.run(run())

    assert result.error is None
    assert result.output == {"text": "hello"}


def test_executor_validation_cannot_be_bypassed_by_mutating_listed_definition() -> None:
    registry = _registry_with(EchoTool())
    definition = registry.list_definitions()[0]
    definition.input_schema["required"] = []
    definition.input_schema["additionalProperties"] = True

    async def run():
        executor = ToolExecutor(registry)
        return await executor.execute(ToolCall(id="call-1", name="echo", arguments={}))

    result = asyncio.run(run())

    assert result.error is not None
    assert result.error.code == ToolErrorCode.INVALID_ARGUMENTS


def test_executor_validation_error_does_not_include_secret_argument_value() -> None:
    async def run():
        executor = ToolExecutor(_registry_with(SecretArgumentTool()))
        return await executor.execute(
            ToolCall(id="call-1", name="secret_arg", arguments={"token": "sk-live-secret"})
        )

    result = asyncio.run(run())

    assert result.error is not None
    assert result.error.code == ToolErrorCode.INVALID_ARGUMENTS
    assert "sk-live-secret" not in result.error.message
    assert result.error.message == "Invalid arguments at token: expected integer"


def test_result_requirements_validates_arguments_before_calling_tool() -> None:
    executor = ToolExecutor(_registry_with(StrictRequirementsTool()))
    call = ToolCall(id="call-1", name="strict_requirements", arguments={})

    requirements = executor.result_requirements(call)
    result = asyncio.run(
        executor.execute_with_result_budget(
            call,
            result_budget=ToolResultBudget(
                max_bytes=requirements.min_bytes,
                max_nodes=requirements.min_nodes,
                max_depth=requirements.min_depth,
            ),
        )
    )

    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_ARGUMENTS


def test_unbudgeted_side_effect_tool_is_not_invoked() -> None:
    tool = UnbudgetedWriteTool()
    executor = ToolExecutor(_registry_with(tool))

    result = asyncio.run(
        executor.execute_with_result_budget(
            ToolCall(id="call-1", name="unbudgeted_write", arguments={}),
            result_budget=ToolResultBudget(max_bytes=1024, max_nodes=20, max_depth=5),
        )
    )

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED
    assert tool.calls == 0


def test_unbudgeted_unspecified_tool_is_not_invoked() -> None:
    tool = UnbudgetedUnspecifiedTool()
    executor = ToolExecutor(_registry_with(tool))

    result = asyncio.run(
        executor.execute_with_result_budget(
            ToolCall(id="call-1", name="unbudgeted_unspecified", arguments={}),
            result_budget=ToolResultBudget(max_bytes=1024, max_nodes=20, max_depth=5),
        )
    )

    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_FAILED
    assert tool.calls == 0


def test_custom_success_requirements_include_standard_errors() -> None:
    executor = ToolExecutor(_registry_with(StrictRequirementsTool()))

    requirements = executor.result_requirements(
        ToolCall(id="call-1", name="strict_requirements", arguments={"text": ""})
    )

    assert requirements.min_bytes > 40
    assert requirements.min_nodes >= 6
    assert requirements.min_depth >= 2


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
