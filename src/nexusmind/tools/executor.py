from __future__ import annotations

import asyncio

from jsonschema import ValidationError
from jsonschema.validators import Draft202012Validator

from nexusmind.tools.contracts import ToolCall, ToolError, ToolErrorCode, ToolResult
from nexusmind.tools.registry import ToolNotFoundError, ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout: float = 30.0) -> None:
        self._registry = registry
        self._timeout = timeout

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            tool = self._registry.get(call.name)
        except ToolNotFoundError:
            return _failure(call, ToolErrorCode.TOOL_NOT_FOUND, f"Tool not found: {call.name}")

        try:
            Draft202012Validator(tool.definition.input_schema).validate(call.arguments)
        except ValidationError as exc:
            return _failure(call, ToolErrorCode.INVALID_ARGUMENTS, _validation_message(exc))

        try:
            output = await asyncio.wait_for(tool.invoke(call.arguments), timeout=self._timeout)
        except asyncio.TimeoutError:
            return _failure(call, ToolErrorCode.EXECUTION_TIMEOUT, f"Tool timed out after {self._timeout:g} seconds")
        except asyncio.CancelledError:
            raise
        except Exception:
            return _failure(call, ToolErrorCode.EXECUTION_FAILED, "Tool execution failed")

        return ToolResult(call_id=call.id, name=call.name, output=output)


def _failure(call: ToolCall, code: ToolErrorCode, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, name=call.name, error=ToolError(code=code, message=message))


def _validation_message(exc: ValidationError) -> str:
    if exc.path:
        path = ".".join(str(part) for part in exc.path)
        return f"Invalid arguments at {path}: {exc.message}"
    return f"Invalid arguments: {exc.message}"

