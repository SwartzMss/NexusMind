from __future__ import annotations

import asyncio

from jsonschema import ValidationError
from nexusmind.tools.contracts import ToolCall, ToolError, ToolErrorCode, ToolResult
from nexusmind.tools.registry import ToolNotFoundError, ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout: float = 30.0) -> None:
        self._registry = registry
        self._timeout = timeout

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            registered = self._registry.get_registered(call.name)
        except ToolNotFoundError:
            return _failure(call, ToolErrorCode.TOOL_NOT_FOUND, f"Tool not found: {call.name}")

        try:
            registered.validator.validate(call.arguments)
        except ValidationError as exc:
            return _failure(call, ToolErrorCode.INVALID_ARGUMENTS, _validation_message(exc))

        try:
            output = await asyncio.wait_for(registered.tool.invoke(call.arguments), timeout=self._timeout)
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
    expected = _expected_type(exc)
    if exc.path:
        path = ".".join(str(part) for part in exc.path)
        if expected:
            return f"Invalid arguments at {path}: expected {expected}"
        return f"Invalid arguments at {path}"
    if expected:
        return f"Invalid arguments: expected {expected}"
    return "Invalid arguments"


def _expected_type(exc: ValidationError) -> str | None:
    if exc.validator == "type":
        expected = exc.validator_value
        if isinstance(expected, list):
            return " or ".join(str(value) for value in expected)
        return str(expected)
    if exc.validator == "required":
        return "required property"
    if exc.validator == "additionalProperties":
        return "no additional properties"
    return None
