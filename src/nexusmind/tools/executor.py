from __future__ import annotations

import asyncio
from typing import Any
from typing import Protocol, runtime_checkable

from jsonschema import ValidationError
from nexusmind.command_errors import CommandError
from nexusmind.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolResultBudget,
    ToolResultRequirements,
    ToolRiskLevel,
    json_result_requirements,
)
from nexusmind.tools.registry import ToolNotFoundError, ToolRegistry
from nexusmind.workspace import WorkspaceError


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    def definition(self, name: str) -> ToolDefinition | None:
        ...

    async def execute(self, call: ToolCall) -> ToolResult:
        ...

    def result_requirements(self, call: ToolCall) -> ToolResultRequirements:
        ...

    async def execute_with_result_budget(
        self,
        call: ToolCall,
        *,
        result_budget: ToolResultBudget,
    ) -> ToolResult:
        ...


class ToolResultBudgetError(RuntimeError):
    pass

TOOL_CANCEL_GRACE_SECONDS = 1.0


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, timeout: float = 30.0) -> None:
        self._registry = registry
        self._timeout = timeout

    def definition(self, name: str) -> ToolDefinition | None:
        try:
            return self._registry.definition(name)
        except ToolNotFoundError:
            return None

    async def execute(self, call: ToolCall) -> ToolResult:
        return await self._execute(call, result_budget=None)

    def result_requirements(self, call: ToolCall) -> ToolResultRequirements:
        try:
            registered = self._registry._get_registered(call.name)
        except ToolNotFoundError:
            return _result_requirements(
                _failure(call, ToolErrorCode.TOOL_NOT_FOUND, f"Tool not found: {call.name}")
            )
        try:
            registered.validator.validate(call.arguments)
        except ValidationError as exc:
            return _result_requirements(
                _failure(call, ToolErrorCode.INVALID_ARGUMENTS, _validation_message(exc))
            )
        requirements = getattr(registered.tool, "result_requirements", None)
        if callable(requirements):
            return _max_requirements(requirements(call.arguments), _default_result_requirements())
        return _default_result_requirements()

    async def execute_with_result_budget(
        self,
        call: ToolCall,
        *,
        result_budget: ToolResultBudget,
    ) -> ToolResult:
        return await self._execute(call, result_budget=result_budget)

    async def _execute(self, call: ToolCall, *, result_budget: ToolResultBudget | None) -> ToolResult:
        try:
            registered = self._registry._get_registered(call.name)
        except ToolNotFoundError:
            return _return_with_budget(
                _failure(call, ToolErrorCode.TOOL_NOT_FOUND, f"Tool not found: {call.name}"),
                result_budget,
            )

        try:
            registered.validator.validate(call.arguments)
        except ValidationError as exc:
            return _return_with_budget(
                _failure(call, ToolErrorCode.INVALID_ARGUMENTS, _validation_message(exc)),
                result_budget,
            )

        requirements_method = getattr(registered.tool, "result_requirements", None)
        invoke_with_result_budget = getattr(registered.tool, "invoke_with_result_budget", None)
        if result_budget is not None and registered.definition.risk_level is not ToolRiskLevel.READ_ONLY and (
            not callable(requirements_method) or not callable(invoke_with_result_budget)
        ):
            return _return_with_budget(
                _failure(call, ToolErrorCode.EXECUTION_FAILED, "Tool does not support result budgets"),
                result_budget,
            )
        if result_budget is not None:
            requirements = self.result_requirements(call)
            if not result_budget.satisfies(requirements):
                raise ToolResultBudgetError("Tool result budget is too small")

        timeout = self._timeout
        timeout_for_call = getattr(registered.tool, "timeout_for_call", None)
        if callable(timeout_for_call):
            try:
                timeout = max(timeout, float(timeout_for_call(call.arguments)))
            except Exception:
                timeout = self._timeout

        if result_budget is not None and callable(invoke_with_result_budget):
            invocation = invoke_with_result_budget(call.arguments, result_budget=result_budget)
        else:
            invocation = registered.tool.invoke(call.arguments)
        invoke_task = asyncio.create_task(invocation)
        try:
            output = await asyncio.wait_for(invoke_task, timeout=timeout)
        except asyncio.TimeoutError:
            return _return_with_budget(
                _failure(call, ToolErrorCode.EXECUTION_TIMEOUT, f"Tool timed out after {timeout:g} seconds"),
                result_budget,
            )
        except asyncio.CancelledError as cancellation:
            invoke_task.cancel()
            current_task = asyncio.current_task()
            uncancel = getattr(current_task, "uncancel", None)
            if callable(uncancel):
                while current_task is not None and current_task.cancelling():
                    uncancel()
            try:
                await asyncio.wait_for(asyncio.shield(invoke_task), timeout=TOOL_CANCEL_GRACE_SECONDS)
            except asyncio.TimeoutError:
                raise cancellation
            except asyncio.CancelledError:
                raise cancellation
            except Exception as exc:
                return _return_with_budget(_failure(call, ToolErrorCode.EXECUTION_FAILED, str(exc) or "Tool execution failed"), result_budget)
            while not invoke_task.done():
                try:
                    await asyncio.shield(invoke_task)
                except asyncio.CancelledError:
                    if invoke_task.done():
                        break
                    if callable(uncancel):
                        while current_task is not None and current_task.cancelling():
                            uncancel()
            try:
                invoke_task.result()
            except CommandError as exc:
                return _return_with_budget(
                    _failure(call, ToolErrorCode.EXECUTION_FAILED, str(exc)),
                    result_budget,
                )
            except asyncio.CancelledError:
                raise cancellation
            raise cancellation
        except CommandError as exc:
            return _return_with_budget(
                _failure(call, ToolErrorCode.EXECUTION_FAILED, str(exc)),
                result_budget,
            )
        except WorkspaceError as exc:
            return _return_with_budget(
                _failure(call, ToolErrorCode.EXECUTION_FAILED, str(exc)),
                result_budget,
            )
        except Exception:
            return _return_with_budget(
                _failure(call, ToolErrorCode.EXECUTION_FAILED, "Tool execution failed"),
                result_budget,
            )

        truncated = isinstance(output, dict) and bool(output.get("truncated") or output.get("stdout_truncated") or output.get("stderr_truncated"))
        if result_budget is not None and not result_budget.satisfies(_result_requirements(ToolResult(call_id=call.id, name=call.name, output=output))):
            output, truncated = _truncate_string_output(output, call, result_budget)
        return _return_with_budget(
            ToolResult(call_id=call.id, name=call.name, output=output, metadata={"result_truncated": truncated}),
            result_budget,
        )


def _failure(call: ToolCall, code: ToolErrorCode, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, name=call.name, error=ToolError(code=code, message=message))


def _return_with_budget(result: ToolResult, budget: ToolResultBudget | None) -> ToolResult:
    if budget is not None and not budget.satisfies(_result_requirements(result)):
        raise ToolResultBudgetError("Tool result budget is too small")
    return result

def _truncate_string_output(output: Any, call: ToolCall, budget: ToolResultBudget) -> tuple[Any, bool]:
    if not isinstance(output, str):
        raise ToolResultBudgetError("Tool result budget is too small")
    low, high = 0, len(output)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = ToolResult(call_id=call.id, name=call.name, output=output[:middle])
        if budget.satisfies(_result_requirements(candidate)): low = middle
        else: high = middle - 1
    if low == 0: raise ToolResultBudgetError("Tool result budget is too small")
    return output[:low], True


def _default_result_requirements() -> ToolResultRequirements:
    results = [
        ToolResult(
            call_id="call",
            name="tool",
            error=ToolError(code=code, message=message),
        )
        for code, message in (
            (ToolErrorCode.INVALID_ARGUMENTS, "Invalid arguments: expected no additional properties"),
            (ToolErrorCode.EXECUTION_FAILED, "Tool execution failed"),
            (ToolErrorCode.EXECUTION_TIMEOUT, "Tool timed out after 300 seconds"),
            (ToolErrorCode.PERMISSION_DENIED, "Tool execution was denied"),
        )
    ]
    requirements = [_result_requirements(result) for result in results]
    return _max_requirements(*requirements)


def _max_requirements(*requirements: ToolResultRequirements) -> ToolResultRequirements:
    return ToolResultRequirements(
        min_bytes=max(item.min_bytes for item in requirements),
        min_nodes=max(item.min_nodes for item in requirements),
        min_depth=max(item.min_depth for item in requirements),
    )


def _result_requirements(result: ToolResult) -> ToolResultRequirements:
    if result.error is None:
        payload: Any = {"ok": True, "output": result.output}
    else:
        payload = {
            "ok": False,
            "error": {
                "code": result.error.code.value,
                "message": result.error.message,
                "retryable": result.error.retryable,
            },
        }
    return json_result_requirements(payload)


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
