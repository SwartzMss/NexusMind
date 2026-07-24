from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

from nexusmind.models.base import ChatModel
from nexusmind.models.tool_calls import ToolCallDelta
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolCall, ToolDefinition, ToolError, ToolErrorCode, ToolResult
from nexusmind.tools.executor import ToolExecutor

_MODEL_EXECUTION_ERROR = "Model execution failed"
_FINISH_REASONS = {
    "stop",
    "tool_calls",
    "length",
    "content_filter",
    "unknown",
    "null",
}
_RUNTIME_ERROR = "Runtime state machine failed"
_LIMIT_ERROR = "Agent loop limit exceeded"


@dataclass(frozen=True, slots=True)
class AgentLoopLimits:
    max_model_turns: int = 8
    max_tool_calls_total: int = 32
    max_tool_arguments_bytes_per_call: int = 1024 * 1024
    max_tool_arguments_bytes_total: int = 4 * 1024 * 1024
    max_tool_result_bytes_per_call: int = 1024 * 1024
    max_tool_result_bytes_total: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        for field_name in (
            "max_model_turns",
            "max_tool_calls_total",
            "max_tool_arguments_bytes_per_call",
            "max_tool_arguments_bytes_total",
            "max_tool_result_bytes_per_call",
            "max_tool_result_bytes_total",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Agent loop limits must be positive integers")


class ChatRuntime:
    def __init__(
        self,
        model: ChatModel,
        tool_executor: ToolExecutor | None = None,
        limits: AgentLoopLimits | None = None,
    ) -> None:
        self._model = model
        self._tool_executor = tool_executor
        self._limits = limits or AgentLoopLimits()

    async def stream_user_message(
        self,
        content: str,
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(Message(role=MessageRole.USER, content=content))

        yield RuntimeEvent(RuntimeEventType.RUN_STARTED)
        try:
            model_turns = 0
            tool_calls_total = 0
            tool_arguments_bytes_total = 0
            tool_result_bytes_total = 0
            executed_tool_call_ids: set[str] = set()
            allowed_tool_names = {tool.name for tool in tools or []}
            while True:
                if model_turns >= self._limits.max_model_turns:
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
                    return
                model_turns += 1
                turn = _ModelTurn()
                try:
                    async for event in self._model.stream(messages, tools=tools):
                        validation_error = _validate_model_event(event, turn.model_started, turn.completed)
                        if validation_error:
                            yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=validation_error)
                            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=validation_error)
                            return
                        if event.type == RuntimeEventType.MODEL_STARTED:
                            turn.model_started = True
                        elif event.type == RuntimeEventType.MODEL_FAILED:
                            yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=_MODEL_EXECUTION_ERROR)
                            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_MODEL_EXECUTION_ERROR)
                            return
                        elif event.type == RuntimeEventType.TEXT_DELTA:
                            turn.text_parts.append(cast(str, event.text))
                        elif event.type == RuntimeEventType.TOOL_CALL_COMPLETED:
                            turn.tool_calls.append(cast(ToolCall, event.tool_call))
                        elif event.type == RuntimeEventType.MODEL_TURN_COMPLETED:
                            turn.completed = True
                            turn.finish_reason = event.finish_reason
                            turn.completed_event = event
                            continue
                        yield event
                except asyncio.CancelledError:
                    raise
                except Exception:
                    yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=_MODEL_EXECUTION_ERROR)
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_MODEL_EXECUTION_ERROR)
                    return
                terminal_error = _validate_completed_turn(turn)
                if terminal_error:
                    yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=terminal_error)
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=terminal_error)
                    return
                completed_event = cast(RuntimeEvent, turn.completed_event)
                yield completed_event
                if turn.finish_reason != "tool_calls":
                    yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)
                    return
                if self._tool_executor is None:
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error="Tool executor is not configured")
                    return
                if _has_duplicate_call_ids(turn.tool_calls):
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                    return
                if any(call.id in executed_tool_call_ids for call in turn.tool_calls):
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                    return
                if any(call.name not in allowed_tool_names for call in turn.tool_calls):
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                    return
                try:
                    safe_tool_calls, arguments_size = _snapshot_tool_calls(
                        turn.tool_calls,
                        max_bytes_per_call=self._limits.max_tool_arguments_bytes_per_call,
                        remaining_total_bytes=self._limits.max_tool_arguments_bytes_total - tool_arguments_bytes_total,
                    )
                except _JsonLimitExceeded:
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
                    return
                except RuntimeError:
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                    return
                if safe_tool_calls is None:
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                    return
                if tool_calls_total + len(turn.tool_calls) > self._limits.max_tool_calls_total:
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
                    return
                tool_arguments_bytes_total += arguments_size
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content="".join(turn.text_parts) or None,
                        tool_calls=tuple(safe_tool_calls),
                    )
                )
                for call in safe_tool_calls:
                    try:
                        result = await self._tool_executor.execute(call)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                        return
                    if result.call_id != call.id or result.name != call.name:
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                        return
                    if not _valid_tool_result(result):
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                        return
                    try:
                        content_json, size = _tool_result_message_content(
                            result,
                            max_bytes_per_call=self._limits.max_tool_result_bytes_per_call,
                            remaining_total_bytes=self._limits.max_tool_result_bytes_total - tool_result_bytes_total,
                        )
                    except _JsonLimitExceeded:
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
                        return
                    except RuntimeError:
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                        return
                    tool_result_bytes_total += size
                    tool_calls_total += 1
                    executed_tool_call_ids.add(call.id)
                    yield RuntimeEvent(RuntimeEventType.TOOL_RESULT, tool_result=result)
                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            name=result.name,
                            tool_call_id=result.call_id,
                            content=content_json,
                        )
                    )
                continue
        except asyncio.CancelledError:
            raise
        except Exception:
            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)


class _ModelTurn:
    def __init__(self) -> None:
        self.model_started = False
        self.completed = False
        self.finish_reason: str | None = None
        self.completed_event: RuntimeEvent | None = None
        self.tool_calls: list[ToolCall] = []
        self.text_parts: list[str] = []


def _validate_completed_turn(turn: _ModelTurn) -> str | None:
    if not turn.model_started:
        return "Model stream ended before model start"
    if not turn.completed:
        return "Model stream ended before model turn completion"
    if turn.finish_reason == "tool_calls" and not turn.tool_calls:
        return "Model turn requested tools without completed tool calls"
    if turn.tool_calls and turn.finish_reason != "tool_calls":
        return "Model turn completed tool calls with an incompatible finish reason"
    if turn.completed_event is None:
        return "Model stream ended without a completion event"
    return None


def _has_duplicate_call_ids(tool_calls: list[ToolCall]) -> bool:
    seen: set[str] = set()
    for call in tool_calls:
        if call.id in seen:
            return True
        seen.add(call.id)
    return False


def _snapshot_tool_calls(
    tool_calls: list[ToolCall],
    *,
    max_bytes_per_call: int,
    remaining_total_bytes: int,
) -> tuple[list[ToolCall], int]:
    snapshotted: list[ToolCall] = []
    total_size = 0
    for call in tool_calls:
        try:
            arguments_json, size = _bounded_json(
                call.arguments,
                max_bytes=min(max_bytes_per_call, remaining_total_bytes - total_size),
            )
            arguments = json.loads(arguments_json)
        except (TypeError, ValueError, RecursionError):
            raise RuntimeError("Tool call arguments are not strict JSON") from None
        if not isinstance(arguments, dict):
            raise RuntimeError("Tool call arguments are not a JSON object")
        snapshotted.append(ToolCall(id=call.id, name=call.name, arguments=arguments))
        total_size += size
    return snapshotted, total_size


def _valid_tool_result(result: ToolResult) -> bool:
    if result.error is None:
        return True
    return (
        result.output is None
        and isinstance(result.error, ToolError)
        and isinstance(result.error.code, ToolErrorCode)
        and isinstance(result.error.message, str)
        and isinstance(result.error.retryable, bool)
    )


def _tool_result_message_content(
    result: ToolResult,
    *,
    max_bytes_per_call: int,
    remaining_total_bytes: int,
) -> tuple[str, int]:
    if result.error is not None:
        payload = {
            "ok": False,
            "error": {
                "code": result.error.code.value,
                "message": result.error.message,
                "retryable": result.error.retryable,
            },
        }
    else:
        payload = {"ok": True, "output": result.output}
    try:
        return _bounded_json(payload, max_bytes=min(max_bytes_per_call, remaining_total_bytes))
    except (TypeError, ValueError, RecursionError) as exc:
        raise RuntimeError("Tool result is not JSON serializable") from exc


class _JsonLimitExceeded(RuntimeError):
    pass


def _bounded_json(payload: object, *, max_bytes: int) -> tuple[str, int]:
    _validate_json_safe_with_budget(payload, remaining=max_bytes)
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    parts: list[str] = []
    size = 0
    for part in encoder.iterencode(payload):
        size += len(part.encode("utf-8"))
        if size > max_bytes:
            raise _JsonLimitExceeded("JSON payload exceeds size limit")
        parts.append(part)
    return "".join(parts), size


def _validate_json_safe_with_budget(value: object, *, remaining: int) -> int:
    remaining = _consume_json_overhead(remaining, 1)
    if value is None or isinstance(value, bool):
        return _consume_json_overhead(remaining, 4)
    if isinstance(value, int) and not isinstance(value, bool):
        return _consume_json_overhead(remaining, len(str(value)))
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("Non-finite JSON number")
        return _consume_json_overhead(remaining, len(str(value)))
    if isinstance(value, str):
        return _consume_json_overhead(remaining, _json_string_size(value))
    if isinstance(value, list):
        current = remaining
        for item in value:
            current = _validate_json_safe_with_budget(item, remaining=current)
        return current
    if isinstance(value, dict):
        current = remaining
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            current = _consume_json_overhead(current, _json_string_size(key))
            current = _validate_json_safe_with_budget(item, remaining=current)
        return current
    raise TypeError("Value is not JSON serializable")


def _consume_json_overhead(remaining: int, size: int) -> int:
    if size > remaining:
        raise _JsonLimitExceeded("JSON payload exceeds size limit")
    return remaining - size


def _json_string_size(value: str) -> int:
    size = 2
    for char in value:
        codepoint = ord(char)
        if char in {'"', "\\"}:
            size += 2
        elif codepoint <= 0x1F:
            size += 6
        elif codepoint <= 0x7F:
            size += 1
        elif codepoint <= 0x7FF:
            size += 2
        elif codepoint <= 0xFFFF:
            size += 3
        else:
            size += 4
    return size


def _validate_model_event(event: RuntimeEvent, model_started: bool, model_turn_completed: bool) -> str | None:
    allowed_types = {
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.TOOL_CALL_DELTA,
        RuntimeEventType.TOOL_CALL_COMPLETED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.MODEL_FAILED,
    }
    if event.type not in allowed_types:
        return "Model emitted an unsupported event"
    if model_turn_completed:
        return "Model emitted events after model turn completion"
    if event.type == RuntimeEventType.MODEL_STARTED:
        if model_started:
            return "Model emitted duplicate start event"
        return None
    if not model_started:
        return "Model emitted events before model start"
    if event.type == RuntimeEventType.TEXT_DELTA and not isinstance(event.text, str):
        return "Model emitted a text delta without text"
    if event.type == RuntimeEventType.TOOL_CALL_DELTA and not isinstance(
        event.tool_call_delta,
        ToolCallDelta,
    ):
        return "Model emitted a tool call delta without a delta"
    if event.type == RuntimeEventType.TOOL_CALL_DELTA:
        delta = cast(ToolCallDelta, event.tool_call_delta)
        if (
            not isinstance(delta.index, int)
            or isinstance(delta.index, bool)
            or delta.index < 0
        ):
            return "Model emitted a tool call delta with an invalid index"
        fragments = (
            delta.call_id_fragment,
            delta.name_fragment,
            delta.arguments_fragment,
            delta.type_fragment,
        )
        if any(not isinstance(fragment, str) for fragment in fragments):
            return "Model emitted a tool call delta with an invalid fragment"
    if event.type == RuntimeEventType.TOOL_CALL_COMPLETED and not isinstance(
        event.tool_call,
        ToolCall,
    ):
        return "Model emitted a completed tool call without a tool call"
    if event.type == RuntimeEventType.TOOL_CALL_COMPLETED:
        tool_call = cast(ToolCall, event.tool_call)
        if not isinstance(tool_call.id, str) or not tool_call.id:
            return "Model emitted a completed tool call with an invalid id"
        if not isinstance(tool_call.name, str) or not tool_call.name:
            return "Model emitted a completed tool call with an invalid name"
        if not isinstance(tool_call.arguments, dict):
            return "Model emitted a completed tool call with invalid arguments"
    if (
        event.type == RuntimeEventType.MODEL_TURN_COMPLETED
        and event.finish_reason not in _FINISH_REASONS
    ):
        return "Model completed a turn with an invalid finish reason"
    if event.type == RuntimeEventType.MODEL_FAILED and not isinstance(event.error, str):
        return "Model failed without an error"
    return None

