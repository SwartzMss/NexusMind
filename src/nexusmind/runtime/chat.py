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
_MIN_TOOL_RESULT_ENVELOPE_BYTES = len('{"ok":true,"output":0}'.encode("utf-8"))
_MIN_TOOL_RESULT_ENVELOPE_NODES = 3


@dataclass(frozen=True, slots=True)
class AgentLoopLimits:
    max_model_turns: int = 8
    max_tool_calls_total: int = 32
    max_tool_arguments_bytes_per_call: int = 1024 * 1024
    max_tool_arguments_bytes_total: int = 4 * 1024 * 1024
    max_tool_result_bytes_per_call: int = 1024 * 1024
    max_tool_result_bytes_total: int = 4 * 1024 * 1024
    max_json_nodes_per_payload: int = 100_000
    max_json_depth: int = 100

    def __post_init__(self) -> None:
        for field_name in (
            "max_model_turns",
            "max_tool_calls_total",
            "max_tool_arguments_bytes_per_call",
            "max_tool_arguments_bytes_total",
            "max_tool_result_bytes_per_call",
            "max_tool_result_bytes_total",
            "max_json_nodes_per_payload",
            "max_json_depth",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Agent loop limits must be positive integers")
        if self.max_json_nodes_per_payload < _MIN_TOOL_RESULT_ENVELOPE_NODES:
            raise ValueError("Agent loop limits must allow a minimal tool result envelope")


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
                    async for event in self._model.stream(_snapshot_messages(messages), tools=tools):
                        validation_error = _validate_model_event(event, turn.model_started, turn.completed)
                        if validation_error:
                            yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=validation_error)
                            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=validation_error)
                            return
                        event = cast(RuntimeEvent, event)
                        if event.type == RuntimeEventType.MODEL_STARTED:
                            turn.model_started = True
                        elif event.type == RuntimeEventType.MODEL_FAILED:
                            yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=_MODEL_EXECUTION_ERROR)
                            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_MODEL_EXECUTION_ERROR)
                            return
                        elif event.type == RuntimeEventType.TEXT_DELTA:
                            turn.text_parts.append(cast(str, event.text))
                        elif event.type == RuntimeEventType.TOOL_CALL_COMPLETED:
                            tool_call = cast(ToolCall, event.tool_call)
                            if type(tool_call.id) is not str or type(tool_call.name) is not str:
                                yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=_RUNTIME_ERROR)
                                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                                return
                            if tool_calls_total + len(turn.tool_calls) + 1 > self._limits.max_tool_calls_total:
                                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
                                return
                            try:
                                safe_tool_call, arguments_size = _snapshot_tool_call(
                                    tool_call,
                                    max_bytes_per_call=self._limits.max_tool_arguments_bytes_per_call,
                                    remaining_total_bytes=(
                                        self._limits.max_tool_arguments_bytes_total
                                        - tool_arguments_bytes_total
                                        - turn.tool_arguments_size
                                    ),
                                    max_nodes=self._limits.max_json_nodes_per_payload,
                                    max_depth=self._limits.max_json_depth,
                                )
                            except _JsonLimitExceeded:
                                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
                                return
                            except RuntimeError:
                                yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=_RUNTIME_ERROR)
                                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                                return
                            turn.tool_calls.append(safe_tool_call)
                            turn.tool_arguments_size += arguments_size
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
                if model_turns >= self._limits.max_model_turns:
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
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
                tool_arguments_bytes_total += turn.tool_arguments_size
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content="".join(turn.text_parts) or None,
                        tool_calls=tuple(turn.tool_calls),
                    )
                )
                for call in turn.tool_calls:
                    remaining_result_bytes = self._limits.max_tool_result_bytes_total - tool_result_bytes_total
                    if (
                        remaining_result_bytes < _MIN_TOOL_RESULT_ENVELOPE_BYTES
                        or self._limits.max_tool_result_bytes_per_call < _MIN_TOOL_RESULT_ENVELOPE_BYTES
                    ):
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_LIMIT_ERROR)
                        return
                    try:
                        result = await self._tool_executor.execute(call)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                        return
                    if (
                        not isinstance(result, ToolResult)
                        or type(result.call_id) is not str
                        or not result.call_id
                        or type(result.name) is not str
                        or not result.name
                    ):
                        yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_RUNTIME_ERROR)
                        return
                    if str.__ne__(result.call_id, call.id) or str.__ne__(result.name, call.name):
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
                            max_nodes=self._limits.max_json_nodes_per_payload,
                            max_depth=self._limits.max_json_depth,
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
        self.tool_arguments_size = 0
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


def _snapshot_tool_call(
    call: ToolCall,
    *,
    max_bytes_per_call: int,
    remaining_total_bytes: int,
    max_nodes: int,
    max_depth: int,
) -> tuple[ToolCall, int]:
    try:
        arguments_json, size = _bounded_json(
            call.arguments,
            max_bytes=min(max_bytes_per_call, remaining_total_bytes),
            max_nodes=max_nodes,
            max_depth=max_depth,
        )
        arguments = json.loads(arguments_json)
    except (TypeError, ValueError, RecursionError):
        raise RuntimeError("Tool call arguments are not strict JSON") from None
    if not isinstance(arguments, dict):
        raise RuntimeError("Tool call arguments are not a JSON object")
    return ToolCall(id=call.id, name=call.name, arguments=arguments), size


def _valid_tool_result(result: ToolResult) -> bool:
    if type(result.metadata) is not dict:
        return False
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
    max_nodes: int,
    max_depth: int,
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
        return _bounded_json(
            payload,
            max_bytes=min(max_bytes_per_call, remaining_total_bytes),
            max_nodes=max_nodes,
            max_depth=max_depth,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise RuntimeError("Tool result is not JSON serializable") from exc


class _JsonLimitExceeded(RuntimeError):
    pass


def _bounded_json(
    payload: object,
    *,
    max_bytes: int,
    max_nodes: int,
    max_depth: int,
) -> tuple[str, int]:
    _JsonBudget(max_bytes=max_bytes, max_nodes=max_nodes, max_depth=max_depth).validate(payload)
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    parts: list[str] = []
    size = 0
    for part in encoder.iterencode(payload):
        size += len(part.encode("utf-8"))
        if size > max_bytes:
            raise _JsonLimitExceeded("JSON payload exceeds size limit")
        parts.append(part)
    return "".join(parts), size


class _JsonBudget:
    def __init__(self, *, max_bytes: int, max_nodes: int, max_depth: int) -> None:
        self.remaining_bytes = max_bytes
        self.remaining_nodes = max_nodes
        self.max_depth = max_depth
        self.seen_containers: set[int] = set()

    def validate(self, value: object, *, depth: int = 0) -> None:
        self._consume_node()
        if depth > self.max_depth:
            raise _JsonLimitExceeded("JSON payload exceeds nesting depth")
        if value is None:
            self._consume_bytes(4)
            return
        if isinstance(value, bool):
            self._consume_bytes(4 if value else 5)
            return
        if isinstance(value, int) and not isinstance(value, bool):
            self._consume_bytes(len(str(value)))
            return
        if isinstance(value, float):
            if value != value or value in {float("inf"), float("-inf")}:
                raise ValueError("Non-finite JSON number")
            self._consume_bytes(len(str(value)))
            return
        if isinstance(value, str):
            self._consume_bytes(_json_string_size(value))
            return
        if isinstance(value, list):
            self._enter_container(value)
            try:
                self._consume_bytes(2)
                if value:
                    self._consume_bytes(len(value) - 1)
                for item in value:
                    self.validate(item, depth=depth + 1)
            finally:
                self._leave_container(value)
            return
        if isinstance(value, dict):
            self._enter_container(value)
            try:
                self._consume_bytes(2)
                if value:
                    self._consume_bytes(len(value) - 1)
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("JSON object keys must be strings")
                    self._consume_bytes(_json_string_size(key))
                    self._consume_bytes(1)
                    self.validate(item, depth=depth + 1)
            finally:
                self._leave_container(value)
            return
        raise TypeError("Value is not JSON serializable")

    def _consume_node(self) -> None:
        if self.remaining_nodes <= 0:
            raise _JsonLimitExceeded("JSON payload exceeds node limit")
        self.remaining_nodes -= 1

    def _consume_bytes(self, size: int) -> None:
        if size > self.remaining_bytes:
            raise _JsonLimitExceeded("JSON payload exceeds size limit")
        self.remaining_bytes -= size

    def _enter_container(self, value: object) -> None:
        container_id = id(value)
        if container_id in self.seen_containers:
            raise ValueError("Circular JSON value")
        self.seen_containers.add(container_id)

    def _leave_container(self, value: object) -> None:
        self.seen_containers.remove(id(value))


def _json_string_size(value: str) -> int:
    short_escapes = {"\b", "\f", "\n", "\r", "\t"}
    size = 2
    for char in value:
        codepoint = ord(char)
        if char in {'"', "\\"}:
            size += 2
        elif char in short_escapes:
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


def _snapshot_messages(messages: list[Message]) -> list[Message]:
    return [
        Message(
            role=message.role,
            content=message.content,
            name=message.name,
            tool_call_id=message.tool_call_id,
            tool_calls=message.tool_calls,
            metadata=message.metadata,
        )
        for message in messages
    ]


def _validate_model_event(event: object, model_started: bool, model_turn_completed: bool) -> str | None:
    if type(event) is not RuntimeEvent:
        return "Model emitted an invalid event DTO"
    event = cast(RuntimeEvent, event)
    if type(event.type) is not RuntimeEventType:
        return "Model emitted an event with an invalid type"
    if type(event.metadata) is not dict:
        return "Model emitted an event with invalid metadata"
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
    payload_error = _validate_event_payload_shape(event)
    if payload_error:
        return payload_error
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


def _validate_event_payload_shape(event: RuntimeEvent) -> str | None:
    payload_fields = {
        "text": event.text,
        "error": event.error,
        "tool_call_delta": event.tool_call_delta,
        "tool_call": event.tool_call,
        "tool_result": event.tool_result,
        "finish_reason": event.finish_reason,
    }
    allowed_fields_by_type = {
        RuntimeEventType.MODEL_STARTED: set(),
        RuntimeEventType.TEXT_DELTA: {"text"},
        RuntimeEventType.TOOL_CALL_DELTA: {"tool_call_delta"},
        RuntimeEventType.TOOL_CALL_COMPLETED: {"tool_call"},
        RuntimeEventType.MODEL_TURN_COMPLETED: {"finish_reason"},
        RuntimeEventType.MODEL_FAILED: {"error"},
    }
    allowed_fields = allowed_fields_by_type.get(event.type, set())
    for field_name, value in payload_fields.items():
        if field_name not in allowed_fields and value is not None:
            return "Model emitted an event with conflicting payload fields"
    return None

