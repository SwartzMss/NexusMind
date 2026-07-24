from __future__ import annotations

from collections.abc import AsyncIterator

from nexusmind.models.base import ChatModel
from nexusmind.models.tool_calls import ToolCallDelta
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolCall, ToolDefinition

_MODEL_EXECUTION_ERROR = "Model execution failed"
_FINISH_REASONS = {
    "stop",
    "tool_calls",
    "length",
    "content_filter",
    "unknown",
    "null",
}


class ChatRuntime:
    def __init__(self, model: ChatModel) -> None:
        self._model = model

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
            model_started = False
            model_turn_completed = False
            model_turn_finish_reason: str | None = None
            has_completed_tool_calls = False
            async for event in self._model.stream(messages, tools=tools):
                validation_error = _validate_model_event(event, model_started, model_turn_completed)
                if validation_error:
                    yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=validation_error)
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=validation_error)
                    return
                if event.type == RuntimeEventType.MODEL_STARTED:
                    model_started = True
                if event.type == RuntimeEventType.MODEL_FAILED:
                    yield RuntimeEvent(
                        RuntimeEventType.MODEL_FAILED,
                        error=_MODEL_EXECUTION_ERROR,
                    )
                    yield RuntimeEvent(
                        RuntimeEventType.RUN_FAILED,
                        error=_MODEL_EXECUTION_ERROR,
                    )
                    return
                if event.type == RuntimeEventType.TOOL_CALL_COMPLETED:
                    has_completed_tool_calls = True
                if event.type == RuntimeEventType.MODEL_TURN_COMPLETED:
                    model_turn_completed = True
                    model_turn_finish_reason = event.finish_reason
                yield event
            if not model_started:
                error = "Model stream ended before model start"
                yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=error)
                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=error)
                return
            if not model_turn_completed:
                error = "Model stream ended before model turn completion"
                yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=error)
                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=error)
                return
            if model_turn_finish_reason == "tool_calls" and not has_completed_tool_calls:
                error = "Model turn requested tools without completed tool calls"
                yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=error)
                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=error)
                return
            if has_completed_tool_calls and model_turn_finish_reason != "tool_calls":
                error = "Model turn completed tool calls with an incompatible finish reason"
                yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=error)
                yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=error)
                return
            if model_turn_finish_reason != "tool_calls" and not has_completed_tool_calls:
                yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)
        except Exception:
            yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=_MODEL_EXECUTION_ERROR)
            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=_MODEL_EXECUTION_ERROR)


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
    if event.type == RuntimeEventType.TOOL_CALL_COMPLETED and not isinstance(
        event.tool_call,
        ToolCall,
    ):
        return "Model emitted a completed tool call without a tool call"
    if (
        event.type == RuntimeEventType.MODEL_TURN_COMPLETED
        and event.finish_reason not in _FINISH_REASONS
    ):
        return "Model completed a turn with an invalid finish reason"
    if event.type == RuntimeEventType.MODEL_FAILED and not isinstance(event.error, str):
        return "Model failed without an error"
    return None

