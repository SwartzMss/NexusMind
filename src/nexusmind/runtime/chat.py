from __future__ import annotations

from collections.abc import AsyncIterator

from nexusmind.models.base import ChatModel
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolDefinition


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
            async for event in self._model.stream(messages, tools=tools):
                validation_error = _validate_model_event(event, model_started, model_turn_completed)
                if validation_error:
                    yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=validation_error)
                    yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=validation_error)
                    return
                if event.type == RuntimeEventType.MODEL_STARTED:
                    model_started = True
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
            if model_turn_finish_reason != "tool_calls":
                yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)
        except Exception as exc:
            yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error=str(exc))
            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=str(exc))


def _validate_model_event(event: RuntimeEvent, model_started: bool, model_turn_completed: bool) -> str | None:
    if model_turn_completed:
        return "Model emitted events after model turn completion"
    if event.type == RuntimeEventType.MODEL_STARTED:
        if model_started:
            return "Model emitted duplicate start event"
        return None
    if not model_started:
        return "Model emitted events before model start"
    if event.type == RuntimeEventType.RUN_COMPLETED or event.type == RuntimeEventType.RUN_FAILED:
        return "Model emitted run-level event"
    return None

