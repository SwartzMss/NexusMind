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
            model_turn_finish_reason: str | None = None
            async for event in self._model.stream(messages, tools=tools):
                if event.type == RuntimeEventType.MODEL_TURN_COMPLETED:
                    model_turn_finish_reason = event.finish_reason
                yield event
            if model_turn_finish_reason != "tool_calls":
                yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)
        except Exception as exc:
            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error=str(exc))

