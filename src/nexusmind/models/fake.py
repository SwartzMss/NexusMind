from __future__ import annotations

from collections.abc import AsyncIterator

from nexusmind.models.base import ChatModel, ChatModelError, ToolDefinition
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message


class FakeChatModel(ChatModel):
    def __init__(self, deltas: list[str] | None = None, error: Exception | None = None) -> None:
        self._deltas = deltas or ["Hello", ", ", "NexusMind"]
        self._error = error

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        if self._error is not None:
            raise ChatModelError(str(self._error)) from self._error
        for delta in self._deltas:
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text=delta)

