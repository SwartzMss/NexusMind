from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.messages import Message
from nexusmind.tools.contracts import ToolDefinition


class ChatModelError(Exception):
    """Provider-neutral model failure."""


class ChatModel(ABC):
    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        """Stream provider-neutral runtime events."""

