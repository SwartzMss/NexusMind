from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.messages import Message


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


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

