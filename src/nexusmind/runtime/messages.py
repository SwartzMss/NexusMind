from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nexusmind.tools.contracts import ToolCall


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.content is None and not (
            self.role == MessageRole.ASSISTANT and self.tool_calls
        ):
            raise ValueError("Only assistant tool call messages may omit content")
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError("Message content must be a string or None")
        if self.tool_calls and self.role != MessageRole.ASSISTANT:
            raise ValueError("Only assistant messages may carry tool calls")
        if self.role == MessageRole.TOOL and not self.tool_call_id:
            raise ValueError("Tool messages must carry tool_call_id")
        if self.role != MessageRole.TOOL and self.tool_call_id is not None:
            raise ValueError("Only tool messages may carry tool_call_id")
        object.__setattr__(
            self,
            "tool_calls",
            tuple(
                ToolCall(id=call.id, name=call.name, arguments=deepcopy(call.arguments))
                for call in self.tool_calls
            ),
        )
