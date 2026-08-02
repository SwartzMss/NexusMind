from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from nexusmind.runtime.messages import Message
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.tools.contracts import ToolDefinition

@dataclass(frozen=True, slots=True)
class HarnessRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    limits: HarnessLimits | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(deepcopy(self.messages)))
        object.__setattr__(self, "tools", tuple(deepcopy(self.tools)))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
