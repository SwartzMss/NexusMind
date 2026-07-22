from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeEventType(str, Enum):
    RUN_STARTED = "run_started"
    MODEL_STARTED = "model_started"
    TEXT_DELTA = "text_delta"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    type: RuntimeEventType
    text: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
