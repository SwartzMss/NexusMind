from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nexusmind.models.tool_calls import ToolCallDelta
    from nexusmind.tools.contracts import ToolCall, ToolResult


class RuntimeEventType(str, Enum):
    RUN_STARTED = "run_started"
    MODEL_STARTED = "model_started"
    TEXT_DELTA = "text_delta"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    TOOL_CALL = "tool_call"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_RESULT = "tool_result"
    MODEL_TURN_COMPLETED = "model_turn_completed"
    MODEL_FAILED = "model_failed"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    type: RuntimeEventType
    text: str | None = None
    error: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
