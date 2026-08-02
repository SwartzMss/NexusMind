from dataclasses import dataclass, field
from enum import Enum
from nexusmind.runtime.messages import Message
from nexusmind.runtime.harness.stop import StopReason

class HarnessStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class HarnessPhase(str, Enum):
    BEFORE_MODEL = "before_model"
    MODEL_RUNNING = "model_running"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    TOOL_RUNNING = "tool_running"
    AFTER_TOOL = "after_tool"
    TERMINAL = "terminal"

@dataclass(slots=True)
class HarnessState:
    messages: list[Message]
    model_turns: int = 0
    tool_calls_total: int = 0
    tool_argument_bytes_total: int = 0
    tool_result_bytes_total: int = 0
    started_tool_call_ids: set[str] = field(default_factory=set)
    executed_tool_call_ids: set[str] = field(default_factory=set)
    status: HarnessStatus = HarnessStatus.RUNNING
    stop_reason: StopReason | None = None
    phase: HarnessPhase = HarnessPhase.BEFORE_MODEL
