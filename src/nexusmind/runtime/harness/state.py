from dataclasses import dataclass, field
from enum import Enum
from nexusmind.runtime.messages import Message
from nexusmind.runtime.harness.stop import StopReason

class HarnessStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

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
