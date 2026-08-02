from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .checkpoint import HarnessCheckpoint
from .context import HarnessRequest
from .limits import HarnessLimits
from .state import HarnessPhase, HarnessState, HarnessStatus
from .stop import StopReason
from nexusmind.tools.contracts import ToolDefinition

class HarnessResumeError(RuntimeError):
    pass

class HarnessResumeCompatibilityError(HarnessResumeError):
    pass

class HarnessResumeStateError(HarnessResumeError):
    pass

@dataclass(frozen=True, slots=True)
class HarnessResumeRequest:
    checkpoint: HarnessCheckpoint
    tools: tuple[ToolDefinition, ...] = ()
    limits: HarnessLimits | None = None
    metadata: dict[str, object] = field(default_factory=dict)

def state_from_checkpoint(checkpoint: HarnessCheckpoint) -> HarnessState:
    checkpoint.validate()
    if checkpoint.state.status is not HarnessStatus.RUNNING:
        raise HarnessResumeStateError("Only running checkpoints can be resumed")
    if checkpoint.state.phase is not HarnessPhase.BEFORE_MODEL:
        raise HarnessResumeStateError("Checkpoint phase is not safely resumable")
    return HarnessState(
        messages=deepcopy(list(checkpoint.state.messages)),
        model_turns=checkpoint.state.model_turns,
        tool_calls_total=checkpoint.state.tool_calls_total,
        tool_argument_bytes_total=checkpoint.state.tool_argument_bytes_total,
        tool_result_bytes_total=checkpoint.state.tool_result_bytes_total,
        started_tool_call_ids=set(checkpoint.state.started_tool_call_ids),
        executed_tool_call_ids=set(checkpoint.state.executed_tool_call_ids),
        status=checkpoint.state.status,
        stop_reason=checkpoint.state.stop_reason,
        phase=checkpoint.state.phase,
    )
