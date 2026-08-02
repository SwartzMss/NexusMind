from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from copy import deepcopy

from nexusmind.runtime.harness.state import HarnessPhase, HarnessState, HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.messages import Message

MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024

class CheckpointBoundary(str, Enum):
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    RUN_TERMINAL = "run_terminal"

_PHASE_TO_BOUNDARY = {
    HarnessPhase.BEFORE_MODEL: CheckpointBoundary.BEFORE_MODEL,
    HarnessPhase.AFTER_MODEL: CheckpointBoundary.AFTER_MODEL,
    HarnessPhase.BEFORE_TOOL: CheckpointBoundary.BEFORE_TOOL,
    HarnessPhase.AFTER_TOOL: CheckpointBoundary.AFTER_TOOL,
    HarnessPhase.TERMINAL: CheckpointBoundary.RUN_TERMINAL,
}

@dataclass(frozen=True, slots=True)
class HarnessStateSnapshot:
    messages: tuple[Message, ...]
    model_turns: int
    tool_calls_total: int
    tool_argument_bytes_total: int
    tool_result_bytes_total: int
    started_tool_call_ids: tuple[str, ...]
    executed_tool_call_ids: tuple[str, ...]
    status: HarnessStatus
    stop_reason: StopReason | None
    phase: HarnessPhase

    @classmethod
    def from_state(cls, state: HarnessState, stop_reason: StopReason | None = None) -> "HarnessStateSnapshot":
        snapshot = cls(
            messages=tuple(deepcopy(state.messages)),
            model_turns=state.model_turns,
            tool_calls_total=state.tool_calls_total,
            tool_argument_bytes_total=state.tool_argument_bytes_total,
            tool_result_bytes_total=state.tool_result_bytes_total,
            started_tool_call_ids=tuple(sorted(state.started_tool_call_ids)),
            executed_tool_call_ids=tuple(sorted(state.executed_tool_call_ids)),
            status=state.status,
            stop_reason=stop_reason if stop_reason is not None else state.stop_reason,
            phase=state.phase,
        )
        _ensure_json_size(snapshot)
        return snapshot

@dataclass(frozen=True, slots=True)
class HarnessCheckpoint:
    schema_version: int
    checkpoint_id: str
    run_id: str
    sequence: int
    boundary: CheckpointBoundary
    state: HarnessStateSnapshot
    created_at: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Invalid checkpoint schema version")
        if type(self.checkpoint_id) is not str or not self.checkpoint_id or type(self.run_id) is not str or not self.run_id:
            raise ValueError("Invalid checkpoint identity or schema version")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Checkpoint sequence must be non-negative")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Checkpoint timestamp must be ISO-8601") from exc
        self.validate()

    def validate(self) -> None:
        if type(self.boundary) is not CheckpointBoundary or not isinstance(self.state, HarnessStateSnapshot):
            raise ValueError("Invalid checkpoint boundary or state")
        if type(self.state.phase) is not HarnessPhase or self.boundary is not _PHASE_TO_BOUNDARY[self.state.phase]:
            raise ValueError("Checkpoint boundary does not match execution phase")
        if type(self.state.status) is not HarnessStatus or (self.state.stop_reason is not None and type(self.state.stop_reason) is not StopReason):
            raise ValueError("Invalid checkpoint status or stop reason")
        active_tools = set(self.state.started_tool_call_ids) - set(self.state.executed_tool_call_ids)
        if active_tools:
            raise ValueError("Cannot create a checkpoint while a tool may be running")
        if self.state.status is HarnessStatus.RUNNING and self.state.stop_reason is not None:
            raise ValueError("Running checkpoint cannot have a stop reason")
        if self.state.status is HarnessStatus.COMPLETED and self.state.stop_reason is not StopReason.MODEL_COMPLETED:
            raise ValueError("Completed checkpoint requires model_completed")
        if self.state.status is HarnessStatus.CANCELLED and self.state.stop_reason is not StopReason.CANCELLED:
            raise ValueError("Cancelled checkpoint requires cancelled")
        if self.state.status is HarnessStatus.FAILED and self.state.stop_reason not in {StopReason.MODEL_FAILED, StopReason.TOOL_FAILED, StopReason.LIMIT_EXCEEDED, StopReason.RUNTIME_ERROR}:
            raise ValueError("Failed checkpoint has an invalid stop reason")
        if self.boundary is CheckpointBoundary.RUN_TERMINAL:
            if self.state.status is HarnessStatus.RUNNING or self.state.stop_reason is None:
                raise ValueError("Terminal checkpoint requires a terminal status and stop reason")
        elif self.state.status is not HarnessStatus.RUNNING:
            raise ValueError("Non-terminal checkpoint requires a running state")
        _ensure_json_size(self)

    @classmethod
    def create(cls, state: HarnessState, run_id: str, sequence: int, boundary: CheckpointBoundary | None = None, stop_reason: StopReason | None = None) -> "HarnessCheckpoint":
        if boundary is None:
            boundary = _PHASE_TO_BOUNDARY[state.phase]
        if type(boundary) is not CheckpointBoundary:
            raise ValueError("Invalid checkpoint boundary")
        if boundary is not _PHASE_TO_BOUNDARY[state.phase]:
            raise ValueError("Checkpoint boundary does not match execution phase")
        active_tools = state.started_tool_call_ids - state.executed_tool_call_ids
        if active_tools:
            raise ValueError("Cannot create a safe checkpoint while a tool may be running")
        effective_reason = stop_reason if stop_reason is not None else state.stop_reason
        if boundary is CheckpointBoundary.RUN_TERMINAL:
            if state.status is HarnessStatus.RUNNING or effective_reason is None:
                raise ValueError("Terminal checkpoint requires a terminal status and stop reason")
        return cls(1, uuid4().hex, run_id, sequence, boundary, HarnessStateSnapshot.from_state(state, effective_reason), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if isinstance(value, (tuple, list)): return [_jsonable(item) for item in value]
    if isinstance(value, Message): return {"role": value.role.value, "content": value.content, "name": value.name, "tool_call_id": value.tool_call_id, "tool_calls": [_jsonable(call) for call in value.tool_calls], "metadata": _jsonable(value.metadata)}
    if hasattr(value, "__dataclass_fields__"): return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("Checkpoint JSON object keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, float) and not __import__("math").isfinite(value):
        raise ValueError("Checkpoint contains a non-finite number")
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    raise ValueError("Checkpoint contains unsupported value")

def _ensure_json_size(value: Any) -> None:
    payload = _jsonable(value)
    _reject_secrets(payload)
    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds maximum size")

def _reject_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("secret", "token", "api_key", "apikey", "password")):
                raise ValueError("Checkpoint contains a secret-like field")
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item)
