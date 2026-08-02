from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from copy import deepcopy

from nexusmind.runtime.harness.state import HarnessState, HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.messages import Message

MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024

class CheckpointBoundary(str, Enum):
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    RUN_TERMINAL = "run_terminal"

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
        if self.schema_version != 1 or not self.checkpoint_id or not self.run_id:
            raise ValueError("Invalid checkpoint identity or schema version")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Checkpoint sequence must be non-negative")
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Checkpoint timestamp must be ISO-8601") from exc
        _ensure_json_size(self)

    @classmethod
    def create(cls, state: HarnessState, run_id: str, sequence: int, boundary: CheckpointBoundary) -> "HarnessCheckpoint":
        if boundary is CheckpointBoundary.BEFORE_TOOL and state.started_tool_call_ids:
            raise ValueError("Cannot checkpoint before a tool while a tool may be running")
        if boundary is CheckpointBoundary.AFTER_TOOL and state.started_tool_call_ids != state.executed_tool_call_ids:
            raise ValueError("Cannot checkpoint after a tool with an incomplete tool call")
        return cls(1, uuid4().hex, run_id, sequence, boundary, HarnessStateSnapshot.from_state(state), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum): return value.value
    if isinstance(value, tuple): return [_jsonable(item) for item in value]
    if isinstance(value, Message): return {"role": value.role.value, "content": value.content, "name": value.name, "tool_call_id": value.tool_call_id, "tool_calls": [_jsonable(call) for call in value.tool_calls], "metadata": _jsonable(value.metadata)}
    if hasattr(value, "__dataclass_fields__"): return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    raise ValueError("Checkpoint contains unsupported value")

def _ensure_json_size(value: Any) -> None:
    payload = _jsonable(value)
    _reject_secrets(payload)
    if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) > MAX_CHECKPOINT_BYTES:
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
