from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

class RunStatus(str, Enum):
    RUNNING = "running"; COMPLETED = "completed"; FAILED = "failed"; CANCELLED = "cancelled"; ABANDONED = "abandoned"
class RunKind(str, Enum):
    CHAT = "chat"; SKILL = "skill"
@dataclass(frozen=True, slots=True)
class RunStartContext:
    kind: RunKind; skill_name: str | None = None; model_name: str | None = None; input_text: str | None = None; record_content: bool = False
@dataclass(frozen=True, slots=True)
class RunTraceEvent:
    event_type: str; occurred_at: datetime; payload: dict[str, Any]
class RunRecorder(Protocol):
    async def start_run(self, context: RunStartContext) -> str: ...
    async def append_event(self, run_id: str, event: RunTraceEvent) -> None: ...
    async def finish_run(self, run_id: str, status: RunStatus, *, error_code: str | None = None, error_message: str | None = None) -> None: ...
