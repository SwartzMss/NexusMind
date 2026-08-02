from __future__ import annotations
import asyncio
from copy import deepcopy
from typing import Protocol
from .checkpoint import HarnessCheckpoint

class CheckpointStore(Protocol):
    async def save(self, checkpoint: HarnessCheckpoint) -> None: ...
    async def load_latest(self, run_id: str) -> HarnessCheckpoint | None: ...
    async def list(self, run_id: str) -> tuple[HarnessCheckpoint, ...]: ...

class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._items: dict[str, list[HarnessCheckpoint]] = {}
        self._lock = asyncio.Lock()

    async def save(self, checkpoint: HarnessCheckpoint) -> None:
        async with self._lock:
            checkpoint.validate()
            items = self._items.setdefault(checkpoint.run_id, [])
            if items and checkpoint.sequence <= items[-1].sequence:
                raise ValueError("Checkpoint sequence must increase")
            items.append(deepcopy(checkpoint))

    async def load_latest(self, run_id: str) -> HarnessCheckpoint | None:
        async with self._lock:
            return deepcopy(self._items.get(run_id, [])[-1]) if self._items.get(run_id) else None

    async def list(self, run_id: str) -> tuple[HarnessCheckpoint, ...]:
        async with self._lock:
            return tuple(deepcopy(item) for item in self._items.get(run_id, ()))
