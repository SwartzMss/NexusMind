from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType

from .checkpoint import CheckpointBoundary
from .checkpoint_store import CheckpointStore
from .runner import HarnessExecution
from .state import HarnessPhase, HarnessStatus
from .stop import StopReason


class CheckpointPersistenceError(RuntimeError):
    """Raised internally when a durability barrier cannot be committed."""


_BARRIER_EVENTS = {
    RuntimeEventType.MODEL_TURN_COMPLETED: CheckpointBoundary.AFTER_MODEL,
    RuntimeEventType.TOOL_RESULT: CheckpointBoundary.AFTER_TOOL,
    RuntimeEventType.RUN_COMPLETED: CheckpointBoundary.RUN_TERMINAL,
    RuntimeEventType.RUN_FAILED: CheckpointBoundary.RUN_TERMINAL,
}


class CheckpointCoordinator:
    """Persist safe execution boundaries before releasing the next event.

    The wrapped execution is an async generator. Receiving an event pauses it
    until the coordinator asks for the next event, which makes the store save
    a real barrier before a sibling tool or the next model turn can run.
    """

    def __init__(
        self,
        execution: HarnessExecution,
        checkpoint_store: CheckpointStore,
        *,
        run_id: str,
        start_sequence: int | None = None,
        save_terminal: bool = True,
    ) -> None:
        if type(run_id) is not str or not run_id:
            raise ValueError("Checkpoint run_id must be a non-empty string")
        if start_sequence is not None and (type(start_sequence) is not int or start_sequence < 0):
            raise ValueError("Checkpoint start sequence must be a non-negative integer")
        self._execution = execution
        self._store = checkpoint_store
        self._run_id = run_id
        self._next_sequence = (
            start_sequence
            if start_sequence is not None
            else (execution.last_checkpoint_sequence + 1 if execution.last_checkpoint_sequence is not None else 0)
        )
        self._save_terminal = save_terminal
        self._terminal_saved = False
        self._stream_started = False

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    async def stream(self) -> AsyncIterator[RuntimeEvent]:
        if self._stream_started:
            raise RuntimeError("CheckpointCoordinator can only be streamed once")
        self._stream_started = True
        try:
            async for event in self._execution.stream():
                boundary = _BARRIER_EVENTS.get(event.type)
                if boundary is None or (boundary is CheckpointBoundary.RUN_TERMINAL and not self._save_terminal):
                    yield event
                    continue
                try:
                    await self._persist(boundary)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    yield self._persistence_failure(boundary, event)
                    return
                yield event
                if boundary is CheckpointBoundary.RUN_TERMINAL:
                    self._terminal_saved = True
        except asyncio.CancelledError:
            # HarnessExecution records cancellation and terminal phase before
            # re-raising. Save it only when no tool could be left in-flight.
            if self._save_terminal and not self._terminal_saved:
                try:
                    await self._persist(CheckpointBoundary.RUN_TERMINAL)
                    self._terminal_saved = True
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            raise

    async def _persist(self, boundary: CheckpointBoundary) -> None:
        checkpoint = self._execution.create_checkpoint(
            run_id=self._run_id,
            sequence=self._next_sequence,
            boundary=boundary,
            commit=False,
        )
        await self._store.save(checkpoint)
        # Only move both cursors after the store has accepted the checkpoint.
        self._execution.commit_checkpoint(checkpoint)
        self._next_sequence += 1

    def _persistence_failure(
        self,
        boundary: CheckpointBoundary,
        original_event: RuntimeEvent | None = None,
    ) -> RuntimeEvent:
        self._execution.state.status = HarnessStatus.FAILED
        self._execution.state.phase = HarnessPhase.TERMINAL
        self._execution.stop_reason = StopReason.RUNTIME_ERROR
        self._execution.state.stop_reason = StopReason.RUNTIME_ERROR
        metadata = dict(original_event.metadata) if original_event is not None else {}
        metadata.update(
            {
                "checkpoint_persistence_failed": True,
                "checkpoint_boundary": boundary.value,
                "stop_reason": StopReason.RUNTIME_ERROR.value,
            }
        )
        error = f"Automatic checkpoint persistence failed at {boundary.value}"
        if original_event is not None and original_event.error:
            error = f"{error}: {original_event.error}"
        return RuntimeEvent(
            RuntimeEventType.RUN_FAILED,
            error=error,
            metadata=metadata,
        )


async def checkpoint_stream(
    execution: HarnessExecution,
    checkpoint_store: CheckpointStore,
    *,
    run_id: str,
    start_sequence: int | None = None,
    save_terminal: bool = True,
) -> AsyncIterator[RuntimeEvent]:
    """Convenience wrapper for callers that do not need the coordinator object."""
    coordinator = CheckpointCoordinator(
        execution,
        checkpoint_store,
        run_id=run_id,
        start_sequence=start_sequence,
        save_terminal=save_terminal,
    )
    async for event in coordinator.stream():
        yield event
