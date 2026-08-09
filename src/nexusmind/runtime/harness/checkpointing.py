from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType

from .checkpoint import CheckpointBoundary, HarnessCheckpoint
from .checkpoint_store import CheckpointStore
from .runner import HarnessExecution
from .state import HarnessPhase, HarnessStatus
from .stop import StopReason


class CheckpointBarrierCancelled(asyncio.CancelledError):
    """Cancellation delivered after an in-flight durability outcome is known."""

    def __init__(
        self,
        boundary: CheckpointBoundary,
        *,
        checkpoint_saved: bool,
        checkpoint_save_failed: bool = False,
        checkpoint_commit_failed: bool = False,
    ) -> None:
        super().__init__(f"Cancelled during checkpoint barrier at {boundary.value}")
        self.boundary = boundary
        self.checkpoint_saved = checkpoint_saved
        self.checkpoint_save_failed = checkpoint_save_failed
        self.checkpoint_commit_failed = checkpoint_commit_failed


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
        self._cancelled_during_barrier = False
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
                if boundary is CheckpointBoundary.RUN_TERMINAL and self._has_active_tools():
                    # A started tool with no trusted result is intentionally not
                    # checkpointable. Preserve the original terminal failure.
                    yield event
                    continue
                try:
                    checkpoint = self._build_checkpoint(boundary)
                except Exception:
                    yield self._creation_failure(boundary, event)
                    return
                try:
                    await self._save_checkpoint(checkpoint, boundary)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    yield self._persistence_failure(boundary, event)
                    return
                try:
                    self._commit_checkpoint(checkpoint)
                except Exception:
                    yield self._commit_failure(boundary, event)
                    return
                yield event
                if boundary is CheckpointBoundary.RUN_TERMINAL:
                    self._terminal_saved = True
        except asyncio.CancelledError:
            # HarnessExecution records cancellation and terminal phase before
            # re-raising. Save it only when no tool could be left in-flight.
            if (
                not self._cancelled_during_barrier
                and self._save_terminal
                and not self._terminal_saved
                and not self._has_active_tools()
            ):
                try:
                    checkpoint = self._build_checkpoint(CheckpointBoundary.RUN_TERMINAL)
                    await self._save_checkpoint(checkpoint, CheckpointBoundary.RUN_TERMINAL)
                    self._commit_checkpoint(checkpoint)
                    self._terminal_saved = True
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            raise

    def _has_active_tools(self) -> bool:
        state = self._execution.state
        return bool(state.started_tool_call_ids - state.executed_tool_call_ids)

    def _build_checkpoint(self, boundary: CheckpointBoundary) -> HarnessCheckpoint:
        return self._execution.create_checkpoint(
            run_id=self._run_id,
            sequence=self._next_sequence,
            boundary=boundary,
            commit=False,
        )

    async def _save_checkpoint(
        self,
        checkpoint: HarnessCheckpoint,
        boundary: CheckpointBoundary,
    ) -> None:
        save_task = asyncio.create_task(self._store.save(checkpoint))
        cancellation: asyncio.CancelledError | None = None
        save_error: Exception | None = None

        while not save_task.done():
            try:
                await asyncio.shield(save_task)
            except asyncio.CancelledError as exc:
                if save_task.cancelled():
                    save_error = RuntimeError("Checkpoint store save was cancelled")
                    break
                if cancellation is None:
                    cancellation = exc
            except Exception:
                break

        if save_error is None:
            try:
                save_task.result()
            except asyncio.CancelledError:
                save_error = RuntimeError("Checkpoint store save was cancelled")
            except Exception as exc:
                save_error = exc

        if cancellation is None:
            if save_error is not None:
                raise save_error
            return

        if boundary is CheckpointBoundary.RUN_TERMINAL:
            # RUN_COMPLETED / RUN_FAILED has already fixed the terminal state.
            # A late cancellation must not make the persisted checkpoint,
            # execution, and run history disagree. Let stream() commit and
            # release the original terminal event after a successful save; a
            # failed save follows the normal persistence-failure path.
            if save_error is not None:
                raise save_error from cancellation
            return

        checkpoint_saved = save_error is None
        commit_failed = False
        if checkpoint_saved:
            try:
                self._commit_checkpoint(checkpoint)
            except Exception:
                commit_failed = True
        self._mark_cancelled()
        self._cancelled_during_barrier = True
        raise CheckpointBarrierCancelled(
            boundary,
            checkpoint_saved=checkpoint_saved,
            checkpoint_save_failed=save_error is not None,
            checkpoint_commit_failed=commit_failed,
        ) from cancellation

    def _commit_checkpoint(self, checkpoint: HarnessCheckpoint) -> None:
        # Only move both cursors after the store has accepted the checkpoint.
        self._execution.commit_checkpoint(checkpoint)
        self._next_sequence += 1

    def _creation_failure(
        self,
        boundary: CheckpointBoundary,
        original_event: RuntimeEvent,
    ) -> RuntimeEvent:
        metadata: dict[str, object] = {"checkpoint_creation_failed": True}
        if boundary is CheckpointBoundary.AFTER_TOOL:
            metadata.update(self._after_tool_durability_loss_metadata())
        return self._runtime_failure(
            boundary,
            original_event,
            error=f"Automatic checkpoint creation failed at {boundary.value}",
            metadata=metadata,
        )

    def _commit_failure(
        self,
        boundary: CheckpointBoundary,
        original_event: RuntimeEvent,
    ) -> RuntimeEvent:
        metadata: dict[str, object] = {"checkpoint_commit_failed": True}
        if boundary is CheckpointBoundary.AFTER_TOOL:
            metadata.update(self._after_tool_trace_incomplete_metadata())
        return self._runtime_failure(
            boundary,
            original_event,
            error=f"Automatic checkpoint commit failed at {boundary.value}",
            metadata=metadata,
        )

    def _persistence_failure(
        self,
        boundary: CheckpointBoundary,
        original_event: RuntimeEvent | None = None,
    ) -> RuntimeEvent:
        metadata: dict[str, object] = {"checkpoint_persistence_failed": True}
        if boundary is CheckpointBoundary.AFTER_TOOL:
            metadata.update(self._after_tool_durability_loss_metadata())
        error = f"Automatic checkpoint persistence failed at {boundary.value}"
        if original_event is not None and original_event.error:
            error = f"{error}: {original_event.error}"
        return self._runtime_failure(boundary, original_event, error=error, metadata=metadata)

    @staticmethod
    def _after_tool_trace_incomplete_metadata() -> dict[str, object]:
        return {
            "tool_execution_started": True,
            "trace_complete": False,
        }

    @classmethod
    def _after_tool_durability_loss_metadata(cls) -> dict[str, object]:
        return {
            **cls._after_tool_trace_incomplete_metadata(),
            "durability_lost_after_tool": True,
        }

    def _mark_cancelled(self) -> None:
        self._execution.state.status = HarnessStatus.CANCELLED
        self._execution.state.phase = HarnessPhase.TERMINAL
        self._execution.stop_reason = StopReason.CANCELLED
        self._execution.state.stop_reason = StopReason.CANCELLED

    def _runtime_failure(
        self,
        boundary: CheckpointBoundary,
        original_event: RuntimeEvent | None,
        *,
        error: str,
        metadata: dict[str, object],
    ) -> RuntimeEvent:
        self._execution.state.status = HarnessStatus.FAILED
        self._execution.state.phase = HarnessPhase.TERMINAL
        self._execution.stop_reason = StopReason.RUNTIME_ERROR
        self._execution.state.stop_reason = StopReason.RUNTIME_ERROR
        merged_metadata = dict(original_event.metadata) if original_event is not None else {}
        merged_metadata.update(metadata)
        merged_metadata.update(
            {
                "checkpoint_boundary": boundary.value,
                "stop_reason": StopReason.RUNTIME_ERROR.value,
            }
        )
        return RuntimeEvent(
            RuntimeEventType.RUN_FAILED,
            error=error,
            metadata=merged_metadata,
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
