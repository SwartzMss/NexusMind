import asyncio
import math
from copy import deepcopy

from nexusmind.runtime.harness import (
    CheckpointBoundary,
    HarnessCheckpoint,
    HarnessState,
    HarnessStateSnapshot,
    InMemoryCheckpointStore,
)
from nexusmind.runtime.harness.state import HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.messages import Message, MessageRole


def test_checkpoint_snapshot_isolated_and_serializable() -> None:
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello", metadata={"files": ["a.c", "b.c"]})])
    snapshot = HarnessStateSnapshot.from_state(state)
    state.messages[0].metadata["changed"] = True
    assert snapshot.messages[0].metadata == {"files": ["a.c", "b.c"]}


def test_checkpoint_rejects_non_finite_numbers() -> None:
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello", metadata={"value": math.nan})])
    try:
        HarnessStateSnapshot.from_state(state)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite numbers must not be checkpointed")


def test_checkpoint_store_enforces_sequence_and_latest() -> None:
    async def run():
        store = InMemoryCheckpointStore()
        state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello")])
        first = HarnessCheckpoint.create(state, "run-1", 0, CheckpointBoundary.BEFORE_MODEL)
        state.status = HarnessStatus.COMPLETED
        state.stop_reason = StopReason.MODEL_COMPLETED
        second = HarnessCheckpoint.create(state, "run-1", 1, CheckpointBoundary.RUN_TERMINAL)
        await store.save(first)
        await store.save(second)
        assert await store.load_latest("run-1") == second
        assert len(await store.list("run-1")) == 2
        second.state.messages[0].metadata["mutated"] = True
        loaded = await store.load_latest("run-1")
        assert loaded.state.messages[0].metadata.get("mutated") is None

    asyncio.run(run())


def test_checkpoint_rejects_secret_like_metadata() -> None:
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello", metadata={"api_key": "hidden"})])
    try:
        HarnessStateSnapshot.from_state(state)
    except ValueError as exc:
        assert "secret" in str(exc)
    else:
        raise AssertionError("secret-like metadata must not be checkpointed")


def test_terminal_checkpoint_contains_stop_reason() -> None:
    state = HarnessState(messages=[], status=HarnessStatus.COMPLETED, stop_reason=StopReason.MODEL_COMPLETED)
    checkpoint = HarnessCheckpoint.create(state, "run-terminal", 0, CheckpointBoundary.RUN_TERMINAL)
    assert checkpoint.state.status.value == "completed"


def test_checkpoint_allows_after_tool_when_previous_tools_are_complete() -> None:
    state = HarnessState(messages=[], started_tool_call_ids={"call-1"}, executed_tool_call_ids={"call-1"})
    checkpoint = HarnessCheckpoint.create(state, "run-tool", 0, CheckpointBoundary.AFTER_TOOL)
    assert checkpoint.boundary is CheckpointBoundary.AFTER_TOOL


def test_checkpoint_rejects_active_tool() -> None:
    state = HarnessState(messages=[], started_tool_call_ids={"call-1"})
    try:
        HarnessCheckpoint.create(state, "run-tool", 0, CheckpointBoundary.AFTER_TOOL)
    except ValueError as exc:
        assert "safe checkpoint" in str(exc)
    else:
        raise AssertionError("active tool must not be checkpointed as after_tool")


def test_direct_checkpoint_construction_cannot_bypass_boundary_validation() -> None:
    state = HarnessState(messages=[], started_tool_call_ids={"call-1"})
    snapshot = HarnessStateSnapshot.from_state(state)
    try:
        HarnessCheckpoint(1, "id", "run", 1, CheckpointBoundary.AFTER_TOOL, snapshot, "2026-08-02T00:00:00Z")
    except ValueError as exc:
        assert "tool" in str(exc)
    else:
        raise AssertionError("direct construction must validate checkpoint safety")


def test_store_revalidates_mutated_checkpoint_before_save() -> None:
    async def run():
        store = InMemoryCheckpointStore()
        state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello")])
        checkpoint = HarnessCheckpoint.create(state, "run-mutated", 0, CheckpointBoundary.BEFORE_MODEL)
        checkpoint.state.messages[0].metadata["huge"] = "x" * (4 * 1024 * 1024)
        try:
            await store.save(checkpoint)
        except ValueError as exc:
            assert "maximum size" in str(exc)
        else:
            raise AssertionError("store must validate a checkpoint immediately before saving")

    asyncio.run(run())


def test_checkpoint_rejects_non_string_object_keys() -> None:
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello", metadata={1: "bad"})])
    try:
        HarnessStateSnapshot.from_state(state)
    except ValueError as exc:
        assert "keys must be strings" in str(exc)
    else:
        raise AssertionError("non-string JSON object keys must be rejected")
