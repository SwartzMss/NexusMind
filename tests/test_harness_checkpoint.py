import asyncio
from copy import deepcopy

from nexusmind.runtime.harness import (
    CheckpointBoundary,
    HarnessCheckpoint,
    HarnessState,
    HarnessStateSnapshot,
    InMemoryCheckpointStore,
)
from nexusmind.runtime.messages import Message, MessageRole


def test_checkpoint_snapshot_isolated_and_serializable() -> None:
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello")])
    snapshot = HarnessStateSnapshot.from_state(state)
    state.messages[0].metadata["changed"] = True
    assert snapshot.messages[0].metadata == {}


def test_checkpoint_store_enforces_sequence_and_latest() -> None:
    async def run():
        store = InMemoryCheckpointStore()
        state = HarnessState(messages=[])
        first = HarnessCheckpoint.create(state, "run-1", 0, CheckpointBoundary.BEFORE_MODEL)
        second = HarnessCheckpoint.create(state, "run-1", 1, CheckpointBoundary.RUN_TERMINAL)
        await store.save(first)
        await store.save(second)
        assert await store.load_latest("run-1") == second
        assert len(await store.list("run-1")) == 2

    asyncio.run(run())
