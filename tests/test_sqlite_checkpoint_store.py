import asyncio
import sqlite3

from nexusmind.runtime.harness import (
    CheckpointBoundary,
    HarnessCheckpoint,
    HarnessPhase,
    HarnessState,
    HarnessStatus,
    SQLiteCheckpointStore,
    StopReason,
)
from nexusmind.runtime.messages import Message, MessageRole


def _checkpoint(sequence: int) -> HarnessCheckpoint:
    state = HarnessState(
        messages=[Message(role=MessageRole.USER, content="你好", metadata={"files": ["a.c"]})],
        status=HarnessStatus.COMPLETED,
        stop_reason=StopReason.MODEL_COMPLETED,
        phase=HarnessPhase.TERMINAL,
    )
    return HarnessCheckpoint.create(state, "run-1", sequence)


def test_sqlite_checkpoint_round_trip_and_reopen(tmp_path) -> None:
    async def run():
        path = tmp_path / "checkpoints.db"
        store = SQLiteCheckpointStore(path)
        await store.initialize()
        checkpoint = _checkpoint(0)
        await store.save(checkpoint)
        reopened = SQLiteCheckpointStore(path)
        await reopened.initialize()
        loaded = await reopened.load_latest("run-1")
        assert loaded is not None
        assert loaded.sequence == 0
        assert loaded.state.messages[0].content == "你好"
        assert await reopened.list("missing") == ()

    asyncio.run(run())


def test_sqlite_checkpoint_rejects_duplicate_or_decreasing_sequence(tmp_path) -> None:
    async def run():
        store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
        await store.initialize()
        await store.save(_checkpoint(1))
        for checkpoint in (_checkpoint(1), _checkpoint(0)):
            try:
                await store.save(checkpoint)
            except Exception as exc:
                assert "sequence" in str(exc) or "conflicts" in str(exc)
            else:
                raise AssertionError("invalid sequence must be rejected")

    asyncio.run(run())


def test_sqlite_checkpoint_rejects_payload_tampering(tmp_path) -> None:
    async def run():
        path = tmp_path / "checkpoints.db"
        store = SQLiteCheckpointStore(path)
        await store.initialize()
        await store.save(_checkpoint(0))
        with sqlite3.connect(path) as db:
            db.execute("UPDATE harness_checkpoints SET payload_json = ?", ('{"broken":true}',))
            db.commit()
        try:
            await store.load_latest("run-1")
        except Exception as exc:
            assert "integrity" in str(exc)
        else:
            raise AssertionError("tampered payload must be rejected")

    asyncio.run(run())


def test_sqlite_checkpoint_rejects_duplicate_checkpoint_id(tmp_path) -> None:
    async def run():
        path = tmp_path / "checkpoints.db"
        store = SQLiteCheckpointStore(path)
        await store.initialize()
        first = _checkpoint(0)
        await store.save(first)
        second = _checkpoint(1)
        with sqlite3.connect(path) as db:
            db.execute("UPDATE harness_checkpoints SET checkpoint_id = ? WHERE sequence = 0", (second.checkpoint_id,))
            db.commit()
        try:
            await store.save(second)
        except Exception:
            pass
        else:
            raise AssertionError("duplicate checkpoint id must be rejected")

    asyncio.run(run())
