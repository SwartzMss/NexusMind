import asyncio
import sqlite3
import json

from nexusmind.runtime.harness import (
    CheckpointBoundary,
    HarnessCheckpoint,
    HarnessPhase,
    HarnessState,
    HarnessStatus,
    SQLiteCheckpointStore,
    StopReason,
    HarnessRequest,
    HarnessRunner,
)
from nexusmind.models.fake import FakeChatModel
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


def test_two_sqlite_stores_allow_only_one_same_sequence(tmp_path) -> None:
    async def run():
        path = tmp_path / "checkpoints.db"
        first_store = SQLiteCheckpointStore(path)
        second_store = SQLiteCheckpointStore(path)
        await first_store.initialize()
        await second_store.initialize()
        results = await asyncio.gather(
            first_store.save(_checkpoint(0)),
            second_store.save(_checkpoint(0)),
            return_exceptions=True,
        )
        assert sum(result is None for result in results) == 1

    asyncio.run(run())


def test_sqlite_checkpoint_rejects_envelope_mismatch(tmp_path) -> None:
    async def run():
        path = tmp_path / "checkpoints.db"
        store = SQLiteCheckpointStore(path)
        await store.initialize()
        await store.save(_checkpoint(0))
        with sqlite3.connect(path) as db:
            db.execute("UPDATE harness_checkpoints SET boundary = 'after_model'")
            db.commit()
        try:
            await store.load_latest("run-1")
        except Exception as exc:
            assert "envelope" in str(exc)
        else:
            raise AssertionError("envelope mismatch must be rejected")

    asyncio.run(run())


def test_sqlite_store_rejects_unknown_database_schema(tmp_path) -> None:
    path = tmp_path / "unknown.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version = 99")
    async def run():
        try:
            await SQLiteCheckpointStore(path).initialize()
        except Exception as exc:
            assert "schema" in str(exc)
        else:
            raise AssertionError("unknown database schema must be rejected")
    asyncio.run(run())


def test_sqlite_store_rejects_unknown_checkpoint_schema(tmp_path) -> None:
    async def run():
        path = tmp_path / "schema.db"
        store = SQLiteCheckpointStore(path)
        await store.initialize()
        await store.save(_checkpoint(0))
        with sqlite3.connect(path) as db:
            db.execute("UPDATE harness_checkpoints SET payload_json = ?", (json.dumps({"schema_version": 99}),))
            db.execute("UPDATE harness_checkpoints SET payload_sha256 = ?", (__import__("hashlib").sha256(json.dumps({"schema_version": 99}).encode()).hexdigest(),))
            db.commit()
        try:
            await store.load_latest("run-1")
        except Exception:
            pass
        else:
            raise AssertionError("unknown checkpoint schema must be rejected")
    asyncio.run(run())


def test_real_harness_execution_checkpoint_round_trip(tmp_path) -> None:
    async def run():
        execution = HarnessRunner(FakeChatModel(["done"])).create_execution(
            HarnessRequest(messages=(Message(role=MessageRole.USER, content="hello"),))
        )
        [event async for event in execution.stream()]
        checkpoint = execution.create_checkpoint("real-run", 0)
        store = SQLiteCheckpointStore(tmp_path / "real.db")
        await store.initialize()
        await store.save(checkpoint)
        assert (await store.load_latest("real-run")).state.status is HarnessStatus.COMPLETED
    asyncio.run(run())
