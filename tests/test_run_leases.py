from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import suppress
from datetime import datetime, timedelta, timezone

import pytest

from nexusmind import cli
from nexusmind.config import ModelConfig
from nexusmind.runtime.lease_guarding import LeaseGuardedChatModel
from nexusmind.runtime.leases import (
    RunLease,
    RunLeaseCoordinator,
    RunLeaseOwnershipLost,
    RunLeaseOwnershipGuard,
    RunLeaseReleaseError,
    RunLeaseStoreError,
    RunLeaseUnavailable,
    SQLiteRunLeaseStore,
)
from nexusmind.models.base import ChatModel
from nexusmind.runtime.chat import ChatRuntime
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.harness import CheckpointBoundary, HarnessStatus, InMemoryCheckpointStore
from nexusmind.runtime.policy import ToolPolicyDecision
from nexusmind.tools.contracts import (
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultBudget,
    ToolResultRequirements,
    ToolRiskLevel,
)
from nexusmind.tools.registry import ToolRegistry


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class BeginImmediateObservedConnection(sqlite3.Connection):
    def __init__(self, *args, begin_immediate_attempted: threading.Event, **kwargs) -> None:
        self._begin_immediate_attempted = begin_immediate_attempted
        super().__init__(*args, **kwargs)

    def execute(self, sql, parameters=()):
        if sql == "BEGIN IMMEDIATE":
            self._begin_immediate_attempted.set()
        return super().execute(sql, parameters)


class ConnectionObservedLeaseStore(SQLiteRunLeaseStore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.begin_immediate_attempted = threading.Event()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(
            self._path,
            timeout=self._timeout,
            isolation_level=None,
            factory=lambda *args, **kwargs: BeginImmediateObservedConnection(
                *args,
                begin_immediate_attempted=self.begin_immediate_attempted,
                **kwargs,
            ),
        )
        db.execute(f"PRAGMA busy_timeout={max(1, int(self._timeout * 1000))}")
        db.execute("PRAGMA synchronous=FULL")
        return db


async def _wait_for_begin_immediate_attempt(
    task: asyncio.Task[RunLease], event: threading.Event
) -> None:
    assert await asyncio.to_thread(event.wait, 1)
    assert not task.done()


async def _drain_lease_task(task: asyncio.Task[RunLease]) -> None:
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
    except asyncio.TimeoutError:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    except BaseException:
        pass


async def _release_holder_and_drain(
    holder: sqlite3.Connection, task: asyncio.Task[RunLease] | None
) -> None:
    try:
        holder.rollback()
    finally:
        try:
            holder.close()
        finally:
            if task is not None:
                await _drain_lease_task(task)


def test_first_owner_acquires_and_heartbeat_renews_lease(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        store = SQLiteRunLeaseStore(tmp_path / "leases.db", clock=clock)
        await store.initialize()

        acquired = await store.acquire("run-1", "owner-1", timedelta(seconds=10))
        clock.advance(3)
        renewed = await store.renew("run-1", "owner-1", timedelta(seconds=10))

        assert acquired.acquired_at == clock.value - timedelta(seconds=3)
        assert renewed.acquired_at == acquired.acquired_at
        assert renewed.heartbeat_at == clock.value
        assert renewed.expires_at == clock.value + timedelta(seconds=10)
        assert await store.inspect("run-1") == renewed
        await store.close()

    asyncio.run(run())


def test_second_owner_cannot_acquire_or_renew_unexpired_lease(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteRunLeaseStore(tmp_path / "leases.db", clock=MutableClock())
        await store.initialize()
        await store.acquire("run-1", "owner-1", timedelta(seconds=10))

        with pytest.raises(RunLeaseUnavailable):
            await store.acquire("run-1", "owner-2", timedelta(seconds=10))
        with pytest.raises(RunLeaseOwnershipLost):
            await store.renew("run-1", "owner-2", timedelta(seconds=10))
        assert (await store.inspect("run-1")).owner_id == "owner-1"

    asyncio.run(run())


def test_concurrent_acquire_allows_exactly_one_owner(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "leases.db"
        first = SQLiteRunLeaseStore(path)
        second = SQLiteRunLeaseStore(path)
        await asyncio.gather(first.initialize(), second.initialize())

        results = await asyncio.gather(
            first.acquire("run-race", "owner-1", timedelta(seconds=30)),
            second.acquire("run-race", "owner-2", timedelta(seconds=30)),
            return_exceptions=True,
        )

        winners = [result for result in results if not isinstance(result, BaseException)]
        losers = [result for result in results if isinstance(result, RunLeaseUnavailable)]
        assert len(winners) == 1
        assert len(losers) == 1
        assert (await first.inspect("run-race")).owner_id == winners[0].owner_id

    asyncio.run(run())


def test_expired_takeover_has_one_winner_and_rejects_stale_owner(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        path = tmp_path / "leases.db"
        stores = [SQLiteRunLeaseStore(path, clock=clock) for _ in range(3)]
        await asyncio.gather(*(store.initialize() for store in stores))
        await stores[0].acquire("run-takeover", "old-owner", timedelta(seconds=5))
        clock.advance(6)

        results = await asyncio.gather(
            stores[1].acquire("run-takeover", "new-owner-1", timedelta(seconds=10)),
            stores[2].acquire("run-takeover", "new-owner-2", timedelta(seconds=10)),
            return_exceptions=True,
        )
        winners = [result for result in results if not isinstance(result, BaseException)]
        assert len(winners) == 1
        assert sum(isinstance(result, RunLeaseUnavailable) for result in results) == 1

        with pytest.raises(RunLeaseOwnershipLost):
            await stores[0].renew("run-takeover", "old-owner", timedelta(seconds=10))
        with pytest.raises(RunLeaseOwnershipLost):
            await stores[0].release("run-takeover", "old-owner")
        assert (await stores[0].inspect("run-takeover")).owner_id == winners[0].owner_id

    asyncio.run(run())


def test_owner_scoped_release_and_different_run_ids_are_isolated(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteRunLeaseStore(tmp_path / "leases.db")
        await store.initialize()
        first, second = await asyncio.gather(
            store.acquire("run-1", "owner-1", timedelta(seconds=10)),
            store.acquire("run-2", "owner-2", timedelta(seconds=10)),
        )
        assert first.run_id == "run-1"
        assert second.run_id == "run-2"

        with pytest.raises(RunLeaseOwnershipLost):
            await store.release("run-1", "owner-2")
        await store.release("run-1", "owner-1")
        assert await store.inspect("run-1") is None
        assert (await store.inspect("run-2")).owner_id == "owner-2"

    asyncio.run(run())


def test_sqlite_reopen_preserves_active_lease(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "leases.db"
        first = SQLiteRunLeaseStore(path)
        await first.initialize()
        lease = await first.acquire("run-1", "owner-1", timedelta(minutes=1))
        await first.close()

        reopened = SQLiteRunLeaseStore(path)
        await reopened.initialize()
        assert await reopened.inspect("run-1") == lease
        with pytest.raises(RunLeaseUnavailable):
            await reopened.acquire("run-1", "owner-2", timedelta(minutes=1))

    asyncio.run(run())


def test_sqlite_lock_failure_is_controlled(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "leases.db"
        store = SQLiteRunLeaseStore(path, timeout=0.01)
        await store.initialize()
        lock = sqlite3.connect(path, isolation_level=None)
        lock.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(RunLeaseStoreError):
                await store.acquire("run-1", "owner-1", timedelta(seconds=10))
        finally:
            lock.rollback()
            lock.close()

    asyncio.run(run())


def test_renew_rechecks_expiry_after_waiting_for_write_lock(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        path = tmp_path / "leases.db"
        store = ConnectionObservedLeaseStore(path, clock=clock, timeout=1)
        await store.initialize()
        old_lease = await store.acquire("run-1", "owner-1", timedelta(seconds=5))
        lock = sqlite3.connect(path, isolation_level=None)
        renew_task: asyncio.Task[RunLease] | None = None
        try:
            lock.execute("BEGIN IMMEDIATE")
            store.begin_immediate_attempted.clear()
            renew_task = asyncio.create_task(
                store.renew("run-1", "owner-1", timedelta(seconds=10))
            )
            await _wait_for_begin_immediate_attempt(
                renew_task, store.begin_immediate_attempted
            )
            clock.advance(6)
        finally:
            await _release_holder_and_drain(lock, renew_task)

        with pytest.raises(RunLeaseOwnershipLost):
            await renew_task
        stored = await store.inspect("run-1")
        assert stored == old_lease
        assert stored.is_expired(clock.value)

    asyncio.run(run())


def test_acquire_rechecks_expiry_after_waiting_for_write_lock(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        path = tmp_path / "leases.db"
        store = ConnectionObservedLeaseStore(path, clock=clock, timeout=1)
        await store.initialize()
        await store.acquire("run-1", "owner-1", timedelta(seconds=5))
        lock = sqlite3.connect(path, isolation_level=None)
        acquire_task: asyncio.Task[RunLease] | None = None
        try:
            lock.execute("BEGIN IMMEDIATE")
            store.begin_immediate_attempted.clear()
            acquire_task = asyncio.create_task(
                store.acquire("run-1", "owner-2", timedelta(seconds=10))
            )
            await _wait_for_begin_immediate_attempt(
                acquire_task, store.begin_immediate_attempted
            )
            clock.advance(6)
        finally:
            await _release_holder_and_drain(lock, acquire_task)

        lease = await acquire_task
        assert lease.owner_id == "owner-2"
        assert lease.acquired_at == clock.value
        assert await store.inspect("run-1") == lease

    asyncio.run(run())


def test_invalid_or_closed_store_fails_with_public_errors(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "leases.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE run_leases (run_id TEXT PRIMARY KEY)")
        connection.close()
        store = SQLiteRunLeaseStore(path)
        with pytest.raises(RunLeaseStoreError):
            await store.initialize()

        healthy = SQLiteRunLeaseStore(tmp_path / "healthy.db")
        await healthy.initialize()
        await healthy.close()
        with pytest.raises(RunLeaseStoreError):
            await healthy.inspect("run-1")

    asyncio.run(run())


@pytest.mark.parametrize("lease_release_timeout", (timedelta(0), 0))
def test_coordinator_rejects_invalid_lease_release_timeout(lease_release_timeout) -> None:
    with pytest.raises(ValueError, match="Lease release timeout must be a positive timedelta"):
        RunLeaseCoordinator(
            object(),
            run_id="run-release-timeout",
            lease_release_timeout=lease_release_timeout,
        )


class CountingTextModel(ChatModel):
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools=None):
        self.calls += 1
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
        yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")


class ToolModel(ChatModel):
    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.calls = 0

    async def stream(self, messages, tools=None):
        self.calls += 1
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=self.call)
        yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")


class RecordingExecutor:
    definition_value = ToolDefinition(
        name="echo",
        description="echo",
        input_schema={"type": "object", "additionalProperties": False},
        risk_level=ToolRiskLevel.READ_ONLY,
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    def definition(self, name: str) -> ToolDefinition | None:
        return self.definition_value if name == "echo" else None

    def result_requirements(self, call: ToolCall) -> ToolResultRequirements:
        return ToolResultRequirements(min_bytes=32, min_nodes=3, min_depth=1)

    async def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call.id)
        return ToolResult(call_id=call.id, name=call.name, output="ok")

    async def execute_with_result_budget(
        self,
        call: ToolCall,
        *,
        result_budget: ToolResultBudget,
    ) -> ToolResult:
        return await self.execute(call)


def test_lease_acquisition_failure_fails_closed_before_model(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteRunLeaseStore(tmp_path / "leases.db")
        await store.initialize()
        await store.acquire("run-1", "existing-owner", timedelta(seconds=30))
        model = CountingTextModel()
        runtime = ChatRuntime(
            model,
            lease_store=store,
            lease_run_id="run-1",
            lease_owner_id="contender",
        )

        with pytest.raises(RunLeaseUnavailable):
            [event async for event in runtime.stream_user_message("hello")]
        assert model.calls == 0
        assert (await store.inspect("run-1")).owner_id == "existing-owner"

    asyncio.run(run())


def test_normal_execution_releases_lease(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteRunLeaseStore(tmp_path / "leases.db")
        await store.initialize()
        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=store,
            lease_run_id="run-1",
            lease_owner_id="owner-1",
        )

        events = [event async for event in runtime.stream_user_message("hello")]
        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert await store.inspect("run-1") is None

    asyncio.run(run())


def test_failed_and_cancelled_executions_release_lease(tmp_path) -> None:
    async def run() -> None:
        class FailingModel(ChatModel):
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                raise RuntimeError("provider failed")

        class CancelModel(ChatModel):
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                raise asyncio.CancelledError()

        store = SQLiteRunLeaseStore(tmp_path / "leases.db")
        await store.initialize()
        failed = ChatRuntime(
            FailingModel(),
            lease_store=store,
            lease_run_id="run-failed",
            lease_owner_id="owner-failed",
        )
        events = [event async for event in failed.stream_user_message("hello")]
        assert events[-1].type is RuntimeEventType.RUN_FAILED
        assert await store.inspect("run-failed") is None

        cancelled = ChatRuntime(
            CancelModel(),
            lease_store=store,
            lease_run_id="run-cancelled",
            lease_owner_id="owner-cancelled",
        )
        with pytest.raises(asyncio.CancelledError):
            [event async for event in cancelled.stream_user_message("hello")]
        assert await store.inspect("run-cancelled") is None

    asyncio.run(run())


def test_heartbeat_ownership_loss_blocks_next_tool() -> None:
    async def run() -> None:
        class HeartbeatFailureStore:
            def __init__(self) -> None:
                self.renew_failed = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat persistence failed")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                return None

        store = HeartbeatFailureStore()
        call = ToolCall(id="call-1", name="echo", arguments={})
        model = ToolModel(call)
        executor = RecordingExecutor()
        runtime = ChatRuntime(
            model,
            tool_executor=executor,
            lease_store=store,
            lease_run_id="run-1",
            lease_owner_id="owner-1",
            lease_ttl=timedelta(seconds=1),
            lease_heartbeat_interval=timedelta(milliseconds=10),
        )

        stream = runtime.stream_user_message("hello", tools=[executor.definition_value])
        seen = []
        with pytest.raises(RunLeaseStoreError, match="heartbeat persistence failed"):
            async for event in stream:
                seen.append(event.type)
                if event.type is RuntimeEventType.MODEL_TURN_COMPLETED:
                    await store.renew_failed.wait()

        assert executor.calls == []
        assert model.calls <= 1

    asyncio.run(run())


def test_release_failure_is_surfaced_after_terminal_event() -> None:
    async def run() -> None:
        class ReleaseFailureStore:
            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                raise RunLeaseStoreError("release persistence failed")

        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=ReleaseFailureStore(),
            lease_run_id="run-1",
            lease_owner_id="owner-1",
            checkpoint_store=(checkpoints := InMemoryCheckpointStore()),
            checkpoint_run_id="run-1",
        )
        events = [event async for event in runtime.stream_user_message("hello")]
        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_release_failed"] is True
        assert isinstance(runtime._lease_coordinator.release_error, RunLeaseReleaseError)
        latest = await checkpoints.load_latest("run-1")
        assert latest is not None
        assert latest.boundary is CheckpointBoundary.RUN_TERMINAL
        assert latest.state.status is HarnessStatus.COMPLETED
        assert runtime._harness.state.status is HarnessStatus.COMPLETED

    asyncio.run(run())


def test_terminal_event_is_not_delivered_until_lease_is_released() -> None:
    async def run() -> None:
        class TrackingStore:
            def __init__(self) -> None:
                self.released = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id):
                self.released.set()

        store = TrackingStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=store,
            lease_run_id="run-terminal-release",
            lease_owner_id="owner-terminal-release",
        )
        stream = runtime.stream_user_message("hello")
        while True:
            event = await anext(stream)
            if event.type is RuntimeEventType.RUN_COMPLETED:
                break

        assert store.released.is_set()
        assert runtime._lease_coordinator.lease is None
        # Deliberately do not request another item or explicitly close here;
        # terminal cleanup must already be complete when the event arrives.

    asyncio.run(run())


def test_cancellation_during_terminal_release_preserves_terminal_outcome() -> None:
    async def run() -> None:
        class PausingReleaseStore:
            def __init__(self) -> None:
                self.release_started = asyncio.Event()
                self.allow_release = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id):
                self.release_started.set()
                await self.allow_release.wait()

        store = PausingReleaseStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=store,
            lease_run_id="run-terminal-cancel",
            lease_owner_id="owner-terminal-cancel",
        )
        task = asyncio.create_task(_collect_events(runtime.stream_user_message("hello")))
        await store.release_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        store.allow_release.set()
        events = await task

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert runtime._harness.state.status is HarnessStatus.COMPLETED
        assert runtime._lease_coordinator.lease is None

    asyncio.run(run())


def test_terminal_release_timeout_preserves_terminal_outcome() -> None:
    async def run() -> None:
        class HangingReleaseStore:
            def __init__(self) -> None:
                self.release_started = asyncio.Event()
                self.release_cancelled = asyncio.Event()
                self.never_release = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                self.release_started.set()
                try:
                    await self.never_release.wait()
                except asyncio.CancelledError:
                    self.release_cancelled.set()
                    raise

        store = HangingReleaseStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=store,
            lease_run_id="run-terminal-release-timeout",
            lease_owner_id="owner-terminal-release-timeout",
            lease_release_timeout=timedelta(milliseconds=20),
        )

        events = [event async for event in runtime.stream_user_message("hello")]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_release_failed"] is True
        assert isinstance(runtime._lease_coordinator.release_error, RunLeaseReleaseError)
        assert runtime._harness.state.status is HarnessStatus.COMPLETED
        await asyncio.wait_for(store.release_cancelled.wait(), timeout=1)
        assert runtime._lease_coordinator.lease is None

    asyncio.run(run())


def test_self_cancelled_terminal_release_preserves_terminal_outcome() -> None:
    async def run() -> None:
        class SelfCancellingReleaseStore:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                self.release_calls += 1
                raise asyncio.CancelledError()

        store = SelfCancellingReleaseStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=store,
            lease_run_id="run-terminal-release-self-cancel",
            lease_owner_id="owner-terminal-release-self-cancel",
        )

        events = [event async for event in runtime.stream_user_message("hello")]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_release_failed"] is True
        assert isinstance(runtime._lease_coordinator.release_error, RunLeaseReleaseError)
        assert store.release_calls == 1

    asyncio.run(run())


def test_timed_out_terminal_release_task_is_retained_until_it_finishes() -> None:
    async def run() -> None:
        class CancellationSuppressingReleaseStore:
            def __init__(self) -> None:
                self.release_started = asyncio.Event()
                self.cancellation_suppressed = asyncio.Event()
                self.allow_finish = asyncio.Event()
                self.never_release = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                self.release_started.set()
                try:
                    await self.never_release.wait()
                except asyncio.CancelledError:
                    self.cancellation_suppressed.set()
                    await self.allow_finish.wait()

        store = CancellationSuppressingReleaseStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=store,
            lease_run_id="run-terminal-release-background-task",
            lease_owner_id="owner-terminal-release-background-task",
            lease_release_timeout=timedelta(milliseconds=20),
        )

        events = [event async for event in runtime.stream_user_message("hello")]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_release_failed"] is True
        await asyncio.wait_for(store.cancellation_suppressed.wait(), timeout=1)
        background_task = next(iter(runtime._lease_coordinator._background_tasks))
        assert not background_task.done()
        store.allow_finish.set()
        await asyncio.wait_for(asyncio.shield(background_task), timeout=1)
        await asyncio.sleep(0)
        assert runtime._lease_coordinator._background_tasks == set()

    asyncio.run(run())


def test_cancellation_cannot_extend_terminal_release_timeout() -> None:
    async def run() -> None:
        class HangingReleaseStore:
            def __init__(self) -> None:
                self.release_started = asyncio.Event()
                self.release_cancelled = asyncio.Event()
                self.never_release = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                self.release_started.set()
                try:
                    await self.never_release.wait()
                except asyncio.CancelledError:
                    self.release_cancelled.set()
                    raise

        store = HangingReleaseStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=store,
            lease_run_id="run-terminal-release-cancel-timeout",
            lease_owner_id="owner-terminal-release-cancel-timeout",
            lease_release_timeout=timedelta(milliseconds=40),
        )
        task = asyncio.create_task(_collect_events(runtime.stream_user_message("hello")))
        await store.release_started.wait()

        cancellation_attempts = 0

        async def cancel_repeatedly() -> None:
            nonlocal cancellation_attempts
            for _ in range(40):
                task.cancel()
                cancellation_attempts += 1
                await asyncio.sleep(0.005)

        cancellation_storm = asyncio.create_task(cancel_repeatedly())
        try:
            events = await asyncio.wait_for(asyncio.shield(task), timeout=0.15)
            assert not cancellation_storm.done()
            assert cancellation_attempts >= 5
        finally:
            await cancellation_storm
        assert cancellation_storm.done()

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_release_failed"] is True
        assert runtime._harness.state.status is HarnessStatus.COMPLETED
        await asyncio.wait_for(store.release_cancelled.wait(), timeout=1)
        assert runtime._lease_coordinator.lease is None

    asyncio.run(run())


@pytest.mark.parametrize(
    ("outcome", "history_status", "harness_status"),
    [
        ("completed", "completed", HarnessStatus.COMPLETED),
        ("failed", "failed", HarnessStatus.FAILED),
    ],
)
def test_heartbeat_loss_during_terminal_checkpoint_preserves_all_outcomes(
    outcome,
    history_status,
    harness_status,
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    async def run() -> tuple[int, InMemoryCheckpointStore, ChatRuntime]:
        class PausingCheckpointStore(InMemoryCheckpointStore):
            def __init__(self) -> None:
                super().__init__()
                self.terminal_started = asyncio.Event()
                self.release_terminal = asyncio.Event()

            async def initialize(self):
                return None

            async def close(self):
                return None

            async def save(self, checkpoint):
                if checkpoint.boundary is CheckpointBoundary.RUN_TERMINAL:
                    self.terminal_started.set()
                    await self.release_terminal.wait()
                await super().save(checkpoint)

        class TerminalHeartbeatFailureStore:
            def __init__(self, checkpoint_store) -> None:
                self.checkpoint_store = checkpoint_store
                self.renew_failed = asyncio.Event()

            async def initialize(self):
                return None

            async def close(self):
                return None

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                await self.checkpoint_store.terminal_started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during terminal checkpoint")

            async def release(self, run_id, owner_id):
                return None

        class OutcomeModel(ChatModel):
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                if outcome == "failed":
                    yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error="provider failed")
                    return
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

        checkpoints = PausingCheckpointStore()
        leases = TerminalHeartbeatFailureStore(checkpoints)
        captured = {}
        original_runtime = ChatRuntime

        def runtime_factory(*args, **kwargs):
            runtime = original_runtime(
                *args,
                **kwargs,
                lease_ttl=timedelta(seconds=1),
                lease_heartbeat_interval=timedelta(milliseconds=10),
            )
            captured["runtime"] = runtime
            return runtime

        monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: OutcomeModel())
        monkeypatch.setattr(cli, "SQLiteCheckpointStore", lambda path: checkpoints)
        monkeypatch.setattr(cli, "SQLiteRunLeaseStore", lambda path: leases)
        monkeypatch.setattr(cli, "ChatRuntime", runtime_factory)
        task = asyncio.create_task(
            cli._run_chat(
                "hello",
                ToolRegistry(),
                model_config=ModelConfig("https://example.test", "test-key", "fake"),
                state_db=str(tmp_path / f"{outcome}-runs.db"),
                checkpoint_db=str(tmp_path / f"{outcome}-checkpoints.db"),
                lease_db=str(tmp_path / f"{outcome}-leases.db"),
            )
        )
        await checkpoints.terminal_started.wait()
        await leases.renew_failed.wait()
        while captured["runtime"]._lease_guard.ownership_error is None:
            await asyncio.sleep(0)
        checkpoints.release_terminal.set()
        return await task, checkpoints, captured["runtime"]

    code, checkpoints, runtime = asyncio.run(run())
    capsys.readouterr()
    connection = sqlite3.connect(tmp_path / f"{outcome}-runs.db")
    run_id, stored_status = connection.execute("SELECT id, status FROM runs LIMIT 1").fetchone()
    event_types = [
        row[0]
        for row in connection.execute("SELECT event_type FROM run_events ORDER BY sequence")
    ]
    connection.close()
    latest = asyncio.run(checkpoints.load_latest(run_id))

    assert code == 1
    assert stored_status == history_status
    assert runtime._harness.state.status is harness_status
    assert latest is not None
    assert latest.state.status is harness_status
    assert "lease_ownership_lost_after_terminal" in event_types


def test_terminal_event_audits_ownership_expiry_while_waiting(tmp_path) -> None:
    async def run() -> None:
        class CountingRenewLeaseStore(SQLiteRunLeaseStore):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.renew_calls = 0

            async def renew(self, run_id, owner_id, ttl):
                self.renew_calls += 1
                return await super().renew(run_id, owner_id, ttl)

        class PausingCheckpointStore(InMemoryCheckpointStore):
            def __init__(self) -> None:
                super().__init__()
                self.terminal_started = asyncio.Event()
                self.release_terminal = asyncio.Event()

            async def save(self, checkpoint):
                if checkpoint.boundary is CheckpointBoundary.RUN_TERMINAL:
                    self.terminal_started.set()
                    await self.release_terminal.wait()
                await super().save(checkpoint)

        clock = MutableClock()
        leases = CountingRenewLeaseStore(tmp_path / "terminal-expiry.db", clock=clock)
        await leases.initialize()
        checkpoints = PausingCheckpointStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            checkpoint_store=checkpoints,
            checkpoint_run_id="run-terminal-expiry",
            lease_store=leases,
            lease_run_id="run-terminal-expiry",
            lease_owner_id="owner-terminal-expiry",
            lease_ttl=timedelta(days=2),
            lease_heartbeat_interval=timedelta(days=1),
            lease_clock=clock,
        )
        task = asyncio.create_task(_collect_events(runtime.stream_user_message("hello")))
        await checkpoints.terminal_started.wait()

        lease = runtime._lease_coordinator.lease
        assert lease is not None
        clock.advance(3 * 24 * 60 * 60)
        assert runtime._lease_guard.ownership_error is None
        stored_lease = await leases.inspect("run-terminal-expiry")
        assert stored_lease is not None
        assert stored_lease.heartbeat_at == lease.heartbeat_at

        checkpoints.release_terminal.set()
        events = await task
        assert leases.renew_calls == 0
        await leases.close()

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert runtime._harness.state.status is HarnessStatus.COMPLETED
        assert events[-1].metadata["lease_ownership_lost_after_terminal"] is True

    asyncio.run(run())


def test_heartbeat_loss_during_policy_blocks_tool_execution() -> None:
    async def run() -> None:
        class HeartbeatFailureStore:
            def __init__(self) -> None:
                self.renew_failed = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                await policy.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during policy")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                return None

        class BlockingPolicy:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def evaluate(self, call, context):
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
                return ToolPolicyDecision.ALLOW

        store = HeartbeatFailureStore()
        policy = BlockingPolicy()
        executor = RecordingExecutor()
        runtime = ChatRuntime(
            ToolModel(ToolCall(id="call-policy", name="echo", arguments={})),
            tool_executor=executor,
            tool_policy=policy,
            lease_store=store,
            lease_run_id="run-policy",
            lease_owner_id="owner-policy",
            lease_ttl=timedelta(seconds=1),
            lease_heartbeat_interval=timedelta(milliseconds=10),
        )
        task = asyncio.create_task(
            _collect_events(runtime.stream_user_message("hello", tools=[executor.definition_value]))
        )
        await policy.started.wait()
        await store.renew_failed.wait()
        while runtime._lease_guard.ownership_error is None:
            await asyncio.sleep(0)

        with pytest.raises(RunLeaseStoreError, match="heartbeat failed during policy"):
            await task
        assert policy.cancelled.is_set()
        assert executor.calls == []

    asyncio.run(run())


def test_heartbeat_loss_cancels_inflight_tool() -> None:
    async def run() -> None:
        class HeartbeatFailureStore:
            def __init__(self) -> None:
                self.renew_failed = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                await executor.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during tool")

            async def release(self, run_id, owner_id):
                return None

        class BlockingExecutor(RecordingExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def execute_with_result_budget(self, call, *, result_budget):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

        store = HeartbeatFailureStore()
        executor = BlockingExecutor()
        runtime = ChatRuntime(
            ToolModel(ToolCall(id="call-inflight", name="echo", arguments={})),
            tool_executor=executor,
            lease_store=store,
            lease_run_id="run-inflight",
            lease_owner_id="owner-inflight",
            lease_ttl=timedelta(seconds=1),
            lease_heartbeat_interval=timedelta(milliseconds=10),
        )
        task = asyncio.create_task(
            _collect_events(runtime.stream_user_message("hello", tools=[executor.definition_value]))
        )
        await executor.started.wait()
        await store.renew_failed.wait()

        with pytest.raises(RunLeaseStoreError, match="heartbeat failed during tool"):
            await asyncio.wait_for(task, timeout=1)
        assert executor.cancelled.is_set()
        assert executor.calls == []
        assert runtime._harness.state.status is HarnessStatus.CANCELLED

    asyncio.run(run())


def test_heartbeat_loss_cancels_inflight_model() -> None:
    async def run() -> None:
        class HeartbeatFailureStore:
            def __init__(self) -> None:
                self.renew_failed = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                await model.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during model")

            async def release(self, run_id, owner_id):
                return None

        class BlockingModel(ChatModel):
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

        store = HeartbeatFailureStore()
        model = BlockingModel()
        runtime = ChatRuntime(
            model,
            lease_store=store,
            lease_run_id="run-model-inflight",
            lease_owner_id="owner-model-inflight",
            lease_ttl=timedelta(seconds=1),
            lease_heartbeat_interval=timedelta(milliseconds=10),
        )
        task = asyncio.create_task(_collect_events(runtime.stream_user_message("hello")))
        await model.started.wait()
        await store.renew_failed.wait()

        with pytest.raises(RunLeaseStoreError, match="heartbeat failed during model"):
            await asyncio.wait_for(task, timeout=1)
        assert model.cancelled.is_set()
        assert runtime._harness.state.status is HarnessStatus.CANCELLED

    asyncio.run(run())


def test_expired_owner_cannot_continue_inflight_tool_after_takeover(tmp_path) -> None:
    async def run() -> None:
        class FailingOwnerStore(SQLiteRunLeaseStore):
            async def renew(self, run_id, owner_id, ttl):
                await executor.started.wait()
                raise RunLeaseStoreError("owner heartbeat failed")

            async def release(self, run_id, owner_id):
                raise RunLeaseStoreError("owner release failed")

        class BlockingExecutor(RecordingExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def execute_with_result_budget(self, call, *, result_budget):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

        clock = MutableClock()
        path = tmp_path / "takeover.db"
        old_store = FailingOwnerStore(path, clock=clock)
        await old_store.initialize()
        executor = BlockingExecutor()
        runtime = ChatRuntime(
            ToolModel(ToolCall(id="call-old-owner", name="echo", arguments={})),
            tool_executor=executor,
            lease_store=old_store,
            lease_run_id="run-takeover-inflight",
            lease_owner_id="owner-old",
            lease_ttl=timedelta(milliseconds=100),
            lease_heartbeat_interval=timedelta(milliseconds=10),
            lease_clock=clock,
        )
        old_task = asyncio.create_task(
            _collect_events(runtime.stream_user_message("hello", tools=[executor.definition_value]))
        )
        await executor.started.wait()
        with pytest.raises(RunLeaseStoreError, match="owner heartbeat failed"):
            await asyncio.wait_for(old_task, timeout=1)
        assert executor.cancelled.is_set()

        clock.advance(1)
        new_store = SQLiteRunLeaseStore(path, clock=clock)
        await new_store.initialize()
        new_lease = await new_store.acquire(
            "run-takeover-inflight",
            "owner-new",
            timedelta(seconds=1),
        )
        assert new_lease.owner_id == "owner-new"
        assert old_task.done()
        assert executor.calls == []
        await new_store.release("run-takeover-inflight", "owner-new")
        await old_store.close()
        await new_store.close()

    asyncio.run(run())


def test_lease_expiry_during_policy_blocks_tool_execution() -> None:
    async def run() -> None:
        class StableStore:
            async def acquire(self, run_id, owner_id, ttl):
                now = clock()
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                await asyncio.Event().wait()

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id):
                return None

        class BlockingPolicy:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def evaluate(self, call, context):
                self.started.set()
                await self.release.wait()
                return ToolPolicyDecision.ALLOW

        clock = MutableClock()
        policy = BlockingPolicy()
        executor = RecordingExecutor()
        runtime = ChatRuntime(
            ToolModel(ToolCall(id="call-expired", name="echo", arguments={})),
            tool_executor=executor,
            tool_policy=policy,
            lease_store=StableStore(),
            lease_run_id="run-expired-policy",
            lease_owner_id="owner-expired-policy",
            lease_ttl=timedelta(milliseconds=100),
            lease_heartbeat_interval=timedelta(milliseconds=90),
            lease_clock=clock,
        )
        task = asyncio.create_task(
            _collect_events(runtime.stream_user_message("hello", tools=[executor.definition_value]))
        )
        await policy.started.wait()
        clock.advance(1)
        policy.release.set()

        events = await task
        assert events[-1].type is RuntimeEventType.RUN_FAILED
        assert executor.calls == []

    asyncio.run(run())


def test_heartbeat_loss_before_next_model_blocks_model_call() -> None:
    async def run() -> None:
        guard = RunLeaseOwnershipGuard()
        now = datetime.now(timezone.utc)
        guard.prove(RunLease("run-model", "owner-model", now, now, now + timedelta(seconds=10)))
        model = CountingTextModel()
        guarded = LeaseGuardedChatModel(model, guard)
        guard.fail(RunLeaseStoreError("heartbeat lost before model"))

        with pytest.raises(RunLeaseStoreError, match="heartbeat lost before model"):
            [event async for event in guarded.stream([])]
        assert model.calls == 0

    asyncio.run(run())


def test_ownership_loss_cannot_be_cleared_by_a_late_renewal() -> None:
    guard = RunLeaseOwnershipGuard()
    now = datetime.now(timezone.utc)
    lease = RunLease("run-irreversible", "owner-1", now, now, now + timedelta(seconds=10))
    guard.prove(lease, initial=True)
    guard.fail(RunLeaseStoreError("ownership lost"))

    with pytest.raises(RunLeaseStoreError, match="ownership lost"):
        guard.prove(
            RunLease(
                "run-irreversible",
                "owner-1",
                now,
                now + timedelta(seconds=1),
                now + timedelta(seconds=11),
            )
        )
    with pytest.raises(RunLeaseStoreError, match="ownership lost"):
        guard.assert_owned()


async def _collect_events(events):
    return [event async for event in events]


def test_cli_execution_owns_and_releases_lease(tmp_path, monkeypatch, capsys) -> None:
    lease_path = tmp_path / "cli-leases.db"
    model = CountingTextModel()
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: model)

    code = asyncio.run(
        cli._run_chat(
            "hello",
            ToolRegistry(),
            model_config=ModelConfig("https://example.test", "test-key", "fake"),
            lease_db=str(lease_path),
            lease_run_id="run-cli",
        )
    )
    capsys.readouterr()

    async def inspect() -> None:
        store = SQLiteRunLeaseStore(lease_path)
        await store.initialize()
        assert await store.inspect("run-cli") is None

    asyncio.run(inspect())
    assert code == 0
    assert model.calls == 1


def test_cli_active_owner_fails_closed_before_model(tmp_path, monkeypatch, capsys) -> None:
    lease_path = tmp_path / "cli-leases.db"

    async def own_run() -> None:
        store = SQLiteRunLeaseStore(lease_path)
        await store.initialize()
        await store.acquire("run-cli", "existing-owner", timedelta(minutes=1))
        await store.close()

    asyncio.run(own_run())
    model = CountingTextModel()
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: model)

    code = asyncio.run(
        cli._run_chat(
            "hello",
            ToolRegistry(),
            model_config=ModelConfig("https://example.test", "test-key", "fake"),
            lease_db=str(lease_path),
            lease_run_id="run-cli",
        )
    )

    assert code == 1
    assert model.calls == 0
    assert "ownership could not be proven" in capsys.readouterr().err


def test_cli_ownership_loss_after_tool_progress_marks_trace_incomplete(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    async def run() -> int:
        class HeartbeatFailureStore:
            def __init__(self) -> None:
                self.renew_failed = asyncio.Event()

            async def initialize(self):
                return None

            async def close(self):
                return None

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl):
                await executor.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed while tool was running")

            async def release(self, run_id, owner_id):
                return None

        class BlockingExecutor(RecordingExecutor):
            definition_value = ToolDefinition(
                name="echo",
                description="echo",
                input_schema={"type": "object", "additionalProperties": False},
                risk_level=ToolRiskLevel.READ_ONLY,
            )

            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()

            async def execute_with_result_budget(self, call, *, result_budget):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise

        store = HeartbeatFailureStore()
        executor = BlockingExecutor()
        model = ToolModel(ToolCall(id="call-cli-progress", name="echo", arguments={}))
        original_runtime = ChatRuntime

        def runtime_factory(*args, **kwargs):
            return original_runtime(
                *args,
                **kwargs,
                lease_ttl=timedelta(seconds=1),
                lease_heartbeat_interval=timedelta(milliseconds=10),
            )

        monkeypatch.setattr(cli, "SQLiteRunLeaseStore", lambda path: store)
        monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: model)
        monkeypatch.setattr(cli, "ToolExecutor", lambda registry, timeout: executor)
        monkeypatch.setattr(cli, "ChatRuntime", runtime_factory)
        task = asyncio.create_task(
            cli._run_chat(
                "hello",
                ToolRegistry(),
                model_config=ModelConfig("https://example.test", "test-key", "fake"),
                tools=[executor.definition_value],
                state_db=str(tmp_path / "runs.db"),
                lease_db=str(tmp_path / "leases.db"),
            )
        )
        await executor.started.wait()
        await store.renew_failed.wait()
        code = await task
        assert executor.cancelled.is_set()
        return code

    code = asyncio.run(run())
    capsys.readouterr()
    connection = sqlite3.connect(tmp_path / "runs.db")
    status, trace_complete, error_code = connection.execute(
        "SELECT status, trace_complete, error_code FROM runs LIMIT 1"
    ).fetchone()
    event_types = [
        row[0]
        for row in connection.execute("SELECT event_type FROM run_events ORDER BY sequence")
    ]
    connection.close()

    assert code == 1
    assert status == "cancelled"
    assert trace_complete == 0
    assert error_code == "lease_lost_after_execution_progress"
    assert "lease_ownership_lost" in event_types
    assert "tool_result_recorded" not in event_types


def test_cli_release_failure_audits_cleanup_without_rewriting_completed_outcome(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class ReleaseFailureStore:
        async def initialize(self):
            return None

        async def close(self):
            return None

        async def acquire(self, run_id, owner_id, ttl):
            now = datetime.now(timezone.utc)
            return RunLease(run_id, owner_id, now, now, now + ttl)

        async def renew(self, run_id, owner_id, ttl):
            raise AssertionError("heartbeat should not run")

        async def release(self, run_id, owner_id):
            raise RunLeaseStoreError("release persistence failed")

    monkeypatch.setattr(cli, "SQLiteRunLeaseStore", lambda path: ReleaseFailureStore())
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: CountingTextModel())
    code = asyncio.run(
        cli._run_chat(
            "hello",
            ToolRegistry(),
            model_config=ModelConfig("https://example.test", "test-key", "fake"),
            state_db=str(tmp_path / "runs.db"),
            lease_db=str(tmp_path / "leases.db"),
        )
    )
    assert "Terminal lease cleanup failed" in capsys.readouterr().err

    connection = sqlite3.connect(tmp_path / "runs.db")
    status, error_code = connection.execute(
        "SELECT status, error_code FROM runs LIMIT 1"
    ).fetchone()
    event_types = [
        row[0]
        for row in connection.execute("SELECT event_type FROM run_events ORDER BY sequence")
    ]
    connection.close()

    assert code == 1
    assert status == "completed"
    assert error_code is None
    assert event_types[-2:] == ["lease_release_failed", "run_completed"]
