from __future__ import annotations

import asyncio
import gc
import sqlite3
import threading
import weakref
from contextlib import closing, suppress
from datetime import datetime, timedelta, timezone

import pytest

from nexusmind import cli
from nexusmind.config import ModelConfig
from nexusmind.runtime.lease_guarding import LeaseGuardedChatModel
import nexusmind.runtime.leases as leases_module
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
    def __init__(
        self,
        *args,
        begin_immediate_attempted: threading.Event,
        connection_closed: threading.Event | None = None,
        **kwargs,
    ) -> None:
        self._begin_immediate_attempted = begin_immediate_attempted
        self._connection_closed = connection_closed
        super().__init__(*args, **kwargs)

    def execute(self, sql, parameters=()):
        if sql == "BEGIN IMMEDIATE":
            self._begin_immediate_attempted.set()
        return super().execute(sql, parameters)

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._connection_closed is not None:
                self._connection_closed.set()


class ConnectionObservedLeaseStore(SQLiteRunLeaseStore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.begin_immediate_attempted = threading.Event()

    def _connect(self, timeout: float | None = None) -> sqlite3.Connection:
        resolved_timeout = self._timeout if timeout is None else timeout
        db = sqlite3.connect(
            self._path,
            timeout=resolved_timeout,
            isolation_level=None,
            factory=lambda *args, **kwargs: BeginImmediateObservedConnection(
                *args,
                begin_immediate_attempted=self.begin_immediate_attempted,
                **kwargs,
            ),
        )
        db.execute(f"PRAGMA busy_timeout={max(1, int(resolved_timeout * 1000))}")
        db.execute("PRAGMA synchronous=FULL")
        return db


class PreacquiredConnectionObservedLeaseStore(SQLiteRunLeaseStore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.begin_immediate_attempted = threading.Event()
        self.release_connection_closed = threading.Event()
        self.preacquired_lease: RunLease | None = None

    async def seed(self, run_id: str, owner_id: str, ttl: timedelta) -> RunLease:
        lease = await super().acquire(run_id, owner_id, ttl)
        self.preacquired_lease = lease
        return lease

    async def acquire(self, run_id: str, owner_id: str, ttl: timedelta) -> RunLease:
        lease = self.preacquired_lease
        if lease is None:
            return await super().acquire(run_id, owner_id, ttl)
        assert lease.run_id == run_id
        assert lease.owner_id == owner_id
        return lease

    def _connect(self, timeout: float | None = None) -> sqlite3.Connection:
        resolved_timeout = self._timeout if timeout is None else timeout
        db = sqlite3.connect(
            self._path,
            timeout=resolved_timeout,
            isolation_level=None,
            factory=lambda *args, **kwargs: BeginImmediateObservedConnection(
                *args,
                begin_immediate_attempted=self.begin_immediate_attempted,
                connection_closed=self.release_connection_closed,
                **kwargs,
            ),
        )
        db.execute(f"PRAGMA busy_timeout={max(1, int(resolved_timeout * 1000))}")
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
        renewed = await store.renew("run-1", "owner-1", timedelta(seconds=10), acquired.generation)

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
        acquired = await store.acquire("run-1", "owner-1", timedelta(seconds=10))

        with pytest.raises(RunLeaseUnavailable):
            await store.acquire("run-1", "owner-2", timedelta(seconds=10))
        with pytest.raises(RunLeaseOwnershipLost):
            await store.renew("run-1", "owner-2", timedelta(seconds=10), acquired.generation)
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
        old_lease = await stores[0].acquire("run-takeover", "old-owner", timedelta(seconds=5))
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
            await stores[0].renew(
                "run-takeover", "old-owner", timedelta(seconds=10), old_lease.generation
            )
        with pytest.raises(RunLeaseOwnershipLost):
            await stores[0].release("run-takeover", "old-owner", old_lease.generation)
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
            await store.release("run-1", "owner-2", first.generation)
        await store.release("run-1", "owner-1", first.generation)
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
                    store.renew("run-1", "owner-1", timedelta(seconds=10), old_lease.generation)
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


def test_same_owner_new_generation_rejects_stale_renew_and_release(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        store = SQLiteRunLeaseStore(tmp_path / "generation.db", clock=clock)
        await store.initialize()
        first = await store.acquire("run-generation", "same-owner", timedelta(seconds=5))
        clock.advance(6)
        second = await store.acquire("run-generation", "same-owner", timedelta(seconds=5))
        assert first.generation != second.generation
        with pytest.raises(RunLeaseOwnershipLost):
            await store.renew(
                "run-generation", "same-owner", timedelta(seconds=5), first.generation
            )
        with pytest.raises(RunLeaseOwnershipLost):
            await store.release("run-generation", "same-owner", first.generation)
        assert await store.inspect("run-generation") == second

    asyncio.run(run())


def test_release_without_generation_cannot_delete_new_same_owner_generation(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        store = SQLiteRunLeaseStore(tmp_path / "missing-release-generation.db", clock=clock)
        await store.initialize()
        await store.acquire("run-missing-generation", "same-owner", timedelta(seconds=5))
        clock.advance(6)
        second = await store.acquire("run-missing-generation", "same-owner", timedelta(seconds=5))

        with pytest.raises(TypeError):
            await store.release("run-missing-generation", "same-owner")
        assert await store.inspect("run-missing-generation") == second

    asyncio.run(run())


def test_renew_without_generation_cannot_adopt_new_same_owner_generation(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        store = SQLiteRunLeaseStore(tmp_path / "missing-renew-generation.db", clock=clock)
        await store.initialize()
        await store.acquire("run-missing-generation", "same-owner", timedelta(seconds=5))
        clock.advance(6)
        second = await store.acquire("run-missing-generation", "same-owner", timedelta(seconds=5))

        with pytest.raises(TypeError):
            await store.renew("run-missing-generation", "same-owner", timedelta(seconds=5))
        assert await store.inspect("run-missing-generation") == second

    asyncio.run(run())


def test_legacy_lease_schema_is_migrated_with_generation(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "legacy.db"
        with closing(sqlite3.connect(path)) as db:
            db.execute(
                "CREATE TABLE run_leases (run_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, "
                "acquired_at TEXT NOT NULL, heartbeat_at TEXT NOT NULL, expires_at TEXT NOT NULL)"
            )
            db.commit()
        store = SQLiteRunLeaseStore(path)
        await store.initialize()
        lease = await store.acquire("run-legacy", "owner-legacy", timedelta(seconds=5))
        assert lease.generation
        assert (await store.inspect("run-legacy")).generation == lease.generation

    asyncio.run(run())


def test_expiry_watchdog_cancels_inflight_tool_while_renew_blocked() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.released = False

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + timedelta(milliseconds=60))

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await asyncio.Event().wait()

            async def release(self, run_id, owner_id, generation=None):
                self.released = True

        class Executor(RecordingExecutor):
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

        store = Store()
        executor = Executor()
        runtime = ChatRuntime(
            ToolModel(ToolCall(id="expiry-call", name="echo", arguments={})),
            tool_executor=executor,
            lease_store=store,
            lease_run_id="run-expiry-watchdog",
            lease_owner_id="owner-expiry-watchdog",
            lease_ttl=timedelta(milliseconds=60),
            lease_heartbeat_interval=timedelta(milliseconds=10),
        )
        task = asyncio.create_task(
            _collect_events(runtime.stream_user_message("hello", tools=[executor.definition_value]))
        )
        await executor.started.wait()
        with pytest.raises(RunLeaseOwnershipLost):
            await asyncio.wait_for(task, timeout=1)
        assert executor.cancelled.is_set()
        assert store.released is False

    asyncio.run(run())


def test_uncertain_tool_cancellation_keeps_unexpired_lease() -> None:
    async def run() -> None:
        tool_started = asyncio.Event()

        class Store:
            def __init__(self) -> None:
                self.released = False

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + timedelta(seconds=5))

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await tool_started.wait()
                raise RunLeaseStoreError("heartbeat failed")

            async def release(self, run_id, owner_id, generation=None):
                self.released = True

        class Executor(RecordingExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.started = tool_started

            async def execute_with_result_budget(self, call, *, result_budget):
                self.started.set()
                await asyncio.Event().wait()

        store = Store()
        executor = Executor()
        runtime = ChatRuntime(
            ToolModel(ToolCall(id="uncertain-call", name="echo", arguments={})),
            tool_executor=executor,
            lease_store=store,
            lease_run_id="run-uncertain-tool",
            lease_owner_id="owner-uncertain-tool",
            lease_ttl=timedelta(seconds=5),
            lease_heartbeat_interval=timedelta(milliseconds=10),
        )
        task = asyncio.create_task(
            _collect_events(runtime.stream_user_message("hello", tools=[executor.definition_value]))
        )
        await executor.started.wait()
        with pytest.raises(RunLeaseStoreError, match="heartbeat failed"):
            await asyncio.wait_for(task, timeout=1)
        assert store.released is False

    asyncio.run(run())


def test_cooperative_tool_cancellation_does_not_permanently_lock_runtime() -> None:
    async def run() -> None:
        clock = MutableClock()

        class Store:
            def __init__(self) -> None:
                self.active: RunLease | None = None
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = clock()
                if self.active is not None and not self.active.is_expired(now):
                    raise RunLeaseUnavailable("Run already has an active execution owner")
                self.active = RunLease(run_id, owner_id, now, now, now + ttl)
                return self.active

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run in this test")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1
                if self.active is not None and self.active.generation == generation:
                    self.active = None

        class OneToolModel(ChatModel):
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, messages, tools=None):
                self.calls += 1
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                if self.calls == 1:
                    yield RuntimeEvent(
                        RuntimeEventType.TOOL_CALL_COMPLETED,
                        tool_call=ToolCall(id="cooperative-call", name="echo", arguments={}),
                    )
                    yield RuntimeEvent(
                        RuntimeEventType.MODEL_TURN_COMPLETED,
                        finish_reason="tool_calls",
                    )
                else:
                    yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
                    yield RuntimeEvent(
                        RuntimeEventType.MODEL_TURN_COMPLETED,
                        finish_reason="stop",
                    )

        class CooperativeExecutor(RecordingExecutor):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.cancelled = asyncio.Event()
                self.calls = 0

            async def execute_with_result_budget(self, call, *, result_budget):
                self.calls += 1
                self.started.set()
                if self.calls == 1:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.cancelled.set()
                        raise
                return ToolResult(call_id=call.id, name=call.name, output="ok")

        store = Store()
        executor = CooperativeExecutor()
        runtime = ChatRuntime(
            OneToolModel(),
            tool_executor=executor,
            lease_store=store,
            lease_run_id="run-cooperative-cancel",
            lease_owner_id="owner-cooperative-cancel",
            lease_ttl=timedelta(seconds=5),
            lease_heartbeat_interval=timedelta(seconds=1),
            lease_clock=clock,
        )
        first = asyncio.create_task(
            _collect_events(runtime.stream_user_message("first", tools=[executor.definition_value]))
        )
        await executor.started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert executor.cancelled.is_set()
        assert store.release_calls == 0
        assert runtime._lease_execution_token is None

        with pytest.raises(RunLeaseUnavailable):
            await _collect_events(runtime.stream_user_message("before-expiry"))

        clock.advance(6)
        events = await _collect_events(runtime.stream_user_message("after-expiry"))
        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert store.release_calls == 1

    asyncio.run(run())


def test_nonterminal_stubborn_aclose_blocks_runtime_reuse_until_resolved() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.release_calls = 0
                self.close_started = asyncio.Event()
                self.allow_close = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class StubbornCloseIterator:
            def __init__(self) -> None:
                self.delivered = False
                self.first_delivered = asyncio.Event()
                self.second_started = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.delivered:
                    self.delivered = True
                    self.first_delivered.set()
                    return RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="partial")
                self.second_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise

            async def aclose(self):
                store.close_started.set()
                while not store.allow_close.is_set():
                    try:
                        await store.allow_close.wait()
                    except asyncio.CancelledError:
                        continue

        store = Store()
        iterator = StubbornCloseIterator()
        resolved = asyncio.Event()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-stubborn-close",
            owner_id="owner-stubborn-close",
            lease_release_timeout=timedelta(milliseconds=20),
            on_execution_resolved=resolved.set,
        )
        task = asyncio.create_task(_collect_events(coordinator.stream(iterator)))
        await asyncio.wait_for(iterator.first_delivered.wait(), timeout=0.2)
        await asyncio.wait_for(iterator.second_started.wait(), timeout=0.2)
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert task in done
        with pytest.raises(asyncio.CancelledError):
            task.result()

        await asyncio.wait_for(store.close_started.wait(), timeout=0.2)
        assert coordinator.execution_unresolved is True
        assert store.release_calls == 0

        store.allow_close.set()
        await asyncio.wait_for(resolved.wait(), timeout=0.2)
        assert coordinator.execution_unresolved is False

    asyncio.run(run())


def test_resolved_anext_during_close_does_not_release_guard_session() -> None:
    async def run() -> None:
        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                raise AssertionError("uncertain execution must not release")

        class StubbornIterator:
            def __init__(self) -> None:
                self.delivered = False
                self.next_started = asyncio.Event()
                self.allow_next = asyncio.Event()
                self.next_finished = asyncio.Event()
                self.close_started = asyncio.Event()
                self.allow_close = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.delivered:
                    self.delivered = True
                    return RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="partial")
                self.next_started.set()
                while not self.allow_next.is_set():
                    try:
                        await self.allow_next.wait()
                    except asyncio.CancelledError:
                        continue
                self.next_finished.set()
                raise StopAsyncIteration

            async def aclose(self):
                self.close_started.set()
                while not self.allow_close.is_set():
                    try:
                        await self.allow_close.wait()
                    except asyncio.CancelledError:
                        continue

        store = Store()
        iterator = StubbornIterator()
        guard = RunLeaseOwnershipGuard()
        resolved = asyncio.Event()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-resolved-anext-during-close",
            owner_id="owner-resolved-anext-during-close",
            lease_release_timeout=timedelta(milliseconds=20),
            guard=guard,
            on_execution_resolved=resolved.set,
        )
        task = asyncio.create_task(_collect_events(coordinator.stream(iterator)))
        await asyncio.wait_for(iterator.next_started.wait(), timeout=0.2)
        task.cancel()
        await asyncio.wait_for(iterator.close_started.wait(), timeout=0.2)
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert task in done
        assert coordinator.execution_unresolved is True

        iterator.allow_next.set()
        await asyncio.wait_for(iterator.next_finished.wait(), timeout=0.2)
        assert coordinator.execution_unresolved is True
        with pytest.raises(RunLeaseUnavailable):
            guard.begin_execution()

        iterator.allow_close.set()
        await asyncio.wait_for(resolved.wait(), timeout=0.2)
        assert coordinator.execution_unresolved is False

    asyncio.run(run())


def test_multiple_unresolved_tasks_do_not_clear_execution_early() -> None:
    async def run() -> None:
        resolved = asyncio.Event()
        guard = RunLeaseOwnershipGuard()
        coordinator = RunLeaseCoordinator(
            object(),
            run_id="run-multiple-unresolved",
            owner_id="owner-multiple-unresolved",
            on_execution_resolved=resolved.set,
            guard=guard,
        )
        coordinator._execution_token = guard.begin_execution()
        first_gate = asyncio.Event()
        second_gate = asyncio.Event()

        async def wait_for(gate: asyncio.Event) -> None:
            await gate.wait()

        first = asyncio.create_task(wait_for(first_gate))
        second = asyncio.create_task(wait_for(second_gate))
        coordinator._mark_execution_uncertain(first)
        coordinator._mark_execution_uncertain(second)
        coordinator._cleanup_complete = True

        second_gate.set()
        await asyncio.sleep(0)
        assert coordinator.execution_unresolved is True
        assert not resolved.is_set()

        first_gate.set()
        await asyncio.wait_for(resolved.wait(), timeout=0.2)
        assert coordinator.execution_unresolved is False

    asyncio.run(run())


def test_shared_guard_cannot_be_reused_while_an_execution_is_active() -> None:
    guard = RunLeaseOwnershipGuard()
    now = datetime.now(timezone.utc)
    first_token = guard.begin_execution()
    guard.prove(
        RunLease("run-shared-guard-a", "owner-a", now, now, now + timedelta(seconds=10)),
        initial=True,
    )
    guard.fail(RunLeaseOwnershipLost("first execution lost ownership"))

    with pytest.raises(RunLeaseUnavailable):
        guard.begin_execution()
    with pytest.raises(RunLeaseOwnershipLost, match="first execution"):
        guard.assert_owned()

    guard.end_execution(first_token)
    second_token = guard.begin_execution()
    guard.prove(
        RunLease("run-shared-guard-b", "owner-b", now, now, now + timedelta(seconds=10)),
        initial=True,
    )
    guard.assert_owned()
    guard.end_execution(second_token)


def test_ended_guard_session_cannot_prove_ownership() -> None:
    guard = RunLeaseOwnershipGuard()
    now = datetime.now(timezone.utc)
    token = guard.begin_execution()
    guard.prove(
        RunLease("run-ended-session", "owner-ended-session", now, now, now + timedelta(seconds=10)),
        initial=True,
    )
    guard.end_execution(token)

    with pytest.raises(RunLeaseOwnershipLost, match="No active lease execution session"):
        guard.assert_owned()


def test_iterator_creation_failure_releases_guard_session() -> None:
    async def run() -> None:
        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                return None

        class BrokenIterator:
            def __aiter__(self):
                raise RuntimeError("iterator construction failed")

        guard = RunLeaseOwnershipGuard()
        first = RunLeaseCoordinator(Store(), run_id="run-broken-iterator", owner_id="owner-a", guard=guard)
        with pytest.raises(RuntimeError, match="iterator construction failed"):
            [event async for event in first.stream(BrokenIterator())]

        async def events():
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        second = RunLeaseCoordinator(Store(), run_id="run-recovered", owner_id="owner-b", guard=guard)
        output = [event async for event in second.stream(events())]
        assert output[-1].type is RuntimeEventType.RUN_COMPLETED

    asyncio.run(run())


def test_nonterminal_release_failure_is_surfaced_when_no_primary_error() -> None:
    async def run() -> None:
        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                raise RunLeaseStoreError("release failed")

        async def empty_events():
            if False:
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="unused")

        coordinator = RunLeaseCoordinator(
            Store(), run_id="run-release-failure", lease_release_timeout=timedelta(milliseconds=20)
        )
        with pytest.raises(RunLeaseReleaseError, match="release failed"):
            [event async for event in coordinator.stream(empty_events())]

    asyncio.run(run())


def test_cancelled_acquire_after_commit_does_not_leave_ghost_lease(tmp_path) -> None:
    class CommitPausedConnection(sqlite3.Connection):
        armed = False
        committed = threading.Event()
        allow_return = threading.Event()

        def commit(self) -> None:
            super().commit()
            if type(self).armed:
                type(self).committed.set()
                type(self).allow_return.wait(1)

    class CommitPausedStore(SQLiteRunLeaseStore):
        def _connect(self, timeout=None):
            resolved_timeout = self._timeout if timeout is None else timeout
            return sqlite3.connect(
                self._path,
                timeout=resolved_timeout,
                isolation_level=None,
                factory=CommitPausedConnection,
            )

    async def run() -> None:
        store = CommitPausedStore(tmp_path / "cancel-acquire.db")
        await store.initialize()
        CommitPausedConnection.armed = True
        task = asyncio.create_task(store.acquire("run-cancel-acquire", "owner", timedelta(seconds=5)))
        assert await asyncio.to_thread(CommitPausedConnection.committed.wait, 1)
        task.cancel()
        CommitPausedConnection.allow_return.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        with closing(sqlite3.connect(tmp_path / "cancel-acquire.db")) as db:
            assert db.execute(
                "SELECT 1 FROM run_leases WHERE run_id = ?", ("run-cancel-acquire",)
            ).fetchone() is None
        CommitPausedConnection.armed = False

    asyncio.run(run())


def test_cancelled_acquire_compensates_after_writer_lock_wait(tmp_path) -> None:
    class CommitPausedConnection(sqlite3.Connection):
        armed = False
        committed = threading.Event()
        allow_return = threading.Event()

        def commit(self) -> None:
            super().commit()
            if type(self).armed:
                type(self).committed.set()
                type(self).allow_return.wait(1)

    class CommitPausedStore(SQLiteRunLeaseStore):
        def _connect(self, timeout=None):
            resolved_timeout = self._timeout if timeout is None else timeout
            return sqlite3.connect(
                self._path,
                timeout=resolved_timeout,
                isolation_level=None,
                factory=CommitPausedConnection,
            )

    async def run() -> None:
        path = tmp_path / "cancel-acquire-lock.db"
        store = CommitPausedStore(path, timeout=1)
        await store.initialize()
        holder = sqlite3.connect(path, isolation_level=None)
        task = None
        CommitPausedConnection.armed = True
        try:
            task = asyncio.create_task(
                store.acquire("run-cancel-acquire-lock", "owner", timedelta(seconds=5))
            )
            assert await asyncio.to_thread(CommitPausedConnection.committed.wait, 1)
            task.cancel()
            holder.execute("BEGIN IMMEDIATE")
            CommitPausedConnection.allow_return.set()
            await asyncio.sleep(0.1)
            holder.rollback()
            holder.close()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await store.inspect("run-cancel-acquire-lock") is None
        finally:
            CommitPausedConnection.armed = False
            CommitPausedConnection.allow_return.set()
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            with suppress(Exception):
                holder.rollback()
            with suppress(Exception):
                holder.close()
            await store.close()

    asyncio.run(run())


def test_cancelling_one_acquire_does_not_cancel_parallel_same_owner_acquire(tmp_path) -> None:
    class BlockingAcquireStore(SQLiteRunLeaseStore):
        def __init__(self, path):
            super().__init__(path)
            self.started = [threading.Event(), threading.Event()]
            self.allow = [threading.Event(), threading.Event()]
            self.calls = 0
            self.calls_lock = threading.Lock()

        def _acquire(
            self, run_id, owner_id, ttl, generation=None, cancellation=None, timeout=None
        ):
            with self.calls_lock:
                index = self.calls
                self.calls += 1
            self.started[index].set()
            while not cancellation.is_set() and not self.allow[index].wait(0.001):
                pass
            if cancellation.is_set():
                return leases_module._SQLiteLeaseAttempt.CANCELLED
            now = datetime.now(timezone.utc)
            return RunLease(
                run_id,
                owner_id,
                now,
                now,
                now + ttl,
                generation=f"generation-{index}",
            )

    async def run() -> None:
        store = BlockingAcquireStore(tmp_path / "parallel-acquire-cancel.db")
        await store.initialize()
        first = asyncio.create_task(
            store.acquire(
                "run-parallel-acquire",
                "same-owner",
                timedelta(seconds=5),
                operation_id="acquire-first",
            )
        )
        second = asyncio.create_task(
            store.acquire(
                "run-parallel-acquire",
                "same-owner",
                timedelta(seconds=5),
                operation_id="acquire-second",
            )
        )
        await asyncio.gather(
            asyncio.to_thread(store.started[0].wait, 1),
            asyncio.to_thread(store.started[1].wait, 1),
        )

        store.cancel_acquire(
            "run-parallel-acquire", "same-owner", operation_id="acquire-first"
        )
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not second.done()

        store.allow[1].set()
        lease = await second
        assert lease.generation == "generation-1"
        await store.close()

    asyncio.run(run())


def test_old_generation_cancel_renew_does_not_cancel_new_generation(tmp_path) -> None:
    class BlockingRenewStore(SQLiteRunLeaseStore):
        def __init__(self, path):
            super().__init__(path)
            self.started = {"generation-1": threading.Event(), "generation-2": threading.Event()}
            self.allow = {"generation-1": threading.Event(), "generation-2": threading.Event()}

        def _renew(
            self, run_id, owner_id, ttl, generation=None, cancellation=None, timeout=None
        ):
            self.started[generation].set()
            while not cancellation.is_set() and not self.allow[generation].wait(0.001):
                pass
            if cancellation.is_set():
                return leases_module._SQLiteLeaseAttempt.CANCELLED
            now = datetime.now(timezone.utc)
            return RunLease(run_id, owner_id, now, now, now + ttl, generation=generation)

    async def run() -> None:
        store = BlockingRenewStore(tmp_path / "generation-renew-cancel.db")
        await store.initialize()
        first = asyncio.create_task(
            store.renew(
                "run-generation-cancel",
                "same-owner",
                timedelta(seconds=5),
                "generation-1",
                operation_id="renew-first",
            )
        )
        second = asyncio.create_task(
            store.renew(
                "run-generation-cancel",
                "same-owner",
                timedelta(seconds=5),
                "generation-2",
                operation_id="renew-second",
            )
        )
        await asyncio.gather(
            asyncio.to_thread(store.started["generation-1"].wait, 1),
            asyncio.to_thread(store.started["generation-2"].wait, 1),
        )

        store.cancel_renew(
            "run-generation-cancel",
            "same-owner",
            generation="generation-1",
            operation_id="renew-first",
        )
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not second.done()

        store.allow["generation-2"].set()
        lease = await second
        assert lease.generation == "generation-2"
        await store.close()

    asyncio.run(run())


def test_old_generation_cancel_release_does_not_cancel_new_generation(tmp_path) -> None:
    class BlockingReleaseStore(SQLiteRunLeaseStore):
        def __init__(self, path):
            super().__init__(path)
            self.started = {"generation-1": threading.Event(), "generation-2": threading.Event()}
            self.allow = {"generation-1": threading.Event(), "generation-2": threading.Event()}

        def _release_attempt(
            self, run_id, owner_id, generation, cancellation, timeout
        ):
            self.started[generation].set()
            while not cancellation.is_set() and not self.allow[generation].wait(0.001):
                pass
            if cancellation.is_set():
                return leases_module._SQLiteReleaseAttempt.CANCELLED
            return leases_module._SQLiteReleaseAttempt.RELEASED

    async def run() -> None:
        store = BlockingReleaseStore(tmp_path / "generation-release-cancel.db")
        await store.initialize()
        first = asyncio.create_task(
            store.release(
                "run-generation-cancel",
                "same-owner",
                "generation-1",
                operation_id="release-first",
            )
        )
        second = asyncio.create_task(
            store.release(
                "run-generation-cancel",
                "same-owner",
                "generation-2",
                operation_id="release-second",
            )
        )
        await asyncio.gather(
            asyncio.to_thread(store.started["generation-1"].wait, 1),
            asyncio.to_thread(store.started["generation-2"].wait, 1),
        )

        store.cancel_release(
            "run-generation-cancel",
            "same-owner",
            generation="generation-1",
            operation_id="release-first",
        )
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not second.done()

        store.allow["generation-2"].set()
        await second
        await store.close()

    asyncio.run(run())


def test_guard_uncertainty_does_not_leak_into_next_execution() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.acquire_calls = 0
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                self.acquire_calls += 1
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        store = Store()
        guard = RunLeaseOwnershipGuard()

        async def first_events():
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            guard.mark_execution_uncertain()

        first = RunLeaseCoordinator(
            store,
            run_id="run-uncertainty-reset",
            owner_id="owner-uncertainty-reset",
            heartbeat_interval=timedelta(seconds=1),
            guard=guard,
        )
        events = [event async for event in first.stream(first_events())]
        assert events[-1].type is RuntimeEventType.MODEL_STARTED
        assert store.release_calls == 0

        async def second_events():
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)

        second = RunLeaseCoordinator(
            store,
            run_id="run-uncertainty-reset",
            owner_id="owner-uncertainty-reset",
            heartbeat_interval=timedelta(seconds=1),
            guard=guard,
        )
        events = [event async for event in second.stream(second_events())]
        assert events[-1].type is RuntimeEventType.MODEL_STARTED
        assert store.acquire_calls == 2
        assert store.release_calls == 1

    asyncio.run(run())


def test_legacy_owner_only_release_is_never_invoked() -> None:
    async def run() -> None:
        class LegacyStore:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation="generation-1",
                )

            async def renew(self, run_id, owner_id, ttl):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id):
                self.release_calls += 1

        async def events():
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        store = LegacyStore()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-legacy-store",
            owner_id="owner-legacy-store",
        )
        output = [event async for event in coordinator.stream(events())]

        assert output[-1].metadata["lease_release_failed"] is True
        assert store.release_calls == 0

    asyncio.run(run())


def test_terminal_does_not_hang_when_renew_suppresses_cancellation() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.renew_started = asyncio.Event()
                self.never_finish = asyncio.Event()
                self.renew_finished = asyncio.Event()
                self.release_called = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation="generation-1",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                self.renew_started.set()
                try:
                    await self.never_finish.wait()
                except asyncio.CancelledError:
                    await self.never_finish.wait()
                finally:
                    self.renew_finished.set()

            async def release(self, run_id, owner_id, generation=None):
                self.release_called.set()

        store = Store()

        async def events():
            await store.renew_started.wait()
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-renew-cancellation",
            owner_id="owner-renew-cancellation",
            heartbeat_interval=timedelta(milliseconds=1),
            lease_release_timeout=timedelta(milliseconds=20),
        )
        output = await asyncio.wait_for(
            _collect_events(coordinator.stream(events())), timeout=0.2
        )

        assert output[-1].type is RuntimeEventType.RUN_COMPLETED
        assert store.release_called.is_set()
        assert coordinator._heartbeat_active is False
        store.never_finish.set()
        await asyncio.wait_for(store.renew_finished.wait(), timeout=0.2)

    asyncio.run(run())


def test_cancelled_stream_with_stubborn_iterator_does_not_renew_forever() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.renew_started = asyncio.Event()
                self.allow_renew_exit = asyncio.Event()
                self.renew_finished = asyncio.Event()
                self.release_called = False

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                self.renew_started.set()
                try:
                    await self.allow_renew_exit.wait()
                except asyncio.CancelledError:
                    await self.allow_renew_exit.wait()
                finally:
                    self.renew_finished.set()
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def release(self, run_id, owner_id, generation=None):
                self.release_called = True

        class StubbornIterator:
            def __init__(self) -> None:
                self.delivered = False
                self.next_started = asyncio.Event()
                self.allow_next_exit = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.delivered:
                    self.delivered = True
                    return RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="partial")
                self.next_started.set()
                try:
                    await self.allow_next_exit.wait()
                except asyncio.CancelledError:
                    await self.allow_next_exit.wait()
                raise StopAsyncIteration

        store = Store()
        iterator = StubbornIterator()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-stubborn-next",
            owner_id="owner-stubborn-next",
            heartbeat_interval=timedelta(milliseconds=1),
            lease_release_timeout=timedelta(milliseconds=20),
        )
        task = asyncio.create_task(_collect_events(coordinator.stream(iterator)))
        await asyncio.wait_for(iterator.next_started.wait(), timeout=0.2)
        await asyncio.wait_for(store.renew_started.wait(), timeout=0.2)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)

        assert coordinator._heartbeat_active is False
        assert coordinator.execution_unresolved is True
        assert store.release_called is False

        iterator.allow_next_exit.set()
        store.allow_renew_exit.set()
        await asyncio.wait_for(store.renew_finished.wait(), timeout=0.2)

    asyncio.run(run())


def test_cancelled_custom_acquire_that_finishes_late_is_compensated() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.acquire_started = asyncio.Event()
                self.allow_commit = asyncio.Event()
                self.release_called = asyncio.Event()
                self.release_generations = []

            async def acquire(self, run_id, owner_id, ttl):
                self.acquire_started.set()
                try:
                    await self.allow_commit.wait()
                except asyncio.CancelledError:
                    await self.allow_commit.wait()
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation="generation-late",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_generations.append(generation)
                self.release_called.set()

        async def events():
            if False:
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="unused")

        store = Store()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-late-acquire",
            owner_id="owner-late-acquire",
            lease_release_timeout=timedelta(milliseconds=20),
        )
        task = asyncio.create_task(_collect_events(coordinator.stream(events())))
        await store.acquire_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)

        store.allow_commit.set()
        await asyncio.wait_for(store.release_called.wait(), timeout=0.2)
        assert store.release_generations == ["generation-late"]

    asyncio.run(run())


def test_stale_background_heartbeat_failure_cannot_poison_next_execution() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.acquire_calls = 0
                self.old_renew_started = asyncio.Event()
                self.allow_old_return = asyncio.Event()
                self.old_finished = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                self.acquire_calls += 1
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation=f"generation-{self.acquire_calls}",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                if generation == "generation-1":
                    self.old_renew_started.set()
                    try:
                        await self.allow_old_return.wait()
                    except asyncio.CancelledError:
                        await self.allow_old_return.wait()
                        self.old_finished.set()
                        raise RunLeaseOwnershipLost("stale heartbeat failed")
                raise AssertionError("next generation heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                return None

        guard = RunLeaseOwnershipGuard()
        store = Store()

        async def first_events():
            await store.old_renew_started.wait()
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        first = RunLeaseCoordinator(
            store,
            run_id="run-stale-heartbeat-failure",
            owner_id="owner-stale-heartbeat-failure",
            heartbeat_interval=timedelta(milliseconds=1),
            lease_release_timeout=timedelta(milliseconds=20),
            guard=guard,
        )
        first_output = await asyncio.wait_for(
            _collect_events(first.stream(first_events())), timeout=0.2
        )
        assert first_output[-1].type is RuntimeEventType.RUN_COMPLETED

        second_ready = asyncio.Event()
        second_finish = asyncio.Event()

        async def second_events():
            second_ready.set()
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="still owned")
            await second_finish.wait()
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        second = RunLeaseCoordinator(
            store,
            run_id="run-stale-heartbeat-failure",
            owner_id="owner-stale-heartbeat-failure",
            heartbeat_interval=timedelta(seconds=1),
            guard=guard,
        )
        second_task = asyncio.create_task(
            _collect_events(second.stream(second_events()))
        )
        await second_ready.wait()
        store.allow_old_return.set()
        await store.old_finished.wait()

        assert guard.lease is not None
        assert guard.lease.generation == "generation-2"
        guard.assert_owned()

        second_finish.set()
        await second_task

    asyncio.run(run())


def test_stale_heartbeat_cancellation_cannot_cancel_next_generation_renew() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.acquire_calls = 0
                self.renew_tasks = set()
                self.cancel_renew_calls = 0
                self.old_renew_started = asyncio.Event()
                self.allow_old_exit = asyncio.Event()
                self.next_renew_started = asyncio.Event()
                self.allow_next_renew = asyncio.Event()
                self.next_renew_cancelled = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                self.acquire_calls += 1
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation=f"generation-{self.acquire_calls}",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                task = asyncio.current_task()
                assert task is not None
                self.renew_tasks.add(task)
                try:
                    if generation == "generation-1":
                        self.old_renew_started.set()
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError:
                            await self.allow_old_exit.wait()
                            raise
                    self.next_renew_started.set()
                    try:
                        await self.allow_next_renew.wait()
                    except asyncio.CancelledError:
                        self.next_renew_cancelled.set()
                        raise
                    now = datetime.now(timezone.utc)
                    return RunLease(
                        run_id,
                        owner_id,
                        now,
                        now,
                        now + ttl,
                        generation=generation,
                    )
                finally:
                    self.renew_tasks.discard(task)

            async def release(self, run_id, owner_id, generation=None):
                return None

            def cancel_renew(self, run_id, owner_id):
                self.cancel_renew_calls += 1
                for task in tuple(self.renew_tasks):
                    task.cancel()

        guard = RunLeaseOwnershipGuard()
        store = Store()

        async def first_events():
            await store.old_renew_started.wait()
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        first = RunLeaseCoordinator(
            store,
            run_id="run-stale-heartbeat-cancel",
            owner_id="owner-stale-heartbeat-cancel",
            heartbeat_interval=timedelta(milliseconds=1),
            lease_release_timeout=timedelta(milliseconds=20),
            guard=guard,
        )
        await asyncio.wait_for(
            _collect_events(first.stream(first_events())), timeout=0.2
        )

        second_ready = asyncio.Event()
        second_finish = asyncio.Event()

        async def second_events():
            second_ready.set()
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="still owned")
            await second_finish.wait()
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        second = RunLeaseCoordinator(
            store,
            run_id="run-stale-heartbeat-cancel",
            owner_id="owner-stale-heartbeat-cancel",
            heartbeat_interval=timedelta(milliseconds=1),
            guard=guard,
        )
        second_task = asyncio.create_task(
            _collect_events(second.stream(second_events()))
        )
        await second_ready.wait()
        await store.next_renew_started.wait()

        store.allow_old_exit.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert store.cancel_renew_calls == 0
        assert not store.next_renew_cancelled.is_set()
        guard.assert_owned()

        store.allow_next_renew.set()
        second_finish.set()
        await second_task

    asyncio.run(run())


def test_nonterminal_iterator_close_timeout_keeps_lease_fenced() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation="generation-1",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class HangingCloseIterator:
            def __init__(self) -> None:
                self.delivered = False
                self.close_started = asyncio.Event()
                self.never_close = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.delivered:
                    self.delivered = True
                    return RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="partial")
                await asyncio.Event().wait()

            async def aclose(self):
                self.close_started.set()
                while not self.never_close.is_set():
                    try:
                        await self.never_close.wait()
                    except asyncio.CancelledError:
                        continue

        store = Store()
        iterator = HangingCloseIterator()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-nonterminal-close-timeout",
            owner_id="owner-nonterminal-close-timeout",
            lease_release_timeout=timedelta(milliseconds=20),
        )
        task = asyncio.create_task(_collect_events(coordinator.stream(iterator)))
        await asyncio.sleep(0)
        while not iterator.delivered:
            await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        assert iterator.close_started.is_set()
        assert store.release_calls == 0
        iterator.never_close.set()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_nonterminal_iterator_close_failure_still_releases_lease() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation="generation-1",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class FailingIterator:
            def __init__(self) -> None:
                self.delivered = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.delivered:
                    raise RuntimeError("stream failed")
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="partial")

            async def aclose(self):
                raise OSError("iterator cleanup failed")

        store = Store()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-nonterminal-close-failure",
            owner_id="owner-nonterminal-close-failure",
        )
        with pytest.raises(RuntimeError, match="stream failed"):
            [event async for event in coordinator.stream(FailingIterator())]
        assert store.release_calls == 1

    asyncio.run(run())


def test_initial_ownership_proof_failure_releases_acquired_lease() -> None:
    async def run() -> None:
        store_clock = MutableClock()
        guard_clock = MutableClock()
        guard_clock.advance(6)

        class Store:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = store_clock()
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation="generation-1",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        async def events():
            if False:
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="unused")

        store = Store()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-initial-proof-failure",
            owner_id="owner-initial-proof-failure",
            ttl=timedelta(seconds=5),
            guard=RunLeaseOwnershipGuard(clock=guard_clock),
        )
        with pytest.raises(RunLeaseOwnershipLost):
            [event async for event in coordinator.stream(events())]
        assert store.release_calls == 1

    asyncio.run(run())


def test_lagging_guard_clock_cannot_execute_after_store_takeover(tmp_path) -> None:
    async def run() -> None:
        store_clock = MutableClock()
        guard_clock = MutableClock()
        path = tmp_path / "clock-binding.db"
        store = SQLiteRunLeaseStore(path, clock=store_clock)
        await store.initialize()
        guard = RunLeaseOwnershipGuard(clock=guard_clock)
        RunLeaseCoordinator(
            store,
            run_id="run-clock-binding",
            owner_id="owner-a",
            ttl=timedelta(seconds=5),
            guard=guard,
        )

        lease_a = await store.acquire("run-clock-binding", "owner-a", timedelta(seconds=5))
        token = guard.begin_execution()
        guard.prove(lease_a, initial=True)

        store_clock.advance(6)
        takeover_store = SQLiteRunLeaseStore(path, clock=store_clock)
        await takeover_store.initialize()
        lease_b = await takeover_store.acquire(
            "run-clock-binding", "owner-b", timedelta(seconds=5)
        )
        assert lease_b.owner_id == "owner-b"

        with pytest.raises(RunLeaseOwnershipLost):
            guard.assert_owned()

        guard.end_execution(token)
        await takeover_store.release("run-clock-binding", "owner-b", lease_b.generation)

    asyncio.run(run())


def test_terminal_closes_wrapped_iterator() -> None:
    async def run() -> None:
        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                return None

        closed = asyncio.Event()

        async def wrapped_events():
            try:
                yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)
            finally:
                closed.set()

        coordinator = RunLeaseCoordinator(Store(), run_id="run-close", owner_id="owner-close")
        events = [event async for event in coordinator.stream(wrapped_events())]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert closed.is_set()

    asyncio.run(run())


def test_cancellation_during_terminal_iterator_close_preserves_outcome() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.released = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.released.set()

        class PausingIterator:
            def __init__(self) -> None:
                self.close_started = asyncio.Event()
                self.allow_close = asyncio.Event()
                self.delivered = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.delivered:
                    raise StopAsyncIteration
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

            async def aclose(self):
                self.close_started.set()
                await self.allow_close.wait()

        store = Store()
        iterator = PausingIterator()
        coordinator = RunLeaseCoordinator(store, run_id="run-close-cancel", owner_id="owner-close-cancel")
        task = asyncio.create_task(_collect_events(coordinator.stream(iterator)))
        await iterator.close_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        iterator.allow_close.set()
        events = await task

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert store.released.is_set()

    asyncio.run(run())


def test_terminal_iterator_close_timeout_still_releases_lease() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class HangingCloseIterator:
            def __init__(self) -> None:
                self.close_started = asyncio.Event()
                self.never_close = asyncio.Event()
                self.delivered = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.delivered:
                    raise StopAsyncIteration
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

            async def aclose(self):
                self.close_started.set()
                while not self.never_close.is_set():
                    try:
                        await self.never_close.wait()
                    except asyncio.CancelledError:
                        continue

        store = Store()
        iterator = HangingCloseIterator()
        guard = RunLeaseOwnershipGuard()
        resolved = asyncio.Event()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-close-timeout",
            owner_id="owner-close-timeout",
            lease_release_timeout=timedelta(milliseconds=20),
            guard=guard,
            on_execution_resolved=resolved.set,
        )
        events = await asyncio.wait_for(
            _collect_events(coordinator.stream(iterator)), timeout=0.2
        )

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_iterator_cleanup_failed"] is True
        assert store.release_calls == 1
        assert coordinator.execution_unresolved is True

        async def second_events():
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        second = RunLeaseCoordinator(
            store,
            run_id="run-close-timeout-second",
            owner_id="owner-close-timeout-second",
            guard=guard,
        )
        with pytest.raises(RunLeaseUnavailable):
            [event async for event in second.stream(second_events())]

        iterator.never_close.set()
        await asyncio.wait_for(resolved.wait(), timeout=0.2)
        assert coordinator.execution_unresolved is False
        second_recovered = RunLeaseCoordinator(
            store,
            run_id="run-close-timeout-second-recovered",
            owner_id="owner-close-timeout-second-recovered",
            guard=guard,
        )
        second_events_result = [
            event async for event in second_recovered.stream(second_events())
        ]
        assert second_events_result[-1].type is RuntimeEventType.RUN_COMPLETED

    asyncio.run(run())


def test_terminal_iterator_close_failure_still_releases_lease() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class FailingCloseIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if hasattr(self, "delivered"):
                    raise StopAsyncIteration
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

            async def aclose(self):
                raise OSError("cleanup failed")

        store = Store()
        coordinator = RunLeaseCoordinator(
            store, run_id="run-close-failure", owner_id="owner-close-failure"
        )
        events = [
            event
            async for event in coordinator.stream(FailingCloseIterator())
        ]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_iterator_cleanup_failed"] is True
        assert store.release_calls == 1

    asyncio.run(run())


def test_terminal_iterator_runtimeerror_is_audited() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class FailingCloseIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if hasattr(self, "delivered"):
                    raise StopAsyncIteration
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

            async def aclose(self):
                raise RuntimeError("provider cleanup failed")

        store = Store()
        coordinator = RunLeaseCoordinator(
            store, run_id="run-close-runtimeerror", owner_id="owner-close-runtimeerror"
        )
        events = [
            event
            async for event in coordinator.stream(FailingCloseIterator())
        ]

        assert events[-1].metadata["lease_iterator_cleanup_failed"] is True
        assert store.release_calls == 1

    asyncio.run(run())


def test_nonterminal_iterator_attributeerror_is_not_swallowed() -> None:
    async def run() -> None:
        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                return None

        class FailingCloseIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if hasattr(self, "delivered"):
                    raise StopAsyncIteration
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="partial")

            async def aclose(self):
                raise AttributeError("provider cleanup failed")

        coordinator = RunLeaseCoordinator(
            Store(), run_id="run-close-attributeerror", owner_id="owner-close-attributeerror"
        )
        with pytest.raises(RunLeaseStoreError, match="iterator cleanup failed"):
            [
                event
                async for event in coordinator.stream(FailingCloseIterator())
            ]

    asyncio.run(run())


def test_terminal_audits_expiry_during_cleanup_barrier() -> None:
    async def run() -> None:
        clock = MutableClock()

        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = clock()
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                return None

        class AdvancingCloseIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if hasattr(self, "delivered"):
                    raise StopAsyncIteration
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

            async def aclose(self):
                clock.advance(6)

        coordinator = RunLeaseCoordinator(
            Store(),
            run_id="run-expiry-during-cleanup",
            owner_id="owner-expiry-during-cleanup",
            ttl=timedelta(seconds=5),
            heartbeat_interval=timedelta(seconds=1),
            clock=clock,
        )
        events = [
            event
            async for event in coordinator.stream(AdvancingCloseIterator())
        ]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_ownership_lost_after_terminal"] is True

    asyncio.run(run())


def test_cancelled_stream_with_stubborn_iterator_does_not_renew_forever() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.renew_calls = 0
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation="generation-stubborn",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                self.renew_calls += 1
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation=generation,
                )

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class StubbornIterator:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.allow_exit = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await self.allow_exit.wait()
                    raise

        store = Store()
        iterator = StubbornIterator()
        coordinator = RunLeaseCoordinator(
            store,
            run_id="run-stubborn-cancel",
            owner_id="owner-stubborn-cancel",
            heartbeat_interval=timedelta(milliseconds=5),
            lease_release_timeout=timedelta(milliseconds=20),
        )
        task = asyncio.create_task(_collect_events(coordinator.stream(iterator)))
        await iterator.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        renew_calls = store.renew_calls
        await asyncio.sleep(0.05)

        assert coordinator.execution_uncertain is True
        assert store.release_calls == 0
        assert store.renew_calls == renew_calls

        iterator.allow_exit.set()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_terminal_audits_expiry_when_release_crosses_ttl_without_yielding() -> None:
    async def run() -> None:
        clock = MutableClock()

        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = clock()
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                clock.advance(6)
                return True

        class TerminalIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if hasattr(self, "delivered"):
                    raise StopAsyncIteration
                self.delivered = True
                return RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        coordinator = RunLeaseCoordinator(
            Store(),
            run_id="run-expiry-during-release",
            owner_id="owner-expiry-during-release",
            ttl=timedelta(seconds=5),
            heartbeat_interval=timedelta(seconds=1),
            clock=clock,
        )
        events = [
            event
            async for event in coordinator.stream(TerminalIterator())
        ]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert events[-1].metadata["lease_ownership_lost_after_terminal"] is True

    asyncio.run(run())


def test_terminal_does_not_report_expiry_after_successful_release() -> None:
    async def run() -> None:
        clock = MutableClock()

        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = clock()
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                clock.advance(6)
                return None

        async def events():
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        coordinator = RunLeaseCoordinator(
            Store(),
            run_id="run-expiry-after-release",
            owner_id="owner-expiry-after-release",
            ttl=timedelta(seconds=5),
            heartbeat_interval=timedelta(seconds=1),
            clock=clock,
        )
        output = [event async for event in coordinator.stream(events())]

        assert output[-1].metadata.get("lease_ownership_lost_after_terminal") is None

    asyncio.run(run())


def test_precise_successful_release_overrides_late_expiry_watchdog() -> None:
    async def run() -> None:
        clock = MutableClock()

        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = clock()
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                clock.advance(6)
                await asyncio.sleep(0.06)
                return False

        async def events():
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        coordinator = RunLeaseCoordinator(
            Store(),
            run_id="run-precise-release",
            owner_id="owner-precise-release",
            ttl=timedelta(seconds=5),
            heartbeat_interval=timedelta(seconds=1),
            clock=clock,
        )
        output = [event async for event in coordinator.stream(events())]

        assert output[-1].metadata.get("lease_ownership_lost_after_terminal") is None

    asyncio.run(run())


def test_terminal_release_ownership_loss_is_audited_separately() -> None:
    async def run() -> None:
        class Store:
            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                raise RunLeaseOwnershipLost("Run lease generation changed")

        async def events():
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        coordinator = RunLeaseCoordinator(
            Store(), run_id="run-release-ownership-loss", owner_id="owner-release-ownership-loss"
        )
        output = [event async for event in coordinator.stream(events())]

        assert output[-1].metadata["lease_release_failed"] is True
        assert output[-1].metadata["lease_ownership_lost_after_terminal"] is True

    asyncio.run(run())


def test_timed_out_old_release_cannot_clear_next_execution_guard() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.acquire_calls = 0
                self.release_calls = 0
                self.first_release_started = asyncio.Event()
                self.allow_first_release = asyncio.Event()

            async def acquire(self, run_id, owner_id, ttl):
                self.acquire_calls += 1
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation=f"generation-{self.acquire_calls}",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1
                if self.release_calls == 1:
                    self.first_release_started.set()
                    try:
                        await self.allow_first_release.wait()
                    except asyncio.CancelledError:
                        await self.allow_first_release.wait()

        store = Store()
        guard = RunLeaseOwnershipGuard()

        async def first_events():
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        first = RunLeaseCoordinator(
            store,
            run_id="run-release-generation",
            owner_id="owner-release-generation",
            lease_release_timeout=timedelta(milliseconds=20),
            guard=guard,
        )
        first_result = await _collect_events(first.stream(first_events()))
        await store.first_release_started.wait()
        assert first_result[-1].type is RuntimeEventType.RUN_COMPLETED

        second_started = asyncio.Event()
        allow_second = asyncio.Event()

        async def second_events():
            second_started.set()
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            await allow_second.wait()
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)

        second = RunLeaseCoordinator(
            store,
            run_id="run-release-generation",
            owner_id="owner-release-generation",
            lease_release_timeout=timedelta(milliseconds=20),
            guard=guard,
        )
        second_task = asyncio.create_task(_collect_events(second.stream(second_events())))
        await second_started.wait()
        assert guard.lease is not None
        assert guard.lease.generation == "generation-2"

        store.allow_first_release.set()
        await asyncio.sleep(0)
        assert guard.lease is not None
        assert guard.lease.generation == "generation-2"

        allow_second.set()
        second_result = await second_task
        assert second_result[-1].type is RuntimeEventType.RUN_COMPLETED
        assert store.release_calls == 2

    asyncio.run(run())


def test_renew_rejects_clock_rollback(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        store = SQLiteRunLeaseStore(tmp_path / "clock-rollback.db", clock=clock)
        await store.initialize()
        lease = await store.acquire("run-clock-rollback", "owner", timedelta(seconds=10))
        clock.value -= timedelta(seconds=1)

        with pytest.raises(RunLeaseStoreError, match="clock moved backwards"):
            await store.renew("run-clock-rollback", "owner", timedelta(seconds=10), lease.generation)
        assert await store.inspect("run-clock-rollback") == lease

    asyncio.run(run())


def test_renew_rejects_nonmonotonic_lease_timestamps(tmp_path) -> None:
    async def run() -> None:
        clock = MutableClock()
        store = SQLiteRunLeaseStore(tmp_path / "nonmonotonic-renew.db", clock=clock)
        await store.initialize()
        lease = await store.acquire("run-nonmonotonic", "owner", timedelta(seconds=10))

        clock.advance(1)
        with pytest.raises(RunLeaseStoreError, match="Lease expiry did not advance"):
            await store.renew("run-nonmonotonic", "owner", timedelta(seconds=1), lease.generation)
        assert await store.inspect("run-nonmonotonic") == lease

        clock.value = lease.heartbeat_at
        with pytest.raises(RunLeaseStoreError, match="Lease heartbeat did not advance"):
            await store.renew("run-nonmonotonic", "owner", timedelta(seconds=10), lease.generation)
        assert await store.inspect("run-nonmonotonic") == lease

    asyncio.run(run())


@pytest.mark.parametrize("timeout", (float("nan"), float("inf"), float("-inf"), True))
def test_sqlite_store_rejects_nonfinite_timeout(tmp_path, timeout) -> None:
    with pytest.raises(ValueError, match="SQLite lease timeout must be positive"):
        SQLiteRunLeaseStore(tmp_path / "invalid-timeout.db", timeout=timeout)


def test_leased_chat_runtime_rejects_concurrent_executions() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.release_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                self.release_calls += 1

        class PausingModel(ChatModel):
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.allow_finish = asyncio.Event()

            async def stream(self, messages, tools=None):
                self.started.set()
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                await self.allow_finish.wait()
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

        store = Store()
        model = PausingModel()
        runtime = ChatRuntime(
            model,
            lease_store=store,
            lease_run_id="run-concurrent-runtime",
            lease_owner_id="owner-concurrent-runtime",
            lease_heartbeat_interval=timedelta(seconds=1),
        )
        first = asyncio.create_task(_collect_events(runtime.stream_user_message("first")))
        await model.started.wait()

        second = asyncio.create_task(_collect_events(runtime.stream_user_message("second")))
        with pytest.raises(RuntimeError, match="does not support concurrent executions"):
            await second

        model.allow_finish.set()
        events = await first
        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert store.release_calls == 1

        second_events = [event async for event in runtime.stream_user_message("second")]
        assert second_events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert store.release_calls == 2

    asyncio.run(run())


def test_old_terminal_stream_close_cannot_overwrite_new_execution_snapshot() -> None:
    async def run() -> None:
        class Store:
            def __init__(self) -> None:
                self.acquire_calls = 0

            async def acquire(self, run_id, owner_id, ttl):
                self.acquire_calls += 1
                now = datetime.now(timezone.utc)
                return RunLease(
                    run_id,
                    owner_id,
                    now,
                    now,
                    now + ttl,
                    generation=f"generation-{self.acquire_calls}",
                )

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
                return None

        async def next_terminal(stream):
            while True:
                event = await anext(stream)
                if event.type is RuntimeEventType.RUN_COMPLETED:
                    return event

        runtime = ChatRuntime(
            CountingTextModel(),
            lease_store=Store(),
            lease_run_id="run-snapshot-race",
            lease_owner_id="owner-snapshot-race",
            lease_heartbeat_interval=timedelta(seconds=1),
        )
        first_stream = runtime.stream_user_message("first")
        await next_terminal(first_stream)
        first_state = runtime._harness.state
        assert first_state is not None

        second_events = [
            event async for event in runtime.stream_user_message("second")
        ]
        assert second_events[-1].type is RuntimeEventType.RUN_COMPLETED
        second_state = runtime._harness.state
        assert second_state is not first_state

        await first_stream.aclose()
        assert runtime._harness.state is second_state

    asyncio.run(run())


def test_leased_runtime_can_be_reused_across_asyncio_run_calls() -> None:
    class Store:
        def __init__(self) -> None:
            self.acquire_calls = 0

        async def acquire(self, run_id, owner_id, ttl):
            self.acquire_calls += 1
            now = datetime.now(timezone.utc)
            return RunLease(
                run_id,
                owner_id,
                now,
                now,
                now + ttl,
                generation=f"generation-{self.acquire_calls}",
            )

        async def renew(self, run_id, owner_id, ttl, generation=None):
            raise AssertionError("heartbeat should not run")

        async def release(self, run_id, owner_id, generation=None):
            return None

    store = Store()
    runtime = ChatRuntime(
        CountingTextModel(),
        lease_store=store,
        lease_run_id="run-cross-loop",
        lease_owner_id="owner-cross-loop",
        lease_heartbeat_interval=timedelta(seconds=1),
    )

    async def collect_once(content):
        return [event async for event in runtime.stream_user_message(content)]

    first = asyncio.run(collect_once("first"))
    second = asyncio.run(collect_once("second"))

    assert first[-1].type is RuntimeEventType.RUN_COMPLETED
    assert second[-1].type is RuntimeEventType.RUN_COMPLETED
    assert store.acquire_calls == 2


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


def test_coordinator_rejects_empty_owner_id() -> None:
    with pytest.raises(ValueError, match="owner_id must be a non-empty string"):
        RunLeaseCoordinator(object(), run_id="run-empty-owner", owner_id="")


def test_coordinator_rejects_zero_heartbeat_interval() -> None:
    with pytest.raises(ValueError, match="Heartbeat interval must be positive"):
        RunLeaseCoordinator(
            object(),
            run_id="run-zero-heartbeat",
            heartbeat_interval=timedelta(0),
        )


class CountingTextModel(ChatModel):
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, messages, tools=None):
        self.calls += 1
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
        yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")


def test_chat_runtime_preserves_prior_positional_lease_clock_slot() -> None:
    async def run() -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        clock_calls = 0

        def lease_clock() -> datetime:
            nonlocal clock_calls
            clock_calls += 1
            return now

        class PositionalLeaseStore:
            def __init__(self) -> None:
                self.released = False

            async def acquire(self, run_id, owner_id, ttl):
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
                self.released = True

        store = PositionalLeaseStore()
        runtime = ChatRuntime(
            CountingTextModel(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            store,
            "run-positional-clock",
            "owner-positional-clock",
            timedelta(seconds=30),
            timedelta(seconds=10),
            lease_clock,
        )

        events = [event async for event in runtime.stream_user_message("hello")]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert store.released is True
        assert clock_calls > 0

        next_positional_runtime = ChatRuntime(
            CountingTextModel(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            PositionalLeaseStore(),
            "run-positional-release-timeout",
            "owner-positional-release-timeout",
            timedelta(seconds=30),
            timedelta(seconds=10),
            lease_clock,
            timedelta(milliseconds=25),
        )
        assert next_positional_runtime._lease_release_timeout == timedelta(milliseconds=25)

    asyncio.run(run())


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


def test_sqlite_terminal_release_timeout_cannot_delete_after_lock_opens(tmp_path) -> None:
    async def run() -> None:
        path = tmp_path / "leases.db"
        store = PreacquiredConnectionObservedLeaseStore(path, timeout=1)
        await store.initialize()
        seeded = await store.seed(
            "run-release-lock-timeout",
            "owner-release-lock-timeout",
            timedelta(seconds=30),
        )
        holder = sqlite3.connect(path, isolation_level=None)
        stream = None
        holder_open = True
        try:
            holder.execute("BEGIN IMMEDIATE")
            store.begin_immediate_attempted.clear()
            store.release_connection_closed.clear()
            runtime = ChatRuntime(
                CountingTextModel(),
                lease_store=store,
                lease_run_id=seeded.run_id,
                lease_owner_id=seeded.owner_id,
                lease_release_timeout=timedelta(milliseconds=20),
            )
            started_at = asyncio.get_running_loop().time()
            stream = runtime.stream_user_message("hello")
            async for event in stream:
                if event.type is not RuntimeEventType.RUN_COMPLETED:
                    continue

                assert asyncio.get_running_loop().time() - started_at < 0.2
                assert store.begin_immediate_attempted.is_set()
                assert holder.in_transaction
                assert event.metadata["lease_release_failed"] is True

                # Do not yield after receiving the terminal event. The
                # coordinator must have signalled the SQLite worker before
                # exposing the fixed outcome to this consumer.
                holder.rollback()
                holder.close()
                holder_open = False
                assert store.release_connection_closed.wait(1)
                with closing(sqlite3.connect(path)) as db:
                    row = db.execute(
                        "SELECT owner_id FROM run_leases WHERE run_id = ?",
                        (seeded.run_id,),
                    ).fetchone()
                assert row == (seeded.owner_id,)
                break
            else:
                pytest.fail("Runtime stream did not produce a terminal event")

            await stream.aclose()
            for _ in range(100):
                if not leases_module._BACKGROUND_RELEASE_TASKS:
                    break
                await asyncio.sleep(0.001)
            assert leases_module._BACKGROUND_RELEASE_TASKS == set()
            assert store._release_cancellations == {}

            stored = await store.inspect(seeded.run_id)
            assert stored is not None
            assert stored.owner_id == seeded.owner_id
            assert stored == seeded
        finally:
            if holder_open:
                holder.rollback()
                holder.close()
            if stream is not None:
                with suppress(BaseException):
                    await stream.aclose()
            stored = await store.inspect(seeded.run_id)
            if stored is not None:
                await store.release(stored.run_id, stored.owner_id, stored.generation)
            await store.close()

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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat persistence failed")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
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
                self.release_task_ref = None

            async def acquire(self, run_id, owner_id, ttl):
                now = datetime.now(timezone.utc)
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
                self.release_started.set()
                task = asyncio.current_task()
                assert task is not None
                self.release_task_ref = weakref.ref(task)
                try:
                    await self.never_release.wait()
                except asyncio.CancelledError:
                    self.cancellation_suppressed.set()
                    await self.allow_finish.wait()
                    raise RunLeaseStoreError("eventual release failure")

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
        release_task_ref = store.release_task_ref
        assert release_task_ref is not None
        assert release_task_ref() is not None
        assert not release_task_ref().done()
        assert release_task_ref() in leases_module._BACKGROUND_RELEASE_TASKS
        del events
        del runtime
        gc.collect()
        assert release_task_ref() is not None
        assert not release_task_ref().done()
        assert release_task_ref() in leases_module._BACKGROUND_RELEASE_TASKS
        store.allow_finish.set()
        for _ in range(10):
            await asyncio.sleep(0)
            if not leases_module._BACKGROUND_RELEASE_TASKS:
                break
        assert leases_module._BACKGROUND_RELEASE_TASKS == set()

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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                raise AssertionError("heartbeat should not run")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
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

        async def cancel_repeatedly() -> None:
            for _ in range(40):
                task.cancel()
                await asyncio.sleep(0.005)

        cancellation_storm = asyncio.create_task(cancel_repeatedly())
        started_at = asyncio.get_running_loop().time()
        try:
            events = await asyncio.wait_for(asyncio.shield(task), timeout=0.15)
            assert asyncio.get_running_loop().time() - started_at < 0.15
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await self.checkpoint_store.terminal_started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during terminal checkpoint")

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                self.renew_calls += 1
                return await super().renew(run_id, owner_id, ttl, generation)

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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await policy.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during policy")

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await executor.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during tool")

            async def release(self, run_id, owner_id, generation=None):
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await model.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed during model")

            async def release(self, run_id, owner_id, generation=None):
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
            async def renew(self, run_id, owner_id, ttl, generation=None):
                await executor.started.wait()
                raise RunLeaseStoreError("owner heartbeat failed")

            async def release(self, run_id, owner_id, generation=None):
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
        await new_store.release("run-takeover-inflight", "owner-new", new_lease.generation)
        await old_store.close()
        await new_store.close()

    asyncio.run(run())


def test_lease_expiry_during_policy_blocks_tool_execution() -> None:
    async def run() -> None:
        class StableStore:
            async def acquire(self, run_id, owner_id, ttl):
                now = clock()
                return RunLease(run_id, owner_id, now, now, now + ttl)

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await asyncio.Event().wait()

            async def inspect(self, run_id):
                return None

            async def release(self, run_id, owner_id, generation=None):
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
        guard.begin_execution()
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
    guard.begin_execution()
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

            async def renew(self, run_id, owner_id, ttl, generation=None):
                await executor.started.wait()
                self.renew_failed.set()
                raise RunLeaseStoreError("heartbeat failed while tool was running")

            async def release(self, run_id, owner_id, generation=None):
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


def test_cli_iterator_cleanup_failure_returns_nonzero_without_rewriting_completed_status(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class FakeHarness:
        def __init__(self) -> None:
            self.state = type("State", (), {"status": HarnessStatus.COMPLETED})()

    class FakeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            self._harness = FakeHarness()

        async def stream_user_message(self, content, **kwargs):
            yield RuntimeEvent(
                RuntimeEventType.RUN_COMPLETED,
                metadata={"lease_iterator_cleanup_failed": True},
            )

    monkeypatch.setattr(cli, "ChatRuntime", FakeRuntime)
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
    assert event_types[-2:] == ["lease_iterator_cleanup_failed", "run_completed"]


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

        async def renew(self, run_id, owner_id, ttl, generation=None):
            raise AssertionError("heartbeat should not run")

        async def release(self, run_id, owner_id, generation=None):
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


def test_cli_release_timeout_audits_cleanup_without_rewriting_completed_outcome(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class HangingReleaseStore:
        def __init__(self) -> None:
            self.release_started = asyncio.Event()
            self.release_cancelled = asyncio.Event()
            self.never_release = asyncio.Event()

        async def initialize(self):
            return None

        async def close(self):
            return None

        async def acquire(self, run_id, owner_id, ttl):
            now = datetime.now(timezone.utc)
            return RunLease(run_id, owner_id, now, now, now + ttl)

        async def renew(self, run_id, owner_id, ttl, generation=None):
            raise AssertionError("heartbeat should not run")

        async def release(self, run_id, owner_id, generation=None):
            self.release_started.set()
            try:
                await self.never_release.wait()
            except asyncio.CancelledError:
                self.release_cancelled.set()
                raise

    store = HangingReleaseStore()
    original_runtime = ChatRuntime

    def runtime_factory(*args, **kwargs):
        kwargs["lease_release_timeout"] = timedelta(milliseconds=30)
        return original_runtime(*args, **kwargs)

    monkeypatch.setattr(cli, "SQLiteRunLeaseStore", lambda path: store)
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: CountingTextModel())
    monkeypatch.setattr(cli, "ChatRuntime", runtime_factory)
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
    assert store.release_started.is_set()
    assert store.release_cancelled.is_set()
