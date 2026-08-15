"""Durable, owner-scoped execution leases for NexusMind runs."""

from __future__ import annotations

import asyncio
import math
import sqlite3
import sys
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType


_BACKGROUND_RELEASE_TASKS: set[asyncio.Task[None]] = set()


def _retain_background_release_task(task: asyncio.Task[None]) -> None:
    _BACKGROUND_RELEASE_TASKS.add(task)
    task.add_done_callback(_consume_background_release_task)


def _consume_background_release_task(task: asyncio.Task[None]) -> None:
    _BACKGROUND_RELEASE_TASKS.discard(task)
    try:
        task.result()
    except BaseException:
        pass


class RunLeaseError(RuntimeError):
    """Base class for controlled lease failures."""


class RunLeaseUnavailable(RunLeaseError):
    """The run is currently owned by another live owner."""


class RunLeaseOwnershipLost(RunLeaseError):
    """The caller can no longer prove that it owns the run."""


class RunLeaseStoreError(RunLeaseError):
    """Lease persistence failed or produced an ambiguous result."""


class RunLeaseReleaseError(RunLeaseStoreError):
    """Execution ended, but releasing its lease could not be proven."""


class _SQLiteReleaseAttempt(Enum):
    RELEASED = auto()
    BUSY = auto()
    CANCELLED = auto()


class _SQLiteLeaseAttempt(Enum):
    BUSY = auto()
    CANCELLED = auto()


@dataclass(frozen=True, slots=True)
class RunLease:
    run_id: str
    owner_id: str
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    generation: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        _validate_identity(self.run_id, "run_id")
        _validate_identity(self.owner_id, "owner_id")
        _validate_identity(self.generation, "generation")
        for value, name in (
            (self.acquired_at, "acquired_at"),
            (self.heartbeat_at, "heartbeat_at"),
            (self.expires_at, "expires_at"),
        ):
            _require_utc(value, name)
        if self.heartbeat_at < self.acquired_at:
            raise ValueError("Run lease heartbeat cannot precede acquisition")
        if self.expires_at <= self.heartbeat_at:
            raise ValueError("Run lease expiry must follow its heartbeat")

    def is_expired(self, at: datetime | None = None) -> bool:
        instant = at or datetime.now(timezone.utc)
        _require_utc(instant, "at")
        return self.expires_at <= instant


class RunLeaseStore(Protocol):
    async def acquire(self, run_id: str, owner_id: str, ttl: timedelta) -> RunLease:
        ...

    async def renew(
        self, run_id: str, owner_id: str, ttl: timedelta, generation: str
    ) -> RunLease:
        ...

    async def inspect(self, run_id: str) -> RunLease | None:
        ...

    async def release(
        self, run_id: str, owner_id: str, generation: str
    ) -> None:
        ...


class ExecutionOwnershipGuard(Protocol):
    def assert_owned(self) -> None:
        ...


class RunLeaseOwnershipGuard:
    """Shared in-process proof consulted immediately before side effects."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lease: RunLease | None = None
        self._ownership_error: RunLeaseError | None = None
        self._execution_uncertain = False
        self._renewal_event = asyncio.Event()

    @property
    def lease(self) -> RunLease | None:
        return self._lease

    @property
    def ownership_error(self) -> RunLeaseError | None:
        return self._ownership_error

    def prove(self, lease: RunLease, *, initial: bool = False) -> None:
        if type(lease) is not RunLease:
            raise RunLeaseStoreError("Run lease store returned an invalid lease")
        if self._ownership_error is not None and not initial:
            raise self._ownership_error
        if not initial and self._lease is not None and lease.generation != self._lease.generation:
            error = RunLeaseOwnershipLost("Run lease generation changed")
            self._ownership_error = error
            raise error
        now = self._now()
        if lease.is_expired(now):
            raise RunLeaseOwnershipLost("Run lease store returned an expired lease")
        self._lease = lease
        self._ownership_error = None
        if not initial:
            self._renewal_event.set()

    def fail(self, error: RunLeaseError) -> None:
        if not isinstance(error, RunLeaseError):
            raise TypeError("Ownership guard failure must be a RunLeaseError")
        self._ownership_error = error

    def clear(self, generation: str) -> None:
        if self._lease is not None and self._lease.generation == generation:
            self._lease = None

    def begin_execution(self) -> None:
        """Reset per-execution proof state before acquiring a new lease."""
        self._lease = None
        self._ownership_error = None
        self._execution_uncertain = False
        self._renewal_event = asyncio.Event()

    @property
    def execution_uncertain(self) -> bool:
        return self._execution_uncertain

    def mark_execution_uncertain(self) -> None:
        # Cancellation is not proof that an external side effect stopped.
        self._execution_uncertain = True

    def consume_renewal_signal(self) -> None:
        self._renewal_event.clear()

    async def wait_for_renewal(self) -> None:
        await self._renewal_event.wait()

    def seconds_until_expiry(self) -> float:
        self.assert_owned()
        assert self._lease is not None
        return max(0.0, (self._lease.expires_at - self._now()).total_seconds())

    def assert_owned(self) -> None:
        if self._ownership_error is not None:
            raise self._ownership_error
        if self._lease is None:
            raise RunLeaseOwnershipLost("Run lease has not been acquired")
        if self._lease.is_expired(self._now()):
            error = RunLeaseOwnershipLost("Run lease expired before execution could continue")
            self._ownership_error = error
            raise error

    def _now(self) -> datetime:
        try:
            now = self._clock()
            _require_utc(now, "clock result")
            return now
        except RunLeaseError:
            raise
        except Exception as exc:
            error = RunLeaseStoreError("Run lease clock failed")
            self._ownership_error = error
            raise error from exc


class RunLeaseCoordinator:
    """Own and heartbeat a run while fail-closed gating an async event stream.

    The coordinator sits outside the Harness state machine. It checks the
    locally proven lease before advancing the wrapped stream, so a lost lease
    cannot advance to another model or tool operation.
    """

    def __init__(
        self,
        store: RunLeaseStore,
        *,
        run_id: str,
        owner_id: str | None = None,
        ttl: timedelta = timedelta(seconds=30),
        heartbeat_interval: timedelta | None = None,
        lease_release_timeout: timedelta = timedelta(seconds=10),
        clock: Callable[[], datetime] | None = None,
        guard: RunLeaseOwnershipGuard | None = None,
    ) -> None:
        _validate_identity(run_id, "run_id")
        resolved_owner_id = uuid4().hex if owner_id is None else owner_id
        _validate_identity(resolved_owner_id, "owner_id")
        if type(ttl) is not timedelta or ttl <= timedelta(0):
            raise ValueError("Run lease ttl must be a positive timedelta")
        interval = ttl / 3 if heartbeat_interval is None else heartbeat_interval
        if type(interval) is not timedelta or interval <= timedelta(0) or interval >= ttl:
            raise ValueError("Heartbeat interval must be positive and shorter than the lease ttl")
        if type(lease_release_timeout) is not timedelta or lease_release_timeout <= timedelta(0):
            raise ValueError("Lease release timeout must be a positive timedelta")
        self._store = store
        self.run_id = run_id
        self.owner_id = resolved_owner_id
        self.ttl = ttl
        self.heartbeat_interval = interval
        self.lease_release_timeout = lease_release_timeout
        self._guard = guard or RunLeaseOwnershipGuard(clock=clock)
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._heartbeat_active = False
        self._stream_started = False
        self.release_error: RunLeaseError | None = None
        self.ownership_lost_after_progress = False
        self._terminal_cleanup_done = False
        self._generation: str | None = None

    @property
    def lease(self) -> RunLease | None:
        return self._guard.lease

    async def stream(self, events: AsyncIterator[RuntimeEvent]) -> AsyncIterator[RuntimeEvent]:
        if self._stream_started:
            raise RuntimeError("RunLeaseCoordinator can only be streamed once")
        self._stream_started = True
        self._guard.begin_execution()
        acquired = False
        iterator = events.__aiter__()
        try:
            try:
                lease = await self._acquire_store()
            except RunLeaseError:
                raise
            except Exception as exc:
                raise RunLeaseStoreError("Run lease acquisition failed") from exc
            lease = self._validate_owned_lease(lease)
            self._generation = lease.generation
            acquired = True
            self._guard.prove(lease, initial=True)
            self._heartbeat_active = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            while True:
                self.assert_owned()
                try:
                    event = await self._next_event(iterator)
                except StopAsyncIteration:
                    break
                if event.type in {RuntimeEventType.RUN_COMPLETED, RuntimeEventType.RUN_FAILED}:
                    event = await self._finalize_terminal(event, iterator)
                    self._terminal_cleanup_done = True
                    yield event
                    break
                try:
                    self.assert_owned()
                except RunLeaseError:
                    self.ownership_lost_after_progress = True
                    raise
                yield event
        finally:
            active_error = sys.exc_info()[1]
            await self._stop_heartbeat()
            close_error: BaseException | None = None
            if not self._terminal_cleanup_done:
                close_error = await self._close_iterator_barrier(iterator)
            if acquired and not self._terminal_cleanup_done and not self._guard.execution_uncertain:
                try:
                    await self._release_with_deadline(propagate_cancellation=True)
                except RunLeaseError as exc:
                    self.release_error = exc
                    if active_error is None:
                        raise
                if self.release_error is not None and active_error is None:
                    raise self.release_error
            if close_error is not None and active_error is None and self.release_error is None:
                raise RunLeaseStoreError("Run lease iterator cleanup failed") from close_error

    @staticmethod
    async def _close_iterator(iterator: AsyncIterator[RuntimeEvent]) -> None:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()

    async def _next_event(self, iterator: AsyncIterator[RuntimeEvent]) -> RuntimeEvent:
        next_task = asyncio.create_task(anext(iterator))
        heartbeat_task = self._heartbeat_task
        if heartbeat_task is None:
            next_task.cancel()
            await self._drain(next_task)
            raise RunLeaseOwnershipLost("Run lease heartbeat is not running")
        self._guard.consume_renewal_signal()
        expiry_task = asyncio.create_task(self._wait_for_expiry())
        renewal_task = asyncio.create_task(self._guard.wait_for_renewal())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {next_task, heartbeat_task, expiry_task, renewal_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if renewal_task in done:
                    renewal_task.result()
                if renewal_task in done and heartbeat_task not in done:
                    if next_task in done:
                        break
                    expiry_task.cancel()
                    await self._drain(expiry_task)
                    self._guard.consume_renewal_signal()
                    expiry_task = asyncio.create_task(self._wait_for_expiry())
                    renewal_task = asyncio.create_task(self._guard.wait_for_renewal())
                    continue
                break
        except asyncio.CancelledError:
            next_task.cancel()
            expiry_task.cancel()
            renewal_task.cancel()
            await self._drain(next_task)
            await self._drain(expiry_task)
            await self._drain(renewal_task)
            raise
        finally:
            if not expiry_task.done():
                expiry_task.cancel()
            if not renewal_task.done():
                renewal_task.cancel()
            await self._drain(expiry_task)
            await self._drain(renewal_task)

        if next_task in done:
            try:
                event = next_task.result()
                if expiry_task in done and event.type not in {
                    RuntimeEventType.RUN_COMPLETED,
                    RuntimeEventType.RUN_FAILED,
                }:
                    ownership_error = RunLeaseOwnershipLost("Run lease expired before progress")
                    self._guard.fail(ownership_error)
                    self.ownership_lost_after_progress = True
                    raise ownership_error
                return event
            except BaseException:
                if heartbeat_task in done and self._guard.ownership_error is not None:
                    raise self._guard.ownership_error
                raise

        if expiry_task in done and heartbeat_task not in done:
            try:
                self._guard.assert_owned()
            except RunLeaseError as exc:
                ownership_error = exc
            else:
                ownership_error = RunLeaseOwnershipLost("Run lease expiry watchdog fired")
                self._guard.fail(ownership_error)
        else:
            ownership_error = self._guard.ownership_error
        if ownership_error is None:
            ownership_error = RunLeaseStoreError("Run lease heartbeat stopped unexpectedly")
            self._guard.fail(ownership_error)
        next_task.cancel()
        try:
            event = await next_task
        except BaseException:
            self.ownership_lost_after_progress = True
            raise ownership_error
        if event.type in {RuntimeEventType.RUN_COMPLETED, RuntimeEventType.RUN_FAILED}:
            return event
        self.ownership_lost_after_progress = True
        raise ownership_error

    async def _wait_for_expiry(self) -> None:
        while True:
            delay = self._guard.seconds_until_expiry()
            if delay <= 0:
                return
            # Short quanta keep injected clocks and real wall-clock expiry
            # observable even while a renew call is blocked.
            await asyncio.sleep(min(delay, 0.05))

    async def _finalize_terminal(
        self, event: RuntimeEvent, iterator: AsyncIterator[RuntimeEvent]
    ) -> RuntimeEvent:
        ownership_lost = False
        try:
            self._guard.assert_owned()
        except RunLeaseError:
            ownership_lost = True
        await self._stop_heartbeat()
        close_error = await self._close_iterator_barrier(iterator)
        try:
            self._guard.assert_owned()
        except RunLeaseError:
            ownership_lost = True
        ownership_lost = ownership_lost or self._guard.ownership_error is not None
        await self._release_with_deadline(propagate_cancellation=False)
        metadata = dict(event.metadata)
        if ownership_lost:
            metadata["lease_ownership_lost_after_terminal"] = True
        if self.release_error is not None:
            metadata["lease_release_failed"] = True
        if close_error is not None:
            metadata["lease_iterator_cleanup_failed"] = True
        return replace(event, metadata=metadata)

    async def _close_iterator_barrier(
        self, iterator: AsyncIterator[RuntimeEvent]
    ) -> BaseException | None:
        close_task = asyncio.create_task(self._close_iterator(iterator))
        deadline = asyncio.get_running_loop().time() + self.lease_release_timeout.total_seconds()
        while not close_task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                close_task.cancel()
                _retain_background_release_task(close_task)
                return RunLeaseStoreError("Run lease iterator close timed out")
            try:
                done, _ = await asyncio.wait({close_task}, timeout=remaining)
            except asyncio.CancelledError:
                continue
            if not done:
                close_task.cancel()
                _retain_background_release_task(close_task)
                return RunLeaseStoreError("Run lease iterator close timed out")
        try:
            close_task.result()
        except BaseException as exc:
            return exc
        return None

    async def _release_with_deadline(self, *, propagate_cancellation: bool) -> None:
        release_task = asyncio.create_task(self._release())
        deadline = asyncio.get_running_loop().time() + self.lease_release_timeout.total_seconds()
        cancelled = False
        while not release_task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait({release_task}, timeout=remaining)
            except asyncio.CancelledError:
                cancelled = True
                continue
        if not release_task.done():
            self.release_error = RunLeaseReleaseError("Run lease release timed out")
            self._cancel_store_release()
            release_task.cancel()
            if self._generation is not None:
                self._guard.clear(self._generation)
            _retain_background_release_task(release_task)
            if cancelled and propagate_cancellation:
                raise asyncio.CancelledError
            return
        try:
            release_task.result()
        except asyncio.CancelledError as exc:
            error = RunLeaseReleaseError("Run lease release cancelled")
            error.__cause__ = exc
            self.release_error = error
        except RunLeaseError as exc:
            self.release_error = exc
        if cancelled and propagate_cancellation:
            raise asyncio.CancelledError

    def _cancel_store_release(self) -> None:
        self._cancel_store_operation("release")

    def _cancel_store_operation(self, operation: str) -> None:
        try:
            cancel = getattr(self._store, f"cancel_{operation}", None)
            if callable(cancel):
                cancel(self.run_id, self.owner_id)
        except Exception:
            pass

    @staticmethod
    async def _drain(task: asyncio.Task[object]) -> None:
        try:
            await task
        except BaseException:
            pass

    def assert_owned(self) -> None:
        self._guard.assert_owned()

    async def _acquire_store(self) -> RunLease:
        task = asyncio.create_task(self._store.acquire(self.run_id, self.owner_id, self.ttl))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._cancel_store_operation("acquire")
            task.cancel()
            compensation_task = asyncio.create_task(self._compensate_late_acquire(task))
            deadline = asyncio.get_running_loop().time() + self.lease_release_timeout.total_seconds()
            while not compensation_task.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    _retain_background_release_task(compensation_task)
                    raise
                try:
                    done, _ = await asyncio.wait({compensation_task}, timeout=remaining)
                except asyncio.CancelledError:
                    continue
                if not done:
                    _retain_background_release_task(compensation_task)
                    raise
            raise

    async def _compensate_late_acquire(self, task: asyncio.Task[RunLease]) -> None:
        try:
            lease = await asyncio.shield(task)
            lease = self._validate_owned_lease(lease)
            await self._store.release(
                self.run_id,
                self.owner_id,
                generation=lease.generation,
            )
        except BaseException:
            return

    async def _heartbeat(self) -> None:
        delay = self.heartbeat_interval.total_seconds()
        while self._heartbeat_active:
            try:
                await asyncio.sleep(delay)
                renewed = await self._renew_store()
                if not self._heartbeat_active:
                    return
                self._guard.prove(self._validate_owned_lease(renewed))
            except asyncio.CancelledError:
                if self._heartbeat_active:
                    self._cancel_store_operation("renew")
                raise
            except RunLeaseError as exc:
                if self._heartbeat_active:
                    self._guard.fail(exc)
                return
            except Exception as exc:
                if self._heartbeat_active:
                    error = RunLeaseStoreError("Run lease heartbeat failed")
                    error.__cause__ = exc
                    self._guard.fail(error)
                return

    def _validate_owned_lease(self, lease: object) -> RunLease:
        if type(lease) is not RunLease:
            raise RunLeaseStoreError("Run lease store returned an invalid lease")
        if lease.run_id != self.run_id or lease.owner_id != self.owner_id:
            raise RunLeaseStoreError("Run lease store returned the wrong owner")
        return lease

    async def _stop_heartbeat(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        self._heartbeat_active = False
        if task is None:
            return
        self._cancel_store_operation("renew")
        task.cancel()
        deadline = asyncio.get_running_loop().time() + self.lease_release_timeout.total_seconds()
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                _retain_background_release_task(task)
                return
            try:
                done, _ = await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError:
                continue
            if not done:
                _retain_background_release_task(task)
                return
        try:
            task.result()
        except BaseException:
            pass

    async def _release(self) -> None:
        try:
            await self._release_store()
        except RunLeaseError as exc:
            raise RunLeaseReleaseError("Run lease release failed") from exc
        except Exception as exc:
            raise RunLeaseReleaseError("Run lease release failed") from exc
        finally:
            if self._generation is not None:
                self._guard.clear(self._generation)

    async def _renew_store(self) -> RunLease:
        return await self._store.renew(
            self.run_id,
            self.owner_id,
            self.ttl,
            generation=self._generation,
        )

    async def _release_store(self) -> None:
        await self._store.release(
            self.run_id,
            self.owner_id,
            generation=self._generation,
        )


class SQLiteRunLeaseStore:
    """SQLite lease store with transactional acquisition and takeover."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._path = str(db_path)
        if self._path in {"", ":memory:"}:
            raise ValueError("SQLiteRunLeaseStore requires a persistent database path")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("SQLite lease timeout must be positive")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timeout = timeout
        self._initialized = False
        self._closed = False
        self._release_cancellations_lock = threading.Lock()
        self._release_cancellations: dict[
            tuple[str, str], set[threading.Event]
        ] = {}
        self._lease_cancellations: dict[
            tuple[str, str, str], set[threading.Event]
        ] = {}

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize)
        self._initialized = True
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    async def acquire(self, run_id: str, owner_id: str, ttl: timedelta) -> RunLease:
        self._require_ready()
        _validate_request(run_id, owner_id, ttl)
        return await self._run_cancellable_lease_operation("acquire", run_id, owner_id, ttl)

    async def renew(
        self, run_id: str, owner_id: str, ttl: timedelta, generation: str
    ) -> RunLease:
        self._require_ready()
        _validate_request(run_id, owner_id, ttl)
        _validate_identity(generation, "generation")
        return await self._run_cancellable_lease_operation("renew", run_id, owner_id, ttl, generation)

    async def _run_cancellable_lease_operation(
        self, operation: str, run_id: str, owner_id: str, ttl: timedelta,
        generation: str | None = None,
    ) -> RunLease:
        cancellation = threading.Event()
        self._register_lease_cancellation(operation, run_id, owner_id, cancellation)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise RunLeaseStoreError(f"SQLite lease {operation} failed: database is locked")
                attempt = loop.run_in_executor(
                    None,
                    getattr(self, f"_{operation}"),
                    run_id,
                    owner_id,
                    ttl,
                    generation,
                    cancellation,
                    min(0.05, remaining),
                )
                try:
                    result = await asyncio.shield(attempt)
                except asyncio.CancelledError as cancelled:
                    cancellation.set()
                    try:
                        result = await self._drain_lease_attempt(attempt, cancellation)
                    except BaseException:
                        raise cancelled
                    if operation == "acquire" and isinstance(result, RunLease):
                        await self._compensating_release(result)
                    raise cancelled
                if isinstance(result, RunLease):
                    return result
                if result is _SQLiteLeaseAttempt.CANCELLED:
                    raise asyncio.CancelledError()
                if loop.time() >= deadline:
                    raise RunLeaseStoreError(f"SQLite lease {operation} failed: database is locked")
                await asyncio.sleep(0)
        finally:
            self._unregister_lease_cancellation(operation, run_id, owner_id, cancellation)

    async def _compensating_release(self, lease: RunLease) -> None:
        """Remove an acquire that committed after its caller was cancelled."""
        release_task = asyncio.create_task(
            self.release(lease.run_id, lease.owner_id, lease.generation)
        )
        deadline = asyncio.get_running_loop().time() + self._timeout
        while not release_task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                _retain_background_release_task(release_task)
                return
            try:
                await asyncio.wait_for(asyncio.shield(release_task), timeout=remaining)
            except asyncio.TimeoutError:
                _retain_background_release_task(release_task)
                return
            except asyncio.CancelledError:
                continue
        try:
            release_task.result()
        except BaseException:
            # The original cancellation remains authoritative. A failed
            # compensation is still bounded by SQLite's own timeout.
            pass

    @staticmethod
    async def _drain_lease_attempt(
        attempt: asyncio.Future[RunLease | _SQLiteLeaseAttempt],
        cancellation: threading.Event,
    ) -> RunLease | _SQLiteLeaseAttempt:
        while not attempt.done():
            try:
                await asyncio.shield(attempt)
            except asyncio.CancelledError:
                cancellation.set()
        return attempt.result()

    def cancel_acquire(self, run_id: str, owner_id: str) -> None:
        self._cancel_lease_operation("acquire", run_id, owner_id)

    def cancel_renew(self, run_id: str, owner_id: str) -> None:
        self._cancel_lease_operation("renew", run_id, owner_id)

    def _cancel_lease_operation(self, operation: str, run_id: str, owner_id: str) -> None:
        _validate_identity(run_id, "run_id")
        _validate_identity(owner_id, "owner_id")
        with self._release_cancellations_lock:
            cancellations = tuple(self._lease_cancellations.get((operation, run_id, owner_id), ()))
        for cancellation in cancellations:
            cancellation.set()

    def _register_lease_cancellation(self, operation, run_id, owner_id, cancellation):
        with self._release_cancellations_lock:
            self._lease_cancellations.setdefault((operation, run_id, owner_id), set()).add(cancellation)

    def _unregister_lease_cancellation(self, operation, run_id, owner_id, cancellation):
        with self._release_cancellations_lock:
            key = (operation, run_id, owner_id)
            cancellations = self._lease_cancellations.get(key)
            if cancellations is None:
                return
            cancellations.discard(cancellation)
            if not cancellations:
                del self._lease_cancellations[key]

    async def inspect(self, run_id: str) -> RunLease | None:
        self._require_ready()
        _validate_identity(run_id, "run_id")
        return await asyncio.to_thread(self._inspect, run_id)

    async def release(self, run_id: str, owner_id: str, generation: str) -> None:
        self._require_ready()
        _validate_identity(run_id, "run_id")
        _validate_identity(owner_id, "owner_id")
        _validate_identity(generation, "generation")
        cancellation = threading.Event()
        self._register_release_cancellation(run_id, owner_id, cancellation)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise RunLeaseStoreError("SQLite lease release failed: database is locked")
                attempt_timeout = min(0.05, remaining)
                attempt = loop.run_in_executor(
                    None,
                    self._release_attempt,
                    run_id,
                    owner_id,
                    generation,
                    cancellation,
                    attempt_timeout,
                )
                try:
                    result = await asyncio.shield(attempt)
                except asyncio.CancelledError as cancelled:
                    cancellation.set()
                    try:
                        result = await self._drain_release_attempt(attempt, cancellation)
                    except BaseException:
                        raise cancelled
                    if result is _SQLiteReleaseAttempt.RELEASED:
                        return
                    raise cancelled
                if result is _SQLiteReleaseAttempt.RELEASED:
                    return
                if result is _SQLiteReleaseAttempt.CANCELLED:
                    raise asyncio.CancelledError()
                if loop.time() >= deadline:
                    raise RunLeaseStoreError("SQLite lease release failed: database is locked")
                await asyncio.sleep(0)
        finally:
            self._unregister_release_cancellation(run_id, owner_id, cancellation)

    def cancel_release(self, run_id: str, owner_id: str) -> None:
        _validate_identity(run_id, "run_id")
        _validate_identity(owner_id, "owner_id")
        with self._release_cancellations_lock:
            cancellations = tuple(
                self._release_cancellations.get((run_id, owner_id), ())
            )
        for cancellation in cancellations:
            cancellation.set()

    def _register_release_cancellation(
        self,
        run_id: str,
        owner_id: str,
        cancellation: threading.Event,
    ) -> None:
        with self._release_cancellations_lock:
            self._release_cancellations.setdefault((run_id, owner_id), set()).add(
                cancellation
            )

    def _unregister_release_cancellation(
        self,
        run_id: str,
        owner_id: str,
        cancellation: threading.Event,
    ) -> None:
        key = (run_id, owner_id)
        with self._release_cancellations_lock:
            cancellations = self._release_cancellations.get(key)
            if cancellations is None:
                return
            cancellations.discard(cancellation)
            if not cancellations:
                del self._release_cancellations[key]

    @staticmethod
    async def _drain_release_attempt(
        attempt: asyncio.Future[_SQLiteReleaseAttempt],
        cancellation: threading.Event,
    ) -> _SQLiteReleaseAttempt:
        while not attempt.done():
            try:
                await asyncio.shield(attempt)
            except asyncio.CancelledError:
                cancellation.set()
        return attempt.result()

    def _connect(self, timeout: float | None = None) -> sqlite3.Connection:
        resolved_timeout = self._timeout if timeout is None else timeout
        db = sqlite3.connect(self._path, timeout=resolved_timeout, isolation_level=None)
        db.execute(f"PRAGMA busy_timeout={max(1, int(resolved_timeout * 1000))}")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_leases (
                        run_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        heartbeat_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        generation TEXT NOT NULL
                    )
                    """
                )
                self._validate_schema(db)
                db.commit()
        except RunLeaseStoreError:
            raise
        except sqlite3.Error as exc:
            raise RunLeaseStoreError("SQLite lease initialization failed") from exc

    @staticmethod
    def _validate_schema(db: sqlite3.Connection) -> None:
        expected = {
            "run_id": ("TEXT", 0, 1),
            "owner_id": ("TEXT", 1, 0),
            "acquired_at": ("TEXT", 1, 0),
            "heartbeat_at": ("TEXT", 1, 0),
            "expires_at": ("TEXT", 1, 0),
            "generation": ("TEXT", 1, 0),
        }
        actual = {
            row[1]: (row[2].upper(), row[3], row[5])
            for row in db.execute("PRAGMA table_info(run_leases)")
        }
        legacy = dict(expected)
        legacy.pop("generation")
        if actual == legacy:
            db.execute("ALTER TABLE run_leases ADD COLUMN generation TEXT")
            rows = db.execute("SELECT run_id FROM run_leases").fetchall()
            for (run_id,) in rows:
                db.execute(
                    "UPDATE run_leases SET generation = ? WHERE run_id = ?",
                    (uuid4().hex, run_id),
                )
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS run_leases_generation_idx ON run_leases(run_id, generation)")
            actual = {
                row[1]: (row[2].upper(), row[3], row[5])
                for row in db.execute("PRAGMA table_info(run_leases)")
            }
        if actual != expected and not (
            actual.get("generation") == ("TEXT", 0, 0)
            and {key: value for key, value in actual.items() if key != "generation"}
            == {key: value for key, value in expected.items() if key != "generation"}
        ):
            raise RunLeaseStoreError("Lease database schema is incomplete or incompatible")

    def _require_ready(self) -> None:
        if not self._initialized or self._closed:
            raise RunLeaseStoreError("Run lease store is not initialized")

    def _now(self) -> datetime:
        try:
            now = self._clock()
            _require_utc(now, "clock result")
            return now
        except RunLeaseError:
            raise
        except Exception as exc:
            raise RunLeaseStoreError("Run lease clock failed") from exc

    def _acquire(
        self,
        run_id: str,
        owner_id: str,
        ttl: timedelta,
        generation: str | None = None,
        cancellation: threading.Event | None = None,
        timeout: float | None = None,
    ) -> RunLease:
        try:
            if cancellation is not None and cancellation.is_set():
                return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
            with closing(self._connect(timeout)) as db:
                if cancellation is not None and cancellation.is_set():
                    return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
                db.execute("BEGIN IMMEDIATE")
                if cancellation is not None and cancellation.is_set():
                    db.rollback()
                    return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
                now = self._now()
                expires_at = now + ttl
                row = db.execute(
                    "SELECT run_id, owner_id, acquired_at, heartbeat_at, expires_at, generation "
                    "FROM run_leases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    lease = RunLease(run_id, owner_id, now, now, expires_at)
                    db.execute(
                        "INSERT INTO run_leases VALUES (?, ?, ?, ?, ?, ?)",
                        _lease_values(lease),
                    )
                else:
                    current = _lease_from_row(row)
                    if not current.is_expired(now):
                        raise RunLeaseUnavailable("Run already has an active execution owner")
                    lease = RunLease(run_id, owner_id, now, now, expires_at)
                    cursor = db.execute(
                        "UPDATE run_leases SET owner_id = ?, acquired_at = ?, heartbeat_at = ?, expires_at = ?, generation = ? "
                        "WHERE run_id = ? AND owner_id = ? AND expires_at = ?",
                        (
                            lease.owner_id,
                            _encode_time(lease.acquired_at),
                            _encode_time(lease.heartbeat_at),
                            _encode_time(lease.expires_at),
                            lease.generation,
                            current.run_id,
                            current.owner_id,
                            _encode_time(current.expires_at),
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RunLeaseStoreError("Lease takeover result was ambiguous")
                if cancellation is not None and cancellation.is_set():
                    db.rollback()
                    return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
                db.commit()
                return lease
        except (RunLeaseUnavailable, RunLeaseStoreError):
            raise
        except sqlite3.Error as exc:
            if _sqlite_is_busy(exc):
                return _SQLiteLeaseAttempt.BUSY  # type: ignore[return-value]
            raise RunLeaseStoreError("SQLite lease acquisition failed") from exc

    def _renew(
        self,
        run_id: str,
        owner_id: str,
        ttl: timedelta,
        generation: str | None = None,
        cancellation: threading.Event | None = None,
        timeout: float | None = None,
    ) -> RunLease:
        try:
            if cancellation is not None and cancellation.is_set():
                return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
            with closing(self._connect(timeout)) as db:
                if cancellation is not None and cancellation.is_set():
                    return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
                db.execute("BEGIN IMMEDIATE")
                if cancellation is not None and cancellation.is_set():
                    db.rollback()
                    return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
                now = self._now()
                row = db.execute(
                    "SELECT run_id, owner_id, acquired_at, heartbeat_at, expires_at, generation "
                    "FROM run_leases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise RunLeaseOwnershipLost("Run lease no longer exists")
                current = _lease_from_row(row)
                if current.owner_id != owner_id:
                    raise RunLeaseOwnershipLost("Run lease is owned by another owner")
                if generation is not None and current.generation != generation:
                    raise RunLeaseOwnershipLost("Run lease generation changed")
                if now < current.heartbeat_at:
                    raise RunLeaseStoreError("Lease clock moved backwards during renewal")
                if now <= current.heartbeat_at:
                    raise RunLeaseStoreError("Lease heartbeat did not advance")
                if current.is_expired(now):
                    raise RunLeaseOwnershipLost("Run lease expired before renewal")
                expires_at = now + ttl
                if expires_at <= current.expires_at:
                    raise RunLeaseStoreError("Lease expiry did not advance")
                lease = RunLease(
                    run_id, owner_id, current.acquired_at, now, expires_at,
                    generation=current.generation,
                )
                cursor = db.execute(
                    "UPDATE run_leases SET heartbeat_at = ?, expires_at = ? "
                    "WHERE run_id = ? AND owner_id = ? AND generation = ? AND expires_at = ?",
                    (
                        _encode_time(lease.heartbeat_at),
                        _encode_time(lease.expires_at),
                        run_id,
                        owner_id,
                        current.generation,
                        _encode_time(current.expires_at),
                    ),
                )
                if cursor.rowcount != 1:
                    raise RunLeaseStoreError("Lease renewal result was ambiguous")
                if cancellation is not None and cancellation.is_set():
                    db.rollback()
                    return _SQLiteLeaseAttempt.CANCELLED  # type: ignore[return-value]
                db.commit()
                return lease
        except (RunLeaseOwnershipLost, RunLeaseStoreError):
            raise
        except sqlite3.Error as exc:
            if _sqlite_is_busy(exc):
                return _SQLiteLeaseAttempt.BUSY  # type: ignore[return-value]
            raise RunLeaseStoreError("SQLite lease renewal failed") from exc

    def _inspect(self, run_id: str) -> RunLease | None:
        try:
            with closing(self._connect()) as db:
                row = db.execute(
                    "SELECT run_id, owner_id, acquired_at, heartbeat_at, expires_at, generation "
                    "FROM run_leases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            return _lease_from_row(row) if row is not None else None
        except RunLeaseStoreError:
            raise
        except sqlite3.Error as exc:
            raise RunLeaseStoreError("SQLite lease inspection failed") from exc

    def _release_attempt(
        self,
        run_id: str,
        owner_id: str,
        generation: str | None,
        cancellation: threading.Event,
        timeout: float,
    ) -> _SQLiteReleaseAttempt:
        if cancellation.is_set():
            return _SQLiteReleaseAttempt.CANCELLED
        try:
            with closing(self._connect(timeout=timeout)) as db:
                if cancellation.is_set():
                    return _SQLiteReleaseAttempt.CANCELLED
                db.execute("BEGIN IMMEDIATE")
                if cancellation.is_set():
                    db.rollback()
                    return _SQLiteReleaseAttempt.CANCELLED
                row = db.execute(
                    "SELECT owner_id, generation FROM run_leases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise RunLeaseOwnershipLost("Run lease no longer exists")
                if row[0] != owner_id:
                    raise RunLeaseOwnershipLost("Run lease is owned by another owner")
                if generation is not None and row[1] != generation:
                    raise RunLeaseOwnershipLost("Run lease generation changed")
                # The legacy two-argument API remains accepted for existing
                # stores, while coordinator callers pass the generation and
                # get ABA-safe owner scoping.
                if cancellation.is_set():
                    db.rollback()
                    return _SQLiteReleaseAttempt.CANCELLED
                sql = "DELETE FROM run_leases WHERE run_id = ? AND owner_id = ?"
                parameters: tuple[object, ...] = (run_id, owner_id)
                if generation is not None:
                    sql += " AND generation = ?"
                    parameters += (generation,)
                cursor = db.execute(sql, parameters)
                if cursor.rowcount != 1:
                    raise RunLeaseStoreError("Lease release result was ambiguous")
                if cancellation.is_set():
                    db.rollback()
                    return _SQLiteReleaseAttempt.CANCELLED
                db.commit()
                return _SQLiteReleaseAttempt.RELEASED
        except (RunLeaseOwnershipLost, RunLeaseStoreError):
            raise
        except sqlite3.OperationalError as exc:
            if _sqlite_is_busy(exc):
                return _SQLiteReleaseAttempt.BUSY
            raise RunLeaseStoreError("SQLite lease release failed") from exc
        except sqlite3.Error as exc:
            raise RunLeaseStoreError("SQLite lease release failed") from exc


def _validate_request(run_id: str, owner_id: str, ttl: timedelta) -> None:
    _validate_identity(run_id, "run_id")
    _validate_identity(owner_id, "owner_id")
    if type(ttl) is not timedelta or ttl <= timedelta(0):
        raise ValueError("Run lease ttl must be a positive timedelta")


def _sqlite_is_busy(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    if type(code) is int and code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _validate_identity(value: str, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"Run lease {name} must be a non-empty string")


def _require_utc(value: datetime, name: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"Run lease {name} must be timezone-aware UTC")


def _encode_time(value: datetime) -> str:
    _require_utc(value, "timestamp")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _decode_time(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise RunLeaseStoreError("Lease database contains an invalid UTC timestamp")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
        _require_utc(result, "stored timestamp")
        return result
    except (TypeError, ValueError) as exc:
        raise RunLeaseStoreError("Lease database contains an invalid UTC timestamp") from exc


def _lease_values(lease: RunLease) -> tuple[str, str, str, str, str, str]:
    return (
        lease.run_id,
        lease.owner_id,
        _encode_time(lease.acquired_at),
        _encode_time(lease.heartbeat_at),
        _encode_time(lease.expires_at),
        lease.generation,
    )


def _lease_from_row(row: tuple[object, ...]) -> RunLease:
    if len(row) != 6 or type(row[0]) is not str or type(row[1]) is not str:
        raise RunLeaseStoreError("Lease database contains an invalid row")
    try:
        return RunLease(
            run_id=row[0],
            owner_id=row[1],
            acquired_at=_decode_time(row[2]),
            heartbeat_at=_decode_time(row[3]),
            expires_at=_decode_time(row[4]),
            generation=row[5],
        )
    except ValueError as exc:
        raise RunLeaseStoreError("Lease database contains invalid lease data") from exc
