"""Durable, owner-scoped execution leases for NexusMind runs."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import closing
from dataclasses import dataclass, replace
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


@dataclass(frozen=True, slots=True)
class RunLease:
    run_id: str
    owner_id: str
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        _validate_identity(self.run_id, "run_id")
        _validate_identity(self.owner_id, "owner_id")
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

    async def renew(self, run_id: str, owner_id: str, ttl: timedelta) -> RunLease:
        ...

    async def inspect(self, run_id: str) -> RunLease | None:
        ...

    async def release(self, run_id: str, owner_id: str) -> None:
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
        now = self._now()
        if lease.is_expired(now):
            raise RunLeaseOwnershipLost("Run lease store returned an expired lease")
        self._lease = lease
        self._ownership_error = None

    def fail(self, error: RunLeaseError) -> None:
        if not isinstance(error, RunLeaseError):
            raise TypeError("Ownership guard failure must be a RunLeaseError")
        self._ownership_error = error

    def clear(self) -> None:
        self._lease = None

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
        resolved_owner_id = owner_id or uuid4().hex
        _validate_identity(resolved_owner_id, "owner_id")
        if type(ttl) is not timedelta or ttl <= timedelta(0):
            raise ValueError("Run lease ttl must be a positive timedelta")
        interval = heartbeat_interval or ttl / 3
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
        self._stream_started = False
        self.release_error: RunLeaseError | None = None
        self.ownership_lost_after_progress = False
        self._terminal_cleanup_done = False

    @property
    def lease(self) -> RunLease | None:
        return self._guard.lease

    async def stream(self, events: AsyncIterator[RuntimeEvent]) -> AsyncIterator[RuntimeEvent]:
        if self._stream_started:
            raise RuntimeError("RunLeaseCoordinator can only be streamed once")
        self._stream_started = True
        acquired = False
        iterator = events.__aiter__()
        try:
            try:
                lease = await self._store.acquire(self.run_id, self.owner_id, self.ttl)
            except RunLeaseError:
                raise
            except Exception as exc:
                raise RunLeaseStoreError("Run lease acquisition failed") from exc
            self._guard.prove(self._validate_owned_lease(lease), initial=True)
            acquired = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            while True:
                self.assert_owned()
                try:
                    event = await self._next_event(iterator)
                except StopAsyncIteration:
                    break
                if event.type in {RuntimeEventType.RUN_COMPLETED, RuntimeEventType.RUN_FAILED}:
                    event = await self._finalize_terminal(event)
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
            try:
                await iterator.aclose()
            except (AttributeError, RuntimeError):
                pass
            if acquired and not self._terminal_cleanup_done:
                try:
                    await self._release()
                except RunLeaseError as exc:
                    self.release_error = exc
                    if active_error is None:
                        raise

    async def _next_event(self, iterator: AsyncIterator[RuntimeEvent]) -> RuntimeEvent:
        next_task = asyncio.create_task(anext(iterator))
        heartbeat_task = self._heartbeat_task
        if heartbeat_task is None:
            next_task.cancel()
            await self._drain(next_task)
            raise RunLeaseOwnershipLost("Run lease heartbeat is not running")
        try:
            done, _ = await asyncio.wait(
                {next_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            next_task.cancel()
            await self._drain(next_task)
            raise

        if next_task in done:
            try:
                return next_task.result()
            except BaseException:
                if heartbeat_task in done and self._guard.ownership_error is not None:
                    raise self._guard.ownership_error
                raise

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

    async def _finalize_terminal(self, event: RuntimeEvent) -> RuntimeEvent:
        ownership_lost = False
        try:
            self._guard.assert_owned()
        except RunLeaseError:
            ownership_lost = True
        await self._stop_heartbeat()
        ownership_lost = ownership_lost or self._guard.ownership_error is not None
        await self._release_terminal_barrier()
        metadata = dict(event.metadata)
        if ownership_lost:
            metadata["lease_ownership_lost_after_terminal"] = True
        if self.release_error is not None:
            metadata["lease_release_failed"] = True
        return replace(event, metadata=metadata)

    async def _release_terminal_barrier(self) -> None:
        release_task = asyncio.create_task(self._release())
        deadline = asyncio.get_running_loop().time() + self.lease_release_timeout.total_seconds()
        while not release_task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait({release_task}, timeout=remaining)
            except asyncio.CancelledError:
                # The execution outcome is already terminal. Resolve the
                # owner-scoped release before exposing that fixed outcome.
                continue
        if not release_task.done():
            self.release_error = RunLeaseReleaseError("Run lease release timed out")
            self._cancel_store_release()
            release_task.cancel()
            self._guard.clear()
            _retain_background_release_task(release_task)
            return
        try:
            release_task.result()
        except asyncio.CancelledError as exc:
            error = RunLeaseReleaseError("Run lease release cancelled")
            error.__cause__ = exc
            self.release_error = error
        except RunLeaseError as exc:
            self.release_error = exc

    def _cancel_store_release(self) -> None:
        try:
            cancel_release = getattr(self._store, "cancel_release", None)
            if callable(cancel_release):
                cancel_release(self.run_id, self.owner_id)
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

    async def _heartbeat(self) -> None:
        delay = self.heartbeat_interval.total_seconds()
        while True:
            try:
                await asyncio.sleep(delay)
                renewed = await self._store.renew(self.run_id, self.owner_id, self.ttl)
                self._guard.prove(self._validate_owned_lease(renewed))
            except asyncio.CancelledError:
                raise
            except RunLeaseError as exc:
                self._guard.fail(exc)
                return
            except Exception as exc:
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
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _release(self) -> None:
        try:
            await self._store.release(self.run_id, self.owner_id)
        except RunLeaseError as exc:
            raise RunLeaseReleaseError("Run lease release failed") from exc
        except Exception as exc:
            raise RunLeaseReleaseError("Run lease release failed") from exc
        finally:
            self._guard.clear()


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
        if timeout <= 0:
            raise ValueError("SQLite lease timeout must be positive")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timeout = timeout
        self._initialized = False
        self._closed = False
        self._release_cancellations_lock = threading.Lock()
        self._release_cancellations: dict[
            tuple[str, str], set[threading.Event]
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
        return await asyncio.to_thread(self._acquire, run_id, owner_id, ttl)

    async def renew(self, run_id: str, owner_id: str, ttl: timedelta) -> RunLease:
        self._require_ready()
        _validate_request(run_id, owner_id, ttl)
        return await asyncio.to_thread(self._renew, run_id, owner_id, ttl)

    async def inspect(self, run_id: str) -> RunLease | None:
        self._require_ready()
        _validate_identity(run_id, "run_id")
        return await asyncio.to_thread(self._inspect, run_id)

    async def release(self, run_id: str, owner_id: str) -> None:
        self._require_ready()
        _validate_identity(run_id, "run_id")
        _validate_identity(owner_id, "owner_id")
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
                        expires_at TEXT NOT NULL
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
        }
        actual = {
            row[1]: (row[2].upper(), row[3], row[5])
            for row in db.execute("PRAGMA table_info(run_leases)")
        }
        if actual != expected:
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
    ) -> RunLease:
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                now = self._now()
                expires_at = now + ttl
                row = db.execute(
                    "SELECT run_id, owner_id, acquired_at, heartbeat_at, expires_at "
                    "FROM run_leases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    lease = RunLease(run_id, owner_id, now, now, expires_at)
                    db.execute(
                        "INSERT INTO run_leases VALUES (?, ?, ?, ?, ?)",
                        _lease_values(lease),
                    )
                else:
                    current = _lease_from_row(row)
                    if not current.is_expired(now):
                        raise RunLeaseUnavailable("Run already has an active execution owner")
                    lease = RunLease(run_id, owner_id, now, now, expires_at)
                    cursor = db.execute(
                        "UPDATE run_leases SET owner_id = ?, acquired_at = ?, heartbeat_at = ?, expires_at = ? "
                        "WHERE run_id = ? AND owner_id = ? AND expires_at = ?",
                        (
                            lease.owner_id,
                            _encode_time(lease.acquired_at),
                            _encode_time(lease.heartbeat_at),
                            _encode_time(lease.expires_at),
                            current.run_id,
                            current.owner_id,
                            _encode_time(current.expires_at),
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RunLeaseStoreError("Lease takeover result was ambiguous")
                db.commit()
                return lease
        except (RunLeaseUnavailable, RunLeaseStoreError):
            raise
        except sqlite3.Error as exc:
            raise RunLeaseStoreError("SQLite lease acquisition failed") from exc

    def _renew(
        self,
        run_id: str,
        owner_id: str,
        ttl: timedelta,
    ) -> RunLease:
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                now = self._now()
                row = db.execute(
                    "SELECT run_id, owner_id, acquired_at, heartbeat_at, expires_at "
                    "FROM run_leases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise RunLeaseOwnershipLost("Run lease no longer exists")
                current = _lease_from_row(row)
                if current.owner_id != owner_id:
                    raise RunLeaseOwnershipLost("Run lease is owned by another owner")
                if current.is_expired(now):
                    raise RunLeaseOwnershipLost("Run lease expired before renewal")
                lease = RunLease(run_id, owner_id, current.acquired_at, now, now + ttl)
                cursor = db.execute(
                    "UPDATE run_leases SET heartbeat_at = ?, expires_at = ? "
                    "WHERE run_id = ? AND owner_id = ? AND expires_at = ?",
                    (
                        _encode_time(lease.heartbeat_at),
                        _encode_time(lease.expires_at),
                        run_id,
                        owner_id,
                        _encode_time(current.expires_at),
                    ),
                )
                if cursor.rowcount != 1:
                    raise RunLeaseStoreError("Lease renewal result was ambiguous")
                db.commit()
                return lease
        except (RunLeaseOwnershipLost, RunLeaseStoreError):
            raise
        except sqlite3.Error as exc:
            raise RunLeaseStoreError("SQLite lease renewal failed") from exc

    def _inspect(self, run_id: str) -> RunLease | None:
        try:
            with closing(self._connect()) as db:
                row = db.execute(
                    "SELECT run_id, owner_id, acquired_at, heartbeat_at, expires_at "
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
                    "SELECT owner_id FROM run_leases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise RunLeaseOwnershipLost("Run lease no longer exists")
                if row[0] != owner_id:
                    raise RunLeaseOwnershipLost("Run lease is owned by another owner")
                if cancellation.is_set():
                    db.rollback()
                    return _SQLiteReleaseAttempt.CANCELLED
                cursor = db.execute(
                    "DELETE FROM run_leases WHERE run_id = ? AND owner_id = ?",
                    (run_id, owner_id),
                )
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


def _lease_values(lease: RunLease) -> tuple[str, str, str, str, str]:
    return (
        lease.run_id,
        lease.owner_id,
        _encode_time(lease.acquired_at),
        _encode_time(lease.heartbeat_at),
        _encode_time(lease.expires_at),
    )


def _lease_from_row(row: tuple[object, ...]) -> RunLease:
    if len(row) != 5 or type(row[0]) is not str or type(row[1]) is not str:
        raise RunLeaseStoreError("Lease database contains an invalid row")
    try:
        return RunLease(
            run_id=row[0],
            owner_id=row[1],
            acquired_at=_decode_time(row[2]),
            heartbeat_at=_decode_time(row[3]),
            expires_at=_decode_time(row[4]),
        )
    except ValueError as exc:
        raise RunLeaseStoreError("Lease database contains invalid lease data") from exc
