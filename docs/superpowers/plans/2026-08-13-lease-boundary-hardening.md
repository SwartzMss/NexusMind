# Lease Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent stale lease renewal, audit terminal-time ownership expiry, and bound terminal release cleanup without rewriting execution outcomes.

**Architecture:** Keep the existing SQLite store and lease coordinator boundaries. Sample wall-clock time only after SQLite grants the write transaction, make terminal ownership validation explicit, and replace the unbounded shield loop with a monotonic deadline controlled by a `lease_release_timeout` propagated from `ChatRuntime`.

**Tech Stack:** Python 3.11+, asyncio, sqlite3, pytest

---

### Task 1: Use post-lock time for SQLite acquire and renew

**Files:**
- Modify: `tests/test_run_leases.py`
- Modify: `src/nexusmind/runtime/leases.py:387-397,468-558`

- [ ] **Step 1: Add deterministic lock-wait regression tests**

Add `threading` to the test imports and define a test-only store that signals every opened connection:

```python
class ConnectionObservedLeaseStore(SQLiteRunLeaseStore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.connection_opened = threading.Event()

    def _connect(self):
        connection = super()._connect()
        self.connection_opened.set()
        return connection
```

Add `test_renew_rechecks_expiry_after_waiting_for_write_lock`: acquire a five-second lease, hold `BEGIN IMMEDIATE` from a separate connection, start renewal, wait for the worker connection signal, advance `MutableClock` past expiry, release the lock, and require `RunLeaseOwnershipLost` with the expired row unchanged.

Add `test_acquire_rechecks_expiry_after_waiting_for_write_lock`: use the same lock choreography for a contender that starts while the old lease is active, advance the clock past expiry before releasing the lock, and require successful takeover by the contender.

- [ ] **Step 2: Run both tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_run_leases.py::test_renew_rechecks_expiry_after_waiting_for_write_lock \
  tests/test_run_leases.py::test_acquire_rechecks_expiry_after_waiting_for_write_lock
```

Expected: renewal incorrectly succeeds and acquisition incorrectly raises `RunLeaseUnavailable` because both use a pre-lock timestamp.

- [ ] **Step 3: Move clock sampling inside the write transaction**

Change the async wrappers to dispatch only validated request values:

```python
return await asyncio.to_thread(self._acquire, run_id, owner_id, ttl)
return await asyncio.to_thread(self._renew, run_id, owner_id, ttl)
```

Remove `now` from both synchronous signatures. Immediately after each successful `db.execute("BEGIN IMMEDIATE")`, add:

```python
now = self._now()
```

Compute `expires_at` only after this point in `_acquire()`.

- [ ] **Step 4: Run the targeted and core SQLite lease tests and verify GREEN**

Run:

```bash
pytest -q tests/test_run_leases.py -k 'waiting_for_write_lock or first_owner or concurrent_acquire or expired_takeover or sqlite_lock_failure'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the SQLite fix**

```bash
git add src/nexusmind/runtime/leases.py tests/test_run_leases.py
git commit -m "fix: evaluate lease expiry after sqlite lock"
```

### Task 2: Audit natural ownership expiry on terminal events

**Files:**
- Modify: `tests/test_run_leases.py`
- Modify: `src/nexusmind/runtime/leases.py:275-284`

- [ ] **Step 1: Add the terminal expiry regression test**

Add `test_terminal_event_audits_ownership_expiry_while_waiting`. Use an injected `MutableClock`, a store whose acquired lease uses that clock, and an async terminal stream that pauses before yielding `RUN_COMPLETED`. Start collection, advance the clock beyond the lease expiry while the stream is paused, then allow the terminal event through.

Assert:

```python
assert events[-1].type is RuntimeEventType.RUN_COMPLETED
assert events[-1].metadata["lease_ownership_lost_after_terminal"] is True
assert runtime._harness.state.status is HarnessStatus.COMPLETED
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/test_run_leases.py::test_terminal_event_audits_ownership_expiry_while_waiting
```

Expected: `lease_ownership_lost_after_terminal` is missing because `_finalize_terminal()` only reads the stored error.

- [ ] **Step 3: Validate the guard during terminal finalization**

At the start of `_finalize_terminal()`, preserve the outcome while actively checking ownership:

```python
ownership_lost = False
try:
    self._guard.assert_owned()
except RunLeaseError:
    ownership_lost = True
await self._stop_heartbeat()
ownership_lost = ownership_lost or self._guard.ownership_error is not None
```

Keep the existing metadata and release logic unchanged.

- [ ] **Step 4: Run terminal ownership tests and verify GREEN**

Run:

```bash
pytest -q tests/test_run_leases.py -k 'terminal and (ownership or heartbeat or expiry)'
```

Expected: all selected tests pass and completed/failed outcomes remain unchanged.

- [ ] **Step 5: Commit the terminal audit fix**

```bash
git add src/nexusmind/runtime/leases.py tests/test_run_leases.py
git commit -m "fix: audit lease expiry at terminal boundary"
```

### Task 3: Bound terminal release cleanup

**Files:**
- Modify: `tests/test_run_leases.py`
- Modify: `src/nexusmind/runtime/leases.py:152-181,275-307`
- Modify: `src/nexusmind/runtime/chat.py:33-51,81-85,114-123`

- [ ] **Step 1: Add timeout validation and hanging-release tests**

Import `RunLeaseCoordinator` in `tests/test_run_leases.py`. Add constructor tests requiring a positive `timedelta` for `lease_release_timeout`.

Add `test_terminal_release_timeout_preserves_terminal_outcome`: a store's `release()` sets `release_started`, waits forever, and records cancellation. Construct `ChatRuntime` with `lease_release_timeout=timedelta(milliseconds=20)`. Require a returned `RUN_COMPLETED` event with `lease_release_failed`, a `RunLeaseReleaseError`, and a cancelled cleanup task.

Add `test_cancellation_cannot_extend_terminal_release_timeout`: start the same runtime, cancel event collection after release starts, and require the fixed terminal outcome to return within one second with `lease_release_failed` rather than hanging.

- [ ] **Step 2: Run the new timeout tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_run_leases.py::test_terminal_release_timeout_preserves_terminal_outcome \
  tests/test_run_leases.py::test_cancellation_cannot_extend_terminal_release_timeout
```

Expected: constructors reject the unknown keyword before timeout behavior exists.

- [ ] **Step 3: Add and propagate the timeout setting**

Add this keyword to `RunLeaseCoordinator.__init__` and `ChatRuntime.__init__`:

```python
lease_release_timeout: timedelta = timedelta(seconds=10)
```

Validate it in the coordinator:

```python
if type(lease_release_timeout) is not timedelta or lease_release_timeout <= timedelta(0):
    raise ValueError("Lease release timeout must be a positive timedelta")
```

Store the value on both objects and pass it from `ChatRuntime` to `RunLeaseCoordinator`.

- [ ] **Step 4: Replace the unbounded release loop with a monotonic deadline**

Use `asyncio.get_running_loop().time()` and `asyncio.wait(..., timeout=remaining)`. Continue after external `CancelledError`, but do not reset the deadline. When remaining time reaches zero, assign `RunLeaseReleaseError("Run lease release timed out")`, cancel the release task, clear the guard, and attach a callback that consumes the eventual task result without blocking terminal delivery.

Keep successful release and ordinary `RunLeaseError` conversion behavior intact.

- [ ] **Step 5: Run release and cancellation tests and verify GREEN**

Run:

```bash
pytest -q tests/test_run_leases.py -k 'terminal_release or cancellation_during_terminal_release or release_failure'
```

Expected: all selected tests pass, including the existing test where release completes before the deadline after cancellation.

- [ ] **Step 6: Commit the bounded barrier**

```bash
git add src/nexusmind/runtime/leases.py src/nexusmind/runtime/chat.py tests/test_run_leases.py
git commit -m "fix: bound terminal lease release cleanup"
```

### Task 4: Verify CLI timeout auditing

**Files:**
- Modify: `tests/test_run_leases.py`

- [ ] **Step 1: Add a CLI release-timeout regression test**

Add `test_cli_release_timeout_audits_cleanup_without_rewriting_completed_outcome`. Monkeypatch a hanging initialized lease store and a `ChatRuntime` factory that supplies a 20ms release timeout. Run `_run_chat()` with state and lease databases.

Require:

```python
assert code == 1
assert status == "completed"
assert error_code is None
assert event_types[-2:] == ["lease_release_failed", "run_completed"]
```

- [ ] **Step 2: Run the test and verify it passes through the new metadata path**

Run:

```bash
pytest -q tests/test_run_leases.py::test_cli_release_timeout_audits_cleanup_without_rewriting_completed_outcome
```

Expected: PASS. If it fails, change only the metadata-to-audit integration needed to preserve completed status and return code 1.

- [ ] **Step 3: Commit CLI regression coverage**

```bash
git add tests/test_run_leases.py src/nexusmind/cli.py
git commit -m "test: cover cli lease release timeout audit"
```

### Task 5: Full verification

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run the lease test module**

```bash
pytest -q tests/test_run_leases.py
```

Expected: all tests pass.

- [ ] **Step 2: Run the complete test suite**

```bash
pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Run repository quality checks**

```bash
git diff --check origin/pr/41...HEAD
git status --short --branch
```

Expected: no whitespace errors; only intentional commits and no uncommitted files.
