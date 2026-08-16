# Lease Boundary Hardening Design

## Goal

Harden PR #41 so an expired lease can never be renewed, terminal events audit
ownership expiry that occurs while the event is being produced, and terminal
lease cleanup always finishes within a configured time bound without changing
the already-determined execution outcome.

## Safety Invariants

- A lease that expires before a renewal obtains SQLite's write lock must not be
  renewed.
- A contender that obtains SQLite's write lock after the current lease expires
  may acquire the lease, even if it started waiting while the lease was active.
- Ownership that cannot be proven after a terminal event is produced is audited
  as `lease_ownership_lost_after_terminal` without rewriting `RUN_COMPLETED` or
  `RUN_FAILED`.
- Terminal lease release must complete, fail, or time out before the terminal
  event is exposed. External cancellation may defer delivery only until that
  bounded barrier resolves.
- Release failure or timeout sets `lease_release_failed`, preserves the terminal
  execution outcome, and causes the CLI to return exit code 1.

## SQLite Transaction Time

`SQLiteRunLeaseStore.acquire()` and `renew()` will continue validating request
arguments before dispatching synchronous SQLite work to a worker thread. They
will no longer sample the clock in the async wrapper.

The synchronous `_acquire()` and `_renew()` methods will call `_now()` only
after `BEGIN IMMEDIATE` succeeds. All expiry decisions and newly written
timestamps therefore use a time sampled while the operation owns SQLite's
write transaction. Clock failures remain controlled `RunLeaseStoreError`
failures and the uncommitted transaction is rolled back when the connection
closes.

This preserves the existing compare-and-swap predicates for takeover and
renewal while removing the stale pre-lock timestamp window.

## Terminal Ownership Audit

`RunLeaseCoordinator._finalize_terminal()` will actively call
`self._guard.assert_owned()` before stopping the heartbeat. It will catch
`RunLeaseError` and remember the loss instead of raising, because the Harness
has already fixed the execution outcome.

After the heartbeat stops, the coordinator will also consider any ownership
error recorded concurrently by the heartbeat. It will then perform terminal
release and attach `lease_ownership_lost_after_terminal` when either check
shows that ownership could not be proven. The terminal event type and Harness
state remain unchanged.

## Bounded Terminal Release

`RunLeaseCoordinator` and `ChatRuntime` will accept a positive
`lease_release_timeout: timedelta`, defaulting to 10 seconds. `ChatRuntime`
will pass the value to its coordinator.

The release barrier will create one owner-scoped release task and wait for it
against a deadline computed from the event loop's monotonic clock. External
`CancelledError` requests received during the barrier will be deferred, as in
the current terminal-outcome preservation behavior, but the same deadline will
continue to apply.

If release finishes, its current success or `RunLeaseReleaseError` handling is
preserved. If the deadline expires, the coordinator will:

1. record a `RunLeaseReleaseError` as `release_error`;
2. cancel the release task and consume its eventual result without waiting
   past the deadline;
3. clear the local ownership proof; and
4. return the terminal event with `lease_release_failed = true`.

The durable lease row is not force-deleted on timeout and remains available for
TTL-based takeover. Existing CLI cleanup auditing turns the metadata into exit
code 1 while retaining the completed or failed run status.

## Testing

The SQLite tests will hold a real `BEGIN IMMEDIATE` lock while an operation is
waiting. A connection hook will signal that the worker reached SQLite so the
test can advance an injected clock deterministically before releasing the lock.
The renewal test must fail with `RunLeaseOwnershipLost`; the acquisition test
must allow takeover using the post-lock time.

A coordinator test will pause production of a terminal event, advance the
injected lease clock past expiry, and verify that the terminal type is
preserved while `lease_ownership_lost_after_terminal` is present.

A hanging release store will verify that a short configured timeout returns a
terminal event with `lease_release_failed`, records `RunLeaseReleaseError`, and
cancels cleanup. A cancellation variant will verify that user cancellation
cannot make the barrier exceed its configured upper bound. CLI coverage will
verify exit code 1 and a preserved completed run status for release timeout.

Existing lease, checkpoint, CLI, and full test suites will be run after the
targeted tests pass.
