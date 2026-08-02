import asyncio
import json
from datetime import datetime, timezone

import pytest
import sqlite3

from nexusmind.persistence import SQLiteRunStore, RunKind, RunStartContext, RunStatus, RunTraceEvent
from nexusmind.persistence.sqlite import StateStoreError


def run(coro):
    return asyncio.run(coro)


def test_store_lifecycle_and_stable_show(tmp_path):
    store = SQLiteRunStore(tmp_path / "state.db")
    run_id = run(store.start_run(RunStartContext(RunKind.CHAT, input_text="hello")))
    run(store.append_event(run_id, RunTraceEvent("tool_result", datetime.now(timezone.utc), {"ok": True})))
    run(store.finish_run(run_id, RunStatus.COMPLETED))
    shown = store.show_run(run_id)
    assert shown["run"]["status"] == "completed"
    assert shown["events"][1]["payload"] == {"ok": True}
    store.close()


def test_read_only_open_does_not_abandon_running_run(tmp_path):
    path = tmp_path / "state.db"
    writer = SQLiteRunStore(path, execution_id="writer")
    run_id = run(writer.start_run(RunStartContext(RunKind.CHAT)))
    reader = SQLiteRunStore(path, execution_id="reader")
    assert reader.show_run(run_id)["run"]["status"] == "running"
    reader.close(); writer.close()


def test_explicit_recovery_updates_event_count(tmp_path):
    path = tmp_path / "state.db"
    writer = SQLiteRunStore(path, execution_id="writer")
    run_id = run(writer.start_run(RunStartContext(RunKind.CHAT)))
    writer.close()
    recovered = SQLiteRunStore(path, execution_id="recovery", recover_abandoned=True)
    shown = recovered.show_run(run_id)
    assert shown["run"]["status"] == "abandoned"
    assert shown["run"]["event_count"] == len(shown["events"])
    recovered.close()


def test_prune_rejects_negative_days(tmp_path):
    store = SQLiteRunStore(tmp_path / "state.db")
    with pytest.raises(ValueError): store.prune(-1)
    store.close()

@pytest.mark.parametrize("version", ["0", "-1", "abc", "2"])
def test_schema_rejects_unsupported_version(tmp_path, version):
    path = tmp_path / "state.db"
    db = sqlite3.connect(path); db.execute("CREATE TABLE schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"); db.execute("INSERT INTO schema_metadata VALUES('version', ?)", (version,)); db.commit(); db.close()
    with pytest.raises(StateStoreError): SQLiteRunStore(path)

def test_trace_failure_marks_trace_incomplete(tmp_path):
    store = SQLiteRunStore(tmp_path / "state.db")
    run_id = run(store.start_run(RunStartContext(RunKind.CHAT)))
    run(store.finish_run(run_id, RunStatus.FAILED, error_code="trace_persist_failed", trace_complete=False))
    assert store.show_run(run_id)["run"]["trace_complete"] == 0
    store.close()

def test_prune_deletes_old_terminal_runs_but_keeps_running(tmp_path):
    store = SQLiteRunStore(tmp_path / "state.db")
    old_id = run(store.start_run(RunStartContext(RunKind.CHAT)))
    run(store.finish_run(old_id, RunStatus.COMPLETED))
    running_id = run(store.start_run(RunStartContext(RunKind.CHAT)))
    store.db.execute("UPDATE runs SET started_at='2020-01-01T00:00:00+00:00' WHERE id IN (?, ?)", (old_id, running_id)); store.db.commit()
    assert store.prune(30) == 1
    assert store.show_run(old_id) is None
    assert store.show_run(running_id) is not None
    store.close()

def test_terminal_run_rejects_events_and_running_finish(tmp_path):
    store = SQLiteRunStore(tmp_path / "state.db")
    run_id = run(store.start_run(RunStartContext(RunKind.CHAT)))
    run(store.finish_run(run_id, RunStatus.COMPLETED))
    with pytest.raises(StateStoreError):
        run(store.append_event(run_id, RunTraceEvent("late", datetime.now(timezone.utc), {})))
    with pytest.raises(StateStoreError):
        run(store.finish_run(run_id, RunStatus.RUNNING))
    store.close()

def test_schema_metadata_without_core_tables_is_rejected(tmp_path):
    path = tmp_path / "state.db"
    db = sqlite3.connect(path); db.execute("CREATE TABLE schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"); db.execute("INSERT INTO schema_metadata VALUES('version','1')"); db.commit(); db.close()
    with pytest.raises(StateStoreError): SQLiteRunStore(path)

def test_schema_rejects_missing_run_constraints(tmp_path):
    path = tmp_path / "invalid-constraints.db"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO schema_metadata VALUES('version','1');
        CREATE TABLE runs(
            id TEXT, schema_version INTEGER, execution_id TEXT, kind TEXT,
            status TEXT, skill_name TEXT, model_name TEXT, input_preview TEXT,
            input_sha256 TEXT, final_text TEXT, error_code TEXT,
            error_message TEXT, trace_complete INTEGER, event_count INTEGER,
            started_at TEXT, updated_at TEXT, finished_at TEXT
        );
        CREATE TABLE run_events(
            run_id TEXT, sequence INTEGER, event_type TEXT,
            occurred_at TEXT, payload_json TEXT, payload_bytes INTEGER
        );
    """)
    db.commit(); db.close()
    with pytest.raises(StateStoreError): SQLiteRunStore(path)
