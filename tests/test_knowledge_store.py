from __future__ import annotations

import sqlite3
import threading

import pytest

from nexusmind import (
    Document,
    KnowledgeCollection,
    KnowledgeSnapshot,
    KnowledgeSnapshotStore,
    KnowledgeSnapshotStoreError,
    KnowledgeSource,
    SQLiteKnowledgeSnapshotStore,
)


def _source(source_id: str, *, metadata: dict | None = None) -> KnowledgeSource:
    return KnowledgeSource(
        source_id=source_id,
        source_type="fake",
        display_name=f"Source {source_id}",
        logical_location=f"{source_id}.logical",
        metadata={} if metadata is None else metadata,
    )


def _document(
    source_id: str,
    logical_path: str,
    content: str,
    *,
    metadata: dict | None = None,
) -> Document:
    return Document(
        source_id=source_id,
        logical_path=logical_path,
        content=content,
        content_type="text/markdown",
        metadata={} if metadata is None else metadata,
    )


def _snapshot(source_id: str, content: str = "searchable") -> KnowledgeSnapshot:
    return KnowledgeSnapshot(
        sources=(_source(source_id),),
        documents=(_document(source_id, "notes.md", content),),
    )


def test_store_contract_and_new_database_load_empty_snapshot(tmp_path) -> None:
    store: KnowledgeSnapshotStore = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")

    assert store.load() == KnowledgeSnapshot((), ())


def test_save_and_load_empty_snapshot(tmp_path) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")

    store.save(KnowledgeSnapshot((), ()))

    assert store.load() == KnowledgeSnapshot((), ())


def test_load_reads_one_committed_point_in_time_during_concurrent_save(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    before = KnowledgeSnapshot(
        (_source("docs", metadata={"version": "before"}),),
        (_document("docs", "notes.md", "before content"),),
    )
    after = KnowledgeSnapshot(
        (_source("docs", metadata={"version": "after"}),),
        (_document("docs", "notes.md", "after content"),),
    )
    writer = SQLiteKnowledgeSnapshotStore(path)
    writer.save(before)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")

    sources_read = threading.Event()
    continue_read = threading.Event()

    class CursorProxy:
        def __init__(self, cursor, pause: bool) -> None:
            self._cursor = cursor
            self._pause = pause

        def fetchall(self):
            rows = self._cursor.fetchall()
            if self._pause:
                sources_read.set()
                if not continue_read.wait(timeout=5):
                    raise AssertionError("concurrent writer did not complete")
            return rows

        def __getattr__(self, name):
            return getattr(self._cursor, name)

        def __iter__(self):
            return iter(self._cursor)

    class ConnectionProxy:
        def __init__(self, connection) -> None:
            self._connection = connection

        def execute(self, sql, parameters=()):
            cursor = self._connection.execute(sql, parameters)
            pause = "FROM sources ORDER BY source_id" in sql
            return CursorProxy(cursor, pause)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    class PausingStore(SQLiteKnowledgeSnapshotStore):
        def _connect(self):
            return ConnectionProxy(super()._connect())

    loader = PausingStore(path)
    loaded: list[KnowledgeSnapshot] = []
    errors: list[BaseException] = []

    def load_snapshot() -> None:
        try:
            loaded.append(loader.load())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=load_snapshot)
    thread.start()
    assert sources_read.wait(timeout=5)
    try:
        writer.save(after)
    finally:
        continue_read.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert loaded == [before]
    assert writer.load() == after


def test_round_trip_multiple_sources_documents_and_nested_metadata(tmp_path) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")
    snapshot = KnowledgeSnapshot(
        sources=(
            _source("z", metadata={"nested": {"items": [None, True, 2, 3.5, "你"]}}),
            _source("a"),
        ),
        documents=(
            _document("z", "z.md", "z", metadata={"tags": ["z"]}),
            _document("a", "b.md", "b"),
            _document("a", "a.md", "a"),
        ),
    )

    store.save(snapshot)
    loaded = store.load()

    assert [source.source_id for source in loaded.sources] == ["a", "z"]
    assert [document.document_id for document in loaded.documents] == [
        document.document_id
        for document in sorted(snapshot.documents, key=lambda item: (item.source_id, item.document_id))
    ]
    assert {source.source_id: source.metadata for source in loaded.sources}["z"] == {
        "nested": {"items": [None, True, 2, 3.5, "你"]}
    }
    expected = store.load()
    loaded.sources[1].metadata["nested"]["items"].append("changed")
    assert store.load() == expected


def test_successful_save_is_always_loadable_as_equal_canonical_snapshot(tmp_path) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")
    snapshot = _snapshot("docs", "canonical")

    store.save(snapshot)

    assert store.load() == snapshot


def test_snapshot_store_restart_restore_and_search_round_trip(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    original = KnowledgeCollection()

    class Adapter:
        def source(self):
            return _source("docs")

        def load_documents(self):
            return (_document("docs", "notes.md", "checkpoint resume"),)

    original.sync(Adapter())
    SQLiteKnowledgeSnapshotStore(path).save(original.snapshot())

    restarted = KnowledgeCollection()
    restarted.restore(SQLiteKnowledgeSnapshotStore(path).load())

    assert restarted.snapshot() == original.snapshot()
    assert restarted.search("checkpoint resume")[0].score == 2


def test_second_save_fully_replaces_old_sources_and_documents(tmp_path) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")
    store.save(_snapshot("old", "old content"))

    new = _snapshot("new", "new content")
    store.save(new)

    assert store.load() == new


@pytest.mark.parametrize(
    "metadata",
    [
        {"set": {1}},
        {"tuple": (1, 2)},
        {1: "non-string key"},
        {"nan": float("nan")},
        {"infinity": float("inf")},
    ],
)
def test_unsupported_metadata_is_rejected(metadata: dict, tmp_path) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")
    snapshot = KnowledgeSnapshot((_source("docs", metadata=metadata),), ())

    with pytest.raises(KnowledgeSnapshotStoreError, match="JSON-compatible"):
        store.save(snapshot)


def test_metadata_encoding_failure_preserves_previous_snapshot(tmp_path) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")
    previous = _snapshot("old")
    store.save(previous)
    invalid = KnowledgeSnapshot((_source("bad", metadata={"value": object()}),), ())

    with pytest.raises(KnowledgeSnapshotStoreError):
        store.save(invalid)

    assert store.load() == previous


def test_sqlite_write_failure_rolls_back_complete_replacement(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    store = SQLiteKnowledgeSnapshotStore(path)
    previous = _snapshot("old")
    store.save(previous)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TRIGGER reject_failed_document BEFORE INSERT ON documents "
            "WHEN NEW.content = 'fail' BEGIN SELECT RAISE(ABORT, 'forced failure'); END"
        )

    with pytest.raises(KnowledgeSnapshotStoreError):
        store.save(_snapshot("new", "fail"))

    assert store.load() == previous


def test_unsupported_and_malformed_schema_fail_closed(tmp_path) -> None:
    unsupported = tmp_path / "unsupported.db"
    store = SQLiteKnowledgeSnapshotStore(unsupported)
    with sqlite3.connect(unsupported) as db:
        db.execute(
            "UPDATE knowledge_store_metadata SET value = '999' WHERE key = 'schema_version'"
        )
    with pytest.raises(KnowledgeSnapshotStoreError, match="unsupported"):
        store.load()

    malformed = tmp_path / "malformed.db"
    with sqlite3.connect(malformed) as db:
        db.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(KnowledgeSnapshotStoreError, match="schema"):
        SQLiteKnowledgeSnapshotStore(malformed)


def test_orphan_database_row_fails_closed(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    store = SQLiteKnowledgeSnapshotStore(path)
    document = _document("missing", "a.md", "orphan")
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=OFF")
        db.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                document.document_id,
                document.source_id,
                document.logical_path,
                document.content,
                document.content_type,
                "{}",
                document.content_hash,
            ),
        )

    with pytest.raises(KnowledgeSnapshotStoreError, match="missing source"):
        store.load()


def test_malformed_stored_metadata_fails_closed(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    store = SQLiteKnowledgeSnapshotStore(path)
    store.save(_snapshot("docs"))
    with sqlite3.connect(path) as db:
        db.execute("UPDATE sources SET metadata_json = '{broken json'")

    with pytest.raises(KnowledgeSnapshotStoreError, match="metadata"):
        store.load()


@pytest.mark.parametrize("column", ["document_id", "content_hash"])
def test_document_integrity_tampering_is_rejected(column: str, tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    store = SQLiteKnowledgeSnapshotStore(path)
    store.save(_snapshot("docs"))
    with sqlite3.connect(path) as db:
        db.execute(f"UPDATE documents SET {column} = ?", ("forged",))

    with pytest.raises(KnowledgeSnapshotStoreError, match="integrity"):
        store.load()


def test_schema_persists_no_derived_retrieval_state(tmp_path) -> None:
    path = tmp_path / "knowledge.db"
    SQLiteKnowledgeSnapshotStore(path).save(_snapshot("docs"))
    with sqlite3.connect(path) as db:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {
            row[1]
            for table in tables
            for row in db.execute(f"PRAGMA table_info({table})")
        }

    assert tables == {"knowledge_store_metadata", "sources", "documents"}
    assert {"chunk_id", "score", "matched_terms"}.isdisjoint(columns)


def test_save_rejects_incoherent_snapshot_without_replacing_old_state(tmp_path) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")
    previous = _snapshot("old")
    store.save(previous)
    forged = _document("docs", "a.md", "content")
    object.__setattr__(forged, "content_hash", "forged")

    with pytest.raises(KnowledgeSnapshotStoreError, match="content hash"):
        store.save(KnowledgeSnapshot((_source("docs"),), (forged,)))

    assert store.load() == previous


@pytest.mark.parametrize(
    "model, field, value",
    [
        ("source", "source_id", ""),
        ("source", "display_name", ""),
        ("source", "logical_location", ""),
        ("source", "metadata", []),
        ("document", "content_type", ""),
        ("document", "metadata", []),
    ],
)
def test_save_revalidates_forged_canonical_model_fields(
    model: str, field: str, value: object, tmp_path
) -> None:
    store = SQLiteKnowledgeSnapshotStore(tmp_path / "knowledge.db")
    previous = _snapshot("old")
    store.save(previous)
    source = _source("docs")
    document = _document("docs", "a.md", "content")
    target = source if model == "source" else document
    object.__setattr__(target, field, value)
    snapshot = KnowledgeSnapshot((source,), () if model == "source" else (document,))

    with pytest.raises(KnowledgeSnapshotStoreError, match="invalid"):
        store.save(snapshot)

    assert store.load() == previous


def test_path_validation_is_controlled() -> None:
    with pytest.raises(ValueError):
        SQLiteKnowledgeSnapshotStore("")
    with pytest.raises(ValueError):
        SQLiteKnowledgeSnapshotStore(":memory:")
