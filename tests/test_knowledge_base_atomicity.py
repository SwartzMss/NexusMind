import os
from pathlib import Path
import sqlite3

import pytest

import nexusmind.knowledge_base as module
from nexusmind import (
    KnowledgeBase,
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBasePersistenceError,
    KnowledgeBaseSourceError,
    LocalFileSourceConfig,
    SQLiteKnowledgeSnapshotStore,
)


def _source_id(tmp_path: Path, name: str) -> str:
    return LocalFileSourceConfig(path=str(tmp_path / name)).source_id


def _ordered_source_ids(tmp_path: Path, *names: str) -> tuple[str, ...]:
    return tuple(sorted(_source_id(tmp_path, name) for name in names))


def _synced_pair(tmp_path: Path) -> tuple[KnowledgeBase, Path]:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha removable", encoding="utf-8")
    second.write_text("beta retained", encoding="utf-8")
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(path=str(first)))
    kb.add_source(LocalFileSourceConfig(path=str(second)))
    kb.sync()
    return kb, root


def test_remove_source_removes_registration_canonical_and_search_state(tmp_path: Path) -> None:
    kb, root = _synced_pair(tmp_path)

    kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert tuple(item.source_id for item in kb.list_sources()) == (_source_id(tmp_path, "second.txt"),)
    assert tuple(item.source_id for item in kb.list_documents()) == (_source_id(tmp_path, "second.txt"),)
    assert kb.search("removable") == ()
    assert kb.search("retained")
    stored = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    assert tuple(item.source_id for item in stored.sources) == (_source_id(tmp_path, "second.txt"),)
    assert tuple(item.source_id for item in stored.documents) == (_source_id(tmp_path, "second.txt"),)
    reopened = KnowledgeBase.open(str(root))
    assert tuple(item.source_id for item in reopened.list_sources()) == (_source_id(tmp_path, "second.txt"),)
    assert tuple(item.source_id for item in reopened.list_documents()) == (_source_id(tmp_path, "second.txt"),)
    assert b"alpha removable" not in root.joinpath("manifest.json").read_bytes()
    with sqlite3.connect(root / "knowledge.db") as database:
        tables = {
            row[0]
            for row in database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {
            table: {row[1] for row in database.execute(f"PRAGMA table_info({table})")}
            for table in tables
        }
    assert tables == {
        "knowledge_store_metadata",
        "sources",
        "documents",
        "document_versions",
    }
    persisted_names = {name for names in columns.values() for name in names}
    assert not persisted_names & {
        "chunk", "chunks", "embedding", "embeddings", "index", "reranker", "config"
    }


@pytest.mark.parametrize("source_id", ["", " ", None, 1])
def test_remove_source_rejects_malformed_ids_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_id: object
) -> None:
    kb, root = _synced_pair(tmp_path)
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    monkeypatch.setattr(module, "write_manifest", lambda *args, **kwargs: pytest.fail("write"))

    with pytest.raises(KnowledgeBaseConfigError):
        kb.remove_source(source_id)  # type: ignore[arg-type]

    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before


def test_remove_source_rejects_unknown_id_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    monkeypatch.setattr(module, "write_manifest", lambda *args, **kwargs: pytest.fail("write"))

    with pytest.raises(KnowledgeBaseSourceError):
        kb.remove_source("missing")

    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before


def test_initial_canonical_failure_preserves_both_stores_and_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    status_before = kb.status()
    documents_before = kb.list_documents()
    search_before = kb.search("removable")
    calls: list[object] = []

    def fail_save(self, snapshot):
        calls.append(snapshot)
        if len(calls) == 1:
            raise RuntimeError("private provider detail")

    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", fail_save)
    monkeypatch.setattr(module, "write_manifest", lambda *args, **kwargs: calls.append("manifest"))

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert tuple(item.source_id for item in calls[0].sources) == (_source_id(tmp_path, "second.txt"),)
    assert calls[1] == snapshot_before
    assert "private" not in str(caught.value)
    assert tuple(item.source_id for item in kb.list_sources()) == _ordered_source_ids(tmp_path, "first.txt", "second.txt")
    assert kb.status() == status_before
    assert kb.list_documents() == documents_before
    assert kb.search("removable") == search_before
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before


def test_post_commit_initial_save_failure_compensates_exact_old_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    status_before = kb.status()
    documents_before = kb.list_documents()
    search_before = kb.search("removable")
    real_save = module.SQLiteKnowledgeSnapshotStore.save
    calls: list[object] = []

    def commit_then_fail(self, snapshot):
        calls.append(snapshot)
        real_save(self, snapshot)
        if len(calls) == 1:
            raise RuntimeError("private post-commit detail")

    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", commit_then_fail)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert tuple(item.source_id for item in calls[0].sources) == (_source_id(tmp_path, "second.txt"),)
    assert calls[1] == snapshot_before
    assert "private" not in str(caught.value)
    assert kb.status() == status_before
    assert kb.list_documents() == documents_before
    assert kb.search("removable") == search_before
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before


def test_initial_save_and_compensation_failure_poisons_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, _ = _synced_pair(tmp_path)
    calls: list[object] = []

    def fail_both_saves(self, snapshot):
        calls.append(snapshot)
        detail = "private initial detail" if len(calls) == 1 else "private recovery detail"
        raise RuntimeError(detail)

    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", fail_both_saves)
    with pytest.raises(KnowledgeBasePersistenceError, match="recovery") as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert tuple(item.source_id for item in calls[0].sources) == (_source_id(tmp_path, "second.txt"),)
    assert tuple(item.source_id for item in calls[1].sources) == _ordered_source_ids(tmp_path, "first.txt", "second.txt")
    assert "private" not in str(caught.value)
    with pytest.raises(KnowledgeBaseClosedError):
        kb.status()


def test_canonical_store_initialization_failure_is_sanitized_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    real_store = module.SQLiteKnowledgeSnapshotStore

    constructions = 0

    class FailingStore:
        def __init__(self, path):
            nonlocal constructions
            constructions += 1
            if constructions == 2:
                raise RuntimeError("private database path")
            self._delegate = real_store(path)

        def load(self):
            return self._delegate.load()

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", FailingStore)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert "private" not in str(caught.value)
    assert tuple(item.source_id for item in kb.list_sources()) == _ordered_source_ids(tmp_path, "first.txt", "second.txt")
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert real_store(root / "knowledge.db").load() == snapshot_before

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", real_store)
    KnowledgeBase.open(str(root)).remove_source(_source_id(tmp_path, "first.txt"))


def test_manifest_failure_compensates_exact_old_canonical_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    real_save = module.SQLiteKnowledgeSnapshotStore.save
    calls: list[object] = []

    def recording_save(self, snapshot):
        calls.append(snapshot)
        return real_save(self, snapshot)

    real_write = module.write_manifest
    writes = 0

    def commit_candidate_then_restore_old(*args, **kwargs):
        nonlocal writes
        writes += 1
        calls.append(("manifest", args[1]))
        real_write(*args, **kwargs)
        if writes == 1:
            raise RuntimeError("private manifest detail")

    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", recording_save)
    monkeypatch.setattr(module, "write_manifest", commit_candidate_then_restore_old)

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert tuple(item.source_id for item in calls[0].sources) == (_source_id(tmp_path, "second.txt"),)
    assert calls[1][0] == "manifest"
    assert tuple(item.source_id for item in calls[1][1].sources) == (_source_id(tmp_path, "second.txt"),)
    assert calls[2][0] == "manifest"
    assert tuple(item.source_id for item in calls[2][1].sources) == _ordered_source_ids(tmp_path, "first.txt", "second.txt")
    assert calls[3] == snapshot_before
    assert "private" not in str(caught.value)
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before
    assert tuple(item.source_id for item in kb.list_sources()) == _ordered_source_ids(tmp_path, "first.txt", "second.txt")
    assert KnowledgeBase.open(str(root)).status() == kb.status()


def test_manifest_recovery_failure_poisons_even_when_database_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    real_write = module.write_manifest
    real_save = module.SQLiteKnowledgeSnapshotStore.save
    calls: list[str] = []
    writes = 0

    def fail_manifest_recovery(*args, **kwargs):
        nonlocal writes
        writes += 1
        calls.append(f"manifest-{writes}")
        real_write(*args, **kwargs)
        if writes == 1:
            raise RuntimeError("private candidate detail")
        raise RuntimeError("private recovery detail")

    def recording_save(self, snapshot):
        calls.append("database")
        return real_save(self, snapshot)

    monkeypatch.setattr(module, "write_manifest", fail_manifest_recovery)
    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", recording_save)
    with pytest.raises(KnowledgeBasePersistenceError, match="recovery") as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert calls == ["database", "manifest-1", "manifest-2", "database"]
    assert "private" not in str(caught.value)
    with pytest.raises(KnowledgeBaseClosedError):
        kb.status()
    KnowledgeBase.open(str(root))


def test_database_recovery_failure_still_attempts_manifest_recovery_and_poisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, _ = _synced_pair(tmp_path)
    real_write = module.write_manifest
    real_save = module.SQLiteKnowledgeSnapshotStore.save
    calls: list[str] = []
    saves = 0
    writes = 0

    def candidate_commit_then_fail(*args, **kwargs):
        nonlocal writes
        writes += 1
        calls.append(f"manifest-{writes}")
        real_write(*args, **kwargs)
        if writes == 1:
            raise RuntimeError("private candidate detail")

    def fail_database_recovery(self, snapshot):
        nonlocal saves
        saves += 1
        calls.append(f"database-{saves}")
        if saves == 2:
            raise RuntimeError("private database recovery")
        return real_save(self, snapshot)

    monkeypatch.setattr(module, "write_manifest", candidate_commit_then_fail)
    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", fail_database_recovery)
    with pytest.raises(KnowledgeBasePersistenceError, match="recovery") as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))

    assert calls == ["database-1", "manifest-1", "manifest-2", "database-2"]
    assert "private" not in str(caught.value)
    with pytest.raises(KnowledgeBaseClosedError):
        kb.list_sources()


def test_failed_compensation_poisons_handle_and_reports_recovery_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, _ = _synced_pair(tmp_path)
    real_save = module.SQLiteKnowledgeSnapshotStore.save
    saves = 0

    def fail_second_save(self, snapshot):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise RuntimeError("private compensation detail")
        return real_save(self, snapshot)

    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", fail_second_save)
    monkeypatch.setattr(module, "write_manifest", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private")))

    with pytest.raises(KnowledgeBasePersistenceError, match="recovery") as caught:
        kb.remove_source(_source_id(tmp_path, "first.txt"))
    assert "private" not in str(caught.value)
    assert saves == 2
    for operation in (kb.status, kb.list_sources, kb.list_documents, lambda: kb.search("x"), kb.sync):
        with pytest.raises(KnowledgeBaseClosedError):
            operation()
    for operation in (
        lambda: kb.add_source(LocalFileSourceConfig(path="third.txt")),
        lambda: kb.unregister_source(_source_id(tmp_path, "second.txt")),
        lambda: kb.remove_source(_source_id(tmp_path, "second.txt")),
        lambda: kb.sync_source(_source_id(tmp_path, "second.txt")),
    ):
        with pytest.raises(KnowledgeBaseClosedError):
            operation()
    kb.close()
    kb.close()


def test_stale_handle_refreshes_before_removal(tmp_path: Path) -> None:
    kb, root = _synced_pair(tmp_path)
    stale = KnowledgeBase.open(str(root))
    third = tmp_path / "third.txt"
    third.write_text("gamma", encoding="utf-8")
    kb.add_source(LocalFileSourceConfig(path=str(third)))
    kb.sync_source(_source_id(tmp_path, "third.txt"))

    stale.remove_source(_source_id(tmp_path, "first.txt"))

    reopened = KnowledgeBase.open(str(root))
    assert tuple(item.source_id for item in reopened.list_sources()) == _ordered_source_ids(tmp_path, "second.txt", "third.txt")
    assert tuple(item.source_id for item in reopened.list_documents()) == _ordered_source_ids(tmp_path, "second.txt", "third.txt")


def test_remove_source_rejects_advisory_lock_contender_and_releases_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    descriptor = os.open(root / ".knowledge-base.lock", os.O_RDWR)
    module._acquire_advisory_lock(descriptor)
    try:
        with pytest.raises(KnowledgeBasePersistenceError):
            kb.remove_source(_source_id(tmp_path, "first.txt"))
    finally:
        module._release_advisory_lock(descriptor)
        os.close(descriptor)

    real_save = module.SQLiteKnowledgeSnapshotStore.save
    monkeypatch.setattr(
        module.SQLiteKnowledgeSnapshotStore,
        "save",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failure")),
    )
    with pytest.raises(KnowledgeBasePersistenceError):
        kb.remove_source(_source_id(tmp_path, "first.txt"))
    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", real_save)
    other = KnowledgeBase.open(str(root))
    other.remove_source(_source_id(tmp_path, "first.txt"))
