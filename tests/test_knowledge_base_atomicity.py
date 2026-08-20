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


def _synced_pair(tmp_path: Path) -> tuple[KnowledgeBase, Path]:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha removable", encoding="utf-8")
    second.write_text("beta retained", encoding="utf-8")
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="first", path=str(first)))
    kb.add_source(LocalFileSourceConfig(source_id="second", path=str(second)))
    kb.sync()
    return kb, root


def test_remove_source_removes_registration_canonical_and_search_state(tmp_path: Path) -> None:
    kb, root = _synced_pair(tmp_path)

    kb.remove_source("first")

    assert tuple(item.source_id for item in kb.list_sources()) == ("second",)
    assert tuple(item.source_id for item in kb.list_documents()) == ("second",)
    assert kb.search("removable") == ()
    assert kb.search("retained")
    stored = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    assert tuple(item.source_id for item in stored.sources) == ("second",)
    assert tuple(item.source_id for item in stored.documents) == ("second",)
    reopened = KnowledgeBase.open(str(root))
    assert tuple(item.source_id for item in reopened.list_sources()) == ("second",)
    assert tuple(item.source_id for item in reopened.list_documents()) == ("second",)
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
    assert tables == {"knowledge_store_metadata", "sources", "documents"}
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
        kb.remove_source("first")

    assert tuple(item.source_id for item in calls[0].sources) == ("second",)
    assert calls[1] == snapshot_before
    assert "private" not in str(caught.value)
    assert tuple(item.source_id for item in kb.list_sources()) == ("first", "second")
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
        kb.remove_source("first")

    assert tuple(item.source_id for item in calls[0].sources) == ("second",)
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
        kb.remove_source("first")

    assert tuple(item.source_id for item in calls[0].sources) == ("second",)
    assert tuple(item.source_id for item in calls[1].sources) == ("first", "second")
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
        kb.remove_source("first")

    assert "private" not in str(caught.value)
    assert tuple(item.source_id for item in kb.list_sources()) == ("first", "second")
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert real_store(root / "knowledge.db").load() == snapshot_before

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", real_store)
    KnowledgeBase.open(str(root)).remove_source("first")


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

    def fail_manifest(*args, **kwargs):
        calls.append("manifest")
        raise RuntimeError("private manifest detail")

    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", recording_save)
    monkeypatch.setattr(module, "write_manifest", fail_manifest)

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.remove_source("first")

    assert calls[1] == "manifest"
    assert calls[2] == snapshot_before
    assert "private" not in str(caught.value)
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before
    assert tuple(item.source_id for item in kb.list_sources()) == ("first", "second")


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
        kb.remove_source("first")
    assert "private" not in str(caught.value)
    assert saves == 2
    for operation in (kb.status, kb.list_sources, kb.list_documents, lambda: kb.search("x"), kb.sync):
        with pytest.raises(KnowledgeBaseClosedError):
            operation()
    for operation in (
        lambda: kb.add_source(LocalFileSourceConfig(source_id="third", path="third.txt")),
        lambda: kb.unregister_source("second"),
        lambda: kb.remove_source("second"),
        lambda: kb.sync_source("second"),
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
    kb.add_source(LocalFileSourceConfig(source_id="third", path=str(third)))
    kb.sync_source("third")

    stale.remove_source("first")

    reopened = KnowledgeBase.open(str(root))
    assert tuple(item.source_id for item in reopened.list_sources()) == ("second", "third")
    assert tuple(item.source_id for item in reopened.list_documents()) == ("second", "third")


def test_remove_source_rejects_advisory_lock_contender_and_releases_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, root = _synced_pair(tmp_path)
    descriptor = os.open(root / ".knowledge-base.lock", os.O_RDWR)
    module._acquire_advisory_lock(descriptor)
    try:
        with pytest.raises(KnowledgeBasePersistenceError):
            kb.remove_source("first")
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
        kb.remove_source("first")
    monkeypatch.setattr(module.SQLiteKnowledgeSnapshotStore, "save", real_save)
    other = KnowledgeBase.open(str(root))
    other.remove_source("first")
