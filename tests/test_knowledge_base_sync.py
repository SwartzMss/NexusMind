from pathlib import Path

import pytest

import nexusmind.knowledge_base as module
from nexusmind import (
    KnowledgeBase,
    KnowledgeBaseConfigError,
    KnowledgeBasePersistenceError,
    KnowledgeBaseSourceError,
    InMemoryChunkIndex,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    SQLiteKnowledgeSnapshotStore,
)


def test_registration_is_sorted_persistent_and_does_not_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "kb"
    file_path = tmp_path / "private.txt"
    file_path.write_text("secret", encoding="utf-8")
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")

    class ForbiddenAdapter:
        def __init__(self, *args, **kwargs):
            raise AssertionError("registration constructed an adapter")

    monkeypatch.setattr(module, "LocalFileAdapter", ForbiddenAdapter)
    kb.add_source(LocalFileSourceConfig(source_id="z", path=str(file_path)))
    kb.add_source(LocalDirectorySourceConfig(source_id="a", path="relative-dir"))

    assert tuple(item.source_id for item in kb.list_sources()) == ("a", "z")
    assert all(Path(item.path).is_absolute() for item in kb.list_sources())
    assert b"secret" not in root.joinpath("manifest.json").read_bytes()
    assert KnowledgeBase.open(str(root)).list_sources() == kb.list_sources()
    with pytest.raises(KnowledgeBaseSourceError):
        kb.add_source(LocalFileSourceConfig(source_id="z", path=str(file_path)))
    with pytest.raises(KnowledgeBaseConfigError):
        kb.add_source(object())  # type: ignore[arg-type]


def test_registration_writes_once_and_does_not_construct_retrieval_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory_calls = 0

    def factory() -> InMemoryChunkIndex:
        nonlocal factory_calls
        factory_calls += 1
        return InMemoryChunkIndex()

    kb = KnowledgeBase.create(
        str(tmp_path / "kb"), knowledge_base_id="kb", index_factory=factory
    )
    calls_before = factory_calls
    writes = 0
    real_write = module.write_manifest

    def recording_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "write_manifest", recording_write)
    kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert writes == 1
    assert factory_calls == calls_before


def test_failed_registration_write_does_not_swap_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")

    def fail_write(*args, **kwargs):
        raise KnowledgeBasePersistenceError("controlled failure")

    monkeypatch.setattr(module, "write_manifest", fail_write)
    with pytest.raises(KnowledgeBasePersistenceError):
        kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert kb.list_sources() == ()


def test_unregister_fails_closed_for_unknown_or_synchronized_sources(tmp_path: Path) -> None:
    path = tmp_path / "one.txt"
    path.write_text("alpha", encoding="utf-8")
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="one", path=str(path)))
    with pytest.raises(KnowledgeBaseSourceError):
        kb.unregister_source("missing")
    kb.sync_source("one")
    with pytest.raises(KnowledgeBaseSourceError):
        kb.unregister_source("one")


def test_unregister_unsynchronized_source_is_persistent(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="unused", path="unused.txt"))

    kb.unregister_source("unused")

    assert kb.list_sources() == ()
    assert KnowledgeBase.open(str(root)).list_sources() == ()


def test_sync_orders_sources_persists_and_reopens_searchable_state(tmp_path: Path) -> None:
    directory = tmp_path / "docs"
    directory.mkdir()
    directory.joinpath("guide.md").write_text("directory knowledge", encoding="utf-8")
    file_path = tmp_path / "note.txt"
    file_path.write_text("file knowledge", encoding="utf-8")
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="z-file", path=str(file_path)))
    kb.add_source(LocalDirectorySourceConfig(source_id="a-dir", path=str(directory)))

    results = kb.sync()
    assert tuple(item.source_id for item in results) == ("a-dir", "z-file")
    documents = kb.list_documents()
    assert tuple(item.source_id for item in documents) == ("a-dir", "z-file")
    documents[0].metadata["mutated"] = True
    assert "mutated" not in kb.list_documents()[0].metadata

    reopened = KnowledgeBase.open(str(root))
    assert tuple(item.source_id for item in reopened.list_documents()) == ("a-dir", "z-file")
    assert reopened.search("knowledge")


def test_sync_source_preserves_other_canonical_sources(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first old", encoding="utf-8")
    second.write_text("second stable", encoding="utf-8")
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="first", path=str(first)))
    kb.add_source(LocalFileSourceConfig(source_id="second", path=str(second)))
    kb.sync()
    first.write_text("first new", encoding="utf-8")

    result = kb.sync_source("first")
    assert result.source_id == "first"
    assert {item.content for item in kb.list_documents()} == {"first new", "second stable"}
    with pytest.raises(KnowledgeBaseSourceError):
        kb.sync_source("unknown")


def test_later_source_failure_rolls_back_staging_and_database(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("old", encoding="utf-8")
    missing = tmp_path / "missing.txt"
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="a-good", path=str(good)))
    kb.sync_source("a-good")
    kb.add_source(LocalFileSourceConfig(source_id="z-missing", path=str(missing)))
    good.write_text("new", encoding="utf-8")
    before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()

    with pytest.raises(KnowledgeBaseSourceError) as caught:
        kb.sync()

    assert str(tmp_path) not in str(caught.value)
    assert kb.list_documents()[0].content == "old"
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == before


def test_empty_sync_skips_store_and_save_failure_keeps_live_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    original = module.SQLiteKnowledgeSnapshotStore

    class FailingStore:
        def __init__(self, path):
            raise AssertionError("empty sync touched sqlite")

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", FailingStore)
    assert kb.sync() == ()
    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", original)

    path = tmp_path / "doc.txt"
    path.write_text("content", encoding="utf-8")
    kb.add_source(LocalFileSourceConfig(source_id="doc", path=str(path)))
    before = kb.list_documents()

    class SaveFailingStore:
        def __init__(self, path):
            pass

        def save(self, snapshot):
            raise RuntimeError("private provider and path")

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", SaveFailingStore)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.sync()
    assert "private" not in str(caught.value)
    assert kb.list_documents() == before
