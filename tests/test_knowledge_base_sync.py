from dataclasses import FrozenInstanceError
import os
from pathlib import Path
from threading import Event, Thread

import pytest

import nexusmind.knowledge_base as module
from nexusmind import (
    KnowledgeBase,
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBaseLimits,
    KnowledgeBasePersistenceError,
    KnowledgeBaseSourceError,
    KnowledgeBaseStatus,
    KnowledgeSyncResult,
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
    registration_started = False

    def factory() -> InMemoryChunkIndex:
        nonlocal factory_calls
        if registration_started:
            raise AssertionError("registration constructed retrieval runtime")
        factory_calls += 1
        return InMemoryChunkIndex()

    kb = KnowledgeBase.create(
        str(tmp_path / "kb"), knowledge_base_id="kb", index_factory=factory
    )
    calls_before = factory_calls
    registration_started = True
    writes = 0
    real_write = module.write_manifest

    def recording_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "write_manifest", recording_write)
    monkeypatch.setattr(
        module, "LocalFileAdapter", lambda *args, **kwargs: pytest.fail("adapter constructed")
    )
    monkeypatch.setattr(
        module,
        "LocalDirectoryAdapter",
        lambda *args, **kwargs: pytest.fail("adapter constructed"),
    )
    kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert writes == 1
    assert factory_calls == calls_before


def test_failed_registration_write_does_not_swap_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    real_write = module.write_manifest

    def fail_write(*args, **kwargs):
        raise KnowledgeBasePersistenceError("private controlled failure")

    monkeypatch.setattr(module, "write_manifest", fail_write)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert "private" not in str(caught.value)
    assert kb.list_sources() == ()
    monkeypatch.setattr(module, "write_manifest", real_write)
    kb.add_source(LocalFileSourceConfig(source_id="two", path="two.txt"))
    assert tuple(item.source_id for item in kb.list_sources()) == ("two",)


def test_registration_bounds_leave_memory_and_manifest_unchanged(tmp_path: Path) -> None:
    cases = (
        (
            KnowledgeBaseLimits(max_sources=1),
            LocalFileSourceConfig(source_id="second", path="second.txt"),
            LocalFileSourceConfig(source_id="first", path="first.txt"),
        ),
        (
            KnowledgeBaseLimits(max_source_id_chars=3),
            LocalFileSourceConfig(source_id="four", path="file.txt"),
            None,
        ),
        (
            KnowledgeBaseLimits(max_path_chars=len(str(tmp_path.resolve()))),
            LocalFileSourceConfig(source_id="path", path="child.txt"),
            None,
        ),
    )
    for index, (limits, rejected, initial) in enumerate(cases):
        root = tmp_path / f"bounded-{index}"
        kb = KnowledgeBase.create(str(root), knowledge_base_id="kb", limits=limits)
        if initial is not None:
            kb.add_source(initial)
        sources_before = kb.list_sources()
        bytes_before = root.joinpath("manifest.json").read_bytes()

        with pytest.raises(KnowledgeBaseConfigError):
            kb.add_source(rejected)

        assert kb.list_sources() == sources_before
        assert root.joinpath("manifest.json").read_bytes() == bytes_before

    seed = tmp_path / "manifest-bound-seed"
    seed_kb = KnowledgeBase.create(str(seed), knowledge_base_id="kb")
    initial_size = len(seed.joinpath("manifest.json").read_bytes())
    seed_kb.close()
    root = tmp_path / "manifest-bound"
    kb = KnowledgeBase.create(
        str(root),
        knowledge_base_id="kb",
        limits=KnowledgeBaseLimits(max_manifest_bytes=initial_size),
    )
    bytes_before = root.joinpath("manifest.json").read_bytes()
    with pytest.raises(KnowledgeBaseConfigError):
        kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))
    assert kb.list_sources() == ()
    assert root.joinpath("manifest.json").read_bytes() == bytes_before


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


def test_failed_unregister_write_preserves_memory_and_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="unused", path="unused.txt"))
    sources_before = kb.list_sources()
    bytes_before = root.joinpath("manifest.json").read_bytes()

    def fail_write(*args, **kwargs):
        raise KnowledgeBasePersistenceError("private path")

    monkeypatch.setattr(module, "write_manifest", fail_write)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.unregister_source("unused")

    assert "private" not in str(caught.value)
    assert kb.list_sources() == sources_before
    assert root.joinpath("manifest.json").read_bytes() == bytes_before


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
    assert kb.status() == KnowledgeBaseStatus("kb", None, 2, 0, 0)

    results = kb.sync()
    assert tuple(item.source_id for item in results) == ("a-dir", "z-file")
    documents = kb.list_documents()
    assert tuple(item.source_id for item in documents) == ("a-dir", "z-file")
    documents[0].metadata["mutated"] = True
    assert "mutated" not in kb.list_documents()[0].metadata

    reopened = KnowledgeBase.open(str(root))
    assert reopened.status() == KnowledgeBaseStatus("kb", None, 2, 2, 2)
    assert tuple(item.source_id for item in reopened.list_documents()) == ("a-dir", "z-file")
    assert reopened.search("knowledge")


def test_real_file_sync_returns_exact_result_type_and_counters(tmp_path: Path) -> None:
    path = tmp_path / "one.txt"
    path.write_text("one short document", encoding="utf-8")
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="one", path=str(path)))

    assert kb.sync_source("one") == KnowledgeSyncResult(
        source_id="one",
        documents_added=1,
        documents_updated=0,
        documents_unchanged=0,
        documents_removed=0,
        chunks_indexed=1,
    )
    result = kb.sync_source("one")
    assert type(result) is KnowledgeSyncResult
    assert result == KnowledgeSyncResult("one", 0, 0, 1, 0, 0)


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


def test_sync_source_constructs_and_loads_only_selected_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected.txt"
    selected.write_text("selected", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    kb.add_source(LocalDirectorySourceConfig(source_id="other", path=str(other)))
    kb.add_source(LocalFileSourceConfig(source_id="selected", path=str(selected)))
    real_adapter = module.LocalFileAdapter
    events: list[str] = []

    class RecordingAdapter:
        def __init__(self, path, *, source_id):
            events.append(f"construct:{source_id}")
            self._delegate = real_adapter(path, source_id=source_id)

        def source(self):
            events.append("source:selected")
            return self._delegate.source()

        def load_documents(self):
            events.append("load:selected")
            return self._delegate.load_documents()

    monkeypatch.setattr(module, "LocalFileAdapter", RecordingAdapter)
    monkeypatch.setattr(
        module,
        "LocalDirectoryAdapter",
        lambda *args, **kwargs: pytest.fail("unselected adapter constructed"),
    )

    kb.sync_source("selected")

    assert events == ["construct:selected", "source:selected", "load:selected"]


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
    stored_before = original(root / "knowledge.db").load()
    save_calls = 0

    class SaveFailingStore:
        def __init__(self, path):
            self._store = original(path)

        def load(self):
            return self._store.load()

        def save(self, snapshot):
            nonlocal save_calls
            save_calls += 1
            if save_calls == 1:
                raise RuntimeError("private provider and path")
            self._store.save(snapshot)

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", SaveFailingStore)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.sync()
    assert "private" not in str(caught.value)
    assert save_calls == 2
    assert kb.list_documents() == before
    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", original)
    assert original(root / "knowledge.db").load() == stored_before


def test_closed_guards_registration_inspection_and_sync(tmp_path: Path) -> None:
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    kb.close()

    operations = (
        lambda: kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt")),
        lambda: kb.unregister_source("one"),
        kb.sync,
        lambda: kb.sync_source("one"),
        kb.list_sources,
    )
    for operation in operations:
        with pytest.raises(KnowledgeBaseClosedError):
            operation()


def test_list_sources_is_sorted_tuple_of_frozen_configs(tmp_path: Path) -> None:
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="z", path="z.txt"))
    kb.add_source(LocalDirectorySourceConfig(source_id="a", path="a"))

    sources = kb.list_sources()
    assert type(sources) is tuple
    assert tuple(item.source_id for item in sources) == ("a", "z")
    with pytest.raises(FrozenInstanceError):
        sources[0].path = "mutated"  # type: ignore[misc]


def test_all_mutations_fail_closed_while_cross_handle_lock_is_held(
    tmp_path: Path,
) -> None:
    root = tmp_path / "kb"
    path = tmp_path / "one.txt"
    path.write_text("one", encoding="utf-8")
    first = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    first.add_source(LocalFileSourceConfig(source_id="one", path=str(path)))
    contender = KnowledgeBase.open(str(root))
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    lock_path = root / ".knowledge-base.lock"
    descriptor = os.open(lock_path, os.O_RDWR)
    module._acquire_advisory_lock(descriptor)

    operations = (
        lambda: contender.add_source(
            LocalFileSourceConfig(source_id="two", path="two.txt")
        ),
        lambda: contender.unregister_source("one"),
        contender.sync,
        lambda: contender.sync_source("one"),
    )
    try:
        for operation in operations:
            with pytest.raises(KnowledgeBasePersistenceError) as caught:
                operation()
            assert str(root) not in str(caught.value)
    finally:
        module._release_advisory_lock(descriptor)
        os.close(descriptor)

    assert lock_path.read_bytes() == b"\0"
    assert contender.list_sources() == first.list_sources()
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before


def test_unlocked_stale_coordination_file_does_not_block_mutation(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    lock_path = root / ".knowledge-base.lock"
    lock_path.write_bytes(b"stale-looking lock contents")

    kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert tuple(item.source_id for item in kb.list_sources()) == ("one",)
    assert lock_path.exists()


def test_advisory_lock_is_released_when_holder_closes_descriptor(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    descriptor = os.open(root / ".knowledge-base.lock", os.O_RDWR)
    module._acquire_advisory_lock(descriptor)
    os.close(descriptor)

    kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert tuple(item.source_id for item in kb.list_sources()) == ("one",)


def test_open_requires_real_coordination_file(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-lock"
    KnowledgeBase.create(str(missing_root), knowledge_base_id="kb").close()
    missing_root.joinpath(".knowledge-base.lock").unlink()
    with pytest.raises(KnowledgeBasePersistenceError):
        KnowledgeBase.open(str(missing_root))

    symlink_root = tmp_path / "symlink-lock"
    KnowledgeBase.create(str(symlink_root), knowledge_base_id="kb").close()
    lock_path = symlink_root / ".knowledge-base.lock"
    lock_path.unlink()
    lock_path.symlink_to(symlink_root / "manifest.json")
    with pytest.raises(KnowledgeBasePersistenceError):
        KnowledgeBase.open(str(symlink_root))


def test_mutation_does_not_recreate_missing_coordination_file(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    root.joinpath(".knowledge-base.lock").unlink()

    with pytest.raises(KnowledgeBasePersistenceError):
        kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert not root.joinpath(".knowledge-base.lock").exists()
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before


def test_coordination_file_replacement_during_acquisition_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    lock_path = root / ".knowledge-base.lock"
    displaced = root / ".displaced-lock"
    manifest_before = root.joinpath("manifest.json").read_bytes()
    snapshot_before = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    real_acquire = module._acquire_advisory_lock

    def replace_after_acquire(descriptor: int) -> None:
        real_acquire(descriptor)
        lock_path.replace(displaced)
        lock_path.write_bytes(b"replacement")

    monkeypatch.setattr(module, "_acquire_advisory_lock", replace_after_acquire)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.add_source(LocalFileSourceConfig(source_id="one", path="one.txt"))

    assert str(root) not in str(caught.value)
    assert root.joinpath("manifest.json").read_bytes() == manifest_before
    assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == snapshot_before


def test_instance_mutations_are_serialized_without_lost_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb = KnowledgeBase.create(str(tmp_path / "kb"), knowledge_base_id="kb")
    first_entered = Event()
    release_first = Event()
    real_write = module.write_manifest
    calls = 0

    def blocking_write(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(module, "write_manifest", blocking_write)
    errors: list[BaseException] = []

    def register(source_id: str) -> None:
        try:
            kb.add_source(LocalFileSourceConfig(source_id=source_id, path=f"{source_id}.txt"))
        except BaseException as exc:
            errors.append(exc)

    first = Thread(target=register, args=("a",))
    second = Thread(target=register, args=("b",))
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    assert calls == 1
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert calls == 2
    assert tuple(item.source_id for item in kb.list_sources()) == ("a", "b")


def test_serialized_stale_handles_refresh_manifest_before_mutation(tmp_path: Path) -> None:
    root = tmp_path / "kb"
    first = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    second = KnowledgeBase.open(str(root))

    first.add_source(LocalFileSourceConfig(source_id="a", path="a.txt"))
    second.add_source(LocalFileSourceConfig(source_id="b", path="b.txt"))

    reopened = KnowledgeBase.open(str(root))
    assert tuple(item.source_id for item in reopened.list_sources()) == ("a", "b")


def test_search_wraps_private_runtime_failure_but_preserves_input_validation(
    tmp_path: Path,
) -> None:
    controlled = KnowledgeBaseSourceError("controlled search failure")

    class ExplodingIndex(InMemoryChunkIndex):
        def search(self, query: str, *, limit: int = 10):
            if type(query) is not str:
                return super().search(query, limit=limit)
            if query == "controlled":
                raise controlled
            raise RuntimeError(f"provider-secret for query {query}")

    kb = KnowledgeBase.create(
        str(tmp_path / "kb"), knowledge_base_id="kb", index_factory=ExplodingIndex
    )

    with pytest.raises(TypeError):
        kb.search(42)  # type: ignore[arg-type]
    with pytest.raises(KnowledgeBaseSourceError) as preserved:
        kb.search("controlled")
    assert preserved.value is not controlled
    assert str(preserved.value) == "unable to search knowledge base"
    assert preserved.value.__cause__ is controlled
    assert "controlled search failure" not in str(preserved.value)
    with pytest.raises(KnowledgeBaseSourceError) as caught:
        kb.search("secret-query")
    assert "provider-secret" not in str(caught.value)
    assert "secret-query" not in str(caught.value)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_sync_and_reopen_reuse_injected_cloneable_index_factory(tmp_path: Path) -> None:
    instances: list[InMemoryChunkIndex] = []

    class AlternativeIndex(InMemoryChunkIndex):
        pass

    def factory() -> AlternativeIndex:
        index = AlternativeIndex()
        instances.append(index)
        return index

    source = tmp_path / "source.txt"
    source.write_text("alternative runtime searchable", encoding="utf-8")
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(
        str(root), knowledge_base_id="kb", index_factory=factory
    )
    kb.add_source(LocalFileSourceConfig(source_id="source", path=str(source)))
    calls_before_sync = len(instances)

    kb.sync()

    assert len(instances) > calls_before_sync
    assert kb.search("searchable")
    kb.close()
    calls_before_open = len(instances)
    reopened = KnowledgeBase.open(str(root), index_factory=factory)
    assert len(instances) > calls_before_open
    assert reopened.search("searchable")


@pytest.mark.parametrize("source_id", [None, 1, ""])
def test_sync_source_rejects_malformed_source_id(
    tmp_path: Path, source_id: object
) -> None:
    kb = KnowledgeBase.create(str(tmp_path / f"kb-{source_id}"), knowledge_base_id="kb")
    with pytest.raises(KnowledgeBaseConfigError):
        kb.sync_source(source_id)  # type: ignore[arg-type]


def test_create_relative_root_remains_stable_after_working_directory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("stable relative create", encoding="utf-8")
    monkeypatch.chdir(first_cwd)
    kb = KnowledgeBase.create("kb", knowledge_base_id="kb")
    monkeypatch.chdir(second_cwd)

    kb.add_source(LocalFileSourceConfig(source_id="source", path=str(source)))
    kb.sync()

    assert kb.status().document_count == 1
    assert first_cwd.joinpath("kb", "manifest.json").is_file()
    assert not second_cwd.joinpath("kb").exists()


def test_open_relative_root_remains_stable_after_working_directory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("stable relative open searchable", encoding="utf-8")
    root = first_cwd / "kb"
    original = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    original.add_source(LocalFileSourceConfig(source_id="source", path=str(source)))
    original.sync()
    original.close()
    monkeypatch.chdir(first_cwd)
    reopened = KnowledgeBase.open("kb")
    monkeypatch.chdir(second_cwd)

    reopened.sync_source("source")

    assert reopened.search("searchable")
    assert reopened.status().document_count == 1


def test_ambiguous_save_is_compensated_to_exact_old_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "kb"
    source = tmp_path / "source.txt"
    source.write_text("old canonical", encoding="utf-8")
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="source", path=str(source)))
    kb.sync()
    old_snapshot = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    source.write_text("new private canonical", encoding="utf-8")
    real_store = module.SQLiteKnowledgeSnapshotStore
    calls = []

    class AmbiguousStore:
        def __init__(self, path):
            self._store = real_store(path)

        def load(self):
            return self._store.load()

        def save(self, snapshot):
            calls.append(snapshot)
            self._store.save(snapshot)
            if len(calls) == 1:
                raise RuntimeError("private ambiguous commit")

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", AmbiguousStore)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.sync_source("source")

    assert "private" not in str(caught.value)
    assert len(calls) == 2
    assert calls[0].documents[0].content == "new private canonical"
    assert calls[1] == old_snapshot
    assert kb.list_documents() == old_snapshot.documents
    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", real_store)
    assert real_store(root / "knowledge.db").load() == old_snapshot
    assert KnowledgeBase.open(str(root)).list_documents() == old_snapshot.documents


def test_failed_ambiguous_save_compensation_poisons_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "kb"
    source = tmp_path / "source.txt"
    source.write_text("old", encoding="utf-8")
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")
    kb.add_source(LocalFileSourceConfig(source_id="source", path=str(source)))
    kb.sync()
    old_snapshot = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    source.write_text("new private provider value", encoding="utf-8")
    real_store = module.SQLiteKnowledgeSnapshotStore
    calls = []

    class UnrecoverableStore:
        def __init__(self, path):
            self._store = real_store(path)

        def load(self):
            return self._store.load()

        def save(self, snapshot):
            calls.append(snapshot)
            if len(calls) == 1:
                self._store.save(snapshot)
                raise RuntimeError("private original ambiguity")
            raise RuntimeError("private compensation failure")

    monkeypatch.setattr(module, "SQLiteKnowledgeSnapshotStore", UnrecoverableStore)
    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        kb.sync()

    assert str(caught.value) == "unable to recover canonical knowledge state"
    assert len(calls) == 2
    assert calls[1] == old_snapshot
    with pytest.raises(KnowledgeBaseClosedError):
        kb.status()
    with pytest.raises(KnowledgeBaseClosedError):
        kb.list_documents()
