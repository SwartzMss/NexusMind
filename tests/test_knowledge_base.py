from __future__ import annotations

from dataclasses import FrozenInstanceError
import errno
import os
from pathlib import Path
import sqlite3

import pytest
import nexusmind.knowledge_base as knowledge_base_module

from nexusmind import (
    Document,
    DocumentVersion,
    InMemoryChunkIndex,
    KnowledgeBase,
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBaseLimits,
    KnowledgeBasePersistenceError,
    KnowledgeBaseStatus,
    KnowledgeSnapshot,
    KnowledgeSnapshotStoreError,
    KnowledgeSource,
    KnowledgeSourceType,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    SQLiteKnowledgeSnapshotStore,
)
from nexusmind.knowledge_base_manifest import KnowledgeBaseManifest, write_manifest


def _write_coordination_file(root: Path) -> None:
    root.joinpath(".knowledge-base.lock").write_bytes(b"\0")


def _write_fixture(
    root: Path,
    *,
    registration: LocalFileSourceConfig | LocalDirectorySourceConfig,
    source_type: KnowledgeSourceType | str,
    content: str = "知识图谱支持语义检索",
) -> None:
    root.mkdir()
    _write_coordination_file(root)
    write_manifest(
        root / "manifest.json",
        KnowledgeBaseManifest(
            knowledge_base_id="fixture",
            display_name="Fixture",
            sources=(registration,),
        ),
        KnowledgeBaseLimits(),
    )
    document = Document(
        source_id=registration.source_id,
        logical_path="guide.txt",
        content=content,
    )
    SQLiteKnowledgeSnapshotStore(root / "knowledge.db").save(
        KnowledgeSnapshot(
            sources=(
                KnowledgeSource(
                    source_id=registration.source_id,
                    source_type=source_type,
                    display_name="Docs",
                ),
            ),
            documents=(document,),
            document_versions=(
                DocumentVersion.from_document(
                    document,
                    created_at="2026-08-27T00:00:00.000000Z",
                    sync_context="fixture",
                ),
            ),
        )
    )


def test_create_new_and_existing_empty_directories_with_exact_layout(tmp_path: Path) -> None:
    for root in (tmp_path / "new", tmp_path / "empty"):
        if root.name == "empty":
            root.mkdir()
        kb = KnowledgeBase.create(
            str(root), knowledge_base_id="security", display_name="Security"
        )
        assert set(item.name for item in root.iterdir()) == {
            "manifest.json",
            "knowledge.db",
            ".knowledge-base.lock",
        }
        assert root.joinpath(".knowledge-base.lock").read_bytes() == b"\0"
        assert root.joinpath("manifest.json").read_bytes() == (
            b'{"display_name":"Security","format_version":"1",'
            b'"knowledge_base_id":"security","sources":[]}\n'
        )
        assert kb.status() == KnowledgeBaseStatus(
            knowledge_base_id="security",
            display_name="Security",
            registered_source_count=0,
            canonical_source_count=0,
            document_count=0,
        )


def test_create_rejects_invalid_roots_and_non_text_paths(tmp_path: Path) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    nonempty.joinpath("mine.txt").write_text("keep", encoding="utf-8")
    file_root = tmp_path / "file"
    file_root.write_text("keep", encoding="utf-8")
    symlink = tmp_path / "link"
    symlink.symlink_to(nonempty, target_is_directory=True)

    for root in (nonempty, file_root, symlink):
        with pytest.raises(KnowledgeBasePersistenceError):
            KnowledgeBase.create(str(root), knowledge_base_id="kb")
    with pytest.raises(KnowledgeBaseConfigError):
        KnowledgeBase.create(tmp_path / "path-object", knowledge_base_id="kb")  # type: ignore[arg-type]
    assert nonempty.joinpath("mine.txt").read_text(encoding="utf-8") == "keep"
    assert file_root.read_text(encoding="utf-8") == "keep"


def test_open_reopens_identity_and_rejects_missing_or_corrupt_layout(tmp_path: Path) -> None:
    root = tmp_path / "base"
    KnowledgeBase.create(str(root), knowledge_base_id="kb", display_name="Name").close()
    assert KnowledgeBase.open(str(root)).status().knowledge_base_id == "kb"
    assert KnowledgeBase.open(str(root)).status().display_name == "Name"

    for missing in ("manifest.json", "knowledge.db"):
        broken = tmp_path / f"missing-{missing}"
        KnowledgeBase.create(str(broken), knowledge_base_id="kb").close()
        broken.joinpath(missing).unlink()
        with pytest.raises(KnowledgeBasePersistenceError):
            KnowledgeBase.open(str(broken))

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    _write_coordination_file(corrupt)
    corrupt.joinpath("manifest.json").write_bytes(b"private document text")
    corrupt.joinpath("knowledge.db").write_bytes(b"private database bytes")
    with pytest.raises(KnowledgeBaseConfigError) as caught:
        KnowledgeBase.open(str(corrupt))
    assert "private" not in str(caught.value)
    assert str(corrupt) not in str(caught.value)


@pytest.mark.parametrize("payload", [b"", b"not a sqlite database: private-token"])
def test_open_rejects_invalid_database_without_mutating_it(
    tmp_path: Path, payload: bytes
) -> None:
    root = tmp_path / "invalid-database-private-path"
    root.mkdir()
    _write_coordination_file(root)
    write_manifest(
        root / "manifest.json",
        KnowledgeBaseManifest(knowledge_base_id="kb"),
        KnowledgeBaseLimits(),
    )
    database = root / "knowledge.db"
    database.write_bytes(payload)

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        KnowledgeBase.open(str(root))

    assert str(caught.value) == "canonical knowledge state is invalid"
    assert database.read_bytes() == payload
    assert "private-token" not in str(caught.value)
    assert str(root) not in str(caught.value)


@pytest.mark.parametrize("sentinel", ["missing", "wrong-version", "wrong-table"])
def test_open_rejects_sqlite_without_valid_store_sentinel_without_mutation(
    tmp_path: Path, sentinel: str
) -> None:
    root = tmp_path / f"private-sentinel-{sentinel}"
    root.mkdir()
    _write_coordination_file(root)
    write_manifest(
        root / "manifest.json",
        KnowledgeBaseManifest(knowledge_base_id="kb"),
        KnowledgeBaseLimits(),
    )
    database = root / "knowledge.db"
    with sqlite3.connect(database) as db:
        if sentinel == "wrong-version":
            db.execute(
                "CREATE TABLE knowledge_store_metadata "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute(
                "INSERT INTO knowledge_store_metadata (key, value) VALUES (?, ?)",
                ("schema_version", "private-version"),
            )
        elif sentinel == "wrong-table":
            db.execute("CREATE TABLE private_table (value TEXT)")
            db.execute("INSERT INTO private_table VALUES ('private-row')")
        else:
            db.execute("PRAGMA user_version = 7")
    before = database.read_bytes()
    with sqlite3.connect(database) as db:
        schema_before = db.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        KnowledgeBase.open(str(root))

    assert database.read_bytes() == before
    with sqlite3.connect(database) as db:
        assert db.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall() == schema_before
    message = str(caught.value)
    assert message == "canonical knowledge state is invalid"
    assert "private-version" not in message
    assert "private-row" not in message
    assert str(root) not in message


def test_create_failure_cleans_only_owned_artifacts_and_preserves_user_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "existing-empty"
    root.mkdir()

    class FailingStore:
        def __init__(self, path: Path) -> None:
            Path(path).write_bytes(b"owned database")
            Path(path).with_name("user-arrived.txt").write_text("keep", encoding="utf-8")
            raise KnowledgeSnapshotStoreError("private injected store detail")

    monkeypatch.setattr(knowledge_base_module, "SQLiteKnowledgeSnapshotStore", FailingStore)

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert "private" not in str(caught.value)
    assert str(root) not in str(caught.value)
    assert root.is_dir()
    assert root.joinpath("user-arrived.txt").read_text(encoding="utf-8") == "keep"
    assert not root.joinpath("knowledge.db").exists()
    assert not root.joinpath("manifest.json").exists()


def test_create_failure_removes_new_owned_directory_when_it_remains_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "new-root"

    class FailingStore:
        def __init__(self, path: Path) -> None:
            Path(path).write_bytes(b"owned database")
            raise KnowledgeSnapshotStoreError("private")

    monkeypatch.setattr(knowledge_base_module, "SQLiteKnowledgeSnapshotStore", FailingStore)

    with pytest.raises(KnowledgeBasePersistenceError):
        KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert not root.exists()


@pytest.mark.parametrize("collision_name", ["manifest.json", "knowledge.db"])
def test_create_does_not_overwrite_or_delete_layout_file_arriving_after_empty_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_name: str,
) -> None:
    root = tmp_path / f"race-{collision_name}"
    root.mkdir()
    private_bytes = f"private concurrent {collision_name}".encode()
    original_iterdir = Path.iterdir
    injected = False

    def racing_iterdir(path: Path):
        nonlocal injected
        if path == root and not injected:
            injected = True

            def empty_then_collide():
                if False:
                    yield path
                root.joinpath(collision_name).write_bytes(private_bytes)

            return empty_then_collide()
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", racing_iterdir)

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert root.joinpath(collision_name).read_bytes() == private_bytes
    assert "private concurrent" not in str(caught.value)
    assert str(root) not in str(caught.value)


def test_create_does_not_clobber_manifest_substituted_after_exclusive_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "manifest-substitution"
    root.mkdir()
    manifest_path = root / "manifest.json"
    private_bytes = b"private substituted manifest"
    original_identity = KnowledgeBase._file_identity
    injected = False
    replacement_identity: tuple[int, int] | None = None

    def substituting_identity(path: Path) -> tuple[int, int]:
        nonlocal injected, replacement_identity
        if path == manifest_path and not injected:
            injected = True
            manifest_path.unlink()
            manifest_path.write_bytes(private_bytes)
            actual = original_identity(path)
            replacement_identity = (actual[0], actual[1] + 1)
        if path == manifest_path and replacement_identity is not None:
            return replacement_identity
        return original_identity(path)

    monkeypatch.setattr(
        KnowledgeBase, "_file_identity", staticmethod(substituting_identity)
    )

    with pytest.raises(KnowledgeBasePersistenceError):
        KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert manifest_path.read_bytes() == private_bytes


def test_create_does_not_initialize_or_delete_substituted_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "database-substitution"
    root.mkdir()
    database_path = root / "knowledge.db"
    private_bytes = b""
    real_store = knowledge_base_module.SQLiteKnowledgeSnapshotStore
    injected = False

    class SubstitutingStore:
        def __init__(self, path: Path) -> None:
            nonlocal injected
            if not injected:
                injected = True
                if database_path.exists():
                    database_path.unlink()
                database_path.write_bytes(private_bytes)
            real_store(path)

    monkeypatch.setattr(
        knowledge_base_module, "SQLiteKnowledgeSnapshotStore", SubstitutingStore
    )

    with pytest.raises(KnowledgeBasePersistenceError):
        KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert database_path.read_bytes() == private_bytes


def test_create_rolls_back_published_database_when_post_link_stat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "post-link-stat"
    root.mkdir()
    database_path = root / "knowledge.db"
    original_identity = KnowledgeBase._file_identity
    failed = False

    def failing_identity(path: Path) -> tuple[int, int]:
        nonlocal failed
        if path == database_path and database_path.exists() and not failed:
            failed = True
            raise OSError("private post-link stat failure")
        return original_identity(path)

    monkeypatch.setattr(KnowledgeBase, "_file_identity", staticmethod(failing_identity))

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert not database_path.exists()
    assert "private post-link stat failure" not in str(caught.value)
    assert str(root) not in str(caught.value)


def test_create_falls_back_to_no_clobber_copy_when_hard_links_are_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "unsupported-link"

    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.ENOTSUP, "private unsupported-link detail")

    monkeypatch.setattr(knowledge_base_module.os, "link", unsupported_link)

    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert kb.status().knowledge_base_id == "kb"
    assert set(item.name for item in root.iterdir()) == {
        "manifest.json",
        "knowledge.db",
        ".knowledge-base.lock",
    }
    assert KnowledgeBase.open(str(root)).status().knowledge_base_id == "kb"


def test_unsupported_link_fallback_preserves_substituted_final_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "unsupported-link-substitution"
    root.mkdir()
    database_path = root / "knowledge.db"
    private_bytes = b"private fallback database"
    original_identity = KnowledgeBase._file_identity

    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.ENOTSUP, "unsupported")

    injected = False
    replacement_identity: tuple[int, int] | None = None

    def substituting_identity(path: Path) -> tuple[int, int]:
        nonlocal injected, replacement_identity
        if path == database_path and not injected:
            injected = True
            database_path.unlink()
            database_path.write_bytes(private_bytes)
            actual = original_identity(path)
            replacement_identity = (actual[0], actual[1] + 1)
        if path == database_path and replacement_identity is not None:
            return replacement_identity
        return original_identity(path)

    monkeypatch.setattr(knowledge_base_module.os, "link", unsupported_link)
    monkeypatch.setattr(
        KnowledgeBase, "_file_identity", staticmethod(substituting_identity)
    )

    with pytest.raises(KnowledgeBasePersistenceError):
        KnowledgeBase.create(str(root), knowledge_base_id="kb")

    assert database_path.read_bytes() == private_bytes


def test_open_rejects_file_and_symlink_roots(tmp_path: Path) -> None:
    file_root = tmp_path / "private-file-root"
    file_root.write_bytes(b"private")
    with pytest.raises(KnowledgeBasePersistenceError) as file_error:
        KnowledgeBase.open(str(file_root))
    assert str(file_root) not in str(file_error.value)

    real_root = tmp_path / "real"
    KnowledgeBase.create(str(real_root), knowledge_base_id="kb").close()
    link_root = tmp_path / "private-link-root"
    try:
        link_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(KnowledgeBasePersistenceError) as link_error:
        KnowledgeBase.open(str(link_root))
    assert str(link_root) not in str(link_error.value)


def test_close_is_idempotent_and_guards_all_inspection_methods(tmp_path: Path) -> None:
    kb = KnowledgeBase.create(str(tmp_path / "base"), knowledge_base_id="kb")
    kb.close()
    kb.close()
    for call in (
        kb.status,
        kb.list_sources,
        kb.list_documents,
        lambda: kb.search("query"),
    ):
        with pytest.raises(KnowledgeBaseClosedError):
            call()


def test_status_is_frozen(tmp_path: Path) -> None:
    value = KnowledgeBase.create(str(tmp_path / "base"), knowledge_base_id="kb").status()
    with pytest.raises(FrozenInstanceError):
        value.document_count = 2  # type: ignore[misc]


def test_open_restores_default_unicode_cjk_index_offline(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    registration = LocalFileSourceConfig(path=str(tmp_path / "docs.txt"))
    _write_fixture(
        root, registration=registration, source_type=KnowledgeSourceType.LOCAL_FILE
    )

    kb = KnowledgeBase.open(str(root))

    assert kb.status() == KnowledgeBaseStatus("fixture", "Fixture", 1, 1, 1)
    result = kb.search("语义检索")[0]
    assert result.document.content == "知识图谱支持语义检索"
    assert result.hit.matched_terms == ("语义", "义检", "检索")


def test_open_accepts_unsynchronized_registration(tmp_path: Path) -> None:
    root = tmp_path / "unsynchronized"
    root.mkdir()
    _write_coordination_file(root)
    write_manifest(
        root / "manifest.json",
        KnowledgeBaseManifest(
            knowledge_base_id="kb",
            sources=(
                LocalFileSourceConfig(path=str(tmp_path / "docs.txt")),
            ),
        ),
        KnowledgeBaseLimits(),
    )
    SQLiteKnowledgeSnapshotStore(root / "knowledge.db")

    assert KnowledgeBase.open(str(root)).status() == KnowledgeBaseStatus(
        "kb", None, 1, 0, 0
    )


def test_open_restores_more_than_collection_default_source_limit(tmp_path: Path) -> None:
    root = tmp_path / "many-sources"
    root.mkdir()
    _write_coordination_file(root)
    registrations = tuple(
        LocalFileSourceConfig(path=str(tmp_path / f"{number}.txt"))
        for number in range(101)
    )
    write_manifest(
        root / "manifest.json",
        KnowledgeBaseManifest(knowledge_base_id="kb", sources=registrations),
        KnowledgeBaseLimits(),
    )
    SQLiteKnowledgeSnapshotStore(root / "knowledge.db").save(
        KnowledgeSnapshot(
            sources=tuple(
                KnowledgeSource(
                    source_id=item.source_id,
                    source_type=KnowledgeSourceType.LOCAL_FILE,
                    display_name=item.source_id,
                )
                for item in registrations
            ),
            documents=(),
        )
    )

    status = KnowledgeBase.open(str(root)).status()

    assert status.registered_source_count == 101
    assert status.canonical_source_count == 101


def test_open_redacts_injected_index_factory_and_restore_failures(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    KnowledgeBase.create(str(empty), knowledge_base_id="kb").close()

    def failing_factory() -> InMemoryChunkIndex:
        raise RuntimeError("private factory token")

    with pytest.raises(KnowledgeBaseConfigError) as factory_error:
        KnowledgeBase.open(str(empty), index_factory=failing_factory)
    assert "private factory token" not in str(factory_error.value)
    assert str(empty) not in str(factory_error.value)

    fixture = tmp_path / "restore-private-path"
    registration = LocalFileSourceConfig(path=str(tmp_path / "docs.txt"))
    _write_fixture(
        fixture, registration=registration, source_type=KnowledgeSourceType.LOCAL_FILE,
        content="private document content",
    )

    class FailingRestoreIndex(InMemoryChunkIndex):
        def replace_document(self, document_id: str, chunks: tuple) -> None:
            raise RuntimeError("private restore token")

    with pytest.raises(KnowledgeBasePersistenceError) as restore_error:
        KnowledgeBase.open(str(fixture), index_factory=FailingRestoreIndex)
    message = str(restore_error.value)
    assert "private restore token" not in message
    assert "private document content" not in message
    assert str(fixture) not in message


def test_open_redacts_injected_store_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store-private-path"
    KnowledgeBase.create(str(root), knowledge_base_id="kb").close()

    class FailingStore:
        def __init__(self, path: Path) -> None:
            raise KnowledgeSnapshotStoreError("private store token")

    monkeypatch.setattr(knowledge_base_module, "SQLiteKnowledgeSnapshotStore", FailingStore)

    with pytest.raises(KnowledgeBasePersistenceError) as caught:
        KnowledgeBase.open(str(root))
    assert "private store token" not in str(caught.value)
    assert str(root) not in str(caught.value)


def test_injected_index_factory_rebuilds_state_and_is_not_persisted(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    registration = LocalDirectorySourceConfig(path=str(tmp_path / "docs"))
    _write_fixture(
        root,
        registration=registration,
        source_type=KnowledgeSourceType.LOCAL_DIRECTORY,
        content="factory searchable",
    )
    calls: list[InMemoryChunkIndex] = []

    def factory() -> InMemoryChunkIndex:
        index = InMemoryChunkIndex()
        calls.append(index)
        return index

    before = root.joinpath("manifest.json").read_bytes()
    kb = KnowledgeBase.open(str(root), index_factory=factory)

    assert len(calls) == 2
    assert kb.search("searchable")
    assert root.joinpath("manifest.json").read_bytes() == before


@pytest.mark.parametrize(
    ("registration", "source_type"),
    [
        (None, KnowledgeSourceType.LOCAL_FILE),
        ("file", KnowledgeSourceType.LOCAL_DIRECTORY),
        ("directory", KnowledgeSourceType.LOCAL_FILE),
    ],
)
def test_open_rejects_orphan_and_type_conflicting_canonical_sources(
    tmp_path: Path,
    registration: str | None,
    source_type: KnowledgeSourceType,
) -> None:
    root = tmp_path / "fixture"
    if registration == "directory":
        config = LocalDirectorySourceConfig(path=str(tmp_path / "docs"))
    else:
        config = LocalFileSourceConfig(path=str(tmp_path / "docs.txt"))
    _write_fixture(root, registration=config, source_type=source_type)
    if registration is None:
        write_manifest(
            root / "manifest.json",
            KnowledgeBaseManifest(knowledge_base_id="fixture"),
            KnowledgeBaseLimits(),
        )

    with pytest.raises(KnowledgeBasePersistenceError, match="incoherent"):
        KnowledgeBase.open(str(root))
