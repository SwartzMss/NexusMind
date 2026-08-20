from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from nexusmind import (
    Document,
    InMemoryChunkIndex,
    KnowledgeBase,
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBaseLimits,
    KnowledgeBasePersistenceError,
    KnowledgeBaseStatus,
    KnowledgeSnapshot,
    KnowledgeSource,
    KnowledgeSourceType,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    SQLiteKnowledgeSnapshotStore,
)
from nexusmind.knowledge_base_manifest import KnowledgeBaseManifest, write_manifest


def _write_fixture(
    root: Path,
    *,
    registration: LocalFileSourceConfig | LocalDirectorySourceConfig,
    source_type: KnowledgeSourceType | str,
    content: str = "知识图谱支持语义检索",
) -> None:
    root.mkdir()
    write_manifest(
        root / "manifest.json",
        KnowledgeBaseManifest(
            knowledge_base_id="fixture",
            display_name="Fixture",
            sources=(registration,),
        ),
        KnowledgeBaseLimits(),
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
            documents=(
                Document(
                    source_id=registration.source_id,
                    logical_path="guide.txt",
                    content=content,
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
        assert set(item.name for item in root.iterdir()) == {"manifest.json", "knowledge.db"}
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
    corrupt.joinpath("manifest.json").write_bytes(b"private document text")
    corrupt.joinpath("knowledge.db").write_bytes(b"private database bytes")
    with pytest.raises(KnowledgeBaseConfigError) as caught:
        KnowledgeBase.open(str(corrupt))
    assert "private" not in str(caught.value)
    assert str(corrupt) not in str(caught.value)


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
    registration = LocalFileSourceConfig(source_id="docs", path=str(tmp_path / "docs.txt"))
    _write_fixture(
        root, registration=registration, source_type=KnowledgeSourceType.LOCAL_FILE
    )

    kb = KnowledgeBase.open(str(root))

    assert kb.status() == KnowledgeBaseStatus("fixture", "Fixture", 1, 1, 1)
    result = kb.search("语义检索")[0]
    assert result.document.content == "知识图谱支持语义检索"
    assert result.hit.matched_terms == ("语义", "义检", "检索")


def test_injected_index_factory_rebuilds_state_and_is_not_persisted(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    registration = LocalDirectorySourceConfig(source_id="docs", path=str(tmp_path / "docs"))
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
        config = LocalDirectorySourceConfig(source_id="docs", path=str(tmp_path / "docs"))
    else:
        config = LocalFileSourceConfig(source_id="docs", path=str(tmp_path / "docs.txt"))
    _write_fixture(root, registration=config, source_type=source_type)
    if registration is None:
        write_manifest(
            root / "manifest.json",
            KnowledgeBaseManifest(knowledge_base_id="fixture"),
            KnowledgeBaseLimits(),
        )

    with pytest.raises(KnowledgeBasePersistenceError, match="incoherent"):
        KnowledgeBase.open(str(root))
