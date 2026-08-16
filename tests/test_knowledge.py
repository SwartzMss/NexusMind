from __future__ import annotations

from datetime import datetime, timezone
import hashlib

import pytest

from nexusmind import Document, KnowledgeSource, KnowledgeSourceType, compute_content_hash


def test_source_has_stable_identity_and_keeps_generic_metadata() -> None:
    metadata = {"directory": "docs", "labels": ["product"]}
    source = KnowledgeSource(
        source_type=KnowledgeSourceType.LOCAL_DIRECTORY,
        display_name="Product docs",
        logical_location="docs",
        metadata=metadata,
    )
    metadata["labels"].append("changed")

    same_source = KnowledgeSource(
        source_type=KnowledgeSourceType.LOCAL_DIRECTORY,
        display_name="Product docs",
        logical_location="docs",
    )
    assert source.source_id == same_source.source_id
    assert source.id == source.source_id
    assert source.metadata == {"directory": "docs", "labels": ["product"]}


def test_document_identity_uses_source_and_logical_path_not_absolute_path() -> None:
    first = Document(source_id="docs", logical_path="README.md", content="# NexusMind")
    second = Document(source_id="docs", logical_path="README.md", content="# NexusMind")
    moved_source = Document(source_id="docs", logical_path="README.md", content="# NexusMind", metadata={"absolute_path": "D:\\other\\README.md"})

    assert first.document_id == second.document_id == moved_source.document_id
    assert first.identity == ("docs", "README.md")
    assert first.source == "docs"
    assert first.path == "README.md"


def test_document_tracks_source_content_metadata_and_timestamps() -> None:
    source = KnowledgeSource(source_id="local-docs", source_type="local_directory", display_name="Docs")
    imported_at = datetime(2026, 8, 16, tzinfo=timezone.utc)
    document = Document(
        source=source,
        path="guides/intro.md",
        content="hello",
        mime_type="text/markdown",
        metadata={"title": "Introduction"},
        imported_at=imported_at,
        updated_at=imported_at,
    )

    assert document.source_id == source.source_id
    assert document.content_type == "text/markdown"
    assert document.mime_type == "text/markdown"
    assert document.metadata == {"title": "Introduction"}
    assert document.content_hash == compute_content_hash("hello")
    assert document.imported_at == imported_at
    assert document.updated_at == imported_at


def test_document_content_hash_detects_changes_for_same_logical_document() -> None:
    original = Document(source_id="docs", logical_path="notes.txt", content="one")
    unchanged = Document(source_id="docs", logical_path="notes.txt", content="one")
    changed = Document(source_id="docs", logical_path="notes.txt", content="two")

    assert not original.has_content_changed(unchanged)
    assert original.has_content_changed(changed)
    assert original.content_changed_from(changed)
    with pytest.raises(ValueError):
        original.has_content_changed(Document(source_id="other", logical_path="notes.txt", content="two"))


def test_content_hash_is_utf8_sha256_and_rejects_inconsistent_values() -> None:
    assert compute_content_hash("你好") == hashlib.sha256("你好".encode("utf-8")).hexdigest()
    assert compute_content_hash(b"hello") == compute_content_hash("hello")
    with pytest.raises(ValueError):
        Document(source_id="docs", logical_path="a.txt", content="one", content_hash="0" * 64)
