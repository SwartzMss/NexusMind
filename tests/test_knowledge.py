from __future__ import annotations

import hashlib

import pytest

from nexusmind import Document, KnowledgeSource, KnowledgeSourceType, compute_content_hash
from nexusmind.knowledge import stable_document_id


def test_source_has_stable_identity_and_keeps_generic_metadata() -> None:
    metadata = {"directory": "docs", "labels": ["product"]}
    source = KnowledgeSource(
        source_id="source-1",
        source_type=KnowledgeSourceType.LOCAL_DIRECTORY,
        display_name="Product docs",
        logical_location="docs",
        metadata=metadata,
    )
    metadata["labels"].append("changed")

    renamed_source = KnowledgeSource(
        source_id="source-1",
        source_type=KnowledgeSourceType.LOCAL_DIRECTORY,
        display_name="Renamed product docs",
        logical_location="docs",
    )
    assert source.source_id == renamed_source.source_id
    assert source.metadata == {"directory": "docs", "labels": ["product"]}


def test_source_requires_explicit_id_but_allows_missing_logical_location() -> None:
    with pytest.raises(TypeError):
        KnowledgeSource(source_type="custom", display_name="Docs")  # type: ignore[call-arg]

    explicit = KnowledgeSource(source_id="source-1", source_type="custom", display_name="Docs")
    assert explicit.source_id == "source-1"
    assert explicit.logical_location is None


def test_document_identity_uses_source_and_logical_path() -> None:
    first = Document(source_id="docs", logical_path="README.md", content="# NexusMind")
    second = Document(source_id="docs", logical_path="README.md", content="# NexusMind")

    assert first.document_id == second.document_id
    assert first.identity == ("docs", "README.md")


def test_stable_id_encoding_preserves_part_boundaries() -> None:
    assert stable_document_id("a\x00b", "c") != stable_document_id("a", "b\x00c")


def test_document_tracks_source_content_and_metadata() -> None:
    document = Document(
        source_id="local-docs",
        logical_path="guides/intro.md",
        content="hello",
        content_type="text/markdown",
        metadata={"title": "Introduction"},
    )

    assert document.source_id == "local-docs"
    assert document.content_type == "text/markdown"
    assert document.metadata == {"title": "Introduction"}
    assert document.content_hash == compute_content_hash("hello")


def test_document_content_hash_detects_changes_for_same_logical_document() -> None:
    original = Document(source_id="docs", logical_path="notes.txt", content="one")
    unchanged = Document(source_id="docs", logical_path="notes.txt", content="one")
    changed = Document(source_id="docs", logical_path="notes.txt", content="two")

    assert not original.has_content_changed(unchanged)
    assert original.has_content_changed(changed)
    with pytest.raises(ValueError):
        original.has_content_changed(Document(source_id="other", logical_path="notes.txt", content="two"))


def test_content_hash_is_utf8_sha256_and_is_derived() -> None:
    assert compute_content_hash("你好") == hashlib.sha256("你好".encode("utf-8")).hexdigest()
    with pytest.raises(TypeError):
        Document(source_id="docs", logical_path="a.txt", content="one", content_hash="0" * 64)  # type: ignore[call-arg]


def test_document_id_is_derived_from_identity() -> None:
    document = Document(source_id="docs", logical_path="a.txt", content="one")

    assert document.document_id == stable_document_id("docs", "a.txt")
    with pytest.raises(TypeError):
        Document(source_id="docs", logical_path="a.txt", content="one", document_id="custom")  # type: ignore[call-arg]


def test_document_rejects_non_text_content() -> None:
    with pytest.raises(TypeError):
        Document(source_id="docs", logical_path="a.bin", content=b"binary")  # type: ignore[arg-type]
