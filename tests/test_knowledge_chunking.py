from __future__ import annotations

from nexusmind import Document
from nexusmind.knowledge_chunking import TextChunker


def _document(content: str) -> Document:
    return Document(source_id="docs", logical_path="notes.txt", content=content)


def test_short_document_produces_one_exact_chunk() -> None:
    document = _document("short")

    chunks = TextChunker(chunk_size=10, overlap=2).chunk(document)

    assert [(chunk.start_offset, chunk.end_offset, chunk.content) for chunk in chunks] == [(0, 5, "short")]
    assert chunks[0].document_id == document.document_id


def test_exact_boundary_produces_one_chunk_without_empty_trailer() -> None:
    chunks = TextChunker(chunk_size=4, overlap=1).chunk(_document("abcd"))

    assert [(chunk.start_offset, chunk.end_offset, chunk.content) for chunk in chunks] == [(0, 4, "abcd")]


def test_multi_chunk_document_preserves_overlap_and_source_slices() -> None:
    document = _document("abcdefghij")

    chunks = TextChunker(chunk_size=4, overlap=1).chunk(document)

    assert [(chunk.start_offset, chunk.end_offset, chunk.content) for chunk in chunks] == [
        (0, 4, "abcd"),
        (3, 7, "defg"),
        (6, 10, "ghij"),
    ]
    assert all(
        chunk.content == document.content[chunk.start_offset : chunk.end_offset]
        for chunk in chunks
    )


def test_empty_document_produces_no_chunks() -> None:
    assert TextChunker(chunk_size=4, overlap=1).chunk(_document("")) == ()


def test_unicode_offsets_are_python_character_offsets() -> None:
    document = _document("你🙂好ab")

    chunks = TextChunker(chunk_size=3, overlap=1).chunk(document)

    assert [(chunk.start_offset, chunk.end_offset, chunk.content) for chunk in chunks] == [
        (0, 3, "你🙂好"),
        (2, 5, "好ab"),
    ]


def test_repeated_chunking_is_deterministic() -> None:
    document = _document("abcdefghij")
    chunker = TextChunker(chunk_size=4, overlap=1)

    assert chunker.chunk(document) == chunker.chunk(document)


def test_same_document_and_configuration_have_stable_chunk_ids() -> None:
    first = _document("abcdefghij")
    second = _document("abcdefghij")
    chunker = TextChunker(chunk_size=4, overlap=1)

    assert [chunk.chunk_id for chunk in chunker.chunk(first)] == [
        chunk.chunk_id for chunk in chunker.chunk(second)
    ]


def test_changed_document_content_does_not_reuse_chunk_identity() -> None:
    original = _document("abcd")
    changed = _document("abce")
    chunker = TextChunker(chunk_size=4, overlap=0)

    assert chunker.chunk(original)[0].chunk_id != chunker.chunk(changed)[0].chunk_id


def test_changed_configuration_changes_chunk_identity() -> None:
    document = _document("abcd")

    without_overlap = TextChunker(chunk_size=4, overlap=0).chunk(document)[0]
    with_overlap = TextChunker(chunk_size=4, overlap=1).chunk(document)[0]

    assert without_overlap.chunk_id != with_overlap.chunk_id
