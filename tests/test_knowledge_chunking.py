from __future__ import annotations

import pytest

from nexusmind import Chunk, ChunkLimitError, Document, StructureAwareChunker, TextChunker


def _document(content: str) -> Document:
    return Document(source_id="docs", logical_path="notes.txt", content=content)


def test_chunking_contracts_are_available_from_package_root() -> None:
    chunk = TextChunker(chunk_size=10, overlap=1).chunk(_document("short"))[0]

    assert isinstance(chunk, Chunk)


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


@pytest.mark.parametrize("chunk_size", [True, 1.5, "4"])
def test_chunk_size_rejects_non_integer_values(chunk_size: object) -> None:
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        TextChunker(chunk_size=chunk_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunk_size_must_be_positive(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        TextChunker(chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [True, 1.5, "1"])
def test_overlap_rejects_non_integer_values(overlap: object) -> None:
    with pytest.raises(TypeError, match="overlap must be an integer"):
        TextChunker(chunk_size=4, overlap=overlap)  # type: ignore[arg-type]


def test_overlap_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        TextChunker(chunk_size=4, overlap=-1)


def test_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        TextChunker(chunk_size=4, overlap=4)


@pytest.mark.parametrize("max_chunks", [True, 1.5, "2"])
def test_max_chunks_rejects_non_integer_values(max_chunks: object) -> None:
    with pytest.raises(TypeError, match="max_chunks must be an integer"):
        TextChunker(max_chunks=max_chunks)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_chunks", [0, -1])
def test_max_chunks_must_be_positive(max_chunks: int) -> None:
    with pytest.raises(ValueError, match="max_chunks must be greater than zero"):
        TextChunker(max_chunks=max_chunks)


def test_chunk_count_limit_fails_without_returning_partial_output() -> None:
    chunker = TextChunker(chunk_size=4, overlap=1, max_chunks=2)

    with pytest.raises(ChunkLimitError, match="document requires 3 chunks; limit is 2"):
        chunker.chunk(_document("abcdefghij"))


def test_chunker_rejects_non_document_input() -> None:
    with pytest.raises(TypeError, match="document must be a Document"):
        TextChunker().chunk("not a document")  # type: ignore[arg-type]


def test_chunk_defaults_keep_legacy_constructor_and_expose_empty_structure() -> None:
    chunk = Chunk("doc", "chunk", "body", 0, 4)

    assert chunk.heading_path == ()
    assert chunk.section_title == ""
    assert chunk.source_location == ""
    assert chunk.retrieval_text == "body"


def test_structural_chunk_retrieval_text_contains_heading_context() -> None:
    chunk = Chunk(
        "doc",
        "chunk",
        "body",
        0,
        4,
        heading_path=("Android Security", "Binder"),
        section_title="Binder",
        source_location="notes.md:L3",
    )

    assert chunk.retrieval_text == "Android Security > Binder\nbody"


@pytest.mark.parametrize("heading_path", [("",), ["Binder"], (1,)])
def test_chunk_rejects_malformed_heading_path(heading_path: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Chunk(
            "doc",
            "chunk",
            "body",
            0,
            4,
            heading_path=heading_path,  # type: ignore[arg-type]
        )


def test_structure_chunker_preserves_nested_heading_metadata_and_locations() -> None:
    content = (
        "Preamble.\n\n"
        "# Android Security\n\nSecurity overview.\n\n"
        "## Binder\n\nBinder manages IPC.\n\n"
        "### oneway\n\nThe oneway transaction uses pid zero.\n\n"
        "```text\n# not a heading\n```\n"
    )
    document = _document(content)

    chunks = StructureAwareChunker(chunk_size=70, overlap=0).chunk(document)

    binder = next(chunk for chunk in chunks if chunk.content.startswith("## Binder"))
    oneway = next(chunk for chunk in chunks if chunk.content.startswith("### oneway"))
    fenced = next(chunk for chunk in chunks if "# not a heading" in chunk.content)

    assert binder.heading_path == ("Android Security", "Binder")
    assert binder.section_title == "Binder"
    assert binder.source_location == "notes.txt:L5"
    assert oneway.heading_path == ("Android Security", "Binder", "oneway")
    assert oneway.section_title == "oneway"
    assert oneway.source_location == "notes.txt:L9"
    assert "not a heading" not in fenced.heading_path
    assert all(
        chunk.content == document.content[chunk.start_offset : chunk.end_offset]
        for chunk in chunks
    )


def test_structure_chunker_keeps_heading_with_following_content() -> None:
    content = (
        "## IAM Token Validation\n\nIAM_Master validates the token.\n\n"
        "### Expiration\n\nExpired tokens are renewed."
    )
    chunks = StructureAwareChunker(chunk_size=70, overlap=10).chunk(_document(content))

    assert chunks[0].content.startswith("## IAM Token Validation")
    assert "IAM_Master validates the token." in chunks[0].content
    assert chunks[1].content.startswith("### Expiration")
    assert "Expired tokens are renewed." in chunks[1].content


def test_structure_chunker_protects_code_table_and_list_blocks() -> None:
    content = (
        "# Guide\n\nIntro.\n\n"
        "```python\nprint('marker')\nprint('done')\n```\n\n"
        "| Key | Value |\n| --- | --- |\n| token | active |\n\n"
        "- first item\n- second item\n- third item\n"
    )
    chunks = StructureAwareChunker(chunk_size=75, overlap=10).chunk(_document(content))

    assert any("```python\nprint('marker')\nprint('done')\n```" in chunk.content for chunk in chunks)
    assert any("| Key | Value |\n| --- | --- |\n| token | active |" in chunk.content for chunk in chunks)
    assert any("- first item\n- second item\n- third item" in chunk.content for chunk in chunks)


def test_structure_chunker_oversized_block_is_bounded_exact_and_deterministic() -> None:
    document = _document("# Large\n\n```text\n" + "abcdefghij " * 20 + "\n```")
    chunker = StructureAwareChunker(chunk_size=50, overlap=5, max_chunks=20)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second
    assert all(len(chunk.content) <= 50 for chunk in first)
    assert all(chunk.content == document.content[chunk.start_offset:chunk.end_offset] for chunk in first)
    assert [chunk.chunk_id for chunk in first] != [
        chunk.chunk_id for chunk in TextChunker(chunk_size=50, overlap=5).chunk(document)
    ]


def test_structure_chunker_packs_small_paragraphs_and_enforces_limit() -> None:
    document = _document("one\n\ntwo\n\nthree\n\nfour")
    assert len(StructureAwareChunker(chunk_size=30, overlap=5).chunk(document)) == 1

    with pytest.raises(ChunkLimitError):
        StructureAwareChunker(chunk_size=8, overlap=1, max_chunks=1).chunk(document)


def test_structure_chunker_does_not_emit_tiny_parent_heading_chunk() -> None:
    document = _document("# Parent\n\n## Child\n\nChild details.")

    chunks = StructureAwareChunker(chunk_size=50, overlap=5).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content


def test_structure_chunker_shrinks_first_body_span_to_keep_heading_context() -> None:
    document = _document("# Heading\n\n" + "body words " * 20)

    chunks = StructureAwareChunker(chunk_size=60, overlap=5).chunk(document)

    assert chunks[0].content.startswith("# Heading\n\nbody")
    assert chunks[0].content != "# Heading\n\n"
    assert min(len(chunk.content) for chunk in chunks) >= 30
    assert all(len(chunk.content) <= 60 for chunk in chunks)
    assert all(chunk.content == document.content[chunk.start_offset:chunk.end_offset] for chunk in chunks)


def test_structure_chunker_fails_during_pathological_overlap_expansion() -> None:
    document = _document("x" * (1024 * 1024 - 1))

    with pytest.raises(ChunkLimitError, match="during fallback"):
        StructureAwareChunker(
            chunk_size=1000,
            overlap=999,
            max_chunks=10,
        ).chunk(document)


@pytest.mark.parametrize(
    "content",
    [
        "```text\nalpha beta gamma\ndelta epsilon zeta\neta theta iota\n```",
        "- alpha beta gamma\n- delta epsilon zeta\n- eta theta iota\n",
        "| Key | Value |\n| --- | --- |\n| alpha | beta gamma |\n| delta | epsilon zeta |\n",
    ],
)
def test_oversized_structures_prefer_line_boundaries_over_later_spaces(content: str) -> None:
    chunks = StructureAwareChunker(chunk_size=32, overlap=0).chunk(_document(content))

    assert len(chunks) > 1
    assert all(chunk.content.endswith("\n") for chunk in chunks[:-1])


@pytest.mark.parametrize(
    "content",
    [
        "# " + "single-heading-token " * 8,
        "# " + "parent-heading-token " * 4 + "\n\n## " + "child-heading-token " * 5,
    ],
)
def test_oversized_headings_are_bounded_exact_and_deterministic(content: str) -> None:
    document = _document(content)
    chunker = StructureAwareChunker(chunk_size=50, overlap=5)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert first == second
    assert len(first) > 1
    assert all(len(chunk.content) <= 50 for chunk in first)
    assert all(chunk.content == content[chunk.start_offset:chunk.end_offset] for chunk in first)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
