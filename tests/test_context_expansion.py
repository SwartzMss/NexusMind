from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexusmind import (
    Chunk,
    Document,
    KnowledgeSearchResult,
    KnowledgeSource,
    SearchHit,
    expand_context_candidates,
)


@dataclass(frozen=True)
class _Fixture:
    source: KnowledgeSource
    document: Document
    chunks: tuple[Chunk, ...]


def _fixture() -> _Fixture:
    content = "definition\nanchor\ncaveat\nsibling"
    document = Document("docs", "guide.md", content)
    source = KnowledgeSource(
        source_id="docs",
        source_type="test",
        display_name="Docs",
    )
    chunks = (
        Chunk(
            document.document_id,
            "definition",
            "definition",
            0,
            10,
            ("Binder",),
            "Binder",
            "guide.md:L1",
        ),
        Chunk(
            document.document_id,
            "anchor",
            "anchor",
            11,
            17,
            ("Binder",),
            "Binder",
            "guide.md:L1",
        ),
        Chunk(
            document.document_id,
            "caveat",
            "caveat",
            18,
            24,
            ("Binder",),
            "Binder",
            "guide.md:L1",
        ),
        Chunk(
            document.document_id,
            "sibling",
            "sibling",
            25,
            32,
            ("SELinux",),
            "SELinux",
            "guide.md:L8",
        ),
    )
    return _Fixture(source, document, chunks)


def _anchor(fixture: _Fixture, chunk_id: str, score: float) -> KnowledgeSearchResult:
    chunk = next(item for item in fixture.chunks if item.chunk_id == chunk_id)
    return KnowledgeSearchResult(
        source=fixture.source,
        document=fixture.document,
        hit=SearchHit(chunk, score, ("anchor",)),
    )


def test_expansion_emits_ranked_anchors_before_same_section_neighbors() -> None:
    fixture = _fixture()

    expanded = expand_context_candidates(
        (_anchor(fixture, "anchor", 5.0),),
        chunk_catalog={fixture.document.document_id: fixture.chunks},
    )

    assert [item.hit.chunk.chunk_id for item in expanded.candidates] == [
        "anchor",
        "definition",
        "caveat",
    ]
    assert expanded.anchor_chunk_ids == ("anchor",)
    assert expanded.expanded_chunk_ids == ("definition", "caveat")
    assert expanded.expanded_document_ids == (fixture.document.document_id,)
    assert expanded.section_boundary_skips == 0
    assert expanded.candidates[1].hit.score == 0.0
    assert expanded.candidates[1].hit.matched_terms == ()


def test_expansion_skips_adjacent_sibling_section() -> None:
    fixture = _fixture()
    expanded = expand_context_candidates(
        (_anchor(fixture, "caveat", 5.0),),
        chunk_catalog={fixture.document.document_id: fixture.chunks},
    )

    assert [item.hit.chunk.chunk_id for item in expanded.candidates] == [
        "caveat",
        "anchor",
    ]
    assert expanded.section_boundary_skips == 1


def test_expansion_deduplicates_neighbors_shared_by_multiple_anchors() -> None:
    fixture = _fixture()
    expanded = expand_context_candidates(
        (
            _anchor(fixture, "anchor", 5.0),
            _anchor(fixture, "caveat", 4.0),
        ),
        chunk_catalog={fixture.document.document_id: fixture.chunks},
    )

    assert [item.hit.chunk.chunk_id for item in expanded.candidates] == [
        "anchor",
        "caveat",
        "definition",
    ]
    neighbor = expanded.candidates[-1]
    assert neighbor.document.content[
        neighbor.hit.chunk.start_offset : neighbor.hit.chunk.end_offset
    ] == "definition"


def test_expansion_rejects_missing_catalog_entries() -> None:
    fixture = _fixture()
    with pytest.raises(ValueError, match="chunk catalog"):
        expand_context_candidates(
            (_anchor(fixture, "anchor", 5.0),),
            chunk_catalog={},
        )
