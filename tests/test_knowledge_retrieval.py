from __future__ import annotations

import pytest

from nexusmind import (
    Chunk,
    ChunkIdentityConflictError,
    ChunkIndex,
    ChunkIndexLimitError,
    ChunkIndexLimits,
    DocumentReplacementError,
    InMemoryChunkIndex,
    SearchHit,
)


def _chunk(chunk_id: str, content: str, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


def test_retrieval_contracts_are_available_from_package_root() -> None:
    index: ChunkIndex = InMemoryChunkIndex()
    index.add((_chunk("chunk-1", "checkpoint resume"),))

    assert index.search("checkpoint") == (
        SearchHit(chunk=_chunk("chunk-1", "checkpoint resume"), score=1, matched_terms=("checkpoint",)),
    )


def test_empty_index_non_match_and_blank_query_return_no_hits() -> None:
    index = InMemoryChunkIndex()
    assert index.search("anything") == ()
    index.add((_chunk("chunk-1", "checkpoint"),))
    assert index.search("missing") == ()
    assert index.search(" \t\n") == ()


def test_multi_term_scoring_casefold_and_duplicate_terms() -> None:
    index = InMemoryChunkIndex()
    index.add(
        (
            _chunk("chunk-b", "Resume from CHECKPOINT"),
            _chunk("chunk-a", "checkpoint only"),
        )
    )

    hits = index.search("CHECKPOINT checkpoint resume")

    assert [(hit.chunk.chunk_id, hit.score, hit.matched_terms) for hit in hits] == [
        ("chunk-b", 2, ("checkpoint", "resume")),
        ("chunk-a", 1, ("checkpoint",)),
    ]


def test_ties_use_chunk_id_and_limit_is_applied() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("chunk-z", "term"), _chunk("chunk-a", "term")))

    assert [hit.chunk.chunk_id for hit in index.search("term", limit=1)] == ["chunk-a"]


def test_unicode_casefold_is_deterministic() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("chunk-1", "STRASSE 中文"), _chunk("chunk-2", "Straße")))

    first = index.search("straße 中文")

    assert [(hit.chunk.chunk_id, hit.score) for hit in first] == [("chunk-1", 2), ("chunk-2", 1)]
    assert index.search("straße 中文") == first


def test_exact_duplicate_add_is_idempotent() -> None:
    chunk = _chunk("chunk-1", "term")
    index = InMemoryChunkIndex()
    index.add((chunk, chunk))
    index.add((chunk,))

    assert len(index.search("term")) == 1


def test_conflicting_chunk_id_is_rejected_without_mutation() -> None:
    original = _chunk("chunk-1", "original")
    index = InMemoryChunkIndex()
    index.add((original,))

    with pytest.raises(ChunkIdentityConflictError):
        index.add((_chunk("chunk-1", "changed"),))

    assert index.search("original")[0].chunk == original
    assert index.search("changed") == ()


def test_remove_document_does_not_affect_another_document() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("chunk-1", "shared", "doc-1"), _chunk("chunk-2", "shared", "doc-2")))

    index.remove_document("doc-1")

    assert [hit.chunk.document_id for hit in index.search("shared")] == ["doc-2"]


def test_replace_document_removes_stale_chunks_and_empty_replacement_removes_all() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("old-a", "stale", "doc-1"), _chunk("old-b", "stale", "doc-1")))

    index.replace_document("doc-1", (_chunk("new-a", "fresh", "doc-1"),))

    assert index.search("stale") == ()
    assert [hit.chunk.chunk_id for hit in index.search("fresh")] == ["new-a"]
    index.replace_document("doc-1", ())
    assert index.search("fresh") == ()


def test_failed_replacement_is_atomic() -> None:
    old = _chunk("old", "old searchable", "doc-1")
    limits = ChunkIndexLimits(max_total_chars=len(old.content), max_chunks=5, max_chunks_per_document=5)
    index = InMemoryChunkIndex(limits=limits)
    index.add((old,))

    with pytest.raises(ChunkIndexLimitError):
        index.replace_document("doc-1", (_chunk("new", "content is too long", "doc-1"),))

    assert index.search("old")[0].chunk == old
    assert index.search("long") == ()


def test_replacement_rejects_reused_chunk_id_with_different_data() -> None:
    old = _chunk("same-id", "old", "doc-1")
    changed = _chunk("same-id", "new", "doc-1")
    index = InMemoryChunkIndex()
    index.add((old,))

    with pytest.raises(ChunkIdentityConflictError):
        index.replace_document("doc-1", (changed,))

    assert index.search("old")[0].chunk == old
    assert index.search("new") == ()


def test_replacement_rejects_mixed_documents_and_duplicate_ids() -> None:
    index = InMemoryChunkIndex()
    with pytest.raises(DocumentReplacementError, match="belong"):
        index.replace_document("doc-1", (_chunk("a", "a", "doc-1"), _chunk("b", "b", "doc-2")))
    duplicate = _chunk("a", "a", "doc-1")
    with pytest.raises(DocumentReplacementError, match="duplicate"):
        index.replace_document("doc-1", (duplicate, duplicate))


def test_query_and_result_limits_are_enforced() -> None:
    limits = ChunkIndexLimits(max_query_chars=10, max_query_terms=2, max_results=1)
    index = InMemoryChunkIndex(limits=limits)
    with pytest.raises(ChunkIndexLimitError, match="max_query_chars"):
        index.search("12345678901")
    with pytest.raises(ChunkIndexLimitError, match="max_query_terms"):
        index.search("a b c")
    with pytest.raises(ChunkIndexLimitError, match="max_results"):
        index.search("a", limit=2)


def test_index_count_content_and_per_document_limits_are_atomic() -> None:
    index = InMemoryChunkIndex(
        limits=ChunkIndexLimits(max_chunks=2, max_total_chars=4, max_chunks_per_document=1)
    )
    index.add((_chunk("a", "ab", "doc-1"),))
    with pytest.raises(ChunkIndexLimitError, match="max_chunks_per_document"):
        index.add((_chunk("b", "c", "doc-1"),))
    with pytest.raises(ChunkIndexLimitError, match="max_total_chars"):
        index.add((_chunk("b", "cde", "doc-2"),))
    index.add((_chunk("b", "cd", "doc-2"),))
    with pytest.raises(ChunkIndexLimitError, match="max_chunks"):
        index.add((_chunk("c", "", "doc-3"),))
    assert [hit.chunk.chunk_id for hit in index.search("ab cd", limit=1)] == ["a"]


@pytest.mark.parametrize("field", ChunkIndexLimits.__dataclass_fields__)
def test_limits_require_positive_plain_integers(field: str) -> None:
    with pytest.raises(TypeError):
        ChunkIndexLimits(**{field: True})
    with pytest.raises(ValueError):
        ChunkIndexLimits(**{field: 0})


def test_invalid_search_limit_is_controlled() -> None:
    index = InMemoryChunkIndex()
    with pytest.raises(TypeError):
        index.search("term", limit=True)
    with pytest.raises(ValueError):
        index.search("term", limit=0)
