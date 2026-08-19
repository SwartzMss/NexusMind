from __future__ import annotations

import math

import pytest

from nexusmind import (
    Chunk,
    EmbeddingVector,
    HybridBackendCoherenceError,
    HybridChunkIndex,
    HybridChunkIndexError,
    HybridChunkIndexLimitError,
    HybridChunkIndexLimits,
    InMemoryChunkIndex,
    InMemorySemanticChunkIndex,
    SearchHit,
)


def _chunk(chunk_id: str, content: str | None = None, document_id: str = "doc") -> Chunk:
    value = chunk_id if content is None else content
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=value,
        start_offset=0,
        end_offset=len(value),
    )


class _ScriptedIndex:
    def __init__(self, hits: tuple[SearchHit, ...] = ()) -> None:
        self.hits = hits
        self.search_calls: list[tuple[str, int]] = []
        self.mutations: list[tuple[str, tuple[object, ...]]] = []
        self.fail_mutation: str | None = None
        self.fail_search = False

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("add", chunks)

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("replace_document", document_id, chunks)

    def remove_document(self, document_id: str) -> None:
        self._mutate("remove_document", document_id)

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        self.search_calls.append((query, limit))
        if self.fail_search:
            raise RuntimeError("private search failure")
        return self.hits[:limit]

    def clone(self) -> "_ScriptedIndex":
        clone = _ScriptedIndex(self.hits)
        clone.search_calls = self.search_calls.copy()
        clone.mutations = self.mutations.copy()
        clone.fail_mutation = self.fail_mutation
        clone.fail_search = self.fail_search
        return clone

    def _mutate(self, method: str, *args: object) -> None:
        self.mutations.append((method, args))
        if self.fail_mutation == method:
            raise RuntimeError("private mutation failure")


def _hybrid(
    lexical: _ScriptedIndex,
    semantic: _ScriptedIndex,
    **kwargs: object,
) -> HybridChunkIndex:
    return HybridChunkIndex(
        lexical_index_factory=lambda: lexical,
        semantic_index_factory=lambda: semantic,
        **kwargs,
    )


@pytest.mark.parametrize("field", HybridChunkIndexLimits.__dataclass_fields__)
@pytest.mark.parametrize("value", [True, 0, -1])
def test_hybrid_limits_require_positive_plain_integers(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        HybridChunkIndexLimits(**{field: value})


@pytest.mark.parametrize("value", [True, 0, -1])
def test_hybrid_rrf_k_and_candidate_depth_require_positive_integers(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _hybrid(_ScriptedIndex(), _ScriptedIndex(), rrf_k=value)
    with pytest.raises((TypeError, ValueError)):
        _hybrid(_ScriptedIndex(), _ScriptedIndex(), candidate_depth=value)


def test_hybrid_rrf_uses_ranks_preserves_lexical_terms_and_ties_by_id() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    lexical = _ScriptedIndex(
        (
            SearchHit(a, 999.0, ("exact",)),
            SearchHit(b, 1.0, ("term",)),
        )
    )
    semantic = _ScriptedIndex(
        (
            SearchHit(c, 0.99),
            SearchHit(b, -1.0),
        )
    )
    index = _hybrid(lexical, semantic, rrf_k=60, candidate_depth=2)

    hits = index.search("query", limit=3)

    assert [hit.chunk.chunk_id for hit in hits] == ["b", "a", "c"]
    assert hits[0].score == pytest.approx(2 / 62)
    assert hits[1].score == pytest.approx(1 / 61)
    assert hits[2].score == pytest.approx(1 / 61)
    assert hits[0].matched_terms == ("term",)
    assert hits[1].matched_terms == ("exact",)
    assert hits[2].matched_terms == ()


def test_hybrid_candidate_depth_exceeds_final_limit_and_is_bounded() -> None:
    lexical = _ScriptedIndex((SearchHit(_chunk("one"), 1.0),))
    semantic = _ScriptedIndex()
    index = _hybrid(
        lexical,
        semantic,
        limits=HybridChunkIndexLimits(max_results=2, max_candidates_per_backend=4),
        candidate_depth=4,
    )

    index.search("query", limit=1)

    assert lexical.search_calls == [("query", 4)]
    assert semantic.search_calls == [("query", 4)]
    with pytest.raises(HybridChunkIndexLimitError, match="max_results"):
        index.search("query", limit=3)
    assert len(lexical.search_calls) == 1


def test_hybrid_rejects_duplicate_and_cross_backend_conflicting_chunks() -> None:
    chunk = _chunk("same")
    duplicate = _hybrid(
        _ScriptedIndex((SearchHit(chunk, 2.0), SearchHit(chunk, 1.0))),
        _ScriptedIndex(),
        candidate_depth=2,
    )
    with pytest.raises(HybridBackendCoherenceError, match="duplicate"):
        duplicate.search("query")

    conflict = _hybrid(
        _ScriptedIndex((SearchHit(chunk, 1.0),)),
        _ScriptedIndex((SearchHit(_chunk("same", "different"), 1.0),)),
        candidate_depth=1,
    )
    with pytest.raises(HybridBackendCoherenceError, match="disagree"):
        conflict.search("query")


@pytest.mark.parametrize(
    "hits",
    [
        [],
        (object(),),
        (SearchHit(_chunk("x"), math.nan),),
        (SearchHit(_chunk("x"), 1.0, (1,)),),
    ],
)
def test_hybrid_rejects_malformed_child_hits(hits: object) -> None:
    index = _hybrid(_ScriptedIndex(hits), _ScriptedIndex())  # type: ignore[arg-type]

    with pytest.raises(HybridBackendCoherenceError):
        index.search("query")


def test_hybrid_mutations_commit_both_clones_and_clone_is_independent() -> None:
    lexical = _ScriptedIndex()
    semantic = _ScriptedIndex()
    index = _hybrid(lexical, semantic)
    chunks = (_chunk("one"),)

    index.add(chunks)
    index.replace_document("doc", chunks)
    clone = index.clone()
    clone.remove_document("doc")

    assert [call[0] for call in index._lexical.mutations] == [
        "add",
        "replace_document",
    ]
    assert [call[0] for call in index._semantic.mutations] == [
        "add",
        "replace_document",
    ]
    assert [call[0] for call in clone._lexical.mutations][-1] == "remove_document"
    assert len(index._lexical.mutations) == 2


@pytest.mark.parametrize("failing_backend", ["lexical", "semantic"])
def test_hybrid_mutation_failure_preserves_both_committed_children(
    failing_backend: str,
) -> None:
    index = _hybrid(_ScriptedIndex(), _ScriptedIndex())
    index.add((_chunk("old"),))
    old_lexical = index._lexical
    old_semantic = index._semantic
    getattr(index, f"_{failing_backend}").fail_mutation = "replace_document"

    with pytest.raises(HybridChunkIndexError, match="mutation failed") as caught:
        index.replace_document("doc", (_chunk("new"),))

    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is not None
    assert index._lexical is old_lexical
    assert index._semantic is old_semantic


@pytest.mark.parametrize("failing_backend", ["lexical", "semantic"])
def test_hybrid_search_fails_closed(failing_backend: str) -> None:
    index = _hybrid(_ScriptedIndex(), _ScriptedIndex())
    getattr(index, f"_{failing_backend}").fail_search = True

    with pytest.raises(HybridChunkIndexError, match="search failed") as caught:
        index.search("private query")

    assert "private" not in str(caught.value)


class _Provider:
    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(
            EmbeddingVector((1.0, 0.0))
            if "alpha" in text
            else EmbeddingVector((0.0, 1.0))
            for text in texts
        )

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector((1.0, 0.0))


def test_hybrid_composes_real_lexical_and_semantic_indexes() -> None:
    index = HybridChunkIndex(
        lexical_index_factory=InMemoryChunkIndex,
        semantic_index_factory=lambda: InMemorySemanticChunkIndex(
            embedding_provider=_Provider()
        ),
    )
    index.add((_chunk("alpha", "alpha exact"), _chunk("beta", "other")))

    hits = index.search("alpha")

    assert hits[0].chunk.chunk_id == "alpha"
    assert hits[0].matched_terms == ("alpha",)
