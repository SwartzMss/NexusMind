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
    RetrievalStage,
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


def test_hybrid_index_exposes_configured_search_capacity_across_clone() -> None:
    index = _hybrid(
        _ScriptedIndex(),
        _ScriptedIndex(),
        limits=HybridChunkIndexLimits(
            max_results=10,
            max_candidates_per_backend=10,
            max_fusion_entries=20,
        ),
        candidate_depth=10,
    )

    assert index.max_search_results == 10
    assert index.clone().max_search_results == 10


def test_hybrid_index_exposes_capacity_safe_for_two_backend_fusion() -> None:
    index = _hybrid(
        _ScriptedIndex(),
        _ScriptedIndex(),
        limits=HybridChunkIndexLimits(
            max_results=20,
            max_candidates_per_backend=10,
            max_fusion_entries=20,
        ),
        candidate_depth=10,
    )

    assert index.max_search_results == 10
    assert index.clone().max_search_results == 10


def test_hybrid_index_does_not_advertise_unsafe_fixed_candidate_depth() -> None:
    index = _hybrid(
        _ScriptedIndex(),
        _ScriptedIndex(),
        limits=HybridChunkIndexLimits(
            max_results=20,
            max_candidates_per_backend=10,
            max_fusion_entries=10,
        ),
        candidate_depth=10,
    )

    assert index.max_search_results is None


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


def test_hybrid_diagnose_emits_backend_and_fusion_trace() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    lexical = _ScriptedIndex(
        (SearchHit(a, 999.0, ("exact",)), SearchHit(b, 1.0, ("term",)))
    )
    semantic = _ScriptedIndex((SearchHit(c, 0.99), SearchHit(b, -1.0)))
    index = _hybrid(lexical, semantic, rrf_k=60, candidate_depth=2)

    diagnostics = index.diagnose("query", limit=2)

    assert [hit.chunk.chunk_id for hit in diagnostics.hits] == ["b", "a"]
    assert lexical.search_calls == [("query", 2)]
    assert semantic.search_calls == [("query", 2)]
    assert [candidate.stage for candidate in diagnostics.candidates] == [
        RetrievalStage.LEXICAL,
        RetrievalStage.LEXICAL,
        RetrievalStage.SEMANTIC,
        RetrievalStage.SEMANTIC,
        RetrievalStage.FUSION,
        RetrievalStage.FUSION,
    ]
    assert [candidate.rank for candidate in diagnostics.candidates] == [1, 2, 1, 2, 1, 2]
    assert [candidate.chunk.chunk_id for candidate in diagnostics.candidates] == [
        "a",
        "b",
        "c",
        "b",
        "b",
        "a",
    ]
    assert [candidate.score for candidate in diagnostics.candidates] == [
        999.0,
        1.0,
        0.99,
        -1.0,
        pytest.approx(2 / 62),
        pytest.approx(1 / 61),
    ]
    assert [candidate.matched_terms for candidate in diagnostics.candidates] == [
        ("exact",),
        ("term",),
        (),
        (),
        ("term",),
        ("exact",),
    ]
    assert [candidate.rrf_contribution for candidate in diagnostics.candidates] == [
        1 / 61,
        1 / 62,
        1 / 61,
        1 / 62,
        None,
        None,
    ]
    assert [candidate.selected for candidate in diagnostics.candidates] == [
        True,
        True,
        False,
        True,
        True,
        True,
    ]


def test_hybrid_diagnose_preserves_empty_and_child_validation_behavior() -> None:
    lexical = _ScriptedIndex()
    semantic = _ScriptedIndex()
    index = _hybrid(lexical, semantic)

    diagnostics = index.diagnose("query")

    assert diagnostics.hits == ()
    assert diagnostics.candidates == ()
    assert lexical.search_calls == [("query", 100)]
    assert semantic.search_calls == [("query", 100)]

    invalid = _hybrid(_ScriptedIndex((SearchHit(_chunk("x"), math.nan),)), _ScriptedIndex())
    with pytest.raises(HybridBackendCoherenceError, match="invalid score"):
        invalid.diagnose("query")

    with pytest.raises(HybridChunkIndexLimitError, match="max_results"):
        index.diagnose("query", limit=101)
    assert lexical.search_calls == [("query", 100)]
    assert semantic.search_calls == [("query", 100)]


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


@pytest.mark.parametrize("method", ["search", "diagnose"])
def test_hybrid_rejects_duplicate_and_cross_backend_conflicting_chunks(
    method: str,
) -> None:
    chunk = _chunk("same")
    duplicate = _hybrid(
        _ScriptedIndex((SearchHit(chunk, 2.0), SearchHit(chunk, 1.0))),
        _ScriptedIndex(),
        candidate_depth=2,
    )
    with pytest.raises(HybridBackendCoherenceError, match="duplicate"):
        getattr(duplicate, method)("query")

    conflict = _hybrid(
        _ScriptedIndex((SearchHit(chunk, 1.0),)),
        _ScriptedIndex((SearchHit(_chunk("same", "different"), 1.0),)),
        candidate_depth=1,
    )
    with pytest.raises(HybridBackendCoherenceError, match="disagree"):
        getattr(conflict, method)("query")


@pytest.mark.parametrize("method", ["search", "diagnose"])
def test_hybrid_rejects_child_results_over_requested_candidate_limit_early(
    method: str,
) -> None:
    class UninspectableHit:
        @property
        def chunk(self) -> object:
            raise AssertionError("oversized results must not be inspected")

    oversized = tuple(UninspectableHit() for _ in range(3))

    class OverReturningIndex(_ScriptedIndex):
        def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
            self.search_calls.append((query, limit))
            return self.hits

    semantic = _ScriptedIndex()
    index = _hybrid(
        OverReturningIndex(oversized),  # type: ignore[arg-type]
        semantic,
        candidate_depth=2,
    )

    with pytest.raises(
        HybridBackendCoherenceError,
        match="exceeded requested candidate limit",
    ):
        getattr(index, method)("query", limit=1)

    assert semantic.search_calls == []


@pytest.mark.parametrize("method", ["search", "diagnose"])
def test_hybrid_checks_combined_fusion_bound_before_hit_validation(
    method: str,
) -> None:
    class UninspectableHit:
        @property
        def chunk(self) -> object:
            raise AssertionError("over-bound results must not be inspected")

    hits = (UninspectableHit(), UninspectableHit())
    index = _hybrid(
        _ScriptedIndex(hits),  # type: ignore[arg-type]
        _ScriptedIndex(hits),  # type: ignore[arg-type]
        limits=HybridChunkIndexLimits(
            max_candidates_per_backend=2,
            max_fusion_entries=3,
        ),
        candidate_depth=2,
    )

    with pytest.raises(HybridChunkIndexLimitError, match="max_fusion_entries"):
        getattr(index, method)("query", limit=1)


@pytest.mark.parametrize("method", ["search", "diagnose"])
@pytest.mark.parametrize(
    "hits",
    [
        [],
        (object(),),
        (SearchHit(_chunk("x"), math.nan),),
        (SearchHit(_chunk("x"), 1.0, (1,)),),
    ],
)
def test_hybrid_rejects_malformed_child_hits(hits: object, method: str) -> None:
    index = _hybrid(_ScriptedIndex(hits), _ScriptedIndex())  # type: ignore[arg-type]

    with pytest.raises(HybridBackendCoherenceError):
        getattr(index, method)("query")


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


def test_hybrid_rejects_cross_child_clone_aliasing_without_mutation() -> None:
    shared_candidate = _ScriptedIndex()

    class AliasingCloneIndex(_ScriptedIndex):
        def clone(self) -> _ScriptedIndex:
            return shared_candidate

    lexical = AliasingCloneIndex()
    semantic = AliasingCloneIndex()
    index = _hybrid(lexical, semantic)

    with pytest.raises(HybridChunkIndexError, match="clone failed"):
        index.clone()
    with pytest.raises(HybridChunkIndexError, match="mutation failed"):
        index.add((_chunk("new"),))

    assert index._lexical is lexical
    assert index._semantic is semantic
    assert shared_candidate.mutations == []


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
