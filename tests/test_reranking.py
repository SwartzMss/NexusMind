from __future__ import annotations

import math

import pytest

from nexusmind import (
    Chunk,
    HybridChunkIndex,
    RetrievalCandidateDiagnostic,
    RetrievalDiagnostics,
    RetrievalStage,
    SearchHit,
)
from nexusmind.reranking import (
    RerankedChunkIndex,
    RerankerCoherenceError,
    RerankerError,
    RerankerLimitError,
    RerankerLimits,
)


def _chunk(chunk_id: str, content: str | None = None) -> Chunk:
    value = chunk_id if content is None else content
    return Chunk("doc", chunk_id, value, 0, len(value))


class _Index:
    def __init__(self, hits: tuple[SearchHit, ...] = ()) -> None:
        self.hits = hits
        self.search_calls: list[tuple[str, int]] = []
        self.mutations: list[tuple[str, tuple[object, ...]]] = []
        self.fail_search = False
        self.fail_mutation: str | None = None
        self.alias_clone = False

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("add", chunks)

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("replace_document", document_id, chunks)

    def remove_document(self, document_id: str) -> None:
        self._mutate("remove_document", document_id)

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        self.search_calls.append((query, limit))
        if self.fail_search:
            raise RuntimeError("private base detail")
        return self.hits

    def clone(self) -> "_Index":
        if self.alias_clone:
            return self
        clone = _Index(self.hits)
        clone.search_calls = self.search_calls.copy()
        clone.mutations = self.mutations.copy()
        clone.fail_search = self.fail_search
        clone.fail_mutation = self.fail_mutation
        return clone

    def _mutate(self, method: str, *args: object) -> None:
        self.mutations.append((method, args))
        if self.fail_mutation == method:
            raise RuntimeError("private mutation detail")


class _DiagnosticIndex(_Index):
    def __init__(self, hits: tuple[SearchHit, ...] = ()) -> None:
        super().__init__(hits)
        self.diagnose_calls: list[tuple[str, int]] = []
        self.trace: object | None = None
        self.fail_diagnose = False

    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics:
        self.diagnose_calls.append((query, limit))
        if self.fail_diagnose:
            raise RuntimeError("private diagnostic detail")
        if self.trace is not None:
            return self.trace  # type: ignore[return-value]
        return RetrievalDiagnostics(
            hits=self.hits,
            candidates=tuple(
                RetrievalCandidateDiagnostic(
                    stage=RetrievalStage.LEXICAL,
                    rank=rank,
                    chunk=hit.chunk,
                    score=hit.score,
                    matched_terms=hit.matched_terms,
                    selected=True,
                )
                for rank, hit in enumerate(self.hits, start=1)
            ),
        )


class _Reranker:
    def __init__(self, result: tuple[SearchHit, ...] | None = None) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[SearchHit, ...], int]] = []
        self.fail = False

    def rerank(
        self, query: str, candidates: tuple[SearchHit, ...], *, limit: int
    ) -> tuple[SearchHit, ...]:
        self.calls.append((query, candidates, limit))
        if self.fail:
            raise RuntimeError("private provider detail")
        return candidates[:limit] if self.result is None else self.result


def _wrapper(base: _Index, reranker: _Reranker, **kwargs: object) -> RerankedChunkIndex:
    return RerankedChunkIndex(
        base_index_factory=lambda: base,
        reranker=reranker,
        candidate_depth=3,
        **kwargs,
    )


@pytest.mark.parametrize("field", RerankerLimits.__dataclass_fields__)
@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "1"])
def test_reranker_limits_require_positive_plain_integers(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RerankerLimits(**{field: value})


def test_reranked_index_exposes_configured_search_capacity_across_clone() -> None:
    index = RerankedChunkIndex(
        base_index_factory=_Index,
        reranker=_Reranker(),
        candidate_depth=10,
        limits=RerankerLimits(max_candidates=10, max_results=10),
    )

    assert index.max_search_results == 10
    assert index.clone().max_search_results == 10


@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "1"])
def test_candidate_depth_requires_positive_plain_integer(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RerankedChunkIndex(
            base_index_factory=_Index,
            reranker=_Reranker(),
            candidate_depth=value,  # type: ignore[arg-type]
        )


def test_constructor_validates_dependencies_and_candidate_bound() -> None:
    with pytest.raises(TypeError, match="factory"):
        RerankedChunkIndex(  # type: ignore[arg-type]
            base_index_factory=object(), reranker=_Reranker(), candidate_depth=1
        )
    with pytest.raises(TypeError, match="rerank"):
        RerankedChunkIndex(base_index_factory=_Index, reranker=object(), candidate_depth=1)
    with pytest.raises(ValueError, match="max_candidates"):
        RerankedChunkIndex(
            base_index_factory=_Index,
            reranker=_Reranker(),
            candidate_depth=2,
            limits=RerankerLimits(max_candidates=1),
        )
    with pytest.raises(RerankerError, match="factory") as caught:
        RerankedChunkIndex(
            base_index_factory=lambda: (_ for _ in ()).throw(RuntimeError("secret")),
            reranker=_Reranker(),
            candidate_depth=1,
        )
    assert "secret" not in str(caught.value)


def test_search_is_fixed_depth_bounded_and_preserves_canonical_fields() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    candidates = (
        SearchHit(a, 3.0, ("a",)),
        SearchHit(b, 2.0, ("b",)),
        SearchHit(c, 1.0),
    )
    reranker = _Reranker((SearchHit(b, 0.9, ("b",)), SearchHit(a, 0.8, ("a",))))
    base = _Index(candidates)

    hits = _wrapper(base, reranker).search("query", limit=2)

    assert base.search_calls == [("query", 3)]
    assert reranker.calls == [("query", candidates, 2)]
    assert hits == (SearchHit(b, 0.9, ("b",)), SearchHit(a, 0.8, ("a",)))


def test_search_preflights_public_limits_before_provider_work() -> None:
    base, reranker = _Index(), _Reranker()
    index = _wrapper(
        base,
        reranker,
        limits=RerankerLimits(
            max_query_chars=2, max_candidates=3, max_total_candidate_chars=10, max_results=2
        ),
    )
    for query, limit, error in (("abc", 1, RerankerLimitError), ("q", 3, RerankerLimitError)):
        with pytest.raises(error):
            index.search(query, limit=limit)
    for value in (True, 0, -1, 1.0):
        with pytest.raises((TypeError, ValueError)):
            index.search("q", limit=value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        index.search(1, limit=1)  # type: ignore[arg-type]
    assert base.search_calls == []
    assert reranker.calls == []


def test_candidate_bounds_and_shape_fail_before_reranker_work() -> None:
    reranker = _Reranker()
    oversized = _Index((SearchHit(_chunk("a", "1234"), 1.0),))
    index = _wrapper(
        oversized,
        reranker,
        limits=RerankerLimits(
            max_query_chars=10,
            max_candidates=3,
            max_total_candidate_chars=3,
            max_results=3,
        ),
    )
    with pytest.raises(RerankerLimitError, match="candidate_chars"):
        index.search("q", limit=1)
    assert reranker.calls == []

    over_returning = _Index(tuple(SearchHit(_chunk(str(i)), 1.0) for i in range(4)))
    with pytest.raises(RerankerCoherenceError, match="result limit"):
        _wrapper(over_returning, reranker).search("q")
    assert reranker.calls == []


def test_base_candidate_must_have_valid_canonical_chunk_fields() -> None:
    malformed = Chunk("doc", "a", "abc", 0, 2)
    with pytest.raises(RerankerCoherenceError, match="chunk"):
        _wrapper(_Index((SearchHit(malformed, 1.0),)), _Reranker()).search("q")


@pytest.mark.parametrize(
    "result",
    [
        [SearchHit(_chunk("a"), 1.0)],
        (SearchHit(_chunk("ghost"), 1.0),),
        (SearchHit(_chunk("a"), 1.0), SearchHit(_chunk("a"), 0.5)),
        (SearchHit(_chunk("a", "changed"), 1.0),),
        (SearchHit(_chunk("a"), math.nan),),
        (SearchHit(_chunk("a"), math.inf),),
        (SearchHit(_chunk("a"), 1, ()),),
        (SearchHit(_chunk("a"), 1.0, ("changed",)),),
    ],
)
def test_reranker_output_must_match_fixed_candidates(result: object) -> None:
    base = _Index((SearchHit(_chunk("a"), 2.0, ("term",)),))
    reranker = _Reranker()
    reranker.result = result  # type: ignore[assignment]
    with pytest.raises(RerankerCoherenceError):
        _wrapper(base, reranker).search("q", limit=1)


def test_ties_use_first_stage_rank_and_underrun_is_not_filled() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    base = _Index((SearchHit(b, 3.0), SearchHit(a, 2.0), SearchHit(c, 1.0)))
    reranker = _Reranker((SearchHit(a, 0.5), SearchHit(b, 0.5)))

    hits = _wrapper(base, reranker).search("q", limit=3)

    assert [hit.chunk.chunk_id for hit in hits] == ["b", "a"]


def test_diagnose_uses_base_trace_once_and_appends_reranker_decisions() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    hits = (
        SearchHit(a, 0.5, ("a",)),
        SearchHit(b, 0.4, ("b",)),
        SearchHit(c, 0.3, ("c",)),
    )
    base = _DiagnosticIndex(hits)
    base.trace = RetrievalDiagnostics(
        hits=hits,
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=a,
                score=3.0,
                matched_terms=("a",),
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=2,
                chunk=b,
                score=2.0,
                matched_terms=("b",),
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=3,
                chunk=c,
                score=1.0,
                matched_terms=("c",),
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.FUSION,
                rank=1,
                chunk=a,
                score=0.5,
                matched_terms=("a",),
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.FUSION,
                rank=2,
                chunk=b,
                score=0.4,
                matched_terms=("b",),
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.FUSION,
                rank=3,
                chunk=c,
                score=0.3,
                matched_terms=("c",),
                selected=True,
            ),
        ),
    )
    reranker = _Reranker((SearchHit(b, 0.9, ("b",)), SearchHit(a, 0.8, ("a",))))

    trace = _wrapper(base, reranker).diagnose("query", limit=2)

    assert base.diagnose_calls == [("query", 3)]
    assert base.search_calls == []
    assert reranker.calls == [("query", hits, 2)]
    assert trace.hits == (SearchHit(b, 0.9, ("b",)), SearchHit(a, 0.8, ("a",)))
    assert [(row.stage, row.rank, row.chunk.chunk_id, row.selected) for row in trace.candidates] == [
        (RetrievalStage.LEXICAL, 1, "a", True),
        (RetrievalStage.LEXICAL, 2, "b", True),
        (RetrievalStage.LEXICAL, 3, "c", False),
        (RetrievalStage.FUSION, 1, "a", True),
        (RetrievalStage.FUSION, 2, "b", True),
        (RetrievalStage.FUSION, 3, "c", False),
        (RetrievalStage.RERANKER, 1, "b", True),
        (RetrievalStage.RERANKER, 2, "a", True),
    ]
    assert trace.candidates[-2:] == (
        RetrievalCandidateDiagnostic(
            stage=RetrievalStage.RERANKER,
            rank=1,
            chunk=b,
            score=0.9,
            matched_terms=("b",),
            selected=True,
        ),
        RetrievalCandidateDiagnostic(
            stage=RetrievalStage.RERANKER,
            rank=2,
            chunk=a,
            score=0.8,
            matched_terms=("a",),
            selected=True,
        ),
    )


def test_diagnose_preserves_reranker_underrun_and_stable_ties() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    base = _DiagnosticIndex((SearchHit(b, 3.0), SearchHit(a, 2.0), SearchHit(c, 1.0)))
    reranker = _Reranker((SearchHit(a, 0.5), SearchHit(b, 0.5)))

    trace = _wrapper(base, reranker).diagnose("query", limit=3)

    assert trace.hits == (SearchHit(b, 0.5), SearchHit(a, 0.5))
    assert [(row.rank, row.chunk.chunk_id) for row in trace.candidates[-2:]] == [
        (1, "b"),
        (2, "a"),
    ]


def test_diagnose_rejects_unsupported_or_malformed_base_trace_before_reranking() -> None:
    base, reranker = _Index((SearchHit(_chunk("a"), 1.0),)), _Reranker()
    with pytest.raises(RerankerError, match="base index does not support diagnostics") as caught:
        _wrapper(base, reranker).diagnose("private query", limit=1)
    assert "private" not in str(caught.value)
    assert base.search_calls == []
    assert reranker.calls == []

    diagnostic = _DiagnosticIndex((SearchHit(_chunk("a"), 1.0),))
    diagnostic.trace = RetrievalDiagnostics(
        hits=(SearchHit(_chunk("ghost"), 1.0),),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=_chunk("a"),
                score=1.0,
                selected=True,
            ),
        ),
    )
    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(diagnostic, reranker).diagnose("query", limit=1)
    assert diagnostic.diagnose_calls == [("query", 3)]
    assert diagnostic.search_calls == []
    assert reranker.calls == []


def test_diagnose_preflights_limits_and_redacts_base_failures() -> None:
    base, reranker = _DiagnosticIndex(), _Reranker()
    index = _wrapper(
        base,
        reranker,
        limits=RerankerLimits(
            max_query_chars=2, max_candidates=3, max_total_candidate_chars=10, max_results=2
        ),
    )
    with pytest.raises(RerankerLimitError):
        index.diagnose("abc", limit=1)
    with pytest.raises(RerankerLimitError):
        index.diagnose("q", limit=3)
    assert base.diagnose_calls == []

    base.fail_diagnose = True
    with pytest.raises(RerankerError, match="base index diagnose failed") as caught:
        index.diagnose("q", limit=1)
    assert "private" not in str(caught.value)
    assert reranker.calls == []


def test_diagnose_rejects_mutated_trace_contract_before_reranking() -> None:
    hit = SearchHit(_chunk("a"), 1.0)
    trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=hit.chunk,
                score=hit.score,
                selected=True,
            ),
        ),
    )
    object.__setattr__(trace, "candidates", list(trace.candidates))
    base, reranker = _DiagnosticIndex((hit,)), _Reranker()
    base.trace = trace

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


@pytest.mark.parametrize("duplicate_kind", ["hits", "candidates"])
def test_diagnose_rejects_duplicate_base_trace_declarations(
    duplicate_kind: str,
) -> None:
    hit = SearchHit(_chunk("a"), 1.0)
    row = RetrievalCandidateDiagnostic(
        stage=RetrievalStage.LEXICAL,
        rank=1,
        chunk=hit.chunk,
        score=hit.score,
        selected=True,
    )
    trace = RetrievalDiagnostics(
        hits=(hit, hit) if duplicate_kind == "hits" else (hit,),
        candidates=(row, row) if duplicate_kind == "candidates" else (row,),
    )
    base, reranker = _DiagnosticIndex((hit,)), _Reranker()
    base.trace = trace

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace|duplicates"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_rejects_reordered_base_final_hits_before_reranking() -> None:
    a, b = _chunk("a"), _chunk("b")
    trace = RetrievalDiagnostics(
        hits=(SearchHit(b, 1.0), SearchHit(a, 1.0)),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=a,
                score=1.0,
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=2,
                chunk=b,
                score=1.0,
                selected=True,
            ),
        ),
    )
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = trace

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=2)
    assert reranker.calls == []


@pytest.mark.parametrize("malformed_kind", ["stage_order", "terminal_unselected"])
def test_diagnose_rejects_incoherent_base_pipeline_declarations(
    malformed_kind: str,
) -> None:
    hit = SearchHit(_chunk("a"), 1.0)
    lexical = RetrievalCandidateDiagnostic(
        stage=RetrievalStage.LEXICAL,
        rank=1,
        chunk=hit.chunk,
        score=hit.score,
        selected=True,
    )
    fusion = RetrievalCandidateDiagnostic(
        stage=RetrievalStage.FUSION,
        rank=1,
        chunk=hit.chunk,
        score=hit.score,
        selected=malformed_kind != "terminal_unselected",
    )
    trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(fusion, lexical) if malformed_kind == "stage_order" else (lexical, fusion),
    )
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = trace

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_composes_nested_reranker_trace_blocks() -> None:
    a, b = _chunk("a"), _chunk("b")
    base = _DiagnosticIndex((SearchHit(a, 2.0, ("a",)), SearchHit(b, 1.0, ("b",))))
    inner_reranker, outer_reranker = _Reranker(), _Reranker()
    inner_reranker.result = (SearchHit(a, 0.8, ("a",)), SearchHit(b, 0.7, ("b",)))
    outer_reranker.result = (SearchHit(b, 0.9, ("b",)),)
    inner = _wrapper(base, inner_reranker)
    outer = RerankedChunkIndex(
        base_index_factory=lambda: inner,
        reranker=outer_reranker,
        candidate_depth=3,
    )

    trace = outer.diagnose("query", limit=2)

    assert base.diagnose_calls == [("query", 3)]
    assert base.search_calls == []
    assert trace.hits == (SearchHit(b, 0.9, ("b",)),)
    assert [(row.stage, row.rank, row.chunk.chunk_id, row.selected) for row in trace.candidates] == [
        (RetrievalStage.LEXICAL, 1, "a", False),
        (RetrievalStage.LEXICAL, 2, "b", True),
        (RetrievalStage.RERANKER, 1, "a", False),
        (RetrievalStage.RERANKER, 2, "b", True),
        (RetrievalStage.RERANKER, 1, "b", True),
    ]


def test_diagnose_accepts_hybrid_stage_specific_matched_terms() -> None:
    chunk = _chunk("a", "term")
    lexical = _Index((SearchHit(chunk, 2.0, ("term",)),))
    semantic = _Index((SearchHit(chunk, 0.9),))
    hybrid = HybridChunkIndex(
        lexical_index_factory=lambda: lexical,
        semantic_index_factory=lambda: semantic,
        candidate_depth=1,
    )
    index = RerankedChunkIndex(
        base_index_factory=lambda: hybrid,
        reranker=_Reranker(),
        candidate_depth=3,
    )

    trace = index.diagnose("term", limit=1)

    assert trace.hits[0].matched_terms == ("term",)
    assert [(row.stage, row.matched_terms) for row in trace.candidates] == [
        (RetrievalStage.LEXICAL, ("term",)),
        (RetrievalStage.SEMANTIC, ()),
        (RetrievalStage.FUSION, ("term",)),
        (RetrievalStage.RERANKER, ("term",)),
    ]


def test_diagnose_rejects_malformed_base_reranker_block_before_outer_reranking() -> None:
    hit = SearchHit(_chunk("a"), 1.0)
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=2,
                chunk=hit.chunk,
                score=hit.score,
                selected=True,
            ),
        ),
    )

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_rejects_reranker_only_base_trace() -> None:
    hit = SearchHit(_chunk("a"), 1.0)
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=1,
                chunk=hit.chunk,
                score=hit.score,
                selected=True,
            ),
        ),
    )

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_rejects_unselected_extra_in_final_reranker_block() -> None:
    a, b = _chunk("a"), _chunk("b")
    hit = SearchHit(a, 1.0)
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=1,
                chunk=a,
                score=1.0,
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=2,
                chunk=b,
                score=0.5,
                selected=False,
            ),
        ),
    )

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_rejects_later_reranker_ghost_after_rank_reset() -> None:
    a, b = _chunk("a"), _chunk("b")
    hit = SearchHit(b, 1.0)
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=b,
                score=2.0,
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=1,
                chunk=a,
                score=1.5,
                selected=False,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=1,
                chunk=b,
                score=1.0,
                selected=True,
            ),
        ),
    )

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_rejects_inconsistent_earlier_selected_flag() -> None:
    b = _chunk("b")
    hit = SearchHit(b, 1.0)
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=b,
                score=2.0,
                selected=False,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=1,
                chunk=b,
                score=1.0,
                selected=True,
            ),
        ),
    )

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_rejects_cross_block_canonical_conflict() -> None:
    original, changed = _chunk("a", "original"), _chunk("a", "changed")
    hit = SearchHit(changed, 1.0, ("term",))
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=original,
                score=2.0,
                matched_terms=("term",),
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=1,
                chunk=changed,
                score=1.0,
                matched_terms=("term",),
                selected=True,
            ),
        ),
    )

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_rejects_reranker_matched_term_drift() -> None:
    chunk = _chunk("a")
    hit = SearchHit(chunk, 1.0, ("outer",))
    base, reranker = _DiagnosticIndex(), _Reranker()
    base.trace = RetrievalDiagnostics(
        hits=(hit,),
        candidates=(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=1,
                chunk=chunk,
                score=2.0,
                matched_terms=("inner",),
                selected=True,
            ),
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=1,
                chunk=chunk,
                score=1.0,
                matched_terms=("outer",),
                selected=True,
            ),
        ),
    )

    with pytest.raises(RerankerCoherenceError, match="diagnostic trace"):
        _wrapper(base, reranker).diagnose("query", limit=1)
    assert reranker.calls == []


def test_diagnose_composes_three_reranker_levels() -> None:
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    base = _DiagnosticIndex((SearchHit(a, 3.0), SearchHit(b, 2.0), SearchHit(c, 1.0)))
    first_reranker, second_reranker, third_reranker = _Reranker(), _Reranker(), _Reranker()
    first_reranker.result = (SearchHit(a, 0.8), SearchHit(b, 0.7), SearchHit(c, 0.6))
    second_reranker.result = (SearchHit(b, 0.9), SearchHit(c, 0.8))
    third_reranker.result = (SearchHit(c, 1.0),)
    first = _wrapper(base, first_reranker)
    second = RerankedChunkIndex(
        base_index_factory=lambda: first,
        reranker=second_reranker,
        candidate_depth=3,
    )
    third = RerankedChunkIndex(
        base_index_factory=lambda: second,
        reranker=third_reranker,
        candidate_depth=3,
    )

    trace = third.diagnose("query", limit=3)

    assert trace.hits == (SearchHit(c, 1.0),)
    assert [(row.rank, row.chunk.chunk_id, row.selected) for row in trace.candidates if row.stage is RetrievalStage.RERANKER] == [
        (1, "a", False),
        (2, "b", False),
        (3, "c", True),
        (1, "b", False),
        (2, "c", True),
        (1, "c", True),
    ]


def test_failures_are_controlled_redacted_and_fail_closed() -> None:
    base, reranker = _Index((SearchHit(_chunk("a"), 1.0),)), _Reranker()
    base.fail_search = True
    with pytest.raises(RerankerError, match="base index search failed") as caught:
        _wrapper(base, reranker).search("private query", limit=1)
    assert "private" not in str(caught.value)

    base.fail_search = False
    reranker.fail = True
    with pytest.raises(RerankerError, match="reranker failed") as caught:
        _wrapper(base, reranker).search("private query", limit=1)
    assert "private" not in str(caught.value)


def test_clone_and_mutations_use_independent_base_and_commit_atomically() -> None:
    base, reranker = _Index(), _Reranker()
    index = _wrapper(base, reranker)
    clone = index.clone()
    clone.add((_chunk("a"),))
    assert base.mutations == []

    base.fail_mutation = "add"
    failing = _wrapper(base, reranker)
    with pytest.raises(RerankerError, match="mutation failed") as caught:
        failing.add((_chunk("b"),))
    assert "private" not in str(caught.value)
    assert base.mutations == []

    base.alias_clone = True
    with pytest.raises(RerankerError, match="clone failed"):
        _wrapper(base, reranker).clone()


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("add", ((_chunk("a"),),)),
        ("replace_document", ("doc", (_chunk("a"),))),
        ("remove_document", ("doc",)),
    ],
)
def test_every_failed_mutation_preserves_committed_base(
    method: str, args: tuple[object, ...]
) -> None:
    base = _Index((SearchHit(_chunk("old"), 1.0),))
    base.fail_mutation = method
    index = _wrapper(base, _Reranker())

    with pytest.raises(RerankerError, match="mutation failed"):
        getattr(index, method)(*args)

    assert index.search("old", limit=1)[0].chunk.chunk_id == "old"
    assert base.mutations == []
