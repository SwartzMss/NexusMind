from __future__ import annotations

import math

import pytest

from nexusmind import Chunk, SearchHit
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
