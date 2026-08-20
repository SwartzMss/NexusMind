from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nexusmind import (
    Chunk,
    ChunkIndexLimitError,
    Document,
    KnowledgeSearchResult,
    KnowledgeSnapshot,
    KnowledgeSource,
    RetrievalCategory,
    RetrievalEvaluationCase,
    RetrievalEvaluationError,
    RetrievalTarget,
    SearchHit,
    evaluate_retrieval,
)


def _source(source_id: str) -> KnowledgeSource:
    return KnowledgeSource(
        source_id=source_id,
        source_type="fake",
        display_name=source_id,
    )


def _document(source_id: str, logical_path: str) -> Document:
    return Document(
        source_id=source_id,
        logical_path=logical_path,
        content=f"content for {logical_path}",
    )


def _result(document: Document, chunk_id: str) -> KnowledgeSearchResult:
    chunk = Chunk(
        document_id=document.document_id,
        chunk_id=chunk_id,
        content=document.content,
        start_offset=0,
        end_offset=len(document.content),
    )
    return KnowledgeSearchResult(
        source=_source(document.source_id),
        document=document,
        hit=SearchHit(chunk=chunk, score=1.0, matched_terms=("term",)),
    )


@dataclass
class FakeCollection:
    documents: tuple[Document, ...]
    results: dict[str, tuple[KnowledgeSearchResult, ...]]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def snapshot(self) -> KnowledgeSnapshot:
        source_ids = sorted({document.source_id for document in self.documents})
        return KnowledgeSnapshot(
            sources=tuple(_source(source_id) for source_id in source_ids),
            documents=self.documents,
        )

    def search(self, query: str, *, limit: int = 10) -> tuple[KnowledgeSearchResult, ...]:
        self.calls.append((query, limit))
        return self.results.get(query, ())[:limit]


@pytest.mark.parametrize("field", ["source_id", "logical_path"])
def test_retrieval_target_requires_non_empty_text(field: str) -> None:
    values = {"source_id": "source", "logical_path": "doc.md"}
    values[field] = " "
    with pytest.raises(ValueError, match=field):
        RetrievalTarget(**values)
    values[field] = 1
    with pytest.raises(ValueError, match=field):
        RetrievalTarget(**values)


def test_evaluation_case_requires_bounded_unique_targets() -> None:
    target = RetrievalTarget("source", "doc.md")
    with pytest.raises(ValueError, match="case_id"):
        RetrievalEvaluationCase("", RetrievalCategory.EXACT_TERM, "query", (target,))
    with pytest.raises(ValueError, match="query"):
        RetrievalEvaluationCase("case", RetrievalCategory.EXACT_TERM, " ", (target,))
    with pytest.raises(TypeError, match="tuple"):
        RetrievalEvaluationCase("case", RetrievalCategory.EXACT_TERM, "query", [target])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        RetrievalEvaluationCase("case", RetrievalCategory.EXACT_TERM, "query", ())
    with pytest.raises(ValueError, match="duplicate"):
        RetrievalEvaluationCase("case", RetrievalCategory.EXACT_TERM, "query", (target, target))


def test_evaluator_preserves_chunk_ranking_and_deduplicates_recall_coverage() -> None:
    a = _document("source", "a.md")
    b = _document("source", "b.md")
    target_a = RetrievalTarget("source", "a.md")
    target_b = RetrievalTarget("source", "b.md")
    collection = FakeCollection(
        (a, b),
        {"query": (_result(a, "a-1"), _result(a, "a-2"), _result(b, "b-1"))},
    )
    case = RetrievalEvaluationCase("case", RetrievalCategory.MULTI_DOCUMENT, "query", (target_a, target_b))
    before = collection.snapshot()

    report = evaluate_retrieval(collection, (case,), k=3)  # type: ignore[arg-type]

    result = report.case_results[0]
    assert result.returned_targets == (target_a, target_a, target_b)
    assert result.returned_chunk_ids == ("a-1", "a-2", "b-1")
    assert result.relevant_targets_found == (target_a, target_b)
    assert result.relevant_targets_missed == ()
    assert result.first_relevant_rank == 1
    assert result.hit_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert collection.calls == [("query", 3)]
    assert collection.snapshot() == before


def test_evaluator_computes_later_rank_miss_and_aggregate_means() -> None:
    relevant = _document("source", "relevant.md")
    distractor = _document("source", "distractor.md")
    target = RetrievalTarget("source", "relevant.md")
    cases = (
        RetrievalEvaluationCase("later", RetrievalCategory.PARAPHRASE, "later query", (target,)),
        RetrievalEvaluationCase("miss", RetrievalCategory.PARAPHRASE, "miss query", (target,)),
    )
    collection = FakeCollection(
        (relevant, distractor),
        {
            "later query": (_result(distractor, "d-1"), _result(relevant, "r-1")),
            "miss query": (_result(distractor, "d-2"),),
        },
    )

    report = evaluate_retrieval(collection, cases, k=5)  # type: ignore[arg-type]

    later, miss = report.case_results
    assert later.first_relevant_rank == 2
    assert later.reciprocal_rank == 0.5
    assert miss.first_relevant_rank is None
    assert miss.hit_at_k == miss.recall_at_k == miss.reciprocal_rank == 0.0
    assert miss.relevant_targets_missed == (target,)
    assert report.hit_at_k == pytest.approx(0.5)
    assert report.recall_at_k == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.25)


def test_evaluator_rejects_invalid_case_sets_and_k() -> None:
    target = RetrievalTarget("source", "doc.md")
    case = RetrievalEvaluationCase("case", RetrievalCategory.EXACT_TERM, "query", (target,))
    collection = FakeCollection((_document("source", "doc.md"),), {})
    with pytest.raises(RetrievalEvaluationError, match="non-empty tuple"):
        evaluate_retrieval(collection, ())  # type: ignore[arg-type]
    with pytest.raises(RetrievalEvaluationError, match="tuple"):
        evaluate_retrieval(collection, [case])  # type: ignore[arg-type]
    with pytest.raises(RetrievalEvaluationError, match="duplicate case_id"):
        evaluate_retrieval(collection, (case, case))  # type: ignore[arg-type]
    for invalid_k in (True, 0, -1, 1.5):
        with pytest.raises(RetrievalEvaluationError, match="positive integer"):
            evaluate_retrieval(collection, (case,), k=invalid_k)  # type: ignore[arg-type]


def test_unknown_relevance_target_fails_before_any_search() -> None:
    collection = FakeCollection((_document("source", "known.md"),), {})
    case = RetrievalEvaluationCase(
        "case",
        RetrievalCategory.EXACT_TERM,
        "query",
        (RetrievalTarget("source", "missing.md"),),
    )

    with pytest.raises(RetrievalEvaluationError, match="unknown relevance target"):
        evaluate_retrieval(collection, (case,))  # type: ignore[arg-type]

    assert collection.calls == []


def test_same_input_produces_equal_reports() -> None:
    document = _document("source", "doc.md")
    target = RetrievalTarget("source", "doc.md")
    case = RetrievalEvaluationCase("case", RetrievalCategory.EXACT_TERM, "query", (target,))
    collection = FakeCollection((document,), {"query": (_result(document, "chunk"),)})

    assert evaluate_retrieval(collection, (case,)) == evaluate_retrieval(  # type: ignore[arg-type]
        collection, (case,)
    )


def test_backend_result_limit_is_reported_as_evaluation_error() -> None:
    document = _document("source", "doc.md")
    target = RetrievalTarget("source", "doc.md")
    case = RetrievalEvaluationCase("case", RetrievalCategory.EXACT_TERM, "query", (target,))

    class LimitedCollection(FakeCollection):
        def search(self, query: str, *, limit: int = 10):
            raise ChunkIndexLimitError("limit exceeds max_results")

    collection = LimitedCollection((document,), {})

    with pytest.raises(
        RetrievalEvaluationError, match="k exceeds retrieval backend result limit"
    ):
        evaluate_retrieval(collection, (case,), k=100)  # type: ignore[arg-type]
