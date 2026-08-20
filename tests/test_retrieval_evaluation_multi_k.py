from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nexusmind import (
    Chunk,
    Document,
    KnowledgeSearchResult,
    KnowledgeSnapshot,
    KnowledgeSource,
    RetrievalCategory,
    RetrievalEvaluationCase,
    RetrievalEvaluationError,
    RetrievalTarget,
    SearchHit,
    evaluate_retrieval_multi_k,
)


def _source() -> KnowledgeSource:
    return KnowledgeSource(
        source_id="source", source_type="fake", display_name="source"
    )


def _document(path: str) -> Document:
    return Document(
        source_id="source", logical_path=path, content=f"content {path}"
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
        source=_source(),
        document=document,
        hit=SearchHit(chunk=chunk, score=1.0, matched_terms=()),
    )


@dataclass
class _Collection:
    documents: tuple[Document, ...]
    ranked: tuple[KnowledgeSearchResult, ...]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def snapshot(self) -> KnowledgeSnapshot:
        return KnowledgeSnapshot((_source(),), self.documents)

    def search(self, query: str, *, limit: int = 10):
        self.calls.append((query, limit))
        return self.ranked[:limit]


def test_multi_k_uses_one_max_ranking_and_strict_prefixes() -> None:
    distractor = _document("distractor.md")
    relevant_a = _document("a.md")
    relevant_b = _document("b.md")
    collection = _Collection(
        (distractor, relevant_a, relevant_b),
        (
            _result(distractor, "rank-1"),
            _result(relevant_a, "rank-2"),
            _result(relevant_b, "rank-3"),
        ),
    )
    case = RetrievalEvaluationCase(
        "multi",
        RetrievalCategory.MULTI_DOCUMENT,
        "query",
        (RetrievalTarget("source", "a.md"), RetrievalTarget("source", "b.md")),
    )

    reports = evaluate_retrieval_multi_k(collection, (case,), ks=(3, 1, 2))

    assert tuple(report.k for report in reports) == (1, 2, 3)
    assert collection.calls == [("query", 3)]
    assert tuple(
        report.case_results[0].returned_chunk_ids for report in reports
    ) == (("rank-1",), ("rank-1", "rank-2"), ("rank-1", "rank-2", "rank-3"))
    assert tuple(report.recall_at_k for report in reports) == (0.0, 0.5, 1.0)
    assert reports[1].case_results[0].category is RetrievalCategory.MULTI_DOCUMENT
    assert reports[1].case_results[0].relevant_targets == case.relevant_documents
    assert reports[2].category_reports[0].category is RetrievalCategory.MULTI_DOCUMENT
    assert reports[2].category_reports[0].case_count == 1
    assert reports[2].category_reports[0].recall_at_k == 1.0


@pytest.mark.parametrize(
    "ks, message",
    [
        ((), "non-empty tuple"),
        ([1], "tuple"),
        ((True,), "positive plain integers"),
        ((0,), "positive plain integers"),
        ((1.5,), "positive plain integers"),
        ((1, 1), "duplicate"),
        ((1, 2, 3, 4, 5, 6, 7, 8, 9), "too many"),
        ((101,), "maximum"),
    ],
)
def test_multi_k_rejects_invalid_or_unbounded_cutoffs(ks, message: str) -> None:
    document = _document("a.md")
    collection = _Collection((document,), (_result(document, "a"),))
    case = RetrievalEvaluationCase(
        "case", RetrievalCategory.EXACT_TERM, "query", (RetrievalTarget("source", "a.md"),)
    )
    with pytest.raises(RetrievalEvaluationError, match=message):
        evaluate_retrieval_multi_k(collection, (case,), ks=ks)
