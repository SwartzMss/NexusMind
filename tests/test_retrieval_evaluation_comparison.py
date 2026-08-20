from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nexusmind import (
    Document,
    KnowledgeSnapshot,
    KnowledgeSource,
    RetrievalBackend,
    RetrievalCategory,
    RetrievalComparisonError,
    RetrievalEvaluationCase,
    RetrievalTarget,
    compare_retrieval_backends,
)


def _source() -> KnowledgeSource:
    return KnowledgeSource(source_id="source", source_type="fake", display_name="source")


def _document(content: str = "same") -> Document:
    return Document(source_id="source", logical_path="doc.md", content=content)


@dataclass
class _Collection:
    document: Document
    calls: list[tuple[str, int]] = field(default_factory=list)

    def snapshot(self) -> KnowledgeSnapshot:
        return KnowledgeSnapshot((_source(),), (self.document,))

    def search(self, query: str, *, limit: int = 10):
        self.calls.append((query, limit))
        return ()


def _case() -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        "case",
        RetrievalCategory.EXACT_TERM,
        "query",
        (RetrievalTarget("source", "doc.md"),),
    )


def test_comparison_preserves_backend_order_and_shared_configuration() -> None:
    left = _Collection(_document())
    right = _Collection(_document())
    report = compare_retrieval_backends(
        (RetrievalBackend("BM25-only", lambda: left), RetrievalBackend("Semantic-only", lambda: right)),
        (_case(),),
        ks=(3, 1),
    )

    assert report.ks == (1, 3)
    assert tuple(item.backend_name for item in report.backend_reports) == (
        "BM25-only",
        "Semantic-only",
    )
    assert left.calls == right.calls == [("query", 3)]


def test_comparison_rejects_snapshot_mismatch_before_search() -> None:
    left = _Collection(_document("left"))
    right = _Collection(_document("right"))
    with pytest.raises(RetrievalComparisonError, match="canonical snapshots differ"):
        compare_retrieval_backends(
            (RetrievalBackend("left", lambda: left), RetrievalBackend("right", lambda: right)),
            (_case(),),
            ks=(1, 3),
        )
    assert left.calls == right.calls == []


def test_comparison_rejects_duplicate_names_and_sanitizes_factory_failure() -> None:
    backend = RetrievalBackend("same", lambda: _Collection(_document()))
    with pytest.raises(RetrievalComparisonError, match="duplicate backend"):
        compare_retrieval_backends((backend, backend), (_case(),), ks=(1,))

    private = RuntimeError("private provider detail")

    def fail():
        raise private

    with pytest.raises(RetrievalComparisonError, match="backend factory failed") as caught:
        compare_retrieval_backends((RetrievalBackend("failed", fail),), (_case(),), ks=(1,))
    assert "private provider detail" not in str(caught.value)
    assert caught.value.__cause__ is private
