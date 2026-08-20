from __future__ import annotations

import pytest

from nexusmind import (
    RetrievalCategory,
    RetrievalEvaluationCaseResult,
    RetrievalFailureKind,
    RetrievalTarget,
    classify_retrieval_failure,
)


def _result(*, category: RetrievalCategory, found: int, total: int, rank: int | None):
    targets = tuple(RetrievalTarget("source", f"{index}.md") for index in range(total))
    return RetrievalEvaluationCaseResult(
        case_id="case",
        category=category,
        query="query",
        relevant_targets=targets,
        returned_targets=targets[:found],
        returned_chunk_ids=tuple(f"chunk-{index}" for index in range(found)),
        relevant_targets_found=targets[:found],
        relevant_targets_missed=targets[found:],
        first_relevant_rank=rank,
        hit_at_k=1.0 if rank else 0.0,
        recall_at_k=found / total,
        reciprocal_rank=1.0 / rank if rank else 0.0,
    )


def test_failure_classification_distinguishes_miss_ranking_and_partial_recall() -> None:
    miss = _result(category=RetrievalCategory.PARAPHRASE, found=0, total=1, rank=None)
    late = _result(category=RetrievalCategory.PARAPHRASE, found=1, total=1, rank=3)
    partial = _result(category=RetrievalCategory.MULTI_DOCUMENT, found=1, total=2, rank=1)

    assert classify_retrieval_failure(miss, smaller_k=1) is RetrievalFailureKind.MISSED
    assert classify_retrieval_failure(late, smaller_k=1) is RetrievalFailureKind.RANKED_BELOW_CUTOFF
    assert classify_retrieval_failure(partial, smaller_k=1) is RetrievalFailureKind.PARTIAL_RECALL
    with pytest.raises(ValueError, match="positive integer"):
        classify_retrieval_failure(late, smaller_k=0)
