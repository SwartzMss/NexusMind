from __future__ import annotations

from pathlib import Path

from nexusmind import (
    KnowledgeCollection,
    LocalDirectoryAdapter,
    TextChunker,
    evaluate_retrieval,
    load_retrieval_evaluation_cases,
)


EVAL_ROOT = Path(__file__).resolve().parents[1] / "evals" / "knowledge"
CORPUS = EVAL_ROOT / "corpus"
CASES = EVAL_ROOT / "cases.json"


def _run_baseline():
    collection = KnowledgeCollection(
        chunker=TextChunker(chunk_size=240, overlap=40)
    )
    collection.sync(LocalDirectoryAdapter(CORPUS, source_id="eval-corpus"))
    cases = load_retrieval_evaluation_cases(CASES)
    return collection, cases, evaluate_retrieval(collection, cases, k=5)


def test_checked_in_retrieval_baseline_is_deterministic() -> None:
    collection, cases, first = _run_baseline()
    second = evaluate_retrieval(collection, cases, k=5)

    assert second == first
    assert first.k == 5
    assert len(cases) == 15
    assert len(first.case_results) == len(cases)
    assert 0.0 <= first.hit_at_k <= 1.0
    assert 0.0 <= first.recall_at_k <= 1.0
    assert 0.0 <= first.mrr <= 1.0
