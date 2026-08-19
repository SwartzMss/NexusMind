from __future__ import annotations

from pathlib import Path

from nexusmind import (
    InMemoryChunkIndex,
    KnowledgeCollection,
    KnowledgeSearchResult,
    LocalDirectoryAdapter,
    TextChunker,
    UnicodeCJKLexicalAnalyzer,
    WhitespaceLexicalAnalyzer,
    evaluate_retrieval,
    load_retrieval_evaluation_cases,
)


EVAL_ROOT = Path(__file__).resolve().parents[1] / "evals" / "knowledge" / "cjk"
CORPUS = EVAL_ROOT / "corpus"
CASES = EVAL_ROOT / "cases.json"
BASELINE = EVAL_ROOT / "baseline.md"
CHUNK_SIZE = 240
CHUNK_OVERLAP = 40
K = 3


def _collection(analyzer: object) -> KnowledgeCollection:
    collection = KnowledgeCollection(
        chunker=TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
        index_factory=lambda: InMemoryChunkIndex(analyzer=analyzer),
    )
    collection.sync(LocalDirectoryAdapter(CORPUS, source_id="cjk-docs"))
    return collection


def _reports():
    cases = load_retrieval_evaluation_cases(CASES)
    unicode_collection = _collection(UnicodeCJKLexicalAnalyzer())
    whitespace_collection = _collection(WhitespaceLexicalAnalyzer())
    unicode_report = evaluate_retrieval(unicode_collection, cases, k=K)
    whitespace_report = evaluate_retrieval(whitespace_collection, cases, k=K)
    return cases, unicode_collection, unicode_report, whitespace_report


def test_cjk_fixture_uses_real_pipeline_and_exposes_whitespace_limitation() -> None:
    cases, collection, unicode_report, whitespace_report = _reports()

    assert evaluate_retrieval(collection, cases, k=K) == unicode_report
    assert len(cases) == 10
    assert len(unicode_report.case_results) == len(cases)
    assert all(
        0.0 <= metric <= 1.0
        for report in (unicode_report, whitespace_report)
        for metric in (report.hit_at_k, report.recall_at_k, report.mrr)
    )
    assert unicode_report.hit_at_k > whitespace_report.hit_at_k
    assert any(
        unicode_case.hit_at_k > whitespace_case.hit_at_k
        for unicode_case, whitespace_case in zip(
            unicode_report.case_results,
            whitespace_report.case_results,
            strict=True,
        )
    )

    snapshot = collection.snapshot()
    assert {
        (document.source_id, document.logical_path) for document in snapshot.documents
    } == {
        ("cjk-docs", "android.md"),
        ("cjk-docs", "trustzone.md"),
        ("cjk-docs", "qnx.md"),
        ("cjk-docs", "cryptography.md"),
    }
    for case in cases:
        results = collection.search(case.query, limit=K)
        assert all(isinstance(result, KnowledgeSearchResult) for result in results)
        assert all(result.source.source_id == result.document.source_id for result in results)


def test_cjk_baseline_documents_fixed_reproduction_policy() -> None:
    baseline = BASELINE.read_text(encoding="utf-8")

    assert "descriptive" in baseline
    assert "non-gate" in baseline
    assert "UnicodeCJKLexicalAnalyzer" in baseline
    assert "WhitespaceLexicalAnalyzer" in baseline
    assert f"chunk_size={CHUNK_SIZE}" in baseline
    assert f"overlap={CHUNK_OVERLAP}" in baseline
    assert f"k={K}" in baseline
    assert "pytest -q tests/test_retrieval_evaluation_cjk.py" in baseline
    assert "bigram" in baseline.lower()
