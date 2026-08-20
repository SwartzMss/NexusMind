from pathlib import Path

from nexusmind import Chunk, RetrievalCategory, SearchHit, load_retrieval_evaluation_cases
from nexusmind.retrieval_benchmark import (
    BENCHMARK_CASES,
    BENCHMARK_CORPUS,
    BENCHMARK_KS,
    BenchmarkReranker,
    run_retrieval_benchmark,
)


def test_authored_benchmark_is_bounded_and_covers_every_category() -> None:
    documents = tuple(sorted(Path(BENCHMARK_CORPUS).glob("*.md")))
    cases = load_retrieval_evaluation_cases(BENCHMARK_CASES)
    assert 10 <= len(documents) <= 20
    assert 30 <= len(cases) <= 50
    assert {case.category for case in cases} == set(RetrievalCategory)
    assert len({case.case_id for case in cases}) == len(cases)
    assert any(len(case.relevant_documents) > 1 for case in cases)


def test_benchmark_compares_four_backends_offline_and_deterministically() -> None:
    first = run_retrieval_benchmark()
    second = run_retrieval_benchmark()
    assert first == second
    assert first.ks == BENCHMARK_KS == (1, 3, 5, 10)
    assert tuple(report.backend_name for report in first.backend_reports) == (
        "BM25-only",
        "Semantic-only",
        "Hybrid-RRF",
        "Hybrid-RRF + Rerank",
    )


def test_benchmark_reranker_is_content_driven_and_preserves_candidates() -> None:
    relevant = Chunk("doc-a", "a", "Binder provides Android IPC", 0, 27)
    distractor = Chunk("doc-b", "b", "unrelated checkpoint notes", 0, 26)
    candidates = (SearchHit(distractor, 2.0), SearchHit(relevant, 1.0))

    result = BenchmarkReranker().rerank("cross process IPC", candidates, limit=2)

    assert [hit.chunk.chunk_id for hit in result] == ["a", "b"]
    assert {hit.chunk for hit in result} == {relevant, distractor}
