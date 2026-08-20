from pathlib import Path

from nexusmind import RetrievalCategory, load_retrieval_evaluation_cases
from nexusmind.retrieval_benchmark import (
    BENCHMARK_CASES,
    BENCHMARK_CORPUS,
    BENCHMARK_KS,
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


def test_benchmark_compares_three_backends_offline_and_deterministically() -> None:
    first = run_retrieval_benchmark()
    second = run_retrieval_benchmark()
    assert first == second
    assert first.ks == BENCHMARK_KS == (1, 3, 5, 10)
    assert tuple(report.backend_name for report in first.backend_reports) == (
        "BM25-only",
        "Semantic-only",
        "Hybrid-RRF",
    )
