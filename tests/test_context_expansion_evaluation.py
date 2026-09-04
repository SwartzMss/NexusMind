from __future__ import annotations

from nexusmind.context_expansion_evaluation import (
    render_context_expansion_benchmark,
    run_context_expansion_benchmark,
)


def test_context_expansion_benchmark_covers_required_metrics() -> None:
    report = run_context_expansion_benchmark()

    assert {item.case_id for item in report.cases} == {
        "multi-document-budget",
        "next-caveat",
        "previous-definition",
        "sibling-boundary",
    }
    assert all(0.0 <= item.anchor_retention <= 1.0 for item in report.cases)
    assert all(0.0 <= item.relevant_coverage <= 1.0 for item in report.cases)
    assert all(0.0 <= item.expansion_precision <= 1.0 for item in report.cases)
    assert all(0.0 <= item.irrelevant_expansion_rate <= 1.0 for item in report.cases)
    assert report.by_case("sibling-boundary").section_boundary_skips == 1
    assert report.by_case("multi-document-budget").anchor_retention == 1.0


def test_context_expansion_benchmark_is_deterministic() -> None:
    first = run_context_expansion_benchmark()
    second = run_context_expansion_benchmark()

    assert first == second
    assert render_context_expansion_benchmark(first) == render_context_expansion_benchmark(second)
