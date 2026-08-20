from pathlib import Path

from nexusmind.retrieval_benchmark import (
    DEFAULT_RENDER_CONFIG,
    render_retrieval_comparison,
    run_retrieval_benchmark,
)


REPORT = Path(__file__).resolve().parents[1] / "evals" / "knowledge" / "benchmark.md"


def test_renderer_is_stable_and_contains_required_diagnostics() -> None:
    report = run_retrieval_benchmark()
    first = render_retrieval_comparison(report, DEFAULT_RENDER_CONFIG)
    second = render_retrieval_comparison(report, DEFAULT_RENDER_CONFIG)
    assert first == second
    assert "| Backend | K | Hit@K | Recall@K | MRR |" in first
    assert "## Per-category metrics" in first
    assert "## Selected diagnostics" in first
    assert "max(K)" in first
    assert "canonical snapshots" in first
    assert "descriptive/non-gate" in first
    assert "/home/" not in first


def test_checked_in_report_matches_generated_output_byte_for_byte() -> None:
    expected = REPORT.read_text(encoding="utf-8")
    actual = render_retrieval_comparison(run_retrieval_benchmark(), DEFAULT_RENDER_CONFIG)
    assert actual == expected
