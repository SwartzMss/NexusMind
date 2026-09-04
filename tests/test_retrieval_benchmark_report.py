from dataclasses import replace
from pathlib import Path

from nexusmind.retrieval_benchmark import (
    DEFAULT_RENDER_CONFIG,
    main,
    render_retrieval_comparison,
    run_retrieval_benchmark,
)


REPORT = Path(__file__).resolve().parents[1] / "evals" / "knowledge" / "benchmark.md"


def test_renderer_is_stable_and_contains_required_diagnostics() -> None:
    report = run_retrieval_benchmark()
    first = render_retrieval_comparison(report, DEFAULT_RENDER_CONFIG)
    second = render_retrieval_comparison(report, DEFAULT_RENDER_CONFIG)
    assert first == second
    assert "| Backend | K | Hit@K | Precision@K | Recall@K | MRR |" in first
    assert "## Per-category metrics" in first
    assert "## Selected diagnostics" in first
    assert "max(K)" in first
    assert "canonical snapshots" in first
    assert "descriptive/non-gate" in first
    assert "/home/" not in first


def test_checked_in_report_matches_generated_output_byte_for_byte() -> None:
    expected = REPORT.read_bytes()
    actual = render_retrieval_comparison(
        run_retrieval_benchmark(), DEFAULT_RENDER_CONFIG
    ).encode("utf-8")
    assert actual == expected


def test_cli_writes_exact_utf8_bytes_without_text_newline_conversion(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "benchmark.md"

    def reject_text_write(*args, **kwargs):
        raise AssertionError("benchmark writer must not use text mode")

    monkeypatch.setattr(Path, "write_text", reject_text_write)

    assert main(("--write", str(output))) == 0
    assert output.read_bytes() == render_retrieval_comparison(
        run_retrieval_benchmark(), DEFAULT_RENDER_CONFIG
    ).encode("utf-8")


def test_diagnostics_include_failures_from_non_first_backend() -> None:
    report = run_retrieval_benchmark()
    case_id = report.backend_reports[0].reports_by_k[0].case_results[0].case_id
    backend_reports = []
    for backend_index, backend in enumerate(report.backend_reports):
        reports_by_k = list(backend.reports_by_k)
        small = reports_by_k[0]
        case_results = tuple(
            replace(
                result,
                recall_at_k=(0.0 if backend_index == 1 and result.case_id == case_id else 1.0),
            )
            for result in small.case_results
        )
        reports_by_k[0] = replace(small, case_results=case_results)
        backend_reports.append(replace(backend, reports_by_k=tuple(reports_by_k)))
    comparison = replace(report, backend_reports=tuple(backend_reports))

    rendered = render_retrieval_comparison(
        comparison, replace(DEFAULT_RENDER_CONFIG, max_diagnostics=1)
    )

    assert f"### case: {case_id}" in rendered
