from pathlib import Path

from nexusmind.search_diversification_benchmark import (
    DIVERSIFICATION_BENCHMARK_CASES,
    DIVERSIFICATION_BENCHMARK_CORPUS,
    DIVERSIFICATION_BENCHMARK_REPORT,
    REPRODUCTION_COMMAND,
    main,
    render_search_diversification_benchmark,
    run_search_diversification_benchmark,
)


def test_authored_dataset_covers_broad_and_precise_queries() -> None:
    documents = tuple(sorted(DIVERSIFICATION_BENCHMARK_CORPUS.glob("*.md")))
    report = run_search_diversification_benchmark()

    assert len(documents) == 5
    assert DIVERSIFICATION_BENCHMARK_CASES.is_file()
    assert {item.query for item in report.cases} >= {
        "Crypto",
        "Binder",
        "QNX",
        "权限校验",
        "lpRpcCrypto ImportFile exact permission flow",
    }


def test_broad_coverage_improves_without_precise_relevance_regression() -> None:
    report = run_search_diversification_benchmark()

    assert (
        report.diversified_broad_unique_relevant_documents
        > report.raw_broad_unique_relevant_documents
    )
    assert report.diversified_precise_mrr >= report.raw_precise_mrr
    assert report.diversified_precise_recall >= report.raw_precise_recall


def test_each_case_exposes_raw_and_diversified_metrics_and_rankings() -> None:
    report = run_search_diversification_benchmark()

    for item in report.cases:
        assert type(item.raw_documents) is tuple
        assert type(item.diversified_documents) is tuple
        assert item.raw_unique_document_count == len(set(item.raw_documents))
        assert item.diversified_unique_document_count == len(
            set(item.diversified_documents)
        )
        assert 0 <= item.raw_relevant_document_count <= len(item.relevant_documents)
        assert 0 <= item.diversified_relevant_document_count <= len(
            item.relevant_documents
        )
        assert item.raw_hit_at_k in (0.0, 1.0)
        assert item.diversified_hit_at_k in (0.0, 1.0)
        assert 0.0 <= item.raw_recall_at_k <= 1.0
        assert 0.0 <= item.diversified_recall_at_k <= 1.0
        assert 0.0 <= item.raw_mrr <= 1.0
        assert 0.0 <= item.diversified_mrr <= 1.0


def test_renderer_is_stable_and_checked_report_matches_byte_for_byte() -> None:
    report = run_search_diversification_benchmark()
    rendered = render_search_diversification_benchmark(report)

    assert rendered == render_search_diversification_benchmark(report)
    assert "Raw documents" in rendered
    assert "Diversified documents" in rendered
    assert "same-query relative score" in rendered
    assert REPRODUCTION_COMMAND in rendered
    assert "/home/" not in rendered
    assert rendered.encode("utf-8") == DIVERSIFICATION_BENCHMARK_REPORT.read_bytes()


def test_cli_writes_exact_utf8_report_bytes(tmp_path: Path) -> None:
    output = tmp_path / "diversification.md"

    assert main(("--write", str(output))) == 0
    assert output.read_bytes() == render_search_diversification_benchmark(
        run_search_diversification_benchmark()
    ).encode("utf-8")
