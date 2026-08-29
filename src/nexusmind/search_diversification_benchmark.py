"""Checked-in raw-vs-diversified retrieval evaluation for search selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .knowledge import Document
from .knowledge_chunking import Chunk
from .knowledge_collection import KnowledgeCollection
from .knowledge_ingestion import LocalDirectoryAdapter
from .retrieval_evaluation import (
    RetrievalCategory,
    RetrievalEvaluationCaseResult,
    evaluate_retrieval,
    load_retrieval_evaluation_cases,
)


_ROOT = Path(__file__).resolve().parents[2]
DIVERSIFICATION_BENCHMARK_ROOT = (
    _ROOT / "evals" / "knowledge" / "diversification"
)
DIVERSIFICATION_BENCHMARK_CORPUS = DIVERSIFICATION_BENCHMARK_ROOT / "corpus"
DIVERSIFICATION_BENCHMARK_CASES = DIVERSIFICATION_BENCHMARK_ROOT / "cases.json"
DIVERSIFICATION_BENCHMARK_REPORT = (
    _ROOT / "evals" / "knowledge" / "diversification.md"
)
DIVERSIFICATION_BENCHMARK_SOURCE_ID = "diversification-docs"
DIVERSIFICATION_BENCHMARK_K = 5
REPRODUCTION_COMMAND = (
    "PYTHONPATH=src python -m nexusmind.search_diversification_benchmark "
    "--write evals/knowledge/diversification.md"
)


@dataclass(frozen=True, slots=True)
class SearchDiversificationCaseReport:
    case_id: str
    category: RetrievalCategory
    query: str
    relevant_documents: tuple[str, ...]
    raw_documents: tuple[str, ...]
    diversified_documents: tuple[str, ...]
    raw_unique_document_count: int
    diversified_unique_document_count: int
    raw_relevant_document_count: int
    diversified_relevant_document_count: int
    raw_hit_at_k: float
    diversified_hit_at_k: float
    raw_recall_at_k: float
    diversified_recall_at_k: float
    raw_mrr: float
    diversified_mrr: float


@dataclass(frozen=True, slots=True)
class SearchDiversificationBenchmarkReport:
    k: int
    cases: tuple[SearchDiversificationCaseReport, ...]
    raw_broad_unique_relevant_documents: int
    diversified_broad_unique_relevant_documents: int
    raw_precise_mrr: float
    diversified_precise_mrr: float
    raw_precise_recall: float
    diversified_precise_recall: float


class _LineChunker:
    """Keep authored benchmark evidence in deterministic paragraph-sized chunks."""

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        search_from = 0
        for line in document.content.splitlines():
            if not line.strip():
                search_from += len(line) + 1
                continue
            start = document.content.index(line, search_from)
            end = start + len(line)
            chunks.append(
                Chunk(
                    document.document_id,
                    f"line:{document.document_id}:{len(chunks)}",
                    line,
                    start,
                    end,
                )
            )
            search_from = end + 1
        return tuple(chunks)


class _RawDiagnosticSearchView:
    def __init__(self, collection: KnowledgeCollection) -> None:
        self._collection = collection

    def snapshot(self):
        return self._collection.snapshot()

    def search(self, query: str, *, limit: int = 10):
        return self._collection.diagnose_search(query, limit=limit).results


def _build_collection() -> KnowledgeCollection:
    collection = KnowledgeCollection(
        chunker=_LineChunker(),
        clock=lambda: datetime(2026, 8, 28, tzinfo=timezone.utc),
    )
    collection.sync(
        LocalDirectoryAdapter(
            DIVERSIFICATION_BENCHMARK_CORPUS,
            source_id=DIVERSIFICATION_BENCHMARK_SOURCE_ID,
        )
    )
    return collection


def _paths(result: RetrievalEvaluationCaseResult) -> tuple[str, ...]:
    return tuple(target.logical_path for target in result.returned_targets)


def run_search_diversification_benchmark() -> SearchDiversificationBenchmarkReport:
    """Compare raw lexical rank with final document-aware selection offline."""

    cases = load_retrieval_evaluation_cases(DIVERSIFICATION_BENCHMARK_CASES)
    collection = _build_collection()
    raw = evaluate_retrieval(
        _RawDiagnosticSearchView(collection),  # type: ignore[arg-type]
        cases,
        k=DIVERSIFICATION_BENCHMARK_K,
    )
    diversified = evaluate_retrieval(
        collection, cases, k=DIVERSIFICATION_BENCHMARK_K
    )
    rows = tuple(
        SearchDiversificationCaseReport(
            case_id=raw_result.case_id,
            category=raw_result.category,
            query=raw_result.query,
            relevant_documents=tuple(
                target.logical_path for target in raw_result.relevant_targets
            ),
            raw_documents=_paths(raw_result),
            diversified_documents=_paths(diversified_result),
            raw_unique_document_count=len(set(raw_result.returned_targets)),
            diversified_unique_document_count=len(
                set(diversified_result.returned_targets)
            ),
            raw_relevant_document_count=len(raw_result.relevant_targets_found),
            diversified_relevant_document_count=len(
                diversified_result.relevant_targets_found
            ),
            raw_hit_at_k=raw_result.hit_at_k,
            diversified_hit_at_k=diversified_result.hit_at_k,
            raw_recall_at_k=raw_result.recall_at_k,
            diversified_recall_at_k=diversified_result.recall_at_k,
            raw_mrr=raw_result.reciprocal_rank,
            diversified_mrr=diversified_result.reciprocal_rank,
        )
        for raw_result, diversified_result in zip(
            raw.case_results, diversified.case_results
        )
    )
    broad = tuple(row for row in rows if row.category is RetrievalCategory.MULTI_DOCUMENT)
    precise = tuple(row for row in rows if row.category is RetrievalCategory.EXACT_TERM)
    precise_count = len(precise)
    return SearchDiversificationBenchmarkReport(
        k=DIVERSIFICATION_BENCHMARK_K,
        cases=rows,
        raw_broad_unique_relevant_documents=sum(
            row.raw_relevant_document_count for row in broad
        ),
        diversified_broad_unique_relevant_documents=sum(
            row.diversified_relevant_document_count for row in broad
        ),
        raw_precise_mrr=sum(row.raw_mrr for row in precise) / precise_count,
        diversified_precise_mrr=sum(row.diversified_mrr for row in precise)
        / precise_count,
        raw_precise_recall=sum(row.raw_recall_at_k for row in precise)
        / precise_count,
        diversified_precise_recall=sum(
            row.diversified_recall_at_k for row in precise
        )
        / precise_count,
    )


def _sequence(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


def render_search_diversification_benchmark(
    report: SearchDiversificationBenchmarkReport,
) -> str:
    """Render deterministic checked-in evidence without performing I/O."""

    if type(report) is not SearchDiversificationBenchmarkReport:
        raise TypeError("report must be a SearchDiversificationBenchmarkReport")
    lines = [
        "# Document-Aware Search Diversification",
        "",
        "This offline lexical evaluation compares raw backend Top-K diagnostics with",
        "the final document-aware search selection. The relevance safeguard uses only",
        "a same-query relative score window; backend scores and raw diagnostics are unchanged.",
        "",
        "## Aggregate safeguards",
        "",
        f"- K: {report.k}",
        "- Broad relevant-document coverage: "
        f"{report.raw_broad_unique_relevant_documents} raw -> "
        f"{report.diversified_broad_unique_relevant_documents} diversified",
        f"- Precise MRR: {report.raw_precise_mrr:.6f} raw -> "
        f"{report.diversified_precise_mrr:.6f} diversified",
        f"- Precise Recall@K: {report.raw_precise_recall:.6f} raw -> "
        f"{report.diversified_precise_recall:.6f} diversified",
        "",
        "## Per-query metrics",
        "",
        "| Case | Category | Raw unique/relevant | Diversified unique/relevant | Raw Hit/Recall/MRR | Diversified Hit/Recall/MRR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report.cases:
        lines.append(
            f"| {item.case_id} | {item.category.value} | "
            f"{item.raw_unique_document_count}/{item.raw_relevant_document_count} | "
            f"{item.diversified_unique_document_count}/"
            f"{item.diversified_relevant_document_count} | "
            f"{item.raw_hit_at_k:.6f}/{item.raw_recall_at_k:.6f}/{item.raw_mrr:.6f} | "
            f"{item.diversified_hit_at_k:.6f}/"
            f"{item.diversified_recall_at_k:.6f}/{item.diversified_mrr:.6f} |"
        )
    lines.extend(["", "## Rankings", ""])
    for item in report.cases:
        lines.extend(
            [
                f"### {item.case_id}: {item.query}",
                f"- Relevant documents: {_sequence(item.relevant_documents)}",
                f"- Raw documents: {_sequence(item.raw_documents)}",
                f"- Diversified documents: {_sequence(item.diversified_documents)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            REPRODUCTION_COMMAND,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the document-aware search diversification report"
    )
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)
    rendered = render_search_diversification_benchmark(
        run_search_diversification_benchmark()
    )
    args.write.write_bytes(rendered.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
