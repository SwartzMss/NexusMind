"""Offline deterministic evaluation for section-aware context expansion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .context_assembly import assemble_context
from .context_expansion import expand_context_candidates
from .knowledge import Document, KnowledgeSource
from .knowledge_chunking import Chunk
from .knowledge_collection import KnowledgeSearchResult
from .knowledge_retrieval import SearchHit


_CASES_PATH = Path(__file__).resolve().parents[2] / "evals/knowledge/context_expansion/cases.json"


@dataclass(frozen=True, slots=True)
class ContextExpansionBenchmarkCaseResult:
    """Metrics for one authored context-expansion fixture."""

    case_id: str
    anchor_retention: float
    relevant_coverage: float
    expansion_precision: float
    irrelevant_expansion_rate: float
    section_boundary_skips: int
    retained_chunk_ids: tuple[str, ...]
    expanded_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContextExpansionBenchmarkReport:
    """Stable ordered results for the complete offline benchmark."""

    cases: tuple[ContextExpansionBenchmarkCaseResult, ...]

    def by_case(self, case_id: str) -> ContextExpansionBenchmarkCaseResult:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)


def run_context_expansion_benchmark(
    cases_path: Path = _CASES_PATH,
) -> ContextExpansionBenchmarkReport:
    """Run the authored benchmark with retrieval anchors held constant."""

    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if type(raw_cases) is not list:
        raise ValueError("context expansion cases must be a JSON list")
    results = tuple(
        _evaluate_case(case)
        for case in sorted(raw_cases, key=lambda item: item["case_id"])
    )
    return ContextExpansionBenchmarkReport(results)


def render_context_expansion_benchmark(
    report: ContextExpansionBenchmarkReport,
) -> str:
    """Render a byte-stable Markdown report."""

    lines = [
        "# Section-Aware Context Expansion Benchmark",
        "",
        "The benchmark holds ranked retrieval anchors constant and evaluates deterministic context expansion.",
        "",
        "| Case | Anchor retention | Relevant coverage | Expansion precision | Irrelevant expansion | Boundary skips |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report.cases:
        lines.append(
            f"| {case.case_id} | {case.anchor_retention:.6f} | "
            f"{case.relevant_coverage:.6f} | {case.expansion_precision:.6f} | "
            f"{case.irrelevant_expansion_rate:.6f} | {case.section_boundary_skips} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "    PYTHONPATH=src python -m nexusmind.context_expansion_evaluation --write evals/knowledge/context_expansion/baseline.md",
            "",
            "All fixtures, anchors, labels, budgets, ordering, and metric formatting are deterministic.",
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_case(case: Any) -> ContextExpansionBenchmarkCaseResult:
    if type(case) is not dict:
        raise ValueError("context expansion case must be an object")
    case_id = _text(case, "case_id")
    query = _text(case, "query")
    documents, chunks_by_id, catalog = _build_fixture(case)
    documents_by_id = {document.document_id: document for document in documents}
    anchor_ids = _string_tuple(case, "anchor_ids")
    relevant_ids = set(_string_tuple(case, "relevant_ids"))
    forbidden_ids = set(_string_tuple(case, "forbidden_ids"))
    if not anchor_ids:
        raise ValueError("context expansion case must contain anchors")
    anchors = tuple(
        _make_anchor(
            documents_by_id[chunks_by_id[chunk_id].document_id],
            chunks_by_id[chunk_id],
            score=float(len(anchor_ids) - index),
            query=query,
        )
        for index, chunk_id in enumerate(anchor_ids)
    )
    expansion = expand_context_candidates(anchors, chunk_catalog=catalog)
    context = assemble_context(
        query,
        expansion.candidates,
        max_passages=_positive_int(case, "max_passages"),
        max_candidates=len(expansion.candidates),
        max_chars=_positive_int(case, "max_chars"),
        max_tokens=_positive_int(case, "max_tokens"),
    )
    retained_ids = tuple(passage.chunk_id for passage in context.passages)
    retained_set = set(retained_ids)
    selected_expanded = set(expansion.expanded_chunk_ids) & retained_set
    anchor_retention = len(set(anchor_ids) & retained_set) / len(set(anchor_ids))
    relevant_coverage = len(relevant_ids & retained_set) / len(relevant_ids)
    expansion_precision = len(selected_expanded & relevant_ids) / max(1, len(selected_expanded))
    irrelevant_rate = len(selected_expanded & forbidden_ids) / max(1, len(selected_expanded))
    return ContextExpansionBenchmarkCaseResult(
        case_id=case_id,
        anchor_retention=anchor_retention,
        relevant_coverage=relevant_coverage,
        expansion_precision=expansion_precision,
        irrelevant_expansion_rate=irrelevant_rate,
        section_boundary_skips=expansion.section_boundary_skips,
        retained_chunk_ids=retained_ids,
        expanded_chunk_ids=tuple(expansion.expanded_chunk_ids),
    )


def _build_fixture(
    case: dict[str, Any],
) -> tuple[tuple[Document, ...], dict[str, Chunk], dict[str, tuple[Chunk, ...]]]:
    documents: list[Document] = []
    chunks_by_id: dict[str, Chunk] = {}
    catalog: dict[str, tuple[Chunk, ...]] = {}
    for document_spec in case["documents"]:
        source_id = _text(document_spec, "source_id")
        logical_path = _text(document_spec, "logical_path")
        chunk_specs = document_spec["chunks"]
        content = "|".join(_text(item, "content") for item in chunk_specs)
        document = Document(source_id, logical_path, content)
        chunks: list[Chunk] = []
        offset = 0
        for index, item in enumerate(chunk_specs):
            chunk_id = _text(item, "chunk_id")
            chunk_content = _text(item, "content")
            heading_path = tuple(_string_tuple(item, "heading_path"))
            chunk = Chunk(
                document.document_id,
                chunk_id,
                chunk_content,
                offset,
                offset + len(chunk_content),
                heading_path,
                heading_path[-1] if heading_path else "",
                f"{logical_path}:L{index + 1}",
            )
            chunks.append(chunk)
            chunks_by_id[chunk_id] = chunk
            offset += len(chunk_content) + 1
        documents.append(document)
        catalog[document.document_id] = tuple(chunks)
    return tuple(documents), chunks_by_id, catalog


def _make_anchor(
    document: Document,
    chunk: Chunk,
    *,
    score: float,
    query: str,
) -> KnowledgeSearchResult:
    return KnowledgeSearchResult(
        source=KnowledgeSource(
            source_id=document.source_id,
            source_type="benchmark",
            display_name=document.logical_path,
        ),
        document=document,
        hit=SearchHit(chunk, score, (query,)),
    )


def _text(value: Any, key: str) -> str:
    result = value.get(key)
    if type(result) is not str or not result.strip():
        raise ValueError(f"context expansion case {key} must be non-empty text")
    return result


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    result = value.get(key)
    if type(result) is not list or any(type(item) is not str or not item.strip() for item in result):
        raise ValueError(f"context expansion case {key} must be a list of non-empty text")
    return tuple(result)


def _positive_int(value: Any, key: str) -> int:
    result = value.get(key)
    if type(result) is not int or result <= 0:
        raise ValueError(f"context expansion case {key} must be positive")
    return result


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the context expansion benchmark")
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)
    args.write.write_text(
        render_context_expansion_benchmark(run_context_expansion_benchmark()),
        encoding="utf-8",
    )
    return 0


__all__ = [
    "ContextExpansionBenchmarkCaseResult",
    "ContextExpansionBenchmarkReport",
    "render_context_expansion_benchmark",
    "run_context_expansion_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
