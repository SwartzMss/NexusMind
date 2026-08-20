"""Checked-in offline benchmark for categorized retrieval comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .embeddings import EmbeddingVector
from .hybrid_retrieval import HybridChunkIndex
from .knowledge_chunking import TextChunker
from .knowledge_collection import KnowledgeCollection
from .knowledge_ingestion import LocalDirectoryAdapter
from .knowledge_retrieval import InMemoryChunkIndex
from .lexical_analysis import UnicodeCJKLexicalAnalyzer
from .knowledge_retrieval import SearchHit
from .reranking import RerankedChunkIndex
from .retrieval_evaluation import (
    RetrievalBackend,
    RetrievalCategory,
    RetrievalComparisonReport,
    compare_retrieval_backends,
    load_retrieval_evaluation_cases,
)
from .semantic_retrieval import InMemorySemanticChunkIndex


_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = _ROOT / "evals" / "knowledge" / "benchmark"
BENCHMARK_CORPUS = BENCHMARK_ROOT / "corpus"
BENCHMARK_CASES = BENCHMARK_ROOT / "cases.json"
BENCHMARK_KS = (1, 3, 5, 10)
BENCHMARK_SOURCE_ID = "benchmark-docs"


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkRenderConfig:
    title: str
    corpus_summary: str
    reproduction_command: str
    max_diagnostics: int = 12


DEFAULT_RENDER_CONFIG = RetrievalBenchmarkRenderConfig(
    title="Categorized Retrieval Backend Comparison",
    corpus_summary="10 UTF-8 documents and 32 authored relevance cases",
    reproduction_command=(
        "PYTHONPATH=src python -m nexusmind.retrieval_benchmark "
        "--write evals/knowledge/benchmark.md"
    ),
)


class BenchmarkEmbeddingProvider:
    """Authored concept projection for architecture tests, not model quality."""

    _concepts = (
        ("api", "knowledgecollection", "snapshot", "canonical"),
        ("identifier", "zxq-417", "alm_xdc", "lp-ichs"),
        ("ipc", "binder", "process", "message passing", "跨进程"),
        ("security", "trustzone", "secure world", "安全世界", "可信执行"),
        ("crypto", "aes-gcm", "hkdf", "key", "密钥", "nonce"),
        ("checkpoint", "checkpoint", "resume", "restart", "恢复", "重启"),
        ("retrieval", "retrieval", "relevant", "evidence", "检索"),
        ("ranking", "ranking", "rank", "mrr", "reciprocal"),
        ("recall", "recall", "coverage", "candidate", "召回"),
        ("hybrid", "hybrid", "fusion", "lexical", "semantic", "混合"),
        ("microkernel", "qnx", "microkernel", "user-space", "微内核", "用户态"),
        ("authentication", "authentication", "credential", "secret", "身份"),
    )

    @classmethod
    def _vector(cls, text: str) -> EmbeddingVector:
        folded = text.casefold()
        values = tuple(
            0.01 + sum(folded.count(term) for term in terms)
            for terms in cls._concepts
        )
        return EmbeddingVector(values)

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        return self._vector(text)


class BenchmarkReranker:
    """Offline content-driven scoring fixture; not a model-quality claim."""

    _concepts = BenchmarkEmbeddingProvider._concepts

    def __init__(self) -> None:
        self._analyzer = UnicodeCJKLexicalAnalyzer()

    def rerank(
        self,
        query: str,
        candidates: tuple[SearchHit, ...],
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        query_folded = query.casefold()
        query_terms = set(self._analyzer.analyze(query))
        scored: list[tuple[SearchHit, int]] = []
        for rank, hit in enumerate(candidates):
            content_folded = hit.chunk.content.casefold()
            content_terms = set(self._analyzer.analyze(hit.chunk.content))
            concept_matches = sum(
                1
                for terms in self._concepts
                if any(term in query_folded for term in terms)
                and any(term in content_folded for term in terms)
            )
            exact_matches = len(query_terms & content_terms)
            score = float(concept_matches * 10 + exact_matches)
            scored.append(
                (SearchHit(hit.chunk, score, hit.matched_terms), rank)
            )
        scored.sort(
            key=lambda item: (-item[0].score, item[1], item[0].chunk.chunk_id)
        )
        return tuple(hit for hit, _ in scored[:limit])


def _lexical_factory():
    return InMemoryChunkIndex(analyzer=UnicodeCJKLexicalAnalyzer())


def _build_collection(mode: str) -> KnowledgeCollection:
    provider = BenchmarkEmbeddingProvider()
    if mode == "lexical":
        index_factory = _lexical_factory
    elif mode == "semantic":
        index_factory = lambda: InMemorySemanticChunkIndex(embedding_provider=provider)
    elif mode in ("hybrid", "reranked"):
        hybrid_factory = lambda: HybridChunkIndex(
            lexical_index_factory=_lexical_factory,
            semantic_index_factory=lambda: InMemorySemanticChunkIndex(
                embedding_provider=provider
            ),
        )
        if mode == "hybrid":
            index_factory = hybrid_factory
        else:
            index_factory = lambda: RerankedChunkIndex(
                base_index_factory=hybrid_factory,
                reranker=BenchmarkReranker(),
                candidate_depth=100,
            )
    else:
        raise ValueError("unknown benchmark backend")
    collection = KnowledgeCollection(
        chunker=TextChunker(chunk_size=1_000, overlap=100),
        index_factory=index_factory,
    )
    collection.sync(LocalDirectoryAdapter(BENCHMARK_CORPUS, source_id=BENCHMARK_SOURCE_ID))
    return collection


def run_retrieval_benchmark() -> RetrievalComparisonReport:
    """Run all checked-in backends without network access or mutable state."""

    cases = load_retrieval_evaluation_cases(BENCHMARK_CASES)
    backends = (
        RetrievalBackend("BM25-only", lambda: _build_collection("lexical")),
        RetrievalBackend("Semantic-only", lambda: _build_collection("semantic")),
        RetrievalBackend("Hybrid-RRF", lambda: _build_collection("hybrid")),
        RetrievalBackend(
            "Hybrid-RRF + Rerank", lambda: _build_collection("reranked")
        ),
    )
    return compare_retrieval_backends(backends, cases, ks=BENCHMARK_KS)


def render_retrieval_comparison(
    report: RetrievalComparisonReport,
    config: RetrievalBenchmarkRenderConfig,
) -> str:
    """Render only supplied immutable values; perform no I/O or retrieval."""

    if not isinstance(report, RetrievalComparisonReport):
        raise TypeError("report must be a RetrievalComparisonReport")
    if not isinstance(config, RetrievalBenchmarkRenderConfig):
        raise TypeError("config must be a RetrievalBenchmarkRenderConfig")
    lines = [
        f"# {config.title}",
        "",
        config.corpus_summary + ". The baseline is descriptive/non-gate.",
        "All cutoffs are strict prefixes of one search at `max(K)` per case.",
        "Backends are compared only after their canonical snapshots are exactly equal.",
        "",
        "## Configuration",
        "",
        f"- Backends: {', '.join(item.backend_name for item in report.backend_reports)}",
        f"- K values: {', '.join(str(k) for k in report.ks)}",
        "- Semantic vectors: deterministic authored concept fixture; not real-model quality",
        "- Relevance labels: authored independently of backend output",
        "",
        "## Overall metrics",
        "",
        "| Backend | K | Hit@K | Recall@K | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for backend in report.backend_reports:
        for item in backend.reports_by_k:
            lines.append(
                f"| {backend.backend_name} | {item.k} | {item.hit_at_k:.6f} | "
                f"{item.recall_at_k:.6f} | {item.mrr:.6f} |"
            )
    lines.extend(["", "## Per-category metrics", ""])
    for category in RetrievalCategory:
        emitted = False
        category_lines: list[str] = []
        for backend in report.backend_reports:
            for item in backend.reports_by_k:
                match = next(
                    (value for value in item.category_reports if value.category is category),
                    None,
                )
                if match is None:
                    continue
                emitted = True
                category_lines.append(
                    f"| {backend.backend_name} | {item.k} | {match.case_count} | "
                    f"{match.hit_at_k:.6f} | {match.recall_at_k:.6f} | {match.mrr:.6f} |"
                )
        if emitted:
            lines.extend(
                [
                    f"### {category.value}",
                    "",
                    "| Backend | K | Cases | Hit@K | Recall@K | MRR |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                    *category_lines,
                    "",
                ]
            )
    lines.extend(["## Selected diagnostics", ""])
    first_backend = report.backend_reports[0]
    small_results = first_backend.reports_by_k[0].case_results
    selected_ids = [
        result.case_id
        for case_index, result in enumerate(small_results)
        if any(
            backend.reports_by_k[0].case_results[case_index].recall_at_k < 1.0
            for backend in report.backend_reports
        )
    ][: config.max_diagnostics]
    for case_id in selected_ids:
        reference = next(
            result
            for result in first_backend.reports_by_k[-1].case_results
            if result.case_id == case_id
        )
        relevant = ", ".join(
            f"{target.source_id}/{target.logical_path}" for target in reference.relevant_targets
        )
        lines.extend(
            [
                f"### case: {case_id}",
                f"- category: {reference.category.value}",
                f"- query: {reference.query}",
                f"- relevant targets: {relevant}",
            ]
        )
        for backend in report.backend_reports:
            maximum = next(
                result
                for result in backend.reports_by_k[-1].case_results
                if result.case_id == case_id
            )
            rank = "missed" if maximum.first_relevant_rank is None else str(maximum.first_relevant_rank)
            found = ", ".join(target.logical_path for target in maximum.relevant_targets_found) or "none"
            missed = ", ".join(target.logical_path for target in maximum.relevant_targets_missed) or "none"
            chunks = ", ".join(maximum.returned_chunk_ids) or "none"
            lines.append(
                f"- {backend.backend_name}: first relevant rank={rank}; "
                f"found={found}; missed={missed}; returned chunks={chunks}"
            )
        lines.append("")
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            config.reproduction_command,
            "```",
            "",
            "This report exposes regressions for review but defines no quality threshold.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the retrieval benchmark report")
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)
    rendered = render_retrieval_comparison(run_retrieval_benchmark(), DEFAULT_RENDER_CONFIG)
    args.write.write_bytes(rendered.encode("utf-8"))
    return 0


__all__ = [
    "BENCHMARK_CASES",
    "BENCHMARK_CORPUS",
    "BENCHMARK_KS",
    "BENCHMARK_ROOT",
    "DEFAULT_RENDER_CONFIG",
    "BenchmarkEmbeddingProvider",
    "RetrievalBenchmarkRenderConfig",
    "render_retrieval_comparison",
    "run_retrieval_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
