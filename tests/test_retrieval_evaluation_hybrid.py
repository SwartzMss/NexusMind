from __future__ import annotations

from pathlib import Path

from nexusmind import (
    EmbeddingVector,
    HybridChunkIndex,
    InMemoryChunkIndex,
    InMemorySemanticChunkIndex,
    KnowledgeCollection,
    LocalDirectoryAdapter,
    RetrievalEvaluationReport,
    TextChunker,
    evaluate_retrieval,
    load_retrieval_evaluation_cases,
)


EVAL_ROOT = Path(__file__).resolve().parents[1] / "evals" / "knowledge" / "hybrid"
CORPUS = EVAL_ROOT / "corpus"
CASES = EVAL_ROOT / "cases.json"
BASELINE = EVAL_ROOT / "baseline.md"
CHUNK_SIZE = 240
CHUNK_OVERLAP = 40
K = 2


class _HybridFixtureProvider:
    """Authored concept vectors for deterministic comparison, not model quality."""

    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector(self._values(text)) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        values = {
            "ZXQ-417": (0.0, 0.0, 0.0, 1.0),
            "How can login keys change during uninterrupted operation?": (
                0.0,
                1.0,
                0.0,
                0.0,
            ),
            "resume a workflow from a saved checkpoint": (0.0, 0.0, 1.0, 0.0),
        }[text]
        return EmbeddingVector(values)

    @staticmethod
    def _values(text: str) -> tuple[float, ...]:
        if "ZXQ-417" in text:
            return (1.0, 0.0, 0.0, 0.0)
        if "authentication secrets" in text:
            return (0.0, 1.0, 0.0, 0.0)
        if "durable checkpoint" in text:
            return (0.0, 0.0, 1.0, 0.0)
        if "dashboard" in text:
            return (0.0, 0.0, 0.0, 1.0)
        raise AssertionError("unmapped hybrid fixture document")


def _collection(mode: str) -> KnowledgeCollection:
    provider = _HybridFixtureProvider()
    if mode == "lexical":
        factory = InMemoryChunkIndex
    elif mode == "semantic":
        factory = lambda: InMemorySemanticChunkIndex(embedding_provider=provider)
    elif mode == "hybrid":
        factory = lambda: HybridChunkIndex(
            lexical_index_factory=InMemoryChunkIndex,
            semantic_index_factory=lambda: InMemorySemanticChunkIndex(
                embedding_provider=provider
            ),
        )
    else:
        raise AssertionError("unknown fixture mode")
    collection = KnowledgeCollection(
        chunker=TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
        index_factory=factory,
    )
    collection.sync(LocalDirectoryAdapter(CORPUS, source_id="hybrid-docs"))
    return collection


def _reports() -> dict[str, RetrievalEvaluationReport]:
    cases = load_retrieval_evaluation_cases(CASES)
    return {
        mode: evaluate_retrieval(_collection(mode), cases, k=K)
        for mode in ("lexical", "semantic", "hybrid")
    }


def test_hybrid_fixture_compares_three_modes_deterministically() -> None:
    first = _reports()
    second = _reports()

    assert first == second
    assert {case.case_id.split("-", 1)[0] for case in first["hybrid"].case_results} == {
        "lexical",
        "semantic",
        "mixed",
    }
    assert first["lexical"] != first["semantic"]
    assert first["hybrid"] != first["lexical"]
    assert first["hybrid"] != first["semantic"]
    for report in first.values():
        assert all(
            0.0 <= metric <= 1.0
            for metric in (report.hit_at_k, report.recall_at_k, report.mrr)
        )


def test_hybrid_baseline_documents_reproduction_and_non_gate_policy() -> None:
    baseline = BASELINE.read_text(encoding="utf-8")

    assert "descriptive" in baseline
    assert "non-gate" in baseline
    assert "BM25-only" in baseline
    assert "Semantic-only" in baseline
    assert "Hybrid-RRF" in baseline
    assert f"chunk_size={CHUNK_SIZE}" in baseline
    assert f"overlap={CHUNK_OVERLAP}" in baseline
    assert f"k={K}" in baseline
    assert "pytest -q tests/test_retrieval_evaluation_hybrid.py" in baseline
