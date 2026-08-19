from __future__ import annotations

from pathlib import Path

from nexusmind import (
    EmbeddingVector,
    InMemorySemanticChunkIndex,
    KnowledgeCollection,
    KnowledgeSearchResult,
    LocalDirectoryAdapter,
    TextChunker,
    evaluate_retrieval,
    load_retrieval_evaluation_cases,
)


EVAL_ROOT = Path(__file__).resolve().parents[1] / "evals" / "knowledge" / "semantic"
CORPUS = EVAL_ROOT / "corpus"
CASES = EVAL_ROOT / "cases.json"
BASELINE = EVAL_ROOT / "baseline.md"
CHUNK_SIZE = 240
CHUNK_OVERLAP = 40
K = 3


class _SemanticFixtureProvider:
    """Fixed concept vectors for offline plumbing tests, not a quality model."""

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        self.document_calls.append(texts)
        return tuple(EmbeddingVector(self._document_values(text)) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        self.query_calls.append(text)
        values = {
            "How do Android apps invoke a service living in another process?": (0.70, 0.71, 0.0, 0.0),
            "Which mobile mechanism transports structured remote calls between applications?": (1.0, 0.1, 0.0, 0.0),
            "Where should Arm devices execute operations hidden from the ordinary operating system?": (0.1, 1.0, 0.0, 0.0),
            "What architecture changes processor security state to reach protected services?": (0.0, 1.0, 0.1, 0.0),
            "Which operating system design can restart a crashed driver without losing the kernel?": (0.0, 0.1, 1.0, 0.0),
            "Where are system services placed when the privileged core is intentionally minimal?": (0.0, 0.0, 1.0, 0.1),
            "How can one root secret produce independent keys for several purposes?": (0.0, 0.0, 0.1, 1.0),
            "Which construction protects confidentiality and detects ciphertext modification?": (0.1, 0.0, 0.0, 1.0),
        }[text]
        return EmbeddingVector(values)

    @staticmethod
    def _document_values(text: str) -> tuple[float, ...]:
        if "Binder" in text:
            return (1.0, 0.0, 0.0, 0.0)
        if "TrustZone" in text:
            return (0.0, 1.0, 0.0, 0.0)
        if "QNX" in text:
            return (0.0, 0.0, 1.0, 0.0)
        if "HKDF" in text:
            return (0.0, 0.0, 0.0, 1.0)
        raise AssertionError("unmapped semantic fixture document")


def _collection() -> tuple[KnowledgeCollection, _SemanticFixtureProvider]:
    provider = _SemanticFixtureProvider()
    collection = KnowledgeCollection(
        chunker=TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP),
        index_factory=lambda: InMemorySemanticChunkIndex(embedding_provider=provider),
    )
    collection.sync(LocalDirectoryAdapter(CORPUS, source_id="semantic-docs"))
    return collection, provider


def test_semantic_fixture_uses_real_pipeline_with_canonical_provenance() -> None:
    cases = load_retrieval_evaluation_cases(CASES)
    collection, provider = _collection()

    first = evaluate_retrieval(collection, cases, k=K)
    second = evaluate_retrieval(collection, cases, k=K)

    assert first == second
    assert len(cases) == 8
    assert len(first.case_results) == len(cases)
    assert all(0.0 <= metric <= 1.0 for metric in (first.hit_at_k, first.recall_at_k, first.mrr))
    assert first.mrr < 1.0
    assert provider.document_calls
    assert len(provider.query_calls) == len(cases) * 2
    for case in cases:
        results = collection.search(case.query, limit=K)
        assert all(isinstance(result, KnowledgeSearchResult) for result in results)
        assert all(result.hit.matched_terms == () for result in results)
        assert all(result.source.source_id == result.document.source_id for result in results)
        assert all(
            result.hit.chunk.content
            == result.document.content[
                result.hit.chunk.start_offset : result.hit.chunk.end_offset
            ]
            for result in results
        )


def test_semantic_baseline_documents_fixed_reproduction_policy() -> None:
    baseline = BASELINE.read_text(encoding="utf-8")

    assert "descriptive" in baseline
    assert "non-gate" in baseline
    assert "fixture" in baseline
    assert f"chunk_size={CHUNK_SIZE}" in baseline
    assert f"overlap={CHUNK_OVERLAP}" in baseline
    assert f"k={K}" in baseline
    assert "pytest -q tests/test_retrieval_evaluation_semantic.py" in baseline
