from __future__ import annotations

import math

import pytest

from nexusmind import (
    Chunk,
    ChunkIndex,
    EmbeddingVector,
    InMemorySemanticChunkIndex,
    SearchHit,
    SemanticChunkIndexLimitError,
    SemanticChunkIndexLimits,
)


def _chunk(chunk_id: str, content: str, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


class _RecordingProvider:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        self.document_calls.append(texts)
        return tuple(EmbeddingVector(self.vectors[text]) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        self.query_calls.append(text)
        return EmbeddingVector(self.vectors[text])


def test_semantic_contracts_are_public_and_implement_chunk_index() -> None:
    provider = _RecordingProvider({})
    index: ChunkIndex = InMemorySemanticChunkIndex(embedding_provider=provider)

    assert index.search("query") == ()
    assert provider.query_calls == []


@pytest.mark.parametrize(
    "field",
    [
        "max_chunks",
        "max_total_chars",
        "max_total_vector_values",
        "max_dimensions",
        "max_chunks_per_document",
        "max_embedding_batch_size",
        "max_query_chars",
        "max_results",
    ],
)
@pytest.mark.parametrize("value", [True, 0, -1])
def test_semantic_limits_require_positive_plain_integers(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SemanticChunkIndexLimits(**{field: value})


def test_semantic_search_skips_embedding_for_empty_index_and_blank_query() -> None:
    provider = _RecordingProvider({"document": (1.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)

    assert index.search("not configured") == ()
    index.add((_chunk("one", "document"),))
    assert index.search(" \t\n") == ()
    assert provider.query_calls == []


def test_semantic_add_batches_and_search_ranks_all_cosine_scores() -> None:
    provider = _RecordingProvider(
        {
            "alpha": (1.0, 0.0),
            "diagonal": (1.0, 1.0),
            "opposite": (-1.0, 0.0),
            "query": (1.0, 0.0),
        }
    )
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    chunks = (
        _chunk("z-alpha", "alpha"),
        _chunk("middle", "diagonal"),
        _chunk("negative", "opposite"),
    )

    index.add(chunks)
    hits = index.search("query")

    assert provider.document_calls == [("alpha", "diagonal", "opposite")]
    assert provider.query_calls == ["query"]
    assert [hit.chunk.chunk_id for hit in hits] == ["z-alpha", "middle", "negative"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 1 / math.sqrt(2), -1.0])
    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
    assert all(hit.matched_terms == () for hit in hits)


def test_semantic_ties_use_chunk_id_and_result_limit() -> None:
    provider = _RecordingProvider({"one": (1.0, 0.0), "two": (2.0, 0.0), "q": (1.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("z", "one"), _chunk("a", "two")))

    hits = index.search("q", limit=1)

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]


def test_semantic_search_validates_query_and_limit_before_provider_call() -> None:
    provider = _RecordingProvider({"one": (1.0,), "q": (1.0,)})
    index = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(max_query_chars=1, max_results=1),
    )
    index.add((_chunk("one", "one"),))

    with pytest.raises(TypeError, match="query"):
        index.search(1)  # type: ignore[arg-type]
    with pytest.raises(SemanticChunkIndexLimitError, match="max_query_chars"):
        index.search("too long")
    with pytest.raises(TypeError, match="limit"):
        index.search("q", limit=True)
    with pytest.raises(ValueError, match="greater than zero"):
        index.search("q", limit=0)
    with pytest.raises(SemanticChunkIndexLimitError, match="max_results"):
        index.search("q", limit=2)
    assert provider.query_calls == []


def test_search_hit_defaults_to_empty_backend_diagnostics() -> None:
    hit = SearchHit(chunk=_chunk("one", "one"), score=0.5)

    assert hit.matched_terms == ()
