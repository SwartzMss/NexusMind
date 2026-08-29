from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from nexusmind import (
    Chunk,
    ChunkIndexLimits,
    Document,
    EmbeddingVector,
    HybridChunkIndex,
    HybridChunkIndexLimits,
    InMemoryChunkIndex,
    InMemorySemanticChunkIndex,
    KnowledgeCollection,
    KnowledgeSource,
    RerankedChunkIndex,
    SearchHit,
    SemanticChunkIndexLimits,
)
from nexusmind.search_diversification import (
    RankedDocumentCandidate,
    select_document_aware_indices,
)


@dataclass
class _Adapter:
    documents: tuple[Document, ...]
    source_id: str = "docs"

    def source(self) -> KnowledgeSource:
        return KnowledgeSource(
            source_id=self.source_id,
            source_type="fixture",
            display_name="Diversification fixture",
        )

    def load_documents(self) -> tuple[Document, ...]:
        return self.documents


class _SegmentChunker:
    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for segment in document.content.split("|"):
            start = document.content.index(segment)
            marker = segment.rsplit(" ", 1)[-1].lower()
            chunks.append(
                Chunk(
                    document.document_id,
                    f"chunk:{marker}",
                    segment,
                    start,
                    start + len(segment),
                )
            )
        return tuple(chunks)


class _EqualEmbeddingProvider:
    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector((1.0, 0.0))


class _StableReranker:
    def rerank(
        self, query: str, candidates: tuple[SearchHit, ...], *, limit: int
    ) -> tuple[SearchHit, ...]:
        ordered = tuple(sorted(candidates, key=lambda item: item.chunk.chunk_id))
        scores = (10.0, 9.0, 8.0, 7.6, 7.5, 7.0)
        return tuple(
            SearchHit(hit.chunk, scores[rank], hit.matched_terms)
            for rank, hit in enumerate(ordered[:limit])
        )


def _semantic_index() -> InMemorySemanticChunkIndex:
    return InMemorySemanticChunkIndex(embedding_provider=_EqualEmbeddingProvider())


def _lexical_factory():
    return InMemoryChunkIndex()


def _semantic_factory():
    return _semantic_index()


def _hybrid_factory():
    return HybridChunkIndex(
        lexical_index_factory=InMemoryChunkIndex,
        semantic_index_factory=_semantic_index,
    )


def _bounded_hybrid_factory():
    return HybridChunkIndex(
        lexical_index_factory=InMemoryChunkIndex,
        semantic_index_factory=_semantic_index,
        limits=HybridChunkIndexLimits(
            max_results=20,
            max_candidates_per_backend=10,
            max_fusion_entries=20,
        ),
        candidate_depth=10,
    )


def _child_bounded_hybrid_factory():
    return HybridChunkIndex(
        lexical_index_factory=lambda: InMemoryChunkIndex(
            limits=ChunkIndexLimits(max_results=5)
        ),
        semantic_index_factory=lambda: InMemorySemanticChunkIndex(
            embedding_provider=_EqualEmbeddingProvider(),
            limits=SemanticChunkIndexLimits(max_results=5),
        ),
        limits=HybridChunkIndexLimits(
            max_results=20,
            max_candidates_per_backend=10,
            max_fusion_entries=20,
        ),
        candidate_depth=5,
    )


def _reranked_factory():
    return RerankedChunkIndex(
        base_index_factory=InMemoryChunkIndex,
        reranker=_StableReranker(),
        candidate_depth=100,
    )


@pytest.mark.parametrize(
    ("index_factory", "minimum_unique_documents"),
    [
        (_lexical_factory, 2),
        (_semantic_factory, 2),
        (_hybrid_factory, 1),
        (_reranked_factory, 2),
    ],
    ids=["lexical", "semantic", "hybrid", "reranked"],
)
def test_every_final_backend_applies_selector_and_preserves_its_own_raw_values(
    index_factory: Callable[[], object], minimum_unique_documents: int
) -> None:
    document_a = Document(
        "docs", "a.md", "broad A0|broad A1|broad A2|broad A3"
    )
    document_b = Document("docs", "b.md", "broad B0")
    document_c = Document("docs", "c.md", "broad C0")
    collection = KnowledgeCollection(
        chunker=_SegmentChunker(),
        index_factory=index_factory,  # type: ignore[arg-type]
    )
    collection.sync(_Adapter((document_a, document_b, document_c)))
    raw = collection.diagnose_search("broad", limit=12).results
    expected_indices = select_document_aware_indices(
        tuple(
            RankedDocumentCandidate(item.document.document_id, item.hit.score)
            for item in raw
        ),
        limit=3,
    )

    results = collection.search("broad", limit=3)

    assert tuple(item.document.document_id for item in raw[:3]) == (
        document_a.document_id,
        document_a.document_id,
        document_a.document_id,
    )
    assert len(results) == 3
    assert tuple(item.hit.chunk.chunk_id for item in results) == tuple(
        raw[index].hit.chunk.chunk_id for index in expected_indices
    )
    assert (
        len({item.document.document_id for item in results})
        >= minimum_unique_documents
    )
    assert results == collection.search("broad", limit=3)
    raw_by_chunk = {item.hit.chunk.chunk_id: item.hit for item in raw}
    raw_positions = {item.hit.chunk.chunk_id: rank for rank, item in enumerate(raw)}
    assert tuple(raw_positions[item.hit.chunk.chunk_id] for item in results) == tuple(
        sorted(raw_positions[item.hit.chunk.chunk_id] for item in results)
    )
    for result in results:
        raw_hit = raw_by_chunk[result.hit.chunk.chunk_id]
        assert result.hit.score == raw_hit.score
        assert result.hit.matched_terms == raw_hit.matched_terms


def test_collection_oversampling_respects_custom_hybrid_capacity() -> None:
    documents = tuple(
        Document("docs", f"{index}.md", f"broad result {index}")
        for index in range(6)
    )
    collection = KnowledgeCollection(
        chunker=_SegmentChunker(),
        index_factory=_bounded_hybrid_factory,
    )
    collection.sync(_Adapter(documents))

    results = collection.search("broad", limit=5)

    assert len(results) == 5


def test_collection_oversampling_respects_hybrid_child_capacities() -> None:
    documents = tuple(
        Document("docs", f"{index}.md", f"broad result {index}")
        for index in range(6)
    )
    collection = KnowledgeCollection(
        chunker=_SegmentChunker(),
        index_factory=_child_bounded_hybrid_factory,
    )
    collection.sync(_Adapter(documents))

    results = collection.search("broad", limit=3)

    assert len(results) == 3
