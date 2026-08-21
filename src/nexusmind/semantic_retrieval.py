"""Bounded in-memory semantic chunk retrieval using cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot

from .embeddings import EmbeddingProvider, EmbeddingVector
from .knowledge_chunking import Chunk
from .knowledge_retrieval import (
    ChunkIdentityConflictError,
    ChunkIndexError,
    DocumentReplacementError,
    RetrievalCandidateDiagnostic,
    RetrievalDiagnostics,
    RetrievalStage,
    SearchHit,
)


class SemanticChunkIndexError(ChunkIndexError):
    """Base class for controlled semantic-index failures."""


class SemanticChunkIndexLimitError(SemanticChunkIndexError):
    """A semantic index mutation or query exceeds a configured bound."""


class SemanticDimensionError(SemanticChunkIndexError):
    """An embedding dimension conflicts with committed index state."""


class SemanticEmbeddingError(SemanticChunkIndexError):
    """A configured embedding provider failed or returned invalid output."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticChunkIndexLimits:
    """Explicit memory, batching, dimension, and query bounds."""

    max_chunks: int = 10_000
    max_total_chars: int = 10_000_000
    max_total_vector_values: int = 2_000_000
    max_dimensions: int = 65_536
    max_chunks_per_document: int = 10_000
    max_embedding_batch_size: int = 2_048
    max_query_chars: int = 1_024
    max_results: int = 100

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


class InMemorySemanticChunkIndex:
    """Process-local brute-force semantic retrieval.

    The embedding provider is immutable runtime configuration and is safe to
    share between clones. Vector and chunk state is independently copied.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        limits: SemanticChunkIndexLimits | None = None,
    ) -> None:
        if not callable(getattr(embedding_provider, "embed_documents", None)):
            raise TypeError("embedding_provider must provide embed_documents")
        if not callable(getattr(embedding_provider, "embed_query", None)):
            raise TypeError("embedding_provider must provide embed_query")
        if limits is not None and not isinstance(limits, SemanticChunkIndexLimits):
            raise TypeError("limits must be SemanticChunkIndexLimits")
        self._provider = embedding_provider
        self._limits = limits or SemanticChunkIndexLimits()
        self._chunks: dict[str, Chunk] = {}
        self._document_chunks: dict[str, set[str]] = {}
        self._vectors: dict[str, EmbeddingVector] = {}
        self._dimension: int | None = None
        self._total_chars = 0

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        self._require_chunk_tuple(chunks)
        additions: dict[str, Chunk] = {}
        for chunk in chunks:
            self._require_chunk(chunk)
            candidate = additions.get(chunk.chunk_id)
            existing = self._chunks.get(chunk.chunk_id)
            if candidate is not None and candidate != chunk:
                raise ChunkIdentityConflictError(
                    f"chunk_id conflicts within input: {chunk.chunk_id}"
                )
            if existing is not None and existing != chunk:
                raise ChunkIdentityConflictError(
                    f"chunk_id conflicts with indexed chunk: {chunk.chunk_id}"
                )
            if existing is None:
                additions[chunk.chunk_id] = chunk
        if not additions:
            return

        candidate_chunks = self._chunks.copy()
        candidate_chunks.update(additions)
        candidate_documents = {
            document_id: chunk_ids.copy()
            for document_id, chunk_ids in self._document_chunks.items()
        }
        for chunk in additions.values():
            candidate_documents.setdefault(chunk.document_id, set()).add(chunk.chunk_id)
        total_chars = self._total_chars + sum(
            len(chunk.content) for chunk in additions.values()
        )
        self._preflight(candidate_chunks, candidate_documents, total_chars)
        vectors = self._embed_documents(
            tuple(chunk.content for chunk in additions.values()),
            retained_vectors=self._vectors,
        )
        candidate_vectors = self._vectors.copy()
        candidate_vectors.update(zip(additions, vectors, strict=True))
        dimension = self._validated_dimension(candidate_vectors)
        self._commit_candidate(
            chunks=candidate_chunks,
            document_chunks=candidate_documents,
            vectors=candidate_vectors,
            dimension=dimension,
            total_chars=total_chars,
        )

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._require_document_id(document_id)
        self._require_chunk_tuple(chunks)
        replacement: dict[str, Chunk] = {}
        for chunk in chunks:
            self._require_chunk(chunk)
            if chunk.document_id != document_id:
                raise DocumentReplacementError(
                    "all replacement chunks must belong to document_id"
                )
            if chunk.chunk_id in replacement:
                raise DocumentReplacementError(
                    "replacement contains duplicate chunk_id values"
                )
            replacement[chunk.chunk_id] = chunk

        old_ids = self._document_chunks.get(document_id, set())
        for chunk in replacement.values():
            existing = self._chunks.get(chunk.chunk_id)
            if existing is not None and existing != chunk:
                raise ChunkIdentityConflictError(
                    f"chunk_id conflicts with indexed chunk: {chunk.chunk_id}"
                )
            if existing is not None and chunk.chunk_id not in old_ids:
                raise ChunkIdentityConflictError(
                    f"chunk_id already belongs to another document: {chunk.chunk_id}"
                )

        candidate_chunks = self._chunks.copy()
        candidate_documents = {
            key: chunk_ids.copy()
            for key, chunk_ids in self._document_chunks.items()
        }
        candidate_vectors = self._vectors.copy()
        removed_chars = 0
        for chunk_id in old_ids:
            removed_chars += len(candidate_chunks[chunk_id].content)
            del candidate_chunks[chunk_id]
            del candidate_vectors[chunk_id]
        if replacement:
            candidate_chunks.update(replacement)
            candidate_documents[document_id] = set(replacement)
        else:
            candidate_documents.pop(document_id, None)
        total_chars = self._total_chars - removed_chars + sum(
            len(chunk.content) for chunk in replacement.values()
        )
        self._preflight(candidate_chunks, candidate_documents, total_chars)

        new_chunks = {
            chunk_id: chunk
            for chunk_id, chunk in replacement.items()
            if self._chunks.get(chunk_id) != chunk
        }
        if new_chunks:
            vectors = self._embed_documents(
                tuple(chunk.content for chunk in new_chunks.values()),
                retained_vectors=candidate_vectors,
            )
            candidate_vectors.update(zip(new_chunks, vectors, strict=True))
        for chunk_id, chunk in replacement.items():
            if chunk_id not in candidate_vectors:
                candidate_vectors[chunk_id] = self._vectors[chunk_id]

        dimension = self._validated_dimension(candidate_vectors)
        self._commit_candidate(
            chunks=candidate_chunks,
            document_chunks=candidate_documents,
            vectors=candidate_vectors,
            dimension=dimension,
            total_chars=total_chars,
        )

    def remove_document(self, document_id: str) -> None:
        self._require_document_id(document_id)
        candidate_chunks = self._chunks.copy()
        candidate_documents = {
            key: chunk_ids.copy()
            for key, chunk_ids in self._document_chunks.items()
        }
        candidate_vectors = self._vectors.copy()
        chunk_ids = candidate_documents.pop(document_id, set())
        removed_chars = 0
        for chunk_id in chunk_ids:
            removed_chars += len(candidate_chunks[chunk_id].content)
            del candidate_chunks[chunk_id]
            del candidate_vectors[chunk_id]
        dimension = self._validated_dimension(candidate_vectors)
        self._commit_candidate(
            chunks=candidate_chunks,
            document_chunks=candidate_documents,
            vectors=candidate_vectors,
            dimension=dimension,
            total_chars=self._total_chars - removed_chars,
        )

    def clone(self) -> "InMemorySemanticChunkIndex":
        """Copy mutable index state while sharing immutable provider config."""

        clone = InMemorySemanticChunkIndex(
            embedding_provider=self._provider,
            limits=self._limits,
        )
        clone._chunks = self._chunks.copy()
        clone._document_chunks = {
            document_id: chunk_ids.copy()
            for document_id, chunk_ids in self._document_chunks.items()
        }
        clone._vectors = self._vectors.copy()
        clone._dimension = self._dimension
        clone._total_chars = self._total_chars
        return clone

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        return self._compute_hits(query, limit=limit)

    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics:
        hits = self._compute_hits(query, limit=limit)
        candidates = tuple(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.SEMANTIC,
                rank=rank,
                chunk=hit.chunk,
                score=hit.score,
                matched_terms=hit.matched_terms,
                selected=True,
            )
            for rank, hit in enumerate(hits, start=1)
        )
        return RetrievalDiagnostics(hits=hits, candidates=candidates)

    def _compute_hits(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        if type(query) is not str:
            raise TypeError("query must be a string")
        if len(query) > self._limits.max_query_chars:
            raise SemanticChunkIndexLimitError("query exceeds max_query_chars")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if limit > self._limits.max_results:
            raise SemanticChunkIndexLimitError("limit exceeds max_results")
        if not self._chunks or not query.strip():
            return ()

        query_vector = self._embed_query(query)
        if len(query_vector.values) != self._dimension:
            raise SemanticDimensionError("query embedding dimension does not match index")
        query_norm = self._norm(query_vector)
        hits: list[SearchHit] = []
        for chunk_id, chunk in self._chunks.items():
            vector = self._vectors[chunk_id]
            vector_norm = self._norm(vector)
            score = fsum(
                (left / query_norm) * (right / vector_norm)
                for left, right in zip(query_vector.values, vector.values, strict=True)
            )
            score = max(-1.0, min(1.0, score))
            hits.append(SearchHit(chunk=chunk, score=float(score)))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(hits[:limit])

    def _embed_documents(
        self,
        texts: tuple[str, ...],
        *,
        retained_vectors: dict[str, EmbeddingVector],
    ) -> tuple[EmbeddingVector, ...]:
        try:
            vectors: list[EmbeddingVector] = []
            batch_size = self._limits.max_embedding_batch_size
            expected_dimension = self._dimension
            vector_values = sum(
                len(vector.values) for vector in retained_vectors.values()
            )
            for offset in range(0, len(texts), batch_size):
                batch = texts[offset : offset + batch_size]
                batch_vectors = self._provider.embed_documents(batch)
                if (
                    type(batch_vectors) is not tuple
                    or len(batch_vectors) != len(batch)
                ):
                    raise TypeError
                if any(
                    type(vector) is not EmbeddingVector for vector in batch_vectors
                ):
                    raise TypeError
                for vector in batch_vectors:
                    dimension = len(vector.values)
                    if dimension > self._limits.max_dimensions:
                        raise SemanticChunkIndexLimitError(
                            "embedding exceeds max_dimensions"
                        )
                    if expected_dimension is None:
                        expected_dimension = dimension
                    elif dimension != expected_dimension:
                        raise SemanticDimensionError(
                            "embedding dimension does not match index"
                        )
                    vector_values += dimension
                    if vector_values > self._limits.max_total_vector_values:
                        raise SemanticChunkIndexLimitError(
                            "index exceeds max_total_vector_values"
                        )
                vectors.extend(batch_vectors)
            return tuple(vectors)
        except SemanticChunkIndexError:
            raise
        except Exception as exc:
            raise SemanticEmbeddingError("document embedding failed") from exc

    def _embed_query(self, query: str) -> EmbeddingVector:
        try:
            vector = self._provider.embed_query(query)
            if type(vector) is not EmbeddingVector:
                raise TypeError
            if len(vector.values) > self._limits.max_dimensions:
                raise SemanticChunkIndexLimitError(
                    "query embedding exceeds max_dimensions"
                )
            return vector
        except SemanticChunkIndexError:
            raise
        except Exception as exc:
            raise SemanticEmbeddingError("query embedding failed") from exc

    def _validated_dimension(self, vectors: dict[str, EmbeddingVector]) -> int | None:
        dimensions = {len(vector.values) for vector in vectors.values()}
        if not dimensions:
            return None
        if len(dimensions) != 1:
            raise SemanticDimensionError("embedding dimensions are inconsistent")
        dimension = next(iter(dimensions))
        if dimension > self._limits.max_dimensions:
            raise SemanticChunkIndexLimitError("embedding exceeds max_dimensions")
        if self._dimension is not None and dimension != self._dimension:
            raise SemanticDimensionError("embedding dimension does not match index")
        if len(vectors) * dimension > self._limits.max_total_vector_values:
            raise SemanticChunkIndexLimitError("index exceeds max_total_vector_values")
        return dimension

    def _preflight(
        self,
        chunks: dict[str, Chunk],
        document_chunks: dict[str, set[str]],
        total_chars: int,
    ) -> None:
        if len(chunks) > self._limits.max_chunks:
            raise SemanticChunkIndexLimitError("index exceeds max_chunks")
        if total_chars > self._limits.max_total_chars:
            raise SemanticChunkIndexLimitError("index exceeds max_total_chars")
        if any(
            len(chunk_ids) > self._limits.max_chunks_per_document
            for chunk_ids in document_chunks.values()
        ):
            raise SemanticChunkIndexLimitError(
                "document exceeds max_chunks_per_document"
            )

    def _commit_candidate(
        self,
        *,
        chunks: dict[str, Chunk],
        document_chunks: dict[str, set[str]],
        vectors: dict[str, EmbeddingVector],
        dimension: int | None,
        total_chars: int,
    ) -> None:
        self._chunks = chunks
        self._document_chunks = document_chunks
        self._vectors = vectors
        self._dimension = dimension
        self._total_chars = total_chars

    @staticmethod
    def _norm(vector: EmbeddingVector) -> float:
        return hypot(*vector.values)

    @staticmethod
    def _require_chunk_tuple(chunks: tuple[Chunk, ...]) -> None:
        if type(chunks) is not tuple:
            raise TypeError("chunks must be a tuple")

    @staticmethod
    def _require_chunk(chunk: Chunk) -> None:
        if not isinstance(chunk, Chunk):
            raise TypeError("chunks must contain only Chunk values")

    @staticmethod
    def _require_document_id(document_id: str) -> None:
        if type(document_id) is not str or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")


__all__ = [
    "InMemorySemanticChunkIndex",
    "SemanticChunkIndexError",
    "SemanticChunkIndexLimitError",
    "SemanticChunkIndexLimits",
    "SemanticDimensionError",
    "SemanticEmbeddingError",
]
