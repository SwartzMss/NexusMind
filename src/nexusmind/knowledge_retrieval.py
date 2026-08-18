"""Source-neutral retrieval contracts and a bounded in-memory lexical index.

The lexical implementation uses Unicode ``str.casefold`` normalization. A
distinct whitespace-delimited query term scores one point when its normalized
text occurs in a chunk. Results are ordered by descending score and then by
ascending ``chunk_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .knowledge_chunking import Chunk


class ChunkIndexError(Exception):
    """Base class for controlled chunk-index failures."""


class ChunkIndexLimitError(ChunkIndexError):
    """An index mutation or query exceeds a configured resource bound."""


class ChunkIdentityConflictError(ChunkIndexError):
    """A chunk ID was reused for different chunk data."""


class DocumentReplacementError(ChunkIndexError):
    """A document replacement input is invalid."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkIndexLimits:
    """Explicit resource bounds for an in-memory chunk index."""

    max_chunks: int = 10_000
    max_total_chars: int = 10_000_000
    max_chunks_per_document: int = 10_000
    max_query_chars: int = 1_024
    max_query_terms: int = 32
    max_results: int = 100

    def __post_init__(self) -> None:
        for name in (
            "max_chunks",
            "max_total_chars",
            "max_chunks_per_document",
            "max_query_chars",
            "max_query_terms",
            "max_results",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One deterministic lexical match and its explicit score details."""

    chunk: Chunk
    score: int
    matched_terms: tuple[str, ...]


class ChunkIndex(Protocol):
    """Source-neutral contract implemented by chunk retrieval backends."""

    def add(self, chunks: tuple[Chunk, ...]) -> None: ...

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None: ...

    def remove_document(self, document_id: str) -> None: ...

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]: ...


class InMemoryChunkIndex:
    """Dependency-free, process-local lexical chunk retrieval."""

    def __init__(self, *, limits: ChunkIndexLimits | None = None) -> None:
        if limits is not None and not isinstance(limits, ChunkIndexLimits):
            raise TypeError("limits must be ChunkIndexLimits")
        self._limits = limits or ChunkIndexLimits()
        self._chunks: dict[str, Chunk] = {}
        self._document_chunks: dict[str, set[str]] = {}
        self._total_chars = 0

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        additions = self._validated_additions(chunks)
        document_counts = {key: len(value) for key, value in self._document_chunks.items()}
        for chunk in additions.values():
            document_counts[chunk.document_id] = document_counts.get(chunk.document_id, 0) + 1
        self._preflight(
            chunk_count=len(self._chunks) + len(additions),
            total_chars=self._total_chars + sum(len(chunk.content) for chunk in additions.values()),
            document_counts=document_counts.values(),
        )
        for chunk in additions.values():
            self._chunks[chunk.chunk_id] = chunk
            self._document_chunks.setdefault(chunk.document_id, set()).add(chunk.chunk_id)
            self._total_chars += len(chunk.content)

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._require_document_id(document_id)
        self._require_chunk_tuple(chunks)
        replacement: dict[str, Chunk] = {}
        for chunk in chunks:
            self._require_chunk(chunk)
            if chunk.document_id != document_id:
                raise DocumentReplacementError("all replacement chunks must belong to document_id")
            if chunk.chunk_id in replacement:
                raise DocumentReplacementError("replacement contains duplicate chunk_id values")
            replacement[chunk.chunk_id] = chunk

        old_ids = self._document_chunks.get(document_id, set())
        for chunk in replacement.values():
            existing = self._chunks.get(chunk.chunk_id)
            if existing is not None and chunk.chunk_id not in old_ids and existing != chunk:
                raise ChunkIdentityConflictError(f"chunk_id conflicts with indexed chunk: {chunk.chunk_id}")
            if existing is not None and chunk.chunk_id not in old_ids:
                raise ChunkIdentityConflictError(f"chunk_id already belongs to another document: {chunk.chunk_id}")

        remaining_count = len(self._chunks) - len(old_ids)
        remaining_chars = self._total_chars - sum(len(self._chunks[key].content) for key in old_ids)
        other_counts = [len(ids) for key, ids in self._document_chunks.items() if key != document_id]
        self._preflight(
            chunk_count=remaining_count + len(replacement),
            total_chars=remaining_chars + sum(len(chunk.content) for chunk in replacement.values()),
            document_counts=(*other_counts, len(replacement)),
        )

        for chunk_id in old_ids:
            del self._chunks[chunk_id]
        if replacement:
            self._document_chunks[document_id] = set(replacement)
            self._chunks.update(replacement)
        else:
            self._document_chunks.pop(document_id, None)
        self._total_chars = remaining_chars + sum(len(chunk.content) for chunk in replacement.values())

    def remove_document(self, document_id: str) -> None:
        self._require_document_id(document_id)
        chunk_ids = self._document_chunks.pop(document_id, set())
        for chunk_id in chunk_ids:
            self._total_chars -= len(self._chunks[chunk_id].content)
            del self._chunks[chunk_id]

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        if type(query) is not str:
            raise TypeError("query must be a string")
        if len(query) > self._limits.max_query_chars:
            raise ChunkIndexLimitError("query exceeds max_query_chars")
        raw_terms = query.split()
        if len(raw_terms) > self._limits.max_query_terms:
            raise ChunkIndexLimitError("query exceeds max_query_terms")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if limit > self._limits.max_results:
            raise ChunkIndexLimitError("limit exceeds max_results")

        terms = tuple(dict.fromkeys(term.casefold() for term in raw_terms))
        if not terms:
            return ()
        hits: list[SearchHit] = []
        for chunk in self._chunks.values():
            normalized_content = chunk.content.casefold()
            matched = tuple(term for term in terms if term in normalized_content)
            if matched:
                hits.append(SearchHit(chunk=chunk, score=len(matched), matched_terms=matched))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(hits[:limit])

    def _validated_additions(self, chunks: tuple[Chunk, ...]) -> dict[str, Chunk]:
        self._require_chunk_tuple(chunks)
        additions: dict[str, Chunk] = {}
        for chunk in chunks:
            self._require_chunk(chunk)
            candidate = additions.get(chunk.chunk_id)
            existing = self._chunks.get(chunk.chunk_id)
            if candidate is not None and candidate != chunk:
                raise ChunkIdentityConflictError(f"chunk_id conflicts within input: {chunk.chunk_id}")
            if existing is not None and existing != chunk:
                raise ChunkIdentityConflictError(f"chunk_id conflicts with indexed chunk: {chunk.chunk_id}")
            if existing is None:
                additions[chunk.chunk_id] = chunk
        return additions

    def _preflight(self, *, chunk_count: int, total_chars: int, document_counts: object) -> None:
        if chunk_count > self._limits.max_chunks:
            raise ChunkIndexLimitError("index exceeds max_chunks")
        if total_chars > self._limits.max_total_chars:
            raise ChunkIndexLimitError("index exceeds max_total_chars")
        if any(count > self._limits.max_chunks_per_document for count in document_counts):
            raise ChunkIndexLimitError("document exceeds max_chunks_per_document")

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
    "ChunkIdentityConflictError",
    "ChunkIndex",
    "ChunkIndexError",
    "ChunkIndexLimitError",
    "ChunkIndexLimits",
    "DocumentReplacementError",
    "InMemoryChunkIndex",
    "SearchHit",
]
