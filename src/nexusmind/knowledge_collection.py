"""Process-local composition of ingestion, chunking, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .knowledge import Document, KnowledgeSource
from .knowledge_chunking import Chunk, TextChunker
from .knowledge_ingestion import KnowledgeSourceAdapter
from .knowledge_retrieval import ChunkIndex, InMemoryChunkIndex, SearchHit


class KnowledgeCollectionError(Exception):
    """Base class for controlled collection synchronization failures."""


class KnowledgeSnapshotError(KnowledgeCollectionError):
    """An adapter returned an invalid or incoherent source snapshot."""


class KnowledgeCollectionLimitError(KnowledgeCollectionError):
    """A synchronization would exceed collection bookkeeping limits."""


class DocumentChunker(Protocol):
    """Source-neutral document-to-chunk transformation contract."""

    def chunk(self, document: Document) -> tuple[Chunk, ...]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeCollectionLimits:
    """Bounds for process-local source and document bookkeeping."""

    max_sources: int = 100
    max_documents: int = 10_000

    def __post_init__(self) -> None:
        for name in ("max_sources", "max_documents"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class KnowledgeSyncResult:
    """Bounded summary of one successfully committed source sync."""

    source_id: str
    documents_added: int
    documents_updated: int
    documents_unchanged: int
    documents_removed: int
    chunks_indexed: int


class KnowledgeCollection:
    """Explicitly synchronize source snapshots into a staged chunk index.

    Synchronization clones the configured index, applies all deterministic
    mutations to that candidate, and only swaps committed state after every
    operation succeeds. Removing an unknown source is a no-op.
    """

    def __init__(
        self,
        *,
        chunker: DocumentChunker | None = None,
        index: ChunkIndex | None = None,
        limits: KnowledgeCollectionLimits | None = None,
    ) -> None:
        self._chunker = TextChunker() if chunker is None else chunker
        self._index = InMemoryChunkIndex() if index is None else index
        self._limits = KnowledgeCollectionLimits() if limits is None else limits
        if not callable(getattr(self._chunker, "chunk", None)):
            raise TypeError("chunker must implement chunk(document)")
        for method in ("add", "replace_document", "remove_document", "search", "clone"):
            if not callable(getattr(self._index, method, None)):
                raise TypeError(f"index must implement {method}()")
        if not isinstance(self._limits, KnowledgeCollectionLimits):
            raise TypeError("limits must be KnowledgeCollectionLimits")
        self._sources: dict[str, KnowledgeSource] = {}
        self._documents: dict[str, dict[str, Document]] = {}

    def sync(self, adapter: KnowledgeSourceAdapter) -> KnowledgeSyncResult:
        if not callable(getattr(adapter, "source", None)) or not callable(
            getattr(adapter, "load_documents", None)
        ):
            raise TypeError("adapter must implement KnowledgeSourceAdapter")
        source = adapter.source()
        if not isinstance(source, KnowledgeSource):
            raise KnowledgeSnapshotError("adapter source must be a KnowledgeSource")
        documents = adapter.load_documents()
        if type(documents) is not tuple:
            raise KnowledgeSnapshotError("adapter documents must be a tuple")

        incoming: dict[str, Document] = {}
        for document in documents:
            if not isinstance(document, Document):
                raise KnowledgeSnapshotError("adapter snapshot must contain only Documents")
            if document.source_id != source.source_id:
                raise KnowledgeSnapshotError("all documents must belong to the adapter source_id")
            if document.document_id in incoming:
                raise KnowledgeSnapshotError("adapter snapshot contains duplicate document_id values")
            incoming[document.document_id] = document

        old = self._documents.get(source.source_id, {})
        old_ids = set(old)
        incoming_ids = set(incoming)
        added = incoming_ids - old_ids
        removed = old_ids - incoming_ids
        shared = old_ids & incoming_ids
        changed = {
            document_id
            for document_id in shared
            if old[document_id].content_hash != incoming[document_id].content_hash
        }
        unchanged = shared - changed
        self._preflight_snapshot(source.source_id, len(incoming))

        prepared: dict[str, tuple[Chunk, ...]] = {}
        for document_id in sorted(added | changed):
            chunks = self._chunker.chunk(incoming[document_id])
            if type(chunks) is not tuple:
                raise KnowledgeSnapshotError("chunker must return a tuple")
            prepared[document_id] = chunks

        staged = self._index.clone()
        for document_id in sorted(removed):
            staged.remove_document(document_id)
        for document_id in sorted(prepared):
            staged.replace_document(document_id, prepared[document_id])

        self._index = staged
        self._sources[source.source_id] = source
        self._documents[source.source_id] = incoming
        return KnowledgeSyncResult(
            source_id=source.source_id,
            documents_added=len(added),
            documents_updated=len(changed),
            documents_unchanged=len(unchanged),
            documents_removed=len(removed),
            chunks_indexed=sum(len(chunks) for chunks in prepared.values()),
        )

    def remove_source(self, source_id: str) -> None:
        if type(source_id) is not str or not source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        documents = self._documents.get(source_id)
        if documents is None:
            return
        staged = self._index.clone()
        for document_id in sorted(documents):
            staged.remove_document(document_id)
        self._index = staged
        del self._documents[source_id]
        del self._sources[source_id]

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        return self._index.search(query, limit=limit)

    def _preflight_snapshot(self, source_id: str, incoming_count: int) -> None:
        source_count = len(self._sources) + (0 if source_id in self._sources else 1)
        if source_count > self._limits.max_sources:
            raise KnowledgeCollectionLimitError("collection exceeds max_sources")
        existing_count = sum(len(documents) for documents in self._documents.values())
        resulting_count = existing_count - len(self._documents.get(source_id, {})) + incoming_count
        if resulting_count > self._limits.max_documents:
            raise KnowledgeCollectionLimitError("collection exceeds max_documents")


__all__ = [
    "DocumentChunker",
    "KnowledgeCollection",
    "KnowledgeCollectionError",
    "KnowledgeCollectionLimitError",
    "KnowledgeCollectionLimits",
    "KnowledgeSnapshotError",
    "KnowledgeSyncResult",
]
