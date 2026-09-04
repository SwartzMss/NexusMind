"""Process-local composition of ingestion, chunking, and retrieval."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from typing import Callable, Protocol

from .context_assembly import ContextPackage, assemble_context

from .knowledge import (
    Document,
    DocumentVersion,
    KnowledgeSource,
    compute_content_hash,
    stable_document_id,
)
from .knowledge_chunking import Chunk, StructureAwareChunker
from .knowledge_inspection import KnowledgeChunkInspection, KnowledgeDocumentInspection
from .knowledge_ingestion import KnowledgeSourceAdapter
from .knowledge_retrieval import (
    ChunkIndex,
    ChunkIndexError,
    InMemoryChunkIndex,
    RetrievalCandidateDiagnostic,
    RetrievalDiagnostics,
    RetrievalStage,
    SearchHit,
)
from .search_diversification import (
    RankedDocumentCandidate,
    search_candidate_depth,
    select_document_aware_indices,
)


class KnowledgeCollectionError(Exception):
    """Base class for controlled collection state failures."""


class KnowledgeSnapshotError(KnowledgeCollectionError):
    """A source or restore snapshot is invalid or incoherent."""


class KnowledgeCollectionLimitError(KnowledgeCollectionError):
    """A synchronization would exceed collection bookkeeping limits."""


class KnowledgeRestoreError(KnowledgeCollectionError):
    """A snapshot cannot be restored into a coherent collection state."""


class KnowledgeSearchResolutionError(KnowledgeCollectionError):
    """A retrieval hit cannot be resolved to committed canonical state."""


class KnowledgeInspectionError(KnowledgeCollectionError):
    """Canonical document chunks cannot be inspected coherently."""


class DocumentChunker(Protocol):
    """Source-neutral document-to-chunk transformation contract."""

    def chunk(self, document: Document) -> tuple[Chunk, ...]: ...


class CloneableChunkIndex(ChunkIndex, Protocol):
    """Collection-specific capability for copy-on-write index staging."""

    def clone(self) -> "CloneableChunkIndex":
        """Return an independent copy suitable for staging atomic updates."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeCollectionLimits:
    """Bounds for process-local source and document bookkeeping."""

    max_sources: int = 100
    max_documents: int = 10_000
    max_document_versions: int = 100_000

    def __post_init__(self) -> None:
        for name in ("max_sources", "max_documents", "max_document_versions"):
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


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    """Frozen container of detached canonical copies, excluding derived state."""

    sources: tuple[KnowledgeSource, ...]
    documents: tuple[Document, ...]
    document_versions: tuple[DocumentVersion, ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeRestoreResult:
    """Bounded summary of one successfully committed snapshot restore."""

    sources_restored: int
    documents_restored: int
    chunks_indexed: int


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    """One retrieval hit with detached canonical source and document provenance."""

    source: KnowledgeSource
    document: Document
    hit: SearchHit


@dataclass(frozen=True, slots=True)
class KnowledgeRetrievalCandidateDiagnostic:
    """One backend diagnostic row with detached canonical provenance."""

    source: KnowledgeSource
    document: Document
    diagnostic: RetrievalCandidateDiagnostic

    def __post_init__(self) -> None:
        if type(self.source) is not KnowledgeSource:
            raise TypeError("source must be a KnowledgeSource")
        if type(self.document) is not Document:
            raise TypeError("document must be a Document")
        if type(self.diagnostic) is not RetrievalCandidateDiagnostic:
            raise TypeError("diagnostic must be a RetrievalCandidateDiagnostic")
        RetrievalCandidateDiagnostic(
            stage=self.diagnostic.stage,
            rank=self.diagnostic.rank,
            chunk=self.diagnostic.chunk,
            score=self.diagnostic.score,
            matched_terms=self.diagnostic.matched_terms,
            rrf_contribution=self.diagnostic.rrf_contribution,
            selected=self.diagnostic.selected,
        )
        if self.document.source_id != self.source.source_id:
            raise ValueError("document must belong to source")
        if self.diagnostic.chunk.document_id != self.document.document_id:
            raise ValueError("diagnostic chunk must belong to document")


@dataclass(frozen=True, slots=True)
class KnowledgeRetrievalDiagnostics:
    """One query's final results and fully provenance-resolved candidate trace."""

    query: str
    results: tuple[KnowledgeSearchResult, ...]
    candidates: tuple[KnowledgeRetrievalCandidateDiagnostic, ...]

    def __post_init__(self) -> None:
        if type(self.query) is not str:
            raise TypeError("query must be a string")
        if type(self.results) is not tuple:
            raise TypeError("results must be a tuple")
        if any(type(result) is not KnowledgeSearchResult for result in self.results):
            raise TypeError("results must contain only KnowledgeSearchResult values")
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        if any(
            type(candidate) is not KnowledgeRetrievalCandidateDiagnostic
            for candidate in self.candidates
        ):
            raise TypeError(
                "candidates must contain only KnowledgeRetrievalCandidateDiagnostic values"
            )


class KnowledgeCollection:
    """Explicitly synchronize source snapshots into a staged chunk index.

    ``index_factory`` must return a new empty index owned by the collection.
    Synchronization clones its private index, applies all deterministic
    mutations to that candidate, and only swaps committed state after every
    operation succeeds. Removing an unknown source is a no-op.
    """

    def __init__(
        self,
        *,
        chunker: DocumentChunker | None = None,
        index_factory: Callable[[], CloneableChunkIndex] | None = None,
        limits: KnowledgeCollectionLimits | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._chunker = StructureAwareChunker() if chunker is None else chunker
        self._limits = KnowledgeCollectionLimits() if limits is None else limits
        if not callable(getattr(self._chunker, "chunk", None)):
            raise TypeError("chunker must implement chunk(document)")
        factory = InMemoryChunkIndex if index_factory is None else index_factory
        if not callable(factory):
            raise TypeError("index_factory must be callable")
        self._index_factory = factory
        self._index = self._index_factory()
        self._require_cloneable_index(self._index)
        if not isinstance(self._limits, KnowledgeCollectionLimits):
            raise TypeError("limits must be KnowledgeCollectionLimits")
        self._clock = self._utc_now if clock is None else clock
        if not callable(self._clock):
            raise TypeError("clock must be callable")
        self._sources: dict[str, KnowledgeSource] = {}
        self._documents: dict[str, dict[str, Document]] = {}
        self._chunks_by_document: dict[str, tuple[Chunk, ...]] = {}
        self._document_versions: dict[str, tuple[DocumentVersion, ...]] = {}

    @property
    def retrieval_backend_name(self) -> str:
        """Return the active process-local retrieval backend type for diagnostics."""

        return type(self._index).__name__

    def sync(self, adapter: KnowledgeSourceAdapter) -> KnowledgeSyncResult:
        if not callable(getattr(adapter, "source", None)) or not callable(
            getattr(adapter, "load_documents", None)
        ):
            raise TypeError("adapter must implement KnowledgeSourceAdapter")
        source = adapter.source()
        if not isinstance(source, KnowledgeSource):
            raise KnowledgeSnapshotError("adapter source must be a KnowledgeSource")
        owned_source = deepcopy(source)
        documents = adapter.load_documents()
        if type(documents) is not tuple:
            raise KnowledgeSnapshotError("adapter documents must be a tuple")

        incoming: dict[str, Document] = {}
        for document in documents:
            if not isinstance(document, Document):
                raise KnowledgeSnapshotError("adapter snapshot must contain only Documents")
            if document.source_id != source.source_id:
                raise KnowledgeSnapshotError("all documents must belong to the adapter source_id")
            self._validate_document_identity(document)
            if document.document_id in incoming:
                raise KnowledgeSnapshotError("adapter snapshot contains duplicate document_id values")
            owned_document = deepcopy(document)
            incoming[owned_document.document_id] = owned_document

        old = self._documents.get(owned_source.source_id, {})
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
        versioned_ids = added | changed
        self._preflight_snapshot(
            owned_source.source_id,
            len(incoming),
            added_versions=len(versioned_ids),
        )

        staged_versions = dict(self._document_versions)
        if versioned_ids:
            created_at = self._format_clock_value(self._clock())
            sync_context = self._sync_context(
                owned_source.source_id, created_at, len(self._all_versions())
            )
            for document_id in sorted(versioned_ids):
                chain = staged_versions.get(document_id, ())
                previous_version_id = chain[-1].version_id if chain else None
                version = DocumentVersion.from_document(
                    incoming[document_id],
                    created_at=created_at,
                    sync_context=sync_context,
                    previous_version_id=previous_version_id,
                )
                staged_versions[document_id] = chain + (version,)

        prepared: dict[str, tuple[Chunk, ...]] = {}
        for document_id in sorted(added | changed):
            chunk_document = deepcopy(incoming[document_id])
            chunks = self._chunker.chunk(chunk_document)
            if type(chunks) is not tuple:
                raise KnowledgeSnapshotError("chunker must return a tuple")
            prepared[document_id] = chunks

        staged = self._index.clone()
        self._require_cloneable_index(staged)
        staged_chunks = dict(self._chunks_by_document)
        for document_id in sorted(removed):
            staged.remove_document(document_id)
            staged_chunks.pop(document_id, None)
        for document_id in sorted(prepared):
            staged.replace_document(document_id, prepared[document_id])
            staged_chunks[document_id] = prepared[document_id]

        self._index = staged
        self._sources[owned_source.source_id] = owned_source
        self._documents[owned_source.source_id] = incoming
        self._chunks_by_document = staged_chunks
        self._document_versions = staged_versions
        return KnowledgeSyncResult(
            source_id=owned_source.source_id,
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
        self._require_cloneable_index(staged)
        staged_chunks = dict(self._chunks_by_document)
        for document_id in sorted(documents):
            staged.remove_document(document_id)
            staged_chunks.pop(document_id, None)
        self._index = staged
        del self._documents[source_id]
        del self._sources[source_id]
        self._chunks_by_document = staged_chunks

    def search(self, query: str, *, limit: int = 10) -> tuple[KnowledgeSearchResult, ...]:
        resolved = self.search_backend(query, limit=limit)
        ranked = tuple(
            RankedDocumentCandidate(item.document.document_id, item.hit.score)
            for item in resolved
        )
        selected = select_document_aware_indices(ranked, limit=limit)
        return tuple(resolved[index] for index in selected)

    def search_backend(self, query: str, *, limit: int = 10) -> tuple[KnowledgeSearchResult, ...]:
        """Return the configured backend's final ranking before document selection."""
        try:
            backend_capacity = getattr(self._index, "max_search_results", None)
        except Exception:
            backend_capacity = None
        candidate_depth = search_candidate_depth(
            limit,
            backend_capacity=backend_capacity,
        )
        hits = self._index.search(query, limit=candidate_depth)
        if type(hits) is not tuple:
            raise KnowledgeSearchResolutionError("index search result must be a tuple")
        resolved = tuple(self._resolve_hit(hit) for hit in hits)
        return resolved

    def context_chunk_catalog(
        self, document_ids: tuple[str, ...]
    ) -> dict[str, tuple[Chunk, ...]]:
        """Return the derived canonical chunks for the requested documents."""

        if type(document_ids) is not tuple:
            raise TypeError("document_ids must be a tuple")
        result: dict[str, tuple[Chunk, ...]] = {}
        for document_id in document_ids:
            if type(document_id) is not str or not document_id.strip():
                raise ValueError("document_ids must contain non-empty strings")
            chunks = self._chunks_by_document.get(document_id)
            if chunks is None:
                raise KnowledgeInspectionError(
                    "chunk catalog references an unknown document"
                )
            result[document_id] = tuple(chunks)
        return result

    def build_context(
        self,
        query: str,
        *,
        retrieval_limit: int = 10,
        max_passages: int = 10,
        max_chars: int | None = None,
        max_tokens: int | None = None,
    ) -> ContextPackage:
        """Retrieve candidates and assemble a bounded, provenance-preserving context.

        Retrieval depth and passage count are separate pipeline controls so callers
        can compare broader candidate retrieval with a bounded model context.
        """

        for name, value in (
            ("retrieval_limit", retrieval_limit),
            ("max_passages", max_passages),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        results = self.search(query, limit=retrieval_limit)
        return assemble_context(
            query,
            results,
            max_passages=max_passages,
            max_candidates=retrieval_limit,
            max_chars=max_chars,
            max_tokens=max_tokens,
        )

    def inspect_document(
        self, document_id: str, *, preview_chars: int = 160
    ) -> KnowledgeDocumentInspection:
        """Inspect one committed canonical document without changing state."""

        if type(document_id) is not str or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        self._validate_preview_chars(preview_chars)
        canonical = self._find_document(document_id)
        if canonical is None:
            raise KnowledgeInspectionError("inspection references an unknown document_id")
        source = self._sources.get(canonical.source_id)
        if source is None:
            raise KnowledgeInspectionError("inspection document has no canonical source")
        return self._inspect_canonical_document(source, canonical, preview_chars)

    def inspect_documents(
        self, *, preview_chars: int = 160
    ) -> tuple[KnowledgeDocumentInspection, ...]:
        """Inspect every canonical document in stable source and identity order."""

        self._validate_preview_chars(preview_chars)
        inspections: list[KnowledgeDocumentInspection] = []
        for source_id in sorted(self._documents):
            source = self._sources.get(source_id)
            if source is None:
                raise KnowledgeInspectionError("inspection document has no canonical source")
            for document_id in sorted(self._documents[source_id]):
                inspections.append(
                    self._inspect_canonical_document(
                        source,
                        self._documents[source_id][document_id],
                        preview_chars,
                    )
                )
        return tuple(inspections)

    def diagnose_search(
        self, query: str, *, limit: int = 10
    ) -> KnowledgeRetrievalDiagnostics:
        """Return one validated backend trace resolved to canonical provenance."""

        if type(query) is not str:
            raise TypeError("query must be a string")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        try:
            diagnose = getattr(self._index, "diagnose", None)
        except Exception as exc:
            raise KnowledgeSearchResolutionError(
                "index does not support diagnostics"
            ) from exc
        if not callable(diagnose):
            raise KnowledgeSearchResolutionError("index does not support diagnostics")
        try:
            trace = diagnose(query, limit=limit)
        except ChunkIndexError:
            raise
        except Exception as exc:
            raise KnowledgeSearchResolutionError("index diagnose failed") from exc
        hits, rows, hit_provenance, row_provenance = self._validate_diagnostic_trace(
            trace
        )
        results = tuple(
            self._detach_diagnostic_hit(hit, provenance)
            for hit, provenance in zip(hits, hit_provenance, strict=True)
        )
        candidates = tuple(
            self._detach_diagnostic_candidate(row, provenance)
            for row, provenance in zip(rows, row_provenance, strict=True)
        )
        return KnowledgeRetrievalDiagnostics(query, results, candidates)

    def _resolve_hit(self, hit: object) -> KnowledgeSearchResult:
        source, document = self._canonical_hit_provenance(hit)
        return KnowledgeSearchResult(
            source=deepcopy(source), document=deepcopy(document), hit=hit
        )

    def _canonical_hit_provenance(
        self,
        hit: object,
        cache: dict[str, tuple[KnowledgeSource, Document]] | None = None,
    ) -> tuple[KnowledgeSource, Document]:
        if not isinstance(hit, SearchHit):
            raise KnowledgeSearchResolutionError(
                "index search result must contain only SearchHit values"
            )
        if not isinstance(hit.chunk, Chunk):
            raise KnowledgeSearchResolutionError("SearchHit must contain a Chunk")
        chunk = hit.chunk
        provenance = None if cache is None else cache.get(chunk.document_id)
        if provenance is None:
            document = self._find_document(chunk.document_id)
            if document is None:
                raise KnowledgeSearchResolutionError(
                    "search hit references an unknown document_id"
                )
        else:
            _, document = provenance
        if type(chunk.start_offset) is not int or type(chunk.end_offset) is not int:
            raise KnowledgeSearchResolutionError("search hit chunk has invalid offsets")
        if not (
            0 <= chunk.start_offset <= chunk.end_offset <= len(document.content)
        ):
            raise KnowledgeSearchResolutionError(
                "search hit chunk has invalid offsets outside canonical document"
            )
        if chunk.content != document.content[chunk.start_offset : chunk.end_offset]:
            raise KnowledgeSearchResolutionError(
                "search hit chunk is incoherent with canonical document"
            )
        if provenance is None:
            source = self._sources.get(document.source_id)
            if source is None:
                raise KnowledgeSearchResolutionError(
                    "search hit references a document with no canonical source"
                )
            provenance = (source, document)
            if cache is not None:
                cache[chunk.document_id] = provenance
        source, document = provenance
        return source, document

    def _detach_diagnostic_hit(
        self,
        hit: SearchHit,
        provenance: tuple[KnowledgeSource, Document],
    ) -> KnowledgeSearchResult:
        canonical_source, canonical_document = provenance
        try:
            source = self._detached_exact_source(canonical_source)
            document = self._detached_exact_document(canonical_document)
        except Exception as exc:
            raise KnowledgeSearchResolutionError(
                "diagnostic provenance is invalid"
            ) from exc
        return KnowledgeSearchResult(source, document, hit)

    def _detach_diagnostic_candidate(
        self,
        row: RetrievalCandidateDiagnostic,
        provenance: tuple[KnowledgeSource, Document],
    ) -> KnowledgeRetrievalCandidateDiagnostic:
        resolved = self._detach_diagnostic_hit(
            SearchHit(row.chunk, row.score, row.matched_terms), provenance
        )
        return KnowledgeRetrievalCandidateDiagnostic(
            resolved.source, resolved.document, row
        )

    def _validate_diagnostic_trace(
        self, trace: object
    ) -> tuple[
        tuple[SearchHit, ...],
        tuple[RetrievalCandidateDiagnostic, ...],
        tuple[tuple[KnowledgeSource, Document], ...],
        tuple[tuple[KnowledgeSource, Document], ...],
    ]:
        if type(trace) is not RetrievalDiagnostics:
            raise KnowledgeSearchResolutionError("index diagnostic trace is invalid")
        if type(trace.hits) is not tuple or type(trace.candidates) is not tuple:
            raise KnowledgeSearchResolutionError("index diagnostic trace is invalid")
        hits = trace.hits
        rows = trace.candidates
        if any(type(hit) is not SearchHit for hit in hits):
            raise KnowledgeSearchResolutionError("index diagnostic trace is invalid")
        if any(type(row) is not RetrievalCandidateDiagnostic for row in rows):
            raise KnowledgeSearchResolutionError("index diagnostic trace is invalid")

        provenance_cache: dict[str, tuple[KnowledgeSource, Document]] = {}
        derived_chunks_cache: dict[str, tuple[Chunk, ...]] = {}
        hit_ids: list[str] = []
        hit_provenance: list[tuple[KnowledgeSource, Document]] = []
        for hit in hits:
            self._validate_diagnostic_hit(hit)
            provenance = self._canonical_hit_provenance(hit, provenance_cache)
            self._require_derived_diagnostic_chunk(
                hit.chunk, provenance[1], derived_chunks_cache
            )
            hit_provenance.append(provenance)
            hit_ids.append(hit.chunk.chunk_id)
        if len(hit_ids) != len(set(hit_ids)):
            raise KnowledgeSearchResolutionError(
                "index diagnostic trace contains duplicate final chunks"
            )
        hit_id_set = set(hit_ids)

        stage_positions = {
            stage: position for position, stage in enumerate(RetrievalStage)
        }
        blocks: list[list[RetrievalCandidateDiagnostic]] = []
        block_stages: list[RetrievalStage] = []
        block_ids: list[set[str]] = []
        canonical_by_id: dict[str, Chunk] = {}
        selected_ids: set[str] = set()
        row_provenance: list[tuple[KnowledgeSource, Document]] = []
        previous_stage_position = -1
        for row in rows:
            self._validate_diagnostic_chunk(row.chunk)
            try:
                RetrievalCandidateDiagnostic(
                    stage=row.stage,
                    rank=row.rank,
                    chunk=row.chunk,
                    score=row.score,
                    matched_terms=row.matched_terms,
                    rrf_contribution=row.rrf_contribution,
                    selected=row.selected,
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSearchResolutionError(
                    "index diagnostic candidate is invalid"
                ) from exc
            provenance = self._canonical_hit_provenance(
                SearchHit(row.chunk, row.score, row.matched_terms),
                provenance_cache,
            )
            self._require_derived_diagnostic_chunk(
                row.chunk, provenance[1], derived_chunks_cache
            )
            row_provenance.append(provenance)
            stage_position = stage_positions[row.stage]
            repeated_reranker = (
                bool(blocks)
                and row.stage is RetrievalStage.RERANKER
                and block_stages[-1] is RetrievalStage.RERANKER
                and row.rank == 1
            )
            new_stage = not blocks or row.stage is not block_stages[-1]
            if new_stage:
                if row.stage is RetrievalStage.RERANKER and not blocks:
                    raise KnowledgeSearchResolutionError(
                        "index diagnostic stage ordering is invalid"
                    )
                if stage_position <= previous_stage_position:
                    raise KnowledgeSearchResolutionError(
                        "index diagnostic stage ordering is invalid"
                    )
                blocks.append([])
                block_stages.append(row.stage)
                block_ids.append(set())
                previous_stage_position = stage_position
            elif repeated_reranker:
                blocks.append([])
                block_stages.append(row.stage)
                block_ids.append(set())
            if row.rank != len(blocks[-1]) + 1:
                raise KnowledgeSearchResolutionError(
                    "index diagnostic candidate rank ordering is invalid"
                )
            if row.chunk.chunk_id in block_ids[-1]:
                raise KnowledgeSearchResolutionError(
                    "index diagnostic block contains duplicate chunks"
                )
            existing = canonical_by_id.setdefault(row.chunk.chunk_id, row.chunk)
            if existing != row.chunk:
                raise KnowledgeSearchResolutionError(
                    "index diagnostic chunk identity is incoherent"
                )
            if row.stage is RetrievalStage.RERANKER:
                if len(blocks) < 2 or row.chunk.chunk_id not in block_ids[-2]:
                    raise KnowledgeSearchResolutionError(
                        "index diagnostic reranker lineage is invalid"
                    )
                previous_row = next(
                    item
                    for item in blocks[-2]
                    if item.chunk.chunk_id == row.chunk.chunk_id
                )
                if row.matched_terms != previous_row.matched_terms:
                    raise KnowledgeSearchResolutionError(
                        "index diagnostic reranker lineage is invalid"
                    )
            blocks[-1].append(row)
            block_ids[-1].add(row.chunk.chunk_id)
            if row.selected != (row.chunk.chunk_id in hit_id_set):
                raise KnowledgeSearchResolutionError(
                    "index diagnostic candidate selection is incoherent"
                )
            if row.selected:
                selected_ids.add(row.chunk.chunk_id)

        if hits and not rows:
            raise KnowledgeSearchResolutionError("index diagnostic trace is incoherent")
        if selected_ids != hit_id_set:
            raise KnowledgeSearchResolutionError(
                "index diagnostic selected candidates do not match final hits"
            )
        terminal_hits = tuple(
            SearchHit(row.chunk, row.score, row.matched_terms)
            for row in (blocks[-1] if blocks else ())
        )
        empty_final_trace = not hits and bool(rows) and all(
            not row.selected for row in rows
        )
        if not empty_final_trace and terminal_hits != hits:
            raise KnowledgeSearchResolutionError(
                "index diagnostic final candidate block does not match final hits"
            )
        return hits, rows, tuple(hit_provenance), tuple(row_provenance)

    @classmethod
    def _validate_diagnostic_hit(cls, hit: object) -> None:
        if type(hit) is not SearchHit:
            raise KnowledgeSearchResolutionError(
                "index diagnostic final hit is invalid"
            )
        cls._validate_diagnostic_chunk(hit.chunk)
        if type(hit.score) is not float or not isfinite(hit.score):
            raise KnowledgeSearchResolutionError(
                "index diagnostic final hit is invalid"
            )
        if type(hit.matched_terms) is not tuple or any(
            type(term) is not str for term in hit.matched_terms
        ):
            raise KnowledgeSearchResolutionError(
                "index diagnostic final hit is invalid"
            )

    @staticmethod
    def _validate_diagnostic_chunk(chunk: object) -> None:
        if type(chunk) is not Chunk:
            raise KnowledgeSearchResolutionError(
                "index diagnostic chunk is invalid"
            )
        if type(chunk.document_id) is not str or not chunk.document_id.strip():
            raise KnowledgeSearchResolutionError(
                "index diagnostic chunk is invalid"
            )
        if type(chunk.chunk_id) is not str or not chunk.chunk_id.strip():
            raise KnowledgeSearchResolutionError(
                "index diagnostic chunk is invalid"
            )
        if type(chunk.content) is not str:
            raise KnowledgeSearchResolutionError(
                "index diagnostic chunk is invalid"
            )
        if type(chunk.start_offset) is not int or type(chunk.end_offset) is not int:
            raise KnowledgeSearchResolutionError(
                "index diagnostic chunk is invalid"
            )
        if chunk.start_offset < 0 or chunk.end_offset <= chunk.start_offset:
            raise KnowledgeSearchResolutionError(
                "index diagnostic chunk is invalid"
            )

    @staticmethod
    def _validate_preview_chars(preview_chars: int) -> None:
        if type(preview_chars) is not int:
            raise TypeError("preview_chars must be an integer")
        if preview_chars <= 0:
            raise ValueError("preview_chars must be greater than zero")

    def _inspect_canonical_document(
        self, source: KnowledgeSource, document: Document, preview_chars: int
    ) -> KnowledgeDocumentInspection:
        chunks = self._derive_verified_chunks(
            document,
            error_type=KnowledgeInspectionError,
            context="inspection",
        )
        validated = self._inspection_chunk_values(chunks, preview_chars)
        try:
            detached_source = self._detached_exact_source(source)
            detached_document = self._detached_exact_document(document)
            return KnowledgeDocumentInspection(
                detached_source, detached_document, validated
            )
        except Exception as exc:
            raise KnowledgeInspectionError(
                "inspection canonical provenance is invalid"
            ) from exc

    @staticmethod
    def _detached_exact_source(source: KnowledgeSource) -> KnowledgeSource:
        return KnowledgeSource(
            source_id=source.source_id,
            source_type=source.source_type,
            display_name=source.display_name,
            logical_location=source.logical_location,
            metadata=source.metadata,
        )

    @staticmethod
    def _detached_exact_document(document: Document) -> Document:
        detached = Document(
            source_id=document.source_id,
            logical_path=document.logical_path,
            content=document.content,
            content_type=document.content_type,
            metadata=document.metadata,
        )
        if (
            detached.document_id != document.document_id
            or detached.content_hash != document.content_hash
        ):
            raise ValueError("canonical document identity is incoherent")
        return detached

    def _require_derived_diagnostic_chunk(
        self,
        chunk: Chunk,
        document: Document,
        cache: dict[str, tuple[Chunk, ...]],
    ) -> None:
        chunks = cache.get(document.document_id)
        if chunks is None:
            chunks = self._derive_verified_chunks(
                document,
                error_type=KnowledgeSearchResolutionError,
                context="diagnostic",
            )
            cache[document.document_id] = chunks
        if chunk not in chunks:
            raise KnowledgeSearchResolutionError(
                "diagnostic chunk is not a legitimately derived canonical chunk"
            )

    def _derive_verified_chunks(
        self,
        document: Document,
        *,
        error_type: type[KnowledgeCollectionError],
        context: str,
    ) -> tuple[Chunk, ...]:
        derived: list[tuple[Chunk, ...]] = []
        for _ in range(2):
            try:
                chunks = self._chunker.chunk(deepcopy(document))
            except Exception as exc:
                raise error_type(f"{context} chunker failed") from exc
            derived.append(
                self._validated_derived_chunks(
                    document,
                    chunks,
                    error_type=error_type,
                    context=context,
                )
            )
        if derived[0] != derived[1]:
            raise error_type(f"{context} chunker must be deterministic")
        return derived[0]

    @staticmethod
    def _validated_derived_chunks(
        document: Document,
        chunks: object,
        *,
        error_type: type[KnowledgeCollectionError],
        context: str,
    ) -> tuple[Chunk, ...]:
        if type(chunks) is not tuple:
            raise error_type(f"{context} chunker must return a tuple")
        chunk_ids: set[str] = set()
        previous_start = -1
        previous_end = -1
        for chunk in chunks:
            if type(chunk) is not Chunk:
                raise error_type(
                    f"{context} chunker tuple must contain only Chunk values"
                )
            if type(chunk.document_id) is not str or chunk.document_id != document.document_id:
                raise error_type(
                    f"{context} chunk has an invalid document_id"
                )
            if type(chunk.chunk_id) is not str or not chunk.chunk_id.strip():
                raise error_type(f"{context} chunk has an invalid chunk_id")
            if chunk.chunk_id in chunk_ids:
                raise error_type(
                    f"{context} chunker returned duplicate chunk_id values"
                )
            if type(chunk.start_offset) is not int or type(chunk.end_offset) is not int:
                raise error_type(f"{context} chunk has invalid offsets")
            if not (
                0 <= chunk.start_offset < chunk.end_offset <= len(document.content)
            ):
                raise error_type(f"{context} chunk has invalid offsets")
            if (
                chunk.start_offset <= previous_start
                or chunk.end_offset <= previous_end
            ):
                raise error_type(
                    f"{context} chunks have invalid stable order"
                )
            if type(chunk.content) is not str or chunk.content != document.content[
                chunk.start_offset : chunk.end_offset
            ]:
                raise error_type(
                    f"{context} chunk content differs from canonical document"
                )
            chunk_ids.add(chunk.chunk_id)
            previous_start = chunk.start_offset
            previous_end = chunk.end_offset
        return chunks

    @staticmethod
    def _inspection_chunk_values(
        chunks: tuple[Chunk, ...], preview_chars: int
    ) -> tuple[KnowledgeChunkInspection, ...]:
        return tuple(
            KnowledgeChunkInspection(
                ordinal,
                chunk.chunk_id,
                chunk.start_offset,
                chunk.end_offset,
                len(chunk.content),
                chunk.content[:preview_chars],
            )
            for ordinal, chunk in enumerate(chunks, start=1)
        )

    def _find_document(self, document_id: str) -> Document | None:
        return next(
            (
                documents[document_id]
                for documents in self._documents.values()
                if document_id in documents
            ),
            None,
        )

    def snapshot(self) -> KnowledgeSnapshot:
        """Export only committed canonical state in stable identity order."""

        sources = tuple(
            deepcopy(self._sources[source_id])
            for source_id in sorted(self._sources)
        )
        documents = tuple(
            deepcopy(document)
            for source_id in sorted(self._documents)
            for _, document in sorted(self._documents[source_id].items())
        )
        versions = tuple(
            deepcopy(version)
            for document_id in sorted(self._document_versions)
            for version in self._document_versions[document_id]
        )
        return KnowledgeSnapshot(
            sources=sources,
            documents=documents,
            document_versions=versions,
        )

    def restore(self, snapshot: KnowledgeSnapshot) -> KnowledgeRestoreResult:
        """Atomically replace collection state and rebuild all derived state."""

        sources, documents = self._validate_restore_snapshot(snapshot)
        restored_versions = self._validate_restore_versions(snapshot, documents)
        prepared: dict[str, tuple[Chunk, ...]] = {}
        for document_id in sorted(documents):
            chunk_document = deepcopy(documents[document_id])
            chunks = self._chunker.chunk(chunk_document)
            if type(chunks) is not tuple:
                raise KnowledgeRestoreError("chunker must return a tuple")
            prepared[document_id] = chunks

        try:
            staged = self._index_factory()
        except Exception as exc:
            raise KnowledgeRestoreError("index_factory failed to create a fresh index") from exc
        try:
            self._require_cloneable_index(staged)
        except TypeError as exc:
            raise KnowledgeRestoreError("index_factory returned an invalid index") from exc
        for document_id in sorted(prepared):
            staged.replace_document(document_id, prepared[document_id])

        restored_sources = {
            source_id: deepcopy(source) for source_id, source in sources.items()
        }
        restored_documents = {source_id: {} for source_id in sources}
        for document_id in sorted(documents):
            document = deepcopy(documents[document_id])
            restored_documents[document.source_id][document_id] = document
        self._index = staged
        self._sources = restored_sources
        self._documents = restored_documents
        self._chunks_by_document = dict(prepared)
        self._document_versions = restored_versions
        return KnowledgeRestoreResult(
            sources_restored=len(sources),
            documents_restored=len(documents),
            chunks_indexed=sum(len(chunks) for chunks in prepared.values()),
        )

    def _validate_restore_snapshot(
        self, snapshot: KnowledgeSnapshot
    ) -> tuple[dict[str, KnowledgeSource], dict[str, Document]]:
        if not isinstance(snapshot, KnowledgeSnapshot):
            raise KnowledgeSnapshotError("snapshot must be a KnowledgeSnapshot")
        if type(snapshot.sources) is not tuple:
            raise KnowledgeSnapshotError("snapshot sources must be a tuple")
        if type(snapshot.documents) is not tuple:
            raise KnowledgeSnapshotError("snapshot documents must be a tuple")
        if type(snapshot.document_versions) is not tuple:
            raise KnowledgeSnapshotError("snapshot document_versions must be a tuple")

        sources: dict[str, KnowledgeSource] = {}
        for source in snapshot.sources:
            if not isinstance(source, KnowledgeSource):
                raise KnowledgeSnapshotError("snapshot sources must contain only KnowledgeSource values")
            if type(source.source_id) is not str or not source.source_id.strip():
                raise KnowledgeSnapshotError("snapshot source_id must be a non-empty string")
            if source.source_id in sources:
                raise KnowledgeSnapshotError("snapshot contains duplicate source_id values")
            sources[source.source_id] = source

        documents: dict[str, Document] = {}
        for document in snapshot.documents:
            if not isinstance(document, Document):
                raise KnowledgeSnapshotError("snapshot documents must contain only Document values")
            if document.source_id not in sources:
                raise KnowledgeSnapshotError("snapshot Document references a missing source_id")
            if document.document_id in documents:
                raise KnowledgeSnapshotError("snapshot contains duplicate document_id values")
            self._validate_document_identity(document)
            documents[document.document_id] = document

        if len(sources) > self._limits.max_sources:
            raise KnowledgeCollectionLimitError("collection exceeds max_sources")
        if len(documents) > self._limits.max_documents:
            raise KnowledgeCollectionLimitError("collection exceeds max_documents")
        return sources, documents

    def _validate_restore_versions(
        self,
        snapshot: KnowledgeSnapshot,
        documents: dict[str, Document],
    ) -> dict[str, tuple[DocumentVersion, ...]]:
        if not snapshot.document_versions:
            if not documents:
                return {}
            raise KnowledgeSnapshotError(
                "snapshot documents require document versions"
            )
        if len(snapshot.document_versions) > self._limits.max_document_versions:
            raise KnowledgeCollectionLimitError(
                "collection exceeds max_document_versions"
            )

        grouped: dict[str, list[DocumentVersion]] = {}
        version_ids: set[str] = set()
        for version in snapshot.document_versions:
            if type(version) is not DocumentVersion:
                raise KnowledgeSnapshotError(
                    "snapshot document_versions must contain only DocumentVersion values"
                )
            try:
                validated = DocumentVersion(
                    version_id=version.version_id,
                    document_id=version.document_id,
                    source_id=version.source_id,
                    logical_path=version.logical_path,
                    content=version.content,
                    content_hash=version.content_hash,
                    created_at=version.created_at,
                    previous_version_id=version.previous_version_id,
                    sync_context=version.sync_context,
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSnapshotError(
                    "snapshot contains an incoherent document version"
                ) from exc
            if validated.version_id in version_ids:
                raise KnowledgeSnapshotError(
                    "snapshot contains duplicate document version IDs"
                )
            version_ids.add(validated.version_id)
            grouped.setdefault(validated.document_id, []).append(validated)

        ordered: dict[str, tuple[DocumentVersion, ...]] = {}
        for document_id, candidates in grouped.items():
            by_id = {version.version_id: version for version in candidates}
            roots = [
                version for version in candidates if version.previous_version_id is None
            ]
            if len(roots) != 1:
                raise KnowledgeSnapshotError(
                    "document version chain must contain exactly one root"
                )
            children: dict[str, DocumentVersion] = {}
            for version in candidates:
                predecessor = version.previous_version_id
                if predecessor is None:
                    continue
                if predecessor not in by_id:
                    raise KnowledgeSnapshotError(
                        "document version chain has a missing predecessor"
                    )
                if predecessor in children:
                    raise KnowledgeSnapshotError("document version chain contains a fork")
                children[predecessor] = version
            chain: list[DocumentVersion] = []
            current = roots[0]
            while True:
                if current in chain:
                    raise KnowledgeSnapshotError("document version chain contains a cycle")
                chain.append(current)
                successor = children.get(current.version_id)
                if successor is None:
                    break
                current = successor
            if len(chain) != len(candidates):
                raise KnowledgeSnapshotError("document version chain is disconnected")
            ordered[document_id] = tuple(deepcopy(version) for version in chain)

        for document_id, document in documents.items():
            chain = ordered.get(document_id)
            if chain is None:
                raise KnowledgeSnapshotError(
                    "current document has no document version history"
                )
            latest = chain[-1]
            if (
                latest.source_id != document.source_id
                or latest.logical_path != document.logical_path
                or latest.content != document.content
                or latest.content_hash != document.content_hash
            ):
                raise KnowledgeSnapshotError(
                    "current document does not match latest document version"
                )
        return ordered

    @staticmethod
    def _validate_document_identity(document: Document) -> None:
        if document.document_id != stable_document_id(document.source_id, document.logical_path):
            raise KnowledgeSnapshotError("Document has an incoherent document_id")
        if document.content_hash != compute_content_hash(document.content):
            raise KnowledgeSnapshotError("Document has an incoherent content_hash")

    def _preflight_snapshot(
        self, source_id: str, incoming_count: int, *, added_versions: int = 0
    ) -> None:
        source_count = len(self._sources) + (0 if source_id in self._sources else 1)
        if source_count > self._limits.max_sources:
            raise KnowledgeCollectionLimitError("collection exceeds max_sources")
        existing_count = sum(len(documents) for documents in self._documents.values())
        resulting_count = existing_count - len(self._documents.get(source_id, {})) + incoming_count
        if resulting_count > self._limits.max_documents:
            raise KnowledgeCollectionLimitError("collection exceeds max_documents")
        if len(self._all_versions()) + added_versions > self._limits.max_document_versions:
            raise KnowledgeCollectionLimitError(
                "collection exceeds max_document_versions"
            )

    def _all_versions(self) -> tuple[DocumentVersion, ...]:
        return tuple(
            version
            for document_id in sorted(self._document_versions)
            for version in self._document_versions[document_id]
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _format_clock_value(value: datetime) -> str:
        if type(value) is not datetime or value.utcoffset() != timezone.utc.utcoffset(value):
            raise KnowledgeSnapshotError("clock must return a UTC datetime")
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def _sync_context(source_id: str, created_at: str, version_count: int) -> str:
        canonical = json.dumps(
            (source_id, created_at, version_count),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sync-{hashlib.sha256(canonical).hexdigest()}"

    @staticmethod
    def _require_cloneable_index(index: object) -> None:
        for method in ("add", "replace_document", "remove_document", "search", "clone"):
            if not callable(getattr(index, method, None)):
                raise TypeError(f"index must implement {method}()")


__all__ = [
    "CloneableChunkIndex",
    "DocumentChunker",
    "KnowledgeCollection",
    "KnowledgeCollectionError",
    "KnowledgeCollectionLimitError",
    "KnowledgeCollectionLimits",
    "KnowledgeInspectionError",
    "KnowledgeRetrievalCandidateDiagnostic",
    "KnowledgeRetrievalDiagnostics",
    "KnowledgeRestoreError",
    "KnowledgeRestoreResult",
    "KnowledgeSearchResolutionError",
    "KnowledgeSearchResult",
    "KnowledgeSnapshot",
    "KnowledgeSnapshotError",
    "KnowledgeSyncResult",
]
