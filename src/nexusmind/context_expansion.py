"""Deterministic, section-aware expansion of retrieved context anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .knowledge_chunking import Chunk
from .knowledge_collection import KnowledgeSearchResult
from .knowledge_retrieval import SearchHit


MAX_CONTEXT_EXPANSION_NEIGHBORS = 1
MAX_CONTEXT_EXPANSION_CANDIDATES = 100


@dataclass(frozen=True, slots=True)
class ContextExpansionResult:
    """Ranked anchors plus bounded context-only neighboring candidates."""

    candidates: tuple[KnowledgeSearchResult, ...]
    anchor_chunk_ids: tuple[str, ...]
    expanded_chunk_ids: tuple[str, ...]
    expanded_document_ids: tuple[str, ...]
    section_boundary_skips: int


def expand_context_candidates(
    anchors: tuple[KnowledgeSearchResult, ...],
    *,
    chunk_catalog: Mapping[str, tuple[Chunk, ...]],
) -> ContextExpansionResult:
    """Append same-section neighbors to ranked, provenance-resolved anchors."""

    if type(anchors) is not tuple:
        raise TypeError("anchors must be a tuple")
    if not isinstance(chunk_catalog, Mapping):
        raise TypeError("chunk_catalog must be a mapping")

    candidates = list(anchors)
    anchor_chunk_ids: list[str] = []
    expanded_chunk_ids: list[str] = []
    expanded_document_ids: list[str] = []
    selected_chunk_ids = {item.hit.chunk.chunk_id for item in anchors}
    resolved_anchors: list[tuple[KnowledgeSearchResult, tuple[Chunk, ...], int]] = []
    section_boundary_skips = 0
    expansion_count = 0

    for anchor in anchors:
        if type(anchor) is not KnowledgeSearchResult:
            raise TypeError("anchors must contain KnowledgeSearchResult values")
        document = anchor.document
        chunk = anchor.hit.chunk
        if type(chunk) is not Chunk or chunk.document_id != document.document_id:
            raise ValueError("anchor chunk provenance is invalid")
        if document.content[chunk.start_offset : chunk.end_offset] != chunk.content:
            raise ValueError("anchor chunk content is not canonical")

        document_chunks = chunk_catalog.get(document.document_id)
        if type(document_chunks) is not tuple:
            raise ValueError("chunk catalog is missing an anchor document")
        _validate_catalog(document.document_id, document.content, document_chunks)
        try:
            ordinal = document_chunks.index(chunk)
        except ValueError as exc:
            raise ValueError("chunk catalog does not contain the anchor chunk") from exc

        anchor_chunk_ids.append(chunk.chunk_id)
        resolved_anchors.append((anchor, document_chunks, ordinal))

    for anchor, document_chunks, ordinal in resolved_anchors:
        chunk = anchor.hit.chunk
        for offset in (-MAX_CONTEXT_EXPANSION_NEIGHBORS, MAX_CONTEXT_EXPANSION_NEIGHBORS):
            neighbor_index = ordinal + offset
            if not 0 <= neighbor_index < len(document_chunks):
                continue
            neighbor = document_chunks[neighbor_index]
            if neighbor.heading_path != chunk.heading_path:
                section_boundary_skips += 1
                continue
            if neighbor.chunk_id in selected_chunk_ids:
                continue
            if expansion_count >= MAX_CONTEXT_EXPANSION_CANDIDATES:
                continue
            selected_chunk_ids.add(neighbor.chunk_id)
            candidates.append(
                KnowledgeSearchResult(
                    source=anchor.source,
                    document=anchor.document,
                    hit=SearchHit(neighbor, 0.0, ()),
                )
            )
            expanded_chunk_ids.append(neighbor.chunk_id)
            if anchor.document.document_id not in expanded_document_ids:
                expanded_document_ids.append(anchor.document.document_id)
            expansion_count += 1

    return ContextExpansionResult(
        candidates=tuple(candidates),
        anchor_chunk_ids=tuple(anchor_chunk_ids),
        expanded_chunk_ids=tuple(expanded_chunk_ids),
        expanded_document_ids=tuple(expanded_document_ids),
        section_boundary_skips=section_boundary_skips,
    )


def _validate_catalog(
    document_id: str,
    document_content: str,
    chunks: tuple[Chunk, ...],
) -> None:
    chunk_ids: set[str] = set()
    previous_start = -1
    previous_end = -1
    for chunk in chunks:
        if type(chunk) is not Chunk or chunk.document_id != document_id:
            raise ValueError("chunk catalog contains incoherent chunks")
        if chunk.chunk_id in chunk_ids:
            raise ValueError("chunk catalog contains duplicate chunk IDs")
        if not 0 <= chunk.start_offset < chunk.end_offset <= len(document_content):
            raise ValueError("chunk catalog contains invalid offsets")
        if document_content[chunk.start_offset : chunk.end_offset] != chunk.content:
            raise ValueError("chunk catalog contains non-canonical content")
        if chunk.start_offset < previous_start or chunk.end_offset < previous_end:
            raise ValueError("chunk catalog is not in canonical order")
        chunk_ids.add(chunk.chunk_id)
        previous_start = chunk.start_offset
        previous_end = chunk.end_offset


__all__ = [
    "ContextExpansionResult",
    "MAX_CONTEXT_EXPANSION_CANDIDATES",
    "MAX_CONTEXT_EXPANSION_NEIGHBORS",
    "expand_context_candidates",
]
