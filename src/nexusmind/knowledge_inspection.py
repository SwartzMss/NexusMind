"""Detached, read-only values for inspecting canonical knowledge state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .knowledge import Document, KnowledgeSource
from .knowledge_base_manifest import LocalDirectorySourceConfig, LocalFileSourceConfig


def _require_nonblank_string(value: object, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_nonnegative_integer(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class KnowledgeBaseStatus:
    """Bounded counters describing the current canonical knowledge state."""

    knowledge_base_id: str
    display_name: str | None
    registered_source_count: int
    canonical_source_count: int
    document_count: int

    def __post_init__(self) -> None:
        _require_nonblank_string(self.knowledge_base_id, "knowledge_base_id")
        if self.display_name is not None:
            _require_nonblank_string(self.display_name, "display_name")
        for field_name in (
            "registered_source_count",
            "canonical_source_count",
            "document_count",
        ):
            _require_nonnegative_integer(getattr(self, field_name), field_name)


class KnowledgeSourceSyncStatus(str, Enum):
    """Whether a registered source has committed canonical state."""

    REGISTERED = "registered"
    SYNCED = "synced"


@dataclass(frozen=True, slots=True)
class KnowledgeSourceInspection:
    """Detached registration state and canonical counters for one source."""

    config: LocalFileSourceConfig | LocalDirectorySourceConfig
    sync_status: KnowledgeSourceSyncStatus
    document_count: int
    chunk_count: int

    def __post_init__(self) -> None:
        if type(self.config) not in (LocalFileSourceConfig, LocalDirectorySourceConfig):
            raise TypeError("config must be a supported local source config")
        if type(self.sync_status) is not KnowledgeSourceSyncStatus:
            raise TypeError("sync_status must be a KnowledgeSourceSyncStatus")
        _require_nonnegative_integer(self.document_count, "document_count")
        _require_nonnegative_integer(self.chunk_count, "chunk_count")
        object.__setattr__(self, "config", deepcopy(self.config))


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentSummary:
    """Detached document metadata and counts without canonical content."""

    source_id: str
    document_id: str
    logical_path: str
    content_type: str
    content_hash: str
    metadata: dict[str, Any]
    character_count: int
    chunk_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "document_id",
            "logical_path",
            "content_type",
            "content_hash",
        ):
            _require_nonblank_string(getattr(self, field_name), field_name)
        if type(self.metadata) is not dict:
            raise TypeError("metadata must be a dict")
        _require_nonnegative_integer(self.character_count, "character_count")
        _require_nonnegative_integer(self.chunk_count, "chunk_count")
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True, slots=True)
class KnowledgeBaseInspection:
    """Detached source and document summaries for one coherent knowledge state."""

    status: KnowledgeBaseStatus
    sources: tuple[KnowledgeSourceInspection, ...]
    documents: tuple[KnowledgeDocumentSummary, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not KnowledgeBaseStatus:
            raise TypeError("status must be a KnowledgeBaseStatus")
        if type(self.sources) is not tuple:
            raise TypeError("sources must be a tuple")
        if any(type(source) is not KnowledgeSourceInspection for source in self.sources):
            raise TypeError("sources must contain only KnowledgeSourceInspection values")
        if type(self.documents) is not tuple:
            raise TypeError("documents must be a tuple")
        if any(
            type(document) is not KnowledgeDocumentSummary
            for document in self.documents
        ):
            raise TypeError("documents must contain only KnowledgeDocumentSummary values")


@dataclass(frozen=True, slots=True)
class KnowledgeChunkInspection:
    """One validated summary of a chunk derived from a canonical document."""

    ordinal: int
    chunk_id: str
    start_offset: int
    end_offset: int
    character_count: int
    preview: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int:
            raise TypeError("ordinal must be an integer")
        if self.ordinal <= 0:
            raise ValueError("ordinal must be greater than zero")
        if type(self.chunk_id) is not str:
            raise TypeError("chunk_id must be a string")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if type(self.start_offset) is not int:
            raise TypeError("start_offset must be an integer")
        if type(self.end_offset) is not int:
            raise TypeError("end_offset must be an integer")
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not precede start_offset")
        if type(self.character_count) is not int:
            raise TypeError("character_count must be an integer")
        if self.character_count != self.end_offset - self.start_offset:
            raise ValueError("character_count must equal the offset span")
        if self.character_count <= 0:
            raise ValueError("character_count must be greater than zero")
        if type(self.preview) is not str:
            raise TypeError("preview must be a string")
        if len(self.preview) > self.character_count:
            raise ValueError("preview cannot exceed character_count")


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentInspection:
    """Detached canonical provenance and derived chunks for one document."""

    source: KnowledgeSource
    document: Document
    chunks: tuple[KnowledgeChunkInspection, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not KnowledgeSource:
            raise TypeError("source must be a KnowledgeSource")
        if type(self.document) is not Document:
            raise TypeError("document must be a Document")
        if self.document.source_id != self.source.source_id:
            raise ValueError("document must belong to source")
        if type(self.chunks) is not tuple:
            raise TypeError("chunks must be a tuple")
        if any(type(chunk) is not KnowledgeChunkInspection for chunk in self.chunks):
            raise TypeError("chunks must contain only KnowledgeChunkInspection values")
        previous_start = -1
        previous_end = -1
        for ordinal, chunk in enumerate(self.chunks, start=1):
            if chunk.ordinal != ordinal:
                raise ValueError("chunk ordinals must be consecutive and one-based")
            if chunk.end_offset > len(self.document.content):
                raise ValueError("chunk offsets must be within document content")
            if chunk.start_offset <= previous_start or chunk.end_offset <= previous_end:
                raise ValueError("chunks must follow stable offset order")
            previous_start = chunk.start_offset
            previous_end = chunk.end_offset


__all__ = [
    "KnowledgeBaseInspection",
    "KnowledgeBaseStatus",
    "KnowledgeChunkInspection",
    "KnowledgeDocumentInspection",
    "KnowledgeDocumentSummary",
    "KnowledgeSourceInspection",
    "KnowledgeSourceSyncStatus",
]
