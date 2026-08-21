"""Detached, read-only values for inspecting canonical knowledge chunks."""

from __future__ import annotations

from dataclasses import dataclass

from .knowledge import Document, KnowledgeSource


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


__all__ = ["KnowledgeChunkInspection", "KnowledgeDocumentInspection"]
