"""Provider-neutral contracts and bounded character-based document chunking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .knowledge import Document


class ChunkLimitError(Exception):
    """The requested operation would exceed the configured chunk limit."""


def _stable_chunk_id(
    document: Document,
    *,
    start_offset: int,
    end_offset: int,
    chunk_size: int,
    overlap: int,
) -> str:
    parts = (
        document.document_id,
        document.content_hash,
        start_offset,
        end_offset,
        chunk_size,
        overlap,
    )
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"chunk-{hashlib.sha256(canonical).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """One exact character slice derived from a source-neutral document."""

    document_id: str
    chunk_id: str
    content: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TextChunker:
    """Split documents into deterministic overlapping character slices."""

    chunk_size: int = 1000
    overlap: int = 100
    max_chunks: int = 10000

    def __post_init__(self) -> None:
        if type(self.chunk_size) is not int:
            raise TypeError("chunk_size must be an integer")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if type(self.overlap) is not int:
            raise TypeError("overlap must be an integer")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if type(self.max_chunks) is not int:
            raise TypeError("max_chunks must be an integer")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero")

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        if not isinstance(document, Document):
            raise TypeError("document must be a Document")
        if not document.content:
            return ()

        step = self.chunk_size - self.overlap
        remaining_after_first = max(0, len(document.content) - self.chunk_size)
        required_chunks = 1 + (remaining_after_first + step - 1) // step
        if required_chunks > self.max_chunks:
            raise ChunkLimitError(
                f"document requires {required_chunks} chunks; limit is {self.max_chunks}"
            )

        chunks: list[Chunk] = []
        start_offset = 0
        while start_offset < len(document.content):
            end_offset = min(start_offset + self.chunk_size, len(document.content))
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=_stable_chunk_id(
                        document,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                    ),
                    content=document.content[start_offset:end_offset],
                    start_offset=start_offset,
                    end_offset=end_offset,
                )
            )
            if end_offset == len(document.content):
                break
            start_offset += step
        return tuple(chunks)


__all__ = ["Chunk", "ChunkLimitError", "TextChunker"]
