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

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        if not document.content:
            return ()

        chunks: list[Chunk] = []
        start_offset = 0
        step = self.chunk_size - self.overlap
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
