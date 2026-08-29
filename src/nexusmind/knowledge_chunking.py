"""Provider-neutral deterministic document chunking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

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
    algorithm: str | None = None,
) -> str:
    parts = (
        document.document_id,
        document.content_hash,
        start_offset,
        end_offset,
        chunk_size,
        overlap,
    )
    if algorithm is not None:
        parts += (algorithm,)
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


_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]+|$)")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_LIST = re.compile(r"^[ \t]{0,3}(?:[-+*]|\d+[.)])[ \t]+")
_TABLE_RULE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)


@dataclass(frozen=True, slots=True)
class _Block:
    start: int
    end: int
    heading: bool = False


def _lines(content: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        result.append((offset, offset + len(line), line.rstrip("\r\n")))
        offset += len(line)
    if offset < len(content):
        result.append((offset, len(content), content[offset:]))
    return result


def _markdown_blocks(content: str) -> list[_Block]:
    lines = _lines(content)
    blocks: list[_Block] = []
    index = 0
    while index < len(lines):
        start, end, text = lines[index]
        if not text.strip():
            if blocks:
                previous = blocks[-1]
                blocks[-1] = _Block(previous.start, end, previous.heading)
            else:
                blocks.append(_Block(start, end))
            index += 1
            continue
        fence = _FENCE.match(text)
        if fence:
            marker = fence.group(1)[0]
            length = len(fence.group(1))
            index += 1
            while index < len(lines):
                end = lines[index][1]
                closing_fence = rf"^[ \t]{{0,3}}{re.escape(marker)}{{{length},}}[ \t]*$"
                if re.match(closing_fence, lines[index][2]):
                    index += 1
                    break
                index += 1
            blocks.append(_Block(start, end))
            continue
        if _HEADING.match(text):
            blocks.append(_Block(start, end, True))
            index += 1
            continue
        if index + 1 < len(lines) and "|" in text and _TABLE_RULE.match(lines[index + 1][2]):
            index += 2
            end = lines[index - 1][1]
            while index < len(lines) and "|" in lines[index][2] and lines[index][2].strip():
                end = lines[index][1]
                index += 1
            blocks.append(_Block(start, end))
            continue
        if _LIST.match(text):
            index += 1
            while index < len(lines):
                candidate = lines[index][2]
                if not candidate.strip() or _LIST.match(candidate) or candidate.startswith(("  ", "\t")):
                    end = lines[index][1]
                    index += 1
                    continue
                break
            blocks.append(_Block(start, end))
            continue
        index += 1
        while index < len(lines):
            candidate = lines[index][2]
            if (
                not candidate.strip()
                or _HEADING.match(candidate)
                or _FENCE.match(candidate)
                or _LIST.match(candidate)
            ):
                break
            end = lines[index][1]
            index += 1
        blocks.append(_Block(start, end))
    return blocks


def _preferred_end(content: str, start: int, limit: int, *, minimum: int = 0) -> int:
    window = content[start:limit]
    candidates = [window.rfind("\n\n"), window.rfind("\n")]
    candidates.append(max(window.rfind(" "), window.rfind("\t")))
    boundary = max(
        (value for value in candidates if start + value + 1 > minimum),
        default=-1,
    )
    if boundary < 0:
        return limit
    return start + boundary + (2 if window[boundary:boundary + 2] == "\n\n" else 1)


def _bounded_spans(
    content: str,
    block: _Block,
    chunk_size: int,
    overlap: int,
    *,
    max_spans: int,
) -> list[_Block]:
    if block.end - block.start <= chunk_size:
        return [block]
    spans: list[_Block] = []
    start = block.start
    while start < block.end:
        if len(spans) >= max_spans:
            raise ChunkLimitError("document exceeds the chunk-count limit during fallback")
        limit = min(start + chunk_size, block.end)
        end = _preferred_end(content, start, limit, minimum=start) if limit < block.end else limit
        spans.append(_Block(start, end, block.heading and start == block.start))
        if end == block.end:
            break
        start = max(start + 1, end - overlap)
    return spans


@dataclass(frozen=True, slots=True, kw_only=True)
class StructureAwareChunker:
    """Pack Markdown structural blocks before bounded recursive fallback."""

    chunk_size: int = 1000
    overlap: int = 100
    max_chunks: int = 10000

    def __post_init__(self) -> None:
        TextChunker(
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            max_chunks=self.max_chunks,
        )

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        if not isinstance(document, Document):
            raise TypeError("document must be a Document")
        if not document.content:
            return ()
        blocks: list[_Block] = []
        oversized_span_count = 0
        for block in _markdown_blocks(document.content):
            is_oversized = block.end - block.start > self.chunk_size
            remaining_budget = self.max_chunks + 1 - oversized_span_count
            spans = _bounded_spans(
                document.content,
                block,
                self.chunk_size,
                self.overlap,
                max_spans=max(1, remaining_budget),
            )
            if is_oversized:
                oversized_span_count += len(spans)
                if oversized_span_count > self.max_chunks:
                    raise ChunkLimitError(
                        "document exceeds the chunk-count limit during fallback"
                    )
            blocks.extend(spans)
        packed: list[tuple[int, int]] = []
        start = blocks[0].start
        end = blocks[0].end
        headings_only = blocks[0].heading
        for block in blocks[1:]:
            if headings_only and not block.heading and block.end - start > self.chunk_size:
                limit = start + self.chunk_size
                combined_end = _preferred_end(
                    document.content,
                    start,
                    limit,
                    minimum=block.start,
                )
                packed.append((start, combined_end))
                start = max(block.start, combined_end - self.overlap)
                end = block.end
                headings_only = False
            elif (block.heading and not headings_only) or block.end - start > self.chunk_size:
                packed.append((start, end))
                start, end = block.start, block.end
                headings_only = block.heading
            else:
                end = block.end
                headings_only = headings_only and block.heading
        packed.append((start, end))
        if len(packed) > self.max_chunks:
            raise ChunkLimitError(
                f"document requires {len(packed)} chunks; limit is {self.max_chunks}"
            )
        return tuple(
            Chunk(
                document_id=document.document_id,
                chunk_id=_stable_chunk_id(
                    document,
                    start_offset=start,
                    end_offset=end,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                    algorithm="structure-v1",
                ),
                content=document.content[start:end],
                start_offset=start,
                end_offset=end,
            )
            for start, end in packed
        )


__all__ = ["Chunk", "ChunkLimitError", "StructureAwareChunker", "TextChunker"]
