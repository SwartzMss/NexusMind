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
    metadata: tuple[str, ...] = (),
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
    if metadata:
        parts += metadata
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
    heading_path: tuple[str, ...] = ()
    section_title: str = ""
    source_location: str = ""

    def __post_init__(self) -> None:
        if type(self.heading_path) is not tuple:
            raise TypeError("heading_path must be a tuple")
        if any(type(title) is not str or not title.strip() for title in self.heading_path):
            raise ValueError("heading_path must contain non-empty strings")
        if type(self.section_title) is not str:
            raise TypeError("section_title must be a string")
        expected_title = self.heading_path[-1] if self.heading_path else ""
        if self.section_title != expected_title:
            raise ValueError("section_title must match the final heading_path item")
        if type(self.source_location) is not str:
            raise TypeError("source_location must be a string")

    @property
    def retrieval_text(self) -> str:
        prefix = " > ".join(self.heading_path)
        return f"{prefix}\n{self.content}" if prefix else self.content

    @property
    def metadata(self) -> dict[str, object]:
        """Return JSON-compatible structural metadata detached from the chunk."""

        return {
            "heading_path": list(self.heading_path),
            "section_title": self.section_title,
            "source_location": self.source_location,
        }


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


@dataclass(frozen=True, slots=True)
class _Heading:
    start: int
    level: int
    title: str
    line_number: int


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


def _markdown_headings(content: str) -> tuple[_Heading, ...]:
    headings: list[_Heading] = []
    lines = _lines(content)
    fence_marker: str | None = None
    fence_length = 0
    for line_number, (start, _end, text) in enumerate(lines, start=1):
        fence = _FENCE.match(text)
        if fence_marker is not None:
            closing = rf"^[ \t]{{0,3}}{re.escape(fence_marker)}{{{fence_length},}}[ \t]*$"
            if re.match(closing, text):
                fence_marker = None
                fence_length = 0
            continue
        if fence:
            fence_marker = fence.group(1)[0]
            fence_length = len(fence.group(1))
            continue
        match = _HEADING.match(text)
        if match is None:
            continue
        marker = text.lstrip()[:6]
        level = len(marker) - len(marker.lstrip("#"))
        title = text[match.end() :].strip()
        title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
        if title:
            headings.append(_Heading(start, level, title, line_number))
    return tuple(headings)


def _heading_metadata(
    document: Document,
    *,
    end: int,
    headings: tuple[_Heading, ...],
) -> tuple[tuple[str, ...], str, str]:
    path: list[str] = []
    location = ""
    active = [heading for heading in headings if heading.start < end]
    for heading in active:
        path = path[: heading.level - 1]
        path.append(heading.title)
        location = f"{document.logical_path}:L{heading.line_number}"
    heading_path = tuple(path)
    return heading_path, heading_path[-1] if heading_path else "", location


def _preferred_end(content: str, start: int, limit: int, *, minimum: int = 0) -> int:
    window = content[start:limit]
    for separator in ("\n\n", "\n"):
        boundary = window.rfind(separator)
        if boundary >= 0 and start + boundary + len(separator) > minimum:
            return start + boundary + len(separator)
    boundary = max(window.rfind(" "), window.rfind("\t"))
    if boundary >= 0 and start + boundary + 1 > minimum:
        return start + boundary + 1
    return limit


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
        minimum_remainder = min(chunk_size // 2, 100)
        remaining_length = block.end - end + overlap
        if end < block.end and remaining_length < minimum_remainder:
            balanced_limit = block.end - minimum_remainder + overlap
            end = _preferred_end(
                content,
                start,
                balanced_limit,
                minimum=start,
            )
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
        raw_blocks = _markdown_blocks(document.content)
        blocks: list[_Block] = []
        oversized_span_count = 0
        index = 0
        while index < len(raw_blocks):
            block = raw_blocks[index]
            if block.heading:
                heading_start = block.start
                heading_end = block.end
                index += 1
                while index < len(raw_blocks) and raw_blocks[index].heading:
                    heading_end = raw_blocks[index].end
                    index += 1
                heading_block = _Block(heading_start, heading_end, True)
                if heading_end - heading_start > self.chunk_size:
                    remaining_budget = self.max_chunks + 1 - oversized_span_count
                    spans = _bounded_spans(
                        document.content,
                        heading_block,
                        self.chunk_size,
                        self.overlap,
                        max_spans=max(1, remaining_budget),
                    )
                    oversized_span_count += len(spans)
                    if oversized_span_count > self.max_chunks:
                        raise ChunkLimitError(
                            "document exceeds the chunk-count limit during fallback"
                        )
                    blocks.extend(spans)
                    continue
                if index < len(raw_blocks):
                    body = raw_blocks[index]
                    heading_length = heading_end - heading_start
                    available = self.chunk_size - heading_length
                    if 0 < available < body.end - body.start:
                        body_limit = body.start + available
                        first_end = _preferred_end(
                            document.content,
                            body.start,
                            body_limit,
                            minimum=body.start,
                        )
                        remaining_length = body.end - first_end + self.overlap
                        minimum_remainder = min(self.chunk_size // 2, 100)
                        if remaining_length < minimum_remainder:
                            balanced_take = max(1, (body.end - body.start + self.overlap) // 2)
                            first_end = min(body.start + available, body.start + balanced_take)
                        blocks.append(_Block(heading_start, first_end))
                        oversized_span_count += 1
                        remainder = _Block(max(body.start, first_end - self.overlap), body.end)
                        remaining_budget = self.max_chunks + 1 - oversized_span_count
                        spans = _bounded_spans(
                            document.content,
                            remainder,
                            self.chunk_size,
                            self.overlap,
                            max_spans=max(1, remaining_budget),
                        )
                        oversized_span_count += len(spans)
                        if oversized_span_count > self.max_chunks:
                            raise ChunkLimitError(
                                "document exceeds the chunk-count limit during fallback"
                            )
                        blocks.extend(spans)
                        index += 1
                        continue
                blocks.append(heading_block)
                continue
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
            index += 1
        packed: list[tuple[int, int]] = []
        start = blocks[0].start
        end = blocks[0].end
        headings_only = blocks[0].heading
        for block in blocks[1:]:
            if (block.heading and not headings_only) or block.end - start > self.chunk_size:
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
        headings = _markdown_headings(document.content)
        structural_chunks: list[Chunk] = []
        for start, end in packed:
            heading_path, section_title, source_location = _heading_metadata(
                document,
                end=end,
                headings=headings,
            )
            structural_chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=_stable_chunk_id(
                        document,
                        start_offset=start,
                        end_offset=end,
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                        algorithm="structure-v2",
                        metadata=heading_path + (source_location,),
                    ),
                    content=document.content[start:end],
                    start_offset=start,
                    end_offset=end,
                    heading_path=heading_path,
                    section_title=section_title,
                    source_location=source_location,
                )
            )
        return tuple(structural_chunks)


__all__ = ["Chunk", "ChunkLimitError", "StructureAwareChunker", "TextChunker"]
