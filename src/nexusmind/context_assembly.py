"""Deterministic, provenance-preserving assembly of retrieval context."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
import re
from typing import TYPE_CHECKING, Any

from .knowledge import Document, KnowledgeSource
from .knowledge_chunking import Chunk

if TYPE_CHECKING:
    from .knowledge_collection import KnowledgeSearchResult


class ContextAssemblyLimitError(ValueError):
    """The candidate input exceeds the public assembler's explicit bound."""


@dataclass(frozen=True, slots=True)
class ContextPassage:
    """One compact canonical provenance reference plus selected content."""

    source_id: str
    document_id: str
    logical_path: str
    document_content_hash: str
    chunk_id: str
    chunk_start_offset: int
    chunk_end_offset: int
    start_offset: int
    end_offset: int
    content: str
    score: float
    matched_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "document_id",
            "logical_path",
            "document_content_hash",
            "chunk_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if re.fullmatch(r"[0-9a-f]{64}", self.document_content_hash) is None:
            raise ValueError("document_content_hash must be a lowercase SHA-256 digest")
        for name in (
            "chunk_start_offset",
            "chunk_end_offset",
            "start_offset",
            "end_offset",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not (
            0
            <= self.chunk_start_offset
            <= self.start_offset
            < self.end_offset
            <= self.chunk_end_offset
        ):
            raise ValueError("passage offsets must be contained by the original chunk")
        if type(self.content) is not str:
            raise TypeError("content must be a string")
        if len(self.content) != self.end_offset - self.start_offset:
            raise ValueError("content length must match passage offsets")
        if type(self.score) is not float:
            raise TypeError("score must be a float")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if type(self.matched_terms) is not tuple or any(
            type(term) is not str for term in self.matched_terms
        ):
            raise TypeError("matched_terms must be a tuple of strings")

@dataclass(frozen=True, slots=True)
class ContextPackage:
    """A bounded context package suitable for downstream RAG consumers."""

    query: str
    passages: tuple[ContextPassage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.query) is not str:
            raise TypeError("query must be a string")
        if type(self.passages) is not tuple:
            raise TypeError("passages must be a tuple")
        if any(type(passage) is not ContextPassage for passage in self.passages):
            raise TypeError("passages must contain only ContextPassage values")
        if type(self.metadata) is not dict:
            raise TypeError("metadata must be a dictionary")
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


def estimate_token_count(text: str) -> int:
    """Return a provider-neutral deterministic token estimate.

    This intentionally does not claim model-tokenizer accuracy. ASCII word
    runs count as one unit; every other non-whitespace character (including
    each CJK character and punctuation mark) counts as one unit.
    """

    if type(text) is not str:
        raise TypeError("text must be a string")
    return len(re.findall(r"[A-Za-z0-9_]+|[^\s]", text, flags=re.UNICODE))


def assemble_context(
    query: str,
    results: tuple["KnowledgeSearchResult", ...],
    *,
    max_passages: int,
    max_candidates: int = 100,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> ContextPackage:
    """Select ranked results using stable first-wins duplicate/overlap policy."""

    if type(query) is not str:
        raise TypeError("query must be a string")
    if type(results) is not tuple:
        raise TypeError("results must be a tuple")
    if type(max_candidates) is not int:
        raise TypeError("max_candidates must be an integer")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be greater than zero")
    if len(results) > max_candidates:
        raise ContextAssemblyLimitError(
            f"results contains {len(results)} candidates; limit is {max_candidates}"
        )
    _validate_optional_limit("max_chars", max_chars)
    _validate_optional_limit("max_tokens", max_tokens)
    if type(max_passages) is not int:
        raise TypeError("max_passages must be an integer")
    if max_passages <= 0:
        raise ValueError("max_passages must be greater than zero")

    selected: list[ContextPassage] = []
    identities: set[tuple[str, int, int, str]] = set()
    intervals: dict[str, list[tuple[int, int]]] = {}
    used_chars = 0
    used_tokens = 0
    candidates = 0
    duplicates_removed = 0
    overlap_characters_removed = 0
    limited = False

    for result in results:
        candidates += 1
        source = getattr(result, "source", None)
        document = getattr(result, "document", None)
        hit = getattr(result, "hit", None)
        chunk = getattr(hit, "chunk", None)
        if not isinstance(source, KnowledgeSource) or not isinstance(document, Document):
            raise TypeError("results must contain provenance-resolved search results")
        if not isinstance(chunk, Chunk):
            raise TypeError("results must contain search hits with chunks")
        _validate_result_provenance(source, document, chunk)

        identity = (chunk.document_id, chunk.start_offset, chunk.end_offset, chunk.content)
        if identity in identities:
            duplicates_removed += 1
            continue
        identities.add(identity)
        uncovered = _subtract_intervals(
            chunk.start_offset,
            chunk.end_offset,
            intervals.get(chunk.document_id, ()),
        )
        if uncovered != ((chunk.start_offset, chunk.end_offset),):
            overlap_characters_removed += (
                chunk.end_offset - chunk.start_offset
            ) - sum(
                end - start for start, end in uncovered
            )
        for start_offset, end_offset in uncovered:
            content = document.content[start_offset:end_offset]
            chars = len(content)
            tokens = estimate_token_count(content)
            if (max_chars is not None and used_chars + chars > max_chars) or (
                max_tokens is not None and used_tokens + tokens > max_tokens
            ):
                limited = True
                continue
            if len(selected) >= max_passages:
                limited = True
                break

            selected.append(
                ContextPassage(
                    source_id=source.source_id,
                    document_id=document.document_id,
                    logical_path=document.logical_path,
                    document_content_hash=document.content_hash,
                    chunk_id=chunk.chunk_id,
                    chunk_start_offset=chunk.start_offset,
                    chunk_end_offset=chunk.end_offset,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    content=content,
                    score=hit.score,
                    matched_terms=hit.matched_terms,
                )
            )
            intervals.setdefault(chunk.document_id, []).append(
                (start_offset, end_offset)
            )
            used_chars += chars
            used_tokens += tokens
        if len(selected) >= max_passages:
            break

    return ContextPackage(
        query=query,
        passages=tuple(selected),
        metadata={
            "candidate_count": candidates,
            "passage_count": len(selected),
            "character_count": used_chars,
            "estimated_token_count": used_tokens,
            "max_passages": max_passages,
            "max_candidates": max_candidates,
            "max_chars": max_chars,
            "max_tokens": max_tokens,
            "duplicates_removed": duplicates_removed,
            "overlap_characters_removed": overlap_characters_removed,
            "limited": limited,
            "token_count_method": "provider_neutral_estimate",
        },
    )


def _validate_optional_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer or None")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_result_provenance(
    source: KnowledgeSource, document: Document, chunk: Chunk
) -> None:
    if document.source_id != source.source_id:
        raise ValueError("result document must belong to source")
    if chunk.document_id != document.document_id:
        raise ValueError("result chunk must belong to document")
    if type(chunk.start_offset) is not int or type(chunk.end_offset) is not int:
        raise TypeError("result chunk offsets must be integers")
    if not (0 <= chunk.start_offset < chunk.end_offset <= len(document.content)):
        raise ValueError("result chunk offsets must reference canonical content")
    if document.content[chunk.start_offset : chunk.end_offset] != chunk.content:
        raise ValueError("result chunk content must match canonical document slice")


def _subtract_intervals(
    start: int,
    end: int,
    covered: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return the document-ordered portions of ``[start, end)`` not covered."""

    remaining = [(start, end)]
    for covered_start, covered_end in sorted(covered):
        next_remaining: list[tuple[int, int]] = []
        for current_start, current_end in remaining:
            if covered_end <= current_start or current_end <= covered_start:
                next_remaining.append((current_start, current_end))
                continue
            if current_start < covered_start:
                next_remaining.append((current_start, covered_start))
            if covered_end < current_end:
                next_remaining.append((covered_end, current_end))
        remaining = next_remaining
    return tuple(remaining)


__all__ = [
    "ContextPackage",
    "ContextPassage",
    "ContextAssemblyLimitError",
    "assemble_context",
    "estimate_token_count",
]
