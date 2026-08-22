"""Deterministic, provenance-preserving assembly of retrieval context."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from .knowledge import Document, KnowledgeSource
from .knowledge_chunking import Chunk


@dataclass(frozen=True, slots=True)
class ContextPassage:
    """One selected retrieval passage with its complete knowledge lineage."""

    source: KnowledgeSource
    document: Document
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.source) is not KnowledgeSource:
            raise TypeError("source must be a KnowledgeSource")
        if type(self.document) is not Document:
            raise TypeError("document must be a Document")
        if type(self.chunk) is not Chunk:
            raise TypeError("chunk must be a Chunk")
        if type(self.score) is not float:
            raise TypeError("score must be a float")
        if type(self.matched_terms) is not tuple or any(
            type(term) is not str for term in self.matched_terms
        ):
            raise TypeError("matched_terms must be a tuple of strings")
        if self.document.source_id != self.source.source_id:
            raise ValueError("document must belong to source")
        if self.chunk.document_id != self.document.document_id:
            raise ValueError("chunk must belong to document")
        if not (
            0 <= self.chunk.start_offset <= self.chunk.end_offset <= len(self.document.content)
        ):
            raise ValueError("chunk offsets must reference the document content")
        if (
            self.document.content[self.chunk.start_offset : self.chunk.end_offset]
            != self.chunk.content
        ):
            raise ValueError("chunk content must match the referenced document slice")

    @property
    def content(self) -> str:
        """Return the exact canonical document slice referenced by the chunk."""

        return self.chunk.content

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def document_id(self) -> str:
        return self.document.document_id

    @property
    def logical_path(self) -> str:
        return self.document.logical_path

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def start_offset(self) -> int:
        return self.chunk.start_offset

    @property
    def end_offset(self) -> int:
        return self.chunk.end_offset


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

    This intentionally does not claim model-tokenizer accuracy. Runs of word
    characters and individual punctuation characters each count as one unit.
    """

    if type(text) is not str:
        raise TypeError("text must be a string")
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def assemble_context(
    query: str,
    results: Iterable[object],
    *,
    max_passages: int,
    max_chars: int | None = None,
    max_tokens: int | None = None,
) -> ContextPackage:
    """Select ranked results using stable first-wins duplicate/overlap policy."""

    if type(query) is not str:
        raise TypeError("query must be a string")
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
    overlaps_removed = 0
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

        identity = (chunk.document_id, chunk.start_offset, chunk.end_offset, chunk.content)
        if identity in identities:
            duplicates_removed += 1
            continue
        if any(
            chunk.start_offset < end and start < chunk.end_offset
            for start, end in intervals.get(chunk.document_id, ())
        ):
            overlaps_removed += 1
            continue

        chars = len(chunk.content)
        tokens = estimate_token_count(chunk.content)
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
                source=deepcopy(source),
                document=deepcopy(document),
                chunk=deepcopy(chunk),
                score=hit.score,
                matched_terms=hit.matched_terms,
            )
        )
        identities.add(identity)
        intervals.setdefault(chunk.document_id, []).append(
            (chunk.start_offset, chunk.end_offset)
        )
        used_chars += chars
        used_tokens += tokens

    return ContextPackage(
        query=query,
        passages=tuple(selected),
        metadata={
            "candidate_count": candidates,
            "passage_count": len(selected),
            "character_count": used_chars,
            "estimated_token_count": used_tokens,
            "max_passages": max_passages,
            "max_chars": max_chars,
            "max_tokens": max_tokens,
            "duplicates_removed": duplicates_removed,
            "overlaps_removed": overlaps_removed,
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


__all__ = [
    "ContextPackage",
    "ContextPassage",
    "assemble_context",
    "estimate_token_count",
]
