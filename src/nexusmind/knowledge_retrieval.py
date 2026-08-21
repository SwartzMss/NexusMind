"""Source-neutral retrieval contracts and a bounded in-memory BM25 index.

The lexical implementation uses a configurable analyzer and positive-IDF
BM25 scoring. Results are ordered by descending score and then by ascending
``chunk_id``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from math import isfinite, log
from typing import Protocol

from .knowledge_chunking import Chunk
from .lexical_analysis import LexicalAnalyzer, UnicodeCJKLexicalAnalyzer


_BM25_K1 = 1.2
_BM25_B = 0.75


class ChunkIndexError(Exception):
    """Base class for controlled chunk-index failures."""


class ChunkIndexLimitError(ChunkIndexError):
    """An index mutation or query exceeds a configured resource bound."""


class LexicalAnalysisError(ChunkIndexError):
    """A configured lexical analyzer failed or returned invalid output."""


class ChunkIdentityConflictError(ChunkIndexError):
    """A chunk ID was reused for different chunk data."""


class DocumentReplacementError(ChunkIndexError):
    """A document replacement input is invalid."""


class RetrievalStage(str, Enum):
    """The retrieval pipeline stage represented by a diagnostic row."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    FUSION = "fusion"
    RERANKER = "reranker"


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkIndexLimits:
    """Explicit resource bounds for an in-memory chunk index.

    Custom analyzers allocate their returned tuple before the index can inspect
    it. Valid returned tokens are bounded here before corpus statistics retain
    them or query scoring uses them.
    """

    max_chunks: int = 10_000
    max_total_chars: int = 10_000_000
    max_total_analyzed_tokens: int = 10_000_000
    max_total_analyzed_token_chars: int = 200_000_000
    max_chunks_per_document: int = 10_000
    max_query_chars: int = 1_024
    max_query_terms: int = 32
    max_query_analyzed_chars: int = 20_480
    max_results: int = 100

    def __post_init__(self) -> None:
        for name in (
            "max_chunks",
            "max_total_chars",
            "max_total_analyzed_tokens",
            "max_total_analyzed_token_chars",
            "max_chunks_per_document",
            "max_query_chars",
            "max_query_terms",
            "max_query_analyzed_chars",
            "max_results",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One deterministic backend match and its explicit score details."""

    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalCandidateDiagnostic:
    """One candidate emitted by a retrieval pipeline stage."""

    stage: RetrievalStage
    rank: int
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()
    rrf_contribution: float | None = None
    selected: bool = False

    def __post_init__(self) -> None:
        if type(self.stage) is not RetrievalStage:
            raise TypeError("stage must be RetrievalStage")
        if type(self.rank) is not int:
            raise TypeError("rank must be an integer")
        if self.rank <= 0:
            raise ValueError("rank must be greater than zero")
        if not isinstance(self.chunk, Chunk):
            raise TypeError("chunk must be a Chunk")
        if type(self.score) is not float:
            raise TypeError("score must be a float")
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        if type(self.matched_terms) is not tuple:
            raise TypeError("matched_terms must be a tuple")
        if any(type(term) is not str for term in self.matched_terms):
            raise TypeError("matched_terms must contain only strings")
        if self.rrf_contribution is not None:
            if type(self.rrf_contribution) is not float:
                raise TypeError("rrf_contribution must be a float or None")
            if not isfinite(self.rrf_contribution):
                raise ValueError("rrf_contribution must be finite")
            if self.rrf_contribution <= 0:
                raise ValueError("rrf_contribution must be greater than zero")
        if type(self.selected) is not bool:
            raise TypeError("selected must be a boolean")


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Final hits and stage candidates captured for one retrieval request."""

    hits: tuple[SearchHit, ...]
    candidates: tuple[RetrievalCandidateDiagnostic, ...]

    def __post_init__(self) -> None:
        if type(self.hits) is not tuple:
            raise TypeError("hits must be a tuple")
        if any(type(hit) is not SearchHit for hit in self.hits):
            raise TypeError("hits must contain only SearchHit values")
        if type(self.candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        if any(
            type(candidate) is not RetrievalCandidateDiagnostic
            for candidate in self.candidates
        ):
            raise TypeError("candidates must contain only RetrievalCandidateDiagnostic values")


class ChunkIndex(Protocol):
    """Source-neutral contract implemented by chunk retrieval backends."""

    def add(self, chunks: tuple[Chunk, ...]) -> None: ...

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None: ...

    def remove_document(self, document_id: str) -> None: ...

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]: ...


class DiagnosticChunkIndex(ChunkIndex, Protocol):
    """Chunk retrieval contract that can expose final candidate diagnostics."""

    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics: ...


class InMemoryChunkIndex:
    """Dependency-free, process-local lexical chunk retrieval.

    Custom analyzers must be deterministic, effectively immutable, stateless
    with respect to analysis results, and safe to share between index clones.
    ``clone()`` shares the analyzer instance while copying mutable index state.
    """

    def __init__(
        self,
        *,
        limits: ChunkIndexLimits | None = None,
        analyzer: LexicalAnalyzer | None = None,
    ) -> None:
        if limits is not None and not isinstance(limits, ChunkIndexLimits):
            raise TypeError("limits must be ChunkIndexLimits")
        if analyzer is not None and not callable(getattr(analyzer, "analyze", None)):
            raise TypeError("analyzer must provide a callable analyze method")
        self._limits = limits or ChunkIndexLimits()
        self._analyzer = (
            UnicodeCJKLexicalAnalyzer() if analyzer is None else analyzer
        )
        self._chunks: dict[str, Chunk] = {}
        self._document_chunks: dict[str, set[str]] = {}
        self._total_chars = 0
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._token_counts: dict[str, int] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._total_tokens = 0

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        additions = self._validated_additions(chunks)
        document_counts = {key: len(value) for key, value in self._document_chunks.items()}
        for chunk in additions.values():
            document_counts[chunk.document_id] = document_counts.get(chunk.document_id, 0) + 1
        self._preflight(
            chunk_count=len(self._chunks) + len(additions),
            total_chars=self._total_chars + sum(len(chunk.content) for chunk in additions.values()),
            document_counts=document_counts.values(),
        )
        candidate_chunks = self._chunks.copy()
        candidate_document_chunks = {
            document_id: chunk_ids.copy()
            for document_id, chunk_ids in self._document_chunks.items()
        }
        for chunk in additions.values():
            candidate_chunks[chunk.chunk_id] = chunk
            candidate_document_chunks.setdefault(chunk.document_id, set()).add(
                chunk.chunk_id
            )
        self._commit_candidate(
            chunks=candidate_chunks,
            document_chunks=candidate_document_chunks,
            total_chars=self._total_chars
            + sum(len(chunk.content) for chunk in additions.values()),
        )

    def clone(self) -> "InMemoryChunkIndex":
        """Return an independent index with the same limits and state.

        Analyzer configuration is immutable runtime configuration and is shared
        between the original and clone. All mutable index data is copied.
        """

        clone = InMemoryChunkIndex(limits=self._limits, analyzer=self._analyzer)
        clone._chunks = self._chunks.copy()
        clone._document_chunks = {
            document_id: chunk_ids.copy()
            for document_id, chunk_ids in self._document_chunks.items()
        }
        clone._total_chars = self._total_chars
        clone._term_frequencies = {
            chunk_id: frequencies.copy()
            for chunk_id, frequencies in self._term_frequencies.items()
        }
        clone._token_counts = self._token_counts.copy()
        clone._document_frequencies = self._document_frequencies.copy()
        clone._total_tokens = self._total_tokens
        return clone

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._require_document_id(document_id)
        self._require_chunk_tuple(chunks)
        replacement: dict[str, Chunk] = {}
        for chunk in chunks:
            self._require_chunk(chunk)
            if chunk.document_id != document_id:
                raise DocumentReplacementError("all replacement chunks must belong to document_id")
            if chunk.chunk_id in replacement:
                raise DocumentReplacementError("replacement contains duplicate chunk_id values")
            replacement[chunk.chunk_id] = chunk

        old_ids = self._document_chunks.get(document_id, set())
        for chunk in replacement.values():
            existing = self._chunks.get(chunk.chunk_id)
            if existing is not None and existing != chunk:
                raise ChunkIdentityConflictError(f"chunk_id conflicts with indexed chunk: {chunk.chunk_id}")
            if existing is not None and chunk.chunk_id not in old_ids:
                raise ChunkIdentityConflictError(f"chunk_id already belongs to another document: {chunk.chunk_id}")

        remaining_count = len(self._chunks) - len(old_ids)
        remaining_chars = self._total_chars - sum(len(self._chunks[key].content) for key in old_ids)
        other_counts = [len(ids) for key, ids in self._document_chunks.items() if key != document_id]
        self._preflight(
            chunk_count=remaining_count + len(replacement),
            total_chars=remaining_chars + sum(len(chunk.content) for chunk in replacement.values()),
            document_counts=(*other_counts, len(replacement)),
        )

        candidate_chunks = self._chunks.copy()
        candidate_document_chunks = {
            key: chunk_ids.copy()
            for key, chunk_ids in self._document_chunks.items()
        }
        for chunk_id in old_ids:
            del candidate_chunks[chunk_id]
        if replacement:
            candidate_document_chunks[document_id] = set(replacement)
            candidate_chunks.update(replacement)
        else:
            candidate_document_chunks.pop(document_id, None)
        self._commit_candidate(
            chunks=candidate_chunks,
            document_chunks=candidate_document_chunks,
            total_chars=remaining_chars
            + sum(len(chunk.content) for chunk in replacement.values()),
        )

    def remove_document(self, document_id: str) -> None:
        self._require_document_id(document_id)
        candidate_chunks = self._chunks.copy()
        candidate_document_chunks = {
            key: chunk_ids.copy()
            for key, chunk_ids in self._document_chunks.items()
        }
        chunk_ids = candidate_document_chunks.pop(document_id, set())
        removed_chars = 0
        for chunk_id in chunk_ids:
            removed_chars += len(candidate_chunks[chunk_id].content)
            del candidate_chunks[chunk_id]
        self._commit_candidate(
            chunks=candidate_chunks,
            document_chunks=candidate_document_chunks,
            total_chars=self._total_chars - removed_chars,
        )

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        return self._compute_hits(query, limit=limit)

    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics:
        hits = self._compute_hits(query, limit=limit)
        candidates = tuple(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.LEXICAL,
                rank=rank,
                chunk=hit.chunk,
                score=hit.score,
                matched_terms=hit.matched_terms,
                selected=True,
            )
            for rank, hit in enumerate(hits, start=1)
        )
        return RetrievalDiagnostics(hits=hits, candidates=candidates)

    def _compute_hits(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        if type(query) is not str:
            raise TypeError("query must be a string")
        if len(query) > self._limits.max_query_chars:
            raise ChunkIndexLimitError("query exceeds max_query_chars")
        raw_terms = self._analyze(query)
        if len(raw_terms) > self._limits.max_query_terms:
            raise ChunkIndexLimitError("query exceeds max_query_terms")
        if sum(len(term) for term in raw_terms) > self._limits.max_query_analyzed_chars:
            raise ChunkIndexLimitError("query exceeds max_query_analyzed_chars")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if limit > self._limits.max_results:
            raise ChunkIndexLimitError("limit exceeds max_results")

        terms = tuple(dict.fromkeys(raw_terms))
        if not terms:
            return ()
        chunk_count = len(self._chunks)
        if chunk_count == 0 or self._total_tokens == 0:
            return ()
        average_length = self._total_tokens / chunk_count
        hits: list[SearchHit] = []
        for chunk_id, chunk in self._chunks.items():
            frequencies = self._term_frequencies[chunk_id]
            matched = tuple(term for term in terms if frequencies.get(term, 0) > 0)
            if matched:
                chunk_length = self._token_counts[chunk_id]
                score = 0.0
                for term in matched:
                    term_frequency = frequencies[term]
                    document_frequency = self._document_frequencies[term]
                    inverse_document_frequency = log(
                        1
                        + (chunk_count - document_frequency + 0.5)
                        / (document_frequency + 0.5)
                    )
                    normalizer = term_frequency + _BM25_K1 * (
                        1
                        - _BM25_B
                        + _BM25_B * chunk_length / average_length
                    )
                    score += (
                        inverse_document_frequency
                        * term_frequency
                        * (_BM25_K1 + 1)
                        / normalizer
                    )
                hits.append(SearchHit(chunk=chunk, score=score, matched_terms=matched))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(hits[:limit])

    def _statistics_for(
        self,
        chunks: dict[str, Chunk],
    ) -> tuple[dict[str, Counter[str]], dict[str, int], Counter[str], int]:
        term_frequencies: dict[str, Counter[str]] = {}
        token_counts: dict[str, int] = {}
        document_frequencies: Counter[str] = Counter()
        total_tokens = 0
        total_token_chars = 0
        for chunk_id, chunk in chunks.items():
            tokens = self._analyze(chunk.content)
            total_tokens += len(tokens)
            if total_tokens > self._limits.max_total_analyzed_tokens:
                raise ChunkIndexLimitError("index exceeds max_total_analyzed_tokens")
            total_token_chars += sum(len(token) for token in tokens)
            if total_token_chars > self._limits.max_total_analyzed_token_chars:
                raise ChunkIndexLimitError("index exceeds max_total_analyzed_token_chars")
            frequencies = Counter(tokens)
            term_frequencies[chunk_id] = frequencies
            token_counts[chunk_id] = len(tokens)
            document_frequencies.update(frequencies.keys())
        return term_frequencies, token_counts, document_frequencies, total_tokens

    def _analyze(self, text: str) -> tuple[str, ...]:
        try:
            tokens = self._analyzer.analyze(text)
            if type(tokens) is not tuple:
                raise TypeError("analyzer must return an exact tuple")
            if any(type(token) is not str or token == "" for token in tokens):
                raise TypeError("analyzer must return non-empty exact str tokens")
            return tuple(tokens)
        except Exception as error:
            raise LexicalAnalysisError("lexical analysis failed") from error

    def _commit_candidate(
        self,
        *,
        chunks: dict[str, Chunk],
        document_chunks: dict[str, set[str]],
        total_chars: int,
    ) -> None:
        (
            term_frequencies,
            token_counts,
            document_frequencies,
            total_tokens,
        ) = self._statistics_for(chunks)
        self._chunks = chunks
        self._document_chunks = document_chunks
        self._total_chars = total_chars
        self._term_frequencies = term_frequencies
        self._token_counts = token_counts
        self._document_frequencies = document_frequencies
        self._total_tokens = total_tokens

    def _validated_additions(self, chunks: tuple[Chunk, ...]) -> dict[str, Chunk]:
        self._require_chunk_tuple(chunks)
        additions: dict[str, Chunk] = {}
        for chunk in chunks:
            self._require_chunk(chunk)
            candidate = additions.get(chunk.chunk_id)
            existing = self._chunks.get(chunk.chunk_id)
            if candidate is not None and candidate != chunk:
                raise ChunkIdentityConflictError(f"chunk_id conflicts within input: {chunk.chunk_id}")
            if existing is not None and existing != chunk:
                raise ChunkIdentityConflictError(f"chunk_id conflicts with indexed chunk: {chunk.chunk_id}")
            if existing is None:
                additions[chunk.chunk_id] = chunk
        return additions

    def _preflight(self, *, chunk_count: int, total_chars: int, document_counts: object) -> None:
        if chunk_count > self._limits.max_chunks:
            raise ChunkIndexLimitError("index exceeds max_chunks")
        if total_chars > self._limits.max_total_chars:
            raise ChunkIndexLimitError("index exceeds max_total_chars")
        if any(count > self._limits.max_chunks_per_document for count in document_counts):
            raise ChunkIndexLimitError("document exceeds max_chunks_per_document")

    @staticmethod
    def _require_chunk_tuple(chunks: tuple[Chunk, ...]) -> None:
        if type(chunks) is not tuple:
            raise TypeError("chunks must be a tuple")

    @staticmethod
    def _require_chunk(chunk: Chunk) -> None:
        if not isinstance(chunk, Chunk):
            raise TypeError("chunks must contain only Chunk values")

    @staticmethod
    def _require_document_id(document_id: str) -> None:
        if type(document_id) is not str or not document_id.strip():
            raise ValueError("document_id must be a non-empty string")


__all__ = [
    "ChunkIdentityConflictError",
    "ChunkIndex",
    "ChunkIndexError",
    "ChunkIndexLimitError",
    "ChunkIndexLimits",
    "DiagnosticChunkIndex",
    "DocumentReplacementError",
    "InMemoryChunkIndex",
    "LexicalAnalysisError",
    "SearchHit",
    "RetrievalCandidateDiagnostic",
    "RetrievalDiagnostics",
    "RetrievalStage",
]
