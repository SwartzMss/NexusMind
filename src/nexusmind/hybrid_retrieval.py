"""Bounded reciprocal-rank fusion over lexical and semantic chunk indexes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, cast

from .knowledge_chunking import Chunk
from .knowledge_collection import CloneableChunkIndex
from .knowledge_retrieval import (
    ChunkIndexError,
    RetrievalCandidateDiagnostic,
    RetrievalDiagnostics,
    RetrievalStage,
    SearchHit,
)


class HybridChunkIndexError(ChunkIndexError):
    """A hybrid child operation or fused result is invalid."""


class HybridChunkIndexLimitError(HybridChunkIndexError):
    """A hybrid query or fusion operation exceeds a configured bound."""


class HybridBackendCoherenceError(HybridChunkIndexError):
    """Child results disagree about canonical chunk identity."""


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridChunkIndexLimits:
    """Explicit final-result, candidate, and temporary fusion bounds."""

    max_results: int = 100
    max_candidates_per_backend: int = 100
    max_fusion_entries: int = 200

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_fusion_entries < self.max_candidates_per_backend:
            raise ValueError(
                "max_fusion_entries must be at least max_candidates_per_backend"
            )


class HybridChunkIndex:
    """Own and atomically compose two cloneable indexes using standard RRF."""

    def __init__(
        self,
        *,
        lexical_index_factory: Callable[[], CloneableChunkIndex],
        semantic_index_factory: Callable[[], CloneableChunkIndex],
        limits: HybridChunkIndexLimits | None = None,
        rrf_k: int = 60,
        candidate_depth: int = 100,
    ) -> None:
        if not callable(lexical_index_factory):
            raise TypeError("lexical_index_factory must be callable")
        if not callable(semantic_index_factory):
            raise TypeError("semantic_index_factory must be callable")
        if limits is not None and not isinstance(limits, HybridChunkIndexLimits):
            raise TypeError("limits must be HybridChunkIndexLimits")
        if type(rrf_k) is not int:
            raise TypeError("rrf_k must be an integer")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if type(candidate_depth) is not int:
            raise TypeError("candidate_depth must be an integer")
        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be greater than zero")
        configured_limits = limits or HybridChunkIndexLimits()
        if candidate_depth > configured_limits.max_candidates_per_backend:
            raise ValueError(
                "candidate_depth exceeds max_candidates_per_backend"
            )
        try:
            lexical = lexical_index_factory()
            semantic = semantic_index_factory()
        except Exception as exc:
            raise HybridChunkIndexError("hybrid index factory failed") from exc
        self._require_child(lexical)
        self._require_child(semantic)
        if lexical is semantic:
            raise HybridChunkIndexError("hybrid child indexes must be independently owned")
        self._lexical_factory = lexical_index_factory
        self._semantic_factory = semantic_index_factory
        self._limits = configured_limits
        self._rrf_k = rrf_k
        self._candidate_depth = candidate_depth
        self._lexical = lexical
        self._semantic = semantic

    @property
    def max_search_results(self) -> int:
        """Return the configured maximum accepted final search limit."""

        return self._limits.max_results

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("add", chunks)

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("replace_document", document_id, chunks)

    def remove_document(self, document_id: str) -> None:
        self._mutate("remove_document", document_id)

    def clone(self) -> "HybridChunkIndex":
        try:
            lexical = self._lexical.clone()
            semantic = self._semantic.clone()
            self._require_child(lexical)
            self._require_child(semantic)
            if (
                lexical is self._lexical
                or semantic is self._semantic
                or lexical is semantic
            ):
                raise TypeError
        except Exception as exc:
            raise HybridChunkIndexError("hybrid backend clone failed") from exc
        clone = object.__new__(HybridChunkIndex)
        clone._lexical_factory = self._lexical_factory
        clone._semantic_factory = self._semantic_factory
        clone._limits = self._limits
        clone._rrf_k = self._rrf_k
        clone._candidate_depth = self._candidate_depth
        clone._lexical = lexical
        clone._semantic = semantic
        return clone

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        return self._compute_fusion(query, limit=limit)[0]

    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics:
        hits, lexical, semantic = self._compute_fusion(query, limit=limit)
        selected_chunk_ids = {hit.chunk.chunk_id for hit in hits}
        candidates = (
            tuple(
                RetrievalCandidateDiagnostic(
                    stage=RetrievalStage.LEXICAL,
                    rank=rank,
                    chunk=hit.chunk,
                    score=hit.score,
                    matched_terms=hit.matched_terms,
                    rrf_contribution=1.0 / (self._rrf_k + rank),
                    selected=hit.chunk.chunk_id in selected_chunk_ids,
                )
                for rank, hit in enumerate(lexical, start=1)
            )
            + tuple(
                RetrievalCandidateDiagnostic(
                    stage=RetrievalStage.SEMANTIC,
                    rank=rank,
                    chunk=hit.chunk,
                    score=hit.score,
                    matched_terms=hit.matched_terms,
                    rrf_contribution=1.0 / (self._rrf_k + rank),
                    selected=hit.chunk.chunk_id in selected_chunk_ids,
                )
                for rank, hit in enumerate(semantic, start=1)
            )
            + tuple(
                RetrievalCandidateDiagnostic(
                    stage=RetrievalStage.FUSION,
                    rank=rank,
                    chunk=hit.chunk,
                    score=hit.score,
                    matched_terms=hit.matched_terms,
                    selected=True,
                )
                for rank, hit in enumerate(hits, start=1)
            )
        )
        return RetrievalDiagnostics(hits=hits, candidates=candidates)

    def _compute_fusion(
        self, query: str, *, limit: int
    ) -> tuple[tuple[SearchHit, ...], tuple[SearchHit, ...], tuple[SearchHit, ...]]:
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if limit > self._limits.max_results:
            raise HybridChunkIndexLimitError("limit exceeds max_results")
        backend_limit = max(limit, self._candidate_depth)
        if backend_limit > self._limits.max_candidates_per_backend:
            raise HybridChunkIndexLimitError(
                "candidate depth exceeds max_candidates_per_backend"
            )
        try:
            lexical_hits = self._lexical.search(query, limit=backend_limit)
        except Exception as exc:
            raise HybridChunkIndexError("hybrid backend search failed") from exc
        lexical = self._bounded_hits(lexical_hits, "lexical", backend_limit)
        try:
            semantic_hits = self._semantic.search(query, limit=backend_limit)
        except Exception as exc:
            raise HybridChunkIndexError("hybrid backend search failed") from exc
        semantic = self._bounded_hits(semantic_hits, "semantic", backend_limit)
        if len(lexical) + len(semantic) > self._limits.max_fusion_entries:
            raise HybridChunkIndexLimitError("fusion exceeds max_fusion_entries")
        lexical = self._validated_hits(lexical, "lexical")
        semantic = self._validated_hits(semantic, "semantic")

        chunks: dict[str, Chunk] = {}
        scores: dict[str, float] = {}
        lexical_terms: dict[str, tuple[str, ...]] = {}
        for backend, is_lexical in ((lexical, True), (semantic, False)):
            for rank, hit in enumerate(backend, start=1):
                chunk_id = hit.chunk.chunk_id
                existing = chunks.get(chunk_id)
                if existing is not None and existing != hit.chunk:
                    raise HybridBackendCoherenceError(
                        "hybrid backends disagree about chunk data"
                    )
                chunks[chunk_id] = hit.chunk
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                    self._rrf_k + rank
                )
                if is_lexical:
                    lexical_terms[chunk_id] = hit.matched_terms
        fused = [
            SearchHit(
                chunk=chunk,
                score=float(scores[chunk_id]),
                matched_terms=lexical_terms.get(chunk_id, ()),
            )
            for chunk_id, chunk in chunks.items()
        ]
        fused.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return tuple(fused[:limit]), lexical, semantic

    def _mutate(self, method: str, *args: object) -> None:
        try:
            lexical = self._lexical.clone()
            semantic = self._semantic.clone()
            self._require_child(lexical)
            self._require_child(semantic)
            if (
                lexical is self._lexical
                or semantic is self._semantic
                or lexical is semantic
            ):
                raise TypeError
            getattr(lexical, method)(*args)
            getattr(semantic, method)(*args)
        except Exception as exc:
            raise HybridChunkIndexError("hybrid backend mutation failed") from exc
        self._lexical = lexical
        self._semantic = semantic

    @staticmethod
    def _bounded_hits(
        hits: object,
        backend: str,
        backend_limit: int,
    ) -> tuple[SearchHit, ...]:
        if type(hits) is not tuple:
            raise HybridBackendCoherenceError(
                f"{backend} backend must return an exact tuple"
            )
        if len(hits) > backend_limit:
            raise HybridBackendCoherenceError(
                f"{backend} backend exceeded requested candidate limit"
            )
        return hits

    @staticmethod
    def _validated_hits(
        hits: tuple[object, ...], backend: str
    ) -> tuple[SearchHit, ...]:
        seen: set[str] = set()
        for hit in hits:
            if type(hit) is not SearchHit or not isinstance(hit.chunk, Chunk):
                raise HybridBackendCoherenceError(
                    f"{backend} backend returned an invalid SearchHit"
                )
            if type(hit.score) is not float or not isfinite(hit.score):
                raise HybridBackendCoherenceError(
                    f"{backend} backend returned an invalid score"
                )
            if type(hit.matched_terms) is not tuple or any(
                type(term) is not str for term in hit.matched_terms
            ):
                raise HybridBackendCoherenceError(
                    f"{backend} backend returned invalid matched_terms"
                )
            if hit.chunk.chunk_id in seen:
                raise HybridBackendCoherenceError(
                    f"{backend} backend returned a duplicate chunk_id"
                )
            seen.add(hit.chunk.chunk_id)
        return cast(tuple[SearchHit, ...], hits)

    @staticmethod
    def _require_child(child: object) -> None:
        for method in (
            "add",
            "replace_document",
            "remove_document",
            "search",
            "clone",
        ):
            if not callable(getattr(child, method, None)):
                raise TypeError("hybrid child index does not implement lifecycle")


__all__ = [
    "HybridBackendCoherenceError",
    "HybridChunkIndex",
    "HybridChunkIndexError",
    "HybridChunkIndexLimitError",
    "HybridChunkIndexLimits",
]
