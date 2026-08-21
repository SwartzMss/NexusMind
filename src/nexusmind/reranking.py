"""Bounded second-stage reranking over a fixed retrieval candidate set."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Protocol, cast

from .knowledge_chunking import Chunk
from .knowledge_collection import CloneableChunkIndex
from .knowledge_retrieval import (
    ChunkIndexError,
    RetrievalCandidateDiagnostic,
    RetrievalDiagnostics,
    RetrievalStage,
    SearchHit,
)


class RerankerError(ChunkIndexError):
    """A reranking composition operation failed."""


class RerankerLimitError(RerankerError):
    """A reranking operation exceeds a configured resource bound."""


class RerankerCoherenceError(RerankerError):
    """A base index or reranker returned incoherent candidate data."""


class Reranker(Protocol):
    """Provider-neutral, search-only second-stage ranking contract."""

    def rerank(
        self,
        query: str,
        candidates: tuple[SearchHit, ...],
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RerankerLimits:
    """Explicit query, candidate, character, and output bounds."""

    max_query_chars: int = 1_024
    max_candidates: int = 100
    max_total_candidate_chars: int = 1_000_000
    max_results: int = 100

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


class RerankedChunkIndex:
    """Decorate one cloneable index with fail-closed bounded reranking."""

    def __init__(
        self,
        *,
        base_index_factory: Callable[[], CloneableChunkIndex],
        reranker: Reranker,
        candidate_depth: int,
        limits: RerankerLimits | None = None,
    ) -> None:
        if not callable(base_index_factory):
            raise TypeError("base_index_factory must be callable")
        if not callable(getattr(reranker, "rerank", None)):
            raise TypeError("reranker must provide a callable rerank method")
        if type(candidate_depth) is not int:
            raise TypeError("candidate_depth must be an integer")
        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be greater than zero")
        if limits is not None and not isinstance(limits, RerankerLimits):
            raise TypeError("limits must be RerankerLimits")
        configured_limits = limits or RerankerLimits()
        if candidate_depth > configured_limits.max_candidates:
            raise ValueError("candidate_depth exceeds max_candidates")
        try:
            base = base_index_factory()
            self._require_base(base)
        except Exception as exc:
            raise RerankerError("base index factory failed") from exc
        self._base_index_factory = base_index_factory
        self._reranker = reranker
        self._candidate_depth = candidate_depth
        self._limits = configured_limits
        self._base = base

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("add", chunks)

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._mutate("replace_document", document_id, chunks)

    def remove_document(self, document_id: str) -> None:
        self._mutate("remove_document", document_id)

    def clone(self) -> "RerankedChunkIndex":
        base = self._clone_base()
        clone = object.__new__(RerankedChunkIndex)
        clone._base_index_factory = self._base_index_factory
        clone._reranker = self._reranker
        clone._candidate_depth = self._candidate_depth
        clone._limits = self._limits
        clone._base = base
        return clone

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        self._validate_request(query, limit)
        try:
            raw_candidates = self._base.search(query, limit=self._candidate_depth)
        except Exception as exc:
            raise RerankerError("base index search failed") from exc
        candidates = self._validated_hits(
            raw_candidates,
            bound=self._candidate_depth,
            source="base index",
        )
        return self._rerank_validated_candidates(query, candidates, limit=limit)

    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics:
        """Rerank one bounded base diagnostic trace without an extra search."""

        self._validate_request(query, limit)
        try:
            diagnose = getattr(self._base, "diagnose", None)
        except Exception as exc:
            raise RerankerError("base index does not support diagnostics") from exc
        if not callable(diagnose):
            raise RerankerError("base index does not support diagnostics")
        try:
            raw_trace = diagnose(query, limit=self._candidate_depth)
        except Exception as exc:
            raise RerankerError("base index diagnose failed") from exc
        candidates, base_rows = self._validated_diagnostic_trace(raw_trace)
        results = self._rerank_validated_candidates(query, candidates, limit=limit)
        selected_chunk_ids = {hit.chunk.chunk_id for hit in results}
        preserved_rows = tuple(
            RetrievalCandidateDiagnostic(
                stage=row.stage,
                rank=row.rank,
                chunk=row.chunk,
                score=row.score,
                matched_terms=row.matched_terms,
                rrf_contribution=row.rrf_contribution,
                selected=row.chunk.chunk_id in selected_chunk_ids,
            )
            for row in base_rows
        )
        reranker_rows = tuple(
            RetrievalCandidateDiagnostic(
                stage=RetrievalStage.RERANKER,
                rank=rank,
                chunk=hit.chunk,
                score=hit.score,
                matched_terms=hit.matched_terms,
                selected=True,
            )
            for rank, hit in enumerate(results, start=1)
        )
        return RetrievalDiagnostics(hits=results, candidates=preserved_rows + reranker_rows)

    def _validate_request(self, query: str, limit: int) -> None:
        if type(query) is not str:
            raise TypeError("query must be a string")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if len(query) > self._limits.max_query_chars:
            raise RerankerLimitError("query exceeds max_query_chars")
        if limit > self._limits.max_results:
            raise RerankerLimitError("limit exceeds max_results")

    def _rerank_validated_candidates(
        self, query: str, candidates: tuple[SearchHit, ...], *, limit: int
    ) -> tuple[SearchHit, ...]:
        total_chars = sum(len(hit.chunk.content) for hit in candidates)
        if total_chars > self._limits.max_total_candidate_chars:
            raise RerankerLimitError("candidates exceed max_total_candidate_chars")
        rerank_limit = min(limit, len(candidates))
        if rerank_limit == 0:
            return ()
        try:
            raw_results = self._reranker.rerank(
                query, candidates, limit=rerank_limit
            )
        except Exception as exc:
            raise RerankerError("reranker failed") from exc
        results = self._validated_hits(
            raw_results,
            bound=rerank_limit,
            source="reranker",
        )
        candidates_by_id = {hit.chunk.chunk_id: hit for hit in candidates}
        first_stage_rank = {
            hit.chunk.chunk_id: rank for rank, hit in enumerate(candidates)
        }
        canonical: list[SearchHit] = []
        for result in results:
            candidate = candidates_by_id.get(result.chunk.chunk_id)
            if candidate is None:
                raise RerankerCoherenceError("reranker returned a ghost chunk_id")
            if result.chunk != candidate.chunk:
                raise RerankerCoherenceError("reranker changed canonical chunk data")
            if result.matched_terms != candidate.matched_terms:
                raise RerankerCoherenceError("reranker changed matched_terms")
            canonical.append(
                SearchHit(candidate.chunk, result.score, candidate.matched_terms)
            )
        canonical.sort(
            key=lambda hit: (
                -hit.score,
                first_stage_rank[hit.chunk.chunk_id],
                hit.chunk.chunk_id,
            )
        )
        return tuple(canonical)

    def _validated_diagnostic_trace(
        self, trace: object
    ) -> tuple[tuple[SearchHit, ...], tuple[RetrievalCandidateDiagnostic, ...]]:
        if type(trace) is not RetrievalDiagnostics:
            raise RerankerCoherenceError("base diagnostic trace is invalid")
        if type(trace.hits) is not tuple or type(trace.candidates) is not tuple:
            raise RerankerCoherenceError("base diagnostic trace is invalid")
        if any(type(row) is not RetrievalCandidateDiagnostic for row in trace.candidates):
            raise RerankerCoherenceError("base diagnostic trace is invalid")
        hits = self._validated_hits(
            trace.hits,
            bound=self._candidate_depth,
            source="base diagnostic trace",
        )
        rows = trace.candidates
        declared: set[SearchHit] = set()
        selected_chunk_ids: set[str] = set()
        blocks: list[tuple[RetrievalStage, list[RetrievalCandidateDiagnostic]]] = []
        stage_positions = {
            stage: position for position, stage in enumerate(RetrievalStage)
        }
        previous_stage_position = -1
        for row in rows:
            try:
                RetrievalCandidateDiagnostic(
                    stage=row.stage,
                    rank=row.rank,
                    chunk=row.chunk,
                    score=row.score,
                    matched_terms=row.matched_terms,
                    rrf_contribution=row.rrf_contribution,
                    selected=row.selected,
                )
                self._validated_hits(
                    (SearchHit(row.chunk, row.score, row.matched_terms),),
                    bound=1,
                    source="base diagnostic trace",
                )
            except (TypeError, ValueError, RerankerCoherenceError) as exc:
                raise RerankerCoherenceError("base diagnostic trace is invalid") from exc
            stage_position = stage_positions[row.stage]
            if not blocks or row.stage is not blocks[-1][0]:
                if stage_position <= previous_stage_position:
                    raise RerankerCoherenceError("base diagnostic trace is incoherent")
                blocks.append((row.stage, []))
                previous_stage_position = stage_position
            elif row.stage is RetrievalStage.RERANKER and row.rank == 1:
                blocks.append((row.stage, []))
            block_rows = blocks[-1][1]
            if row.rank != len(block_rows) + 1:
                raise RerankerCoherenceError("base diagnostic trace is incoherent")
            if any(existing.chunk.chunk_id == row.chunk.chunk_id for existing in block_rows):
                raise RerankerCoherenceError("base diagnostic trace contains duplicates")
            block_rows.append(row)
            if row.selected:
                selected_chunk_ids.add(row.chunk.chunk_id)
                declared.add(SearchHit(row.chunk, row.score, row.matched_terms))
        hit_chunk_ids = {hit.chunk.chunk_id for hit in hits}
        if selected_chunk_ids != hit_chunk_ids or any(hit not in declared for hit in hits):
            raise RerankerCoherenceError("base diagnostic trace is incoherent")
        if rows:
            final_rows = blocks[-1][1]
            if hits and not all(row.selected for row in final_rows):
                raise RerankerCoherenceError("base diagnostic trace is incoherent")
            if tuple(
                SearchHit(row.chunk, row.score, row.matched_terms)
                for row in final_rows
                if row.selected
            ) != hits:
                raise RerankerCoherenceError("base diagnostic trace is incoherent")
        elif hits:
            raise RerankerCoherenceError("base diagnostic trace is incoherent")
        return hits, rows

    def _clone_base(self) -> CloneableChunkIndex:
        try:
            base = self._base.clone()
            self._require_base(base)
            if base is self._base:
                raise TypeError
        except Exception as exc:
            raise RerankerError("base index clone failed") from exc
        return base

    def _mutate(self, method: str, *args: object) -> None:
        try:
            candidate = self._clone_base()
            getattr(candidate, method)(*args)
        except Exception as exc:
            raise RerankerError("base index mutation failed") from exc
        self._base = candidate

    @staticmethod
    def _validated_hits(
        hits: object, *, bound: int, source: str
    ) -> tuple[SearchHit, ...]:
        if type(hits) is not tuple:
            raise RerankerCoherenceError(f"{source} must return an exact tuple")
        if len(hits) > bound:
            raise RerankerCoherenceError(f"{source} exceeded its result limit")
        seen: set[str] = set()
        for hit in hits:
            if type(hit) is not SearchHit or not isinstance(hit.chunk, Chunk):
                raise RerankerCoherenceError(f"{source} returned an invalid SearchHit")
            chunk = hit.chunk
            if (
                type(chunk.document_id) is not str
                or not chunk.document_id
                or type(chunk.chunk_id) is not str
                or not chunk.chunk_id
                or type(chunk.content) is not str
                or type(chunk.start_offset) is not int
                or type(chunk.end_offset) is not int
                or chunk.start_offset < 0
                or chunk.end_offset < chunk.start_offset
                or chunk.end_offset - chunk.start_offset != len(chunk.content)
            ):
                raise RerankerCoherenceError(
                    f"{source} returned invalid canonical chunk data"
                )
            if type(hit.score) is not float or not isfinite(hit.score):
                raise RerankerCoherenceError(f"{source} returned an invalid score")
            if type(hit.matched_terms) is not tuple or any(
                type(term) is not str for term in hit.matched_terms
            ):
                raise RerankerCoherenceError(f"{source} returned invalid matched_terms")
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                raise RerankerCoherenceError(f"{source} returned a duplicate chunk_id")
            seen.add(chunk_id)
        return cast(tuple[SearchHit, ...], hits)

    @staticmethod
    def _require_base(base: object) -> None:
        for method in ("add", "replace_document", "remove_document", "search", "clone"):
            if not callable(getattr(base, method, None)):
                raise TypeError("base index does not implement cloneable lifecycle")


__all__ = [
    "RerankedChunkIndex",
    "Reranker",
    "RerankerCoherenceError",
    "RerankerError",
    "RerankerLimitError",
    "RerankerLimits",
]
