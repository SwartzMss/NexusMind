"""Deterministic document-relevance evaluation over Knowledge search results."""

from __future__ import annotations

from dataclasses import dataclass

from .knowledge_collection import KnowledgeCollection


class RetrievalEvaluationError(Exception):
    """Evaluation input or canonical relevance state is invalid."""


def _require_text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RetrievalTarget:
    source_id: str
    logical_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _require_text(self.source_id, "source_id"))
        object.__setattr__(self, "logical_path", _require_text(self.logical_path, "logical_path"))


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    case_id: str
    query: str
    relevant_documents: tuple[RetrievalTarget, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_text(self.case_id, "case_id"))
        object.__setattr__(self, "query", _require_text(self.query, "query"))
        if type(self.relevant_documents) is not tuple:
            raise TypeError("relevant_documents must be a tuple")
        if not self.relevant_documents:
            raise ValueError("relevant_documents must contain at least one target")
        if any(not isinstance(target, RetrievalTarget) for target in self.relevant_documents):
            raise TypeError("relevant_documents must contain only RetrievalTarget values")
        if len(set(self.relevant_documents)) != len(self.relevant_documents):
            raise ValueError("relevant_documents contains duplicate targets")


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCaseResult:
    case_id: str
    query: str
    returned_targets: tuple[RetrievalTarget, ...]
    returned_chunk_ids: tuple[str, ...]
    relevant_targets_found: tuple[RetrievalTarget, ...]
    relevant_targets_missed: tuple[RetrievalTarget, ...]
    first_relevant_rank: int | None
    hit_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    k: int
    case_results: tuple[RetrievalEvaluationCaseResult, ...]
    hit_at_k: float
    recall_at_k: float
    mrr: float


def evaluate_retrieval(
    collection: KnowledgeCollection,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    k: int = 5,
) -> RetrievalEvaluationReport:
    if type(cases) is not tuple or not cases:
        raise RetrievalEvaluationError("cases must be a non-empty tuple")
    if any(not isinstance(case, RetrievalEvaluationCase) for case in cases):
        raise RetrievalEvaluationError("cases must contain only RetrievalEvaluationCase values")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise RetrievalEvaluationError("cases contain duplicate case_id values")
    if type(k) is not int or k <= 0:
        raise RetrievalEvaluationError("k must be a positive integer")
    if not callable(getattr(collection, "snapshot", None)) or not callable(
        getattr(collection, "search", None)
    ):
        raise RetrievalEvaluationError("collection must implement snapshot() and search()")

    snapshot = collection.snapshot()
    canonical_targets = {
        RetrievalTarget(document.source_id, document.logical_path)
        for document in snapshot.documents
    }
    for case in cases:
        for target in case.relevant_documents:
            if target not in canonical_targets:
                raise RetrievalEvaluationError(
                    f"unknown relevance target in case {case.case_id}: "
                    f"{target.source_id}/{target.logical_path}"
                )

    case_results: list[RetrievalEvaluationCaseResult] = []
    for case in cases:
        search_results = collection.search(case.query, limit=k)
        returned_targets = tuple(
            RetrievalTarget(result.source.source_id, result.document.logical_path)
            for result in search_results
        )
        returned_chunk_ids = tuple(result.hit.chunk.chunk_id for result in search_results)
        relevant = set(case.relevant_documents)
        found: list[RetrievalTarget] = []
        seen: set[RetrievalTarget] = set()
        first_relevant_rank: int | None = None
        for rank, target in enumerate(returned_targets, start=1):
            if target not in relevant:
                continue
            if first_relevant_rank is None:
                first_relevant_rank = rank
            if target not in seen:
                found.append(target)
                seen.add(target)
        missed = tuple(target for target in case.relevant_documents if target not in seen)
        hit_at_k = 1.0 if first_relevant_rank is not None else 0.0
        recall_at_k = len(found) / len(case.relevant_documents)
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        case_results.append(
            RetrievalEvaluationCaseResult(
                case_id=case.case_id,
                query=case.query,
                returned_targets=returned_targets,
                returned_chunk_ids=returned_chunk_ids,
                relevant_targets_found=tuple(found),
                relevant_targets_missed=missed,
                first_relevant_rank=first_relevant_rank,
                hit_at_k=hit_at_k,
                recall_at_k=recall_at_k,
                reciprocal_rank=reciprocal_rank,
            )
        )

    results = tuple(case_results)
    count = len(results)
    return RetrievalEvaluationReport(
        k=k,
        case_results=results,
        hit_at_k=sum(result.hit_at_k for result in results) / count,
        recall_at_k=sum(result.recall_at_k for result in results) / count,
        mrr=sum(result.reciprocal_rank for result in results) / count,
    )


__all__ = [
    "RetrievalEvaluationCase",
    "RetrievalEvaluationCaseResult",
    "RetrievalEvaluationError",
    "RetrievalEvaluationReport",
    "RetrievalTarget",
    "evaluate_retrieval",
]
