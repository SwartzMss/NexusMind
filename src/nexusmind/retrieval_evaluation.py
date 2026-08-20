"""Deterministic document-relevance evaluation over Knowledge search results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .knowledge_collection import KnowledgeCollection
from .knowledge_retrieval import ChunkIndexLimitError


class RetrievalEvaluationError(Exception):
    """Evaluation input or canonical relevance state is invalid."""


class RetrievalEvaluationDatasetError(RetrievalEvaluationError):
    """A retrieval evaluation dataset is malformed or unreadable."""


class RetrievalCategory(str, Enum):
    """Strict primary failure-mode category for one relevance case."""

    EXACT_TERM = "exact_term"
    IDENTIFIER = "identifier"
    CJK = "cjk"
    PARAPHRASE = "paraphrase"
    CROSS_LANGUAGE = "cross_language"
    MULTI_DOCUMENT = "multi_document"
    DISTRACTOR_HEAVY = "distractor_heavy"
    MIXED_SIGNAL = "mixed_signal"


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
    category: RetrievalCategory
    query: str
    relevant_documents: tuple[RetrievalTarget, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _require_text(self.case_id, "case_id"))
        if not isinstance(self.category, RetrievalCategory):
            raise TypeError("category must be a RetrievalCategory")
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
    category: RetrievalCategory
    query: str
    relevant_targets: tuple[RetrievalTarget, ...]
    returned_targets: tuple[RetrievalTarget, ...]
    returned_chunk_ids: tuple[str, ...]
    relevant_targets_found: tuple[RetrievalTarget, ...]
    relevant_targets_missed: tuple[RetrievalTarget, ...]
    first_relevant_rank: int | None
    hit_at_k: float
    recall_at_k: float
    reciprocal_rank: float


class RetrievalFailureKind(str, Enum):
    MISSED = "missed"
    RANKED_BELOW_CUTOFF = "ranked_below_cutoff"
    PARTIAL_RECALL = "partial_recall"


@dataclass(frozen=True, slots=True)
class RetrievalCategoryReport:
    category: RetrievalCategory
    case_count: int
    hit_at_k: float
    recall_at_k: float
    mrr: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    k: int
    case_results: tuple[RetrievalEvaluationCaseResult, ...]
    category_reports: tuple[RetrievalCategoryReport, ...]
    hit_at_k: float
    recall_at_k: float
    mrr: float


MAX_EVALUATION_K_VALUES = 8
MAX_EVALUATION_K = 100


def load_retrieval_evaluation_cases(
    path: str | Path,
) -> tuple[RetrievalEvaluationCase, ...]:
    """Load and strictly validate a UTF-8 JSON evaluation dataset."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, TypeError) as exc:
        raise RetrievalEvaluationDatasetError("failed to read dataset") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RetrievalEvaluationDatasetError("dataset must contain valid JSON") from exc
    if type(payload) is not dict:
        raise RetrievalEvaluationDatasetError("dataset root must be an object")
    if set(payload) != {"cases"}:
        raise RetrievalEvaluationDatasetError("dataset root fields must be exactly: cases")
    raw_cases = payload["cases"]
    if type(raw_cases) is not list:
        raise RetrievalEvaluationDatasetError("cases must be an array")
    if not raw_cases:
        raise RetrievalEvaluationDatasetError("dataset must contain at least one case")

    cases: list[RetrievalEvaluationCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in raw_cases:
        if type(raw_case) is not dict:
            raise RetrievalEvaluationDatasetError("each case must be an object")
        if set(raw_case) != {"case_id", "category", "query", "relevant_documents"}:
            raise RetrievalEvaluationDatasetError(
                "case fields must be exactly: case_id, category, query, relevant_documents"
            )
        raw_targets = raw_case["relevant_documents"]
        if type(raw_targets) is not list:
            raise RetrievalEvaluationDatasetError(
                "relevant_documents must be an array"
            )
        targets: list[RetrievalTarget] = []
        for raw_target in raw_targets:
            if type(raw_target) is not dict:
                raise RetrievalEvaluationDatasetError("each target must be an object")
            if set(raw_target) != {"source_id", "logical_path"}:
                raise RetrievalEvaluationDatasetError(
                    "target fields must be exactly: source_id, logical_path"
                )
            try:
                targets.append(
                    RetrievalTarget(
                        source_id=raw_target["source_id"],
                        logical_path=raw_target["logical_path"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RetrievalEvaluationDatasetError(str(exc)) from exc
        if len(set(targets)) != len(targets):
            raise RetrievalEvaluationDatasetError(
                "relevant_documents contains duplicate targets"
            )
        try:
            try:
                category = RetrievalCategory(raw_case["category"])
            except (TypeError, ValueError) as exc:
                raise RetrievalEvaluationDatasetError("unknown category") from exc
            case = RetrievalEvaluationCase(
                case_id=raw_case["case_id"],
                category=category,
                query=raw_case["query"],
                relevant_documents=tuple(targets),
            )
        except (TypeError, ValueError) as exc:
            raise RetrievalEvaluationDatasetError(str(exc)) from exc
        if case.case_id in seen_case_ids:
            raise RetrievalEvaluationDatasetError("dataset contains duplicate case_id values")
        seen_case_ids.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def _validate_cases(
    cases: tuple[RetrievalEvaluationCase, ...],
) -> None:
    if type(cases) is not tuple or not cases:
        raise RetrievalEvaluationError("cases must be a non-empty tuple")
    if any(not isinstance(case, RetrievalEvaluationCase) for case in cases):
        raise RetrievalEvaluationError("cases must contain only RetrievalEvaluationCase values")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise RetrievalEvaluationError("cases contain duplicate case_id values")


def _validate_ks(ks: tuple[int, ...]) -> tuple[int, ...]:
    if type(ks) is not tuple or not ks:
        raise RetrievalEvaluationError("ks must be a non-empty tuple")
    if any(type(k) is not int or k <= 0 for k in ks):
        raise RetrievalEvaluationError("ks must contain positive plain integers")
    if len(set(ks)) != len(ks):
        raise RetrievalEvaluationError("ks contains duplicate values")
    if len(ks) > MAX_EVALUATION_K_VALUES:
        raise RetrievalEvaluationError("ks contains too many values")
    if max(ks) > MAX_EVALUATION_K:
        raise RetrievalEvaluationError("k exceeds configured maximum")
    return tuple(sorted(ks))


def _snapshot_and_validate_relevance(
    collection: KnowledgeCollection,
    cases: tuple[RetrievalEvaluationCase, ...],
):
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
    return snapshot


def _report_from_rankings(
    cases: tuple[RetrievalEvaluationCase, ...],
    rankings: tuple[tuple[object, ...], ...],
    *,
    k: int,
) -> RetrievalEvaluationReport:
    case_results: list[RetrievalEvaluationCaseResult] = []
    for case, full_ranking in zip(cases, rankings):
        search_results = full_ranking[:k]
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
        case_results.append(
            RetrievalEvaluationCaseResult(
                case_id=case.case_id,
                category=case.category,
                query=case.query,
                relevant_targets=case.relevant_documents,
                returned_targets=returned_targets,
                returned_chunk_ids=returned_chunk_ids,
                relevant_targets_found=tuple(found),
                relevant_targets_missed=missed,
                first_relevant_rank=first_relevant_rank,
                hit_at_k=1.0 if first_relevant_rank is not None else 0.0,
                recall_at_k=len(found) / len(case.relevant_documents),
                reciprocal_rank=1.0 / first_relevant_rank if first_relevant_rank else 0.0,
            )
        )
    results = tuple(case_results)
    count = len(results)
    category_reports: list[RetrievalCategoryReport] = []
    for category in RetrievalCategory:
        members = tuple(result for result in results if result.category is category)
        if not members:
            continue
        member_count = len(members)
        category_reports.append(
            RetrievalCategoryReport(
                category=category,
                case_count=member_count,
                hit_at_k=sum(result.hit_at_k for result in members) / member_count,
                recall_at_k=sum(result.recall_at_k for result in members) / member_count,
                mrr=sum(result.reciprocal_rank for result in members) / member_count,
            )
        )
    return RetrievalEvaluationReport(
        k=k,
        case_results=results,
        category_reports=tuple(category_reports),
        hit_at_k=sum(result.hit_at_k for result in results) / count,
        recall_at_k=sum(result.recall_at_k for result in results) / count,
        mrr=sum(result.reciprocal_rank for result in results) / count,
    )


def classify_retrieval_failure(
    result: RetrievalEvaluationCaseResult,
    *,
    smaller_k: int,
) -> RetrievalFailureKind | None:
    """Classify deterministic evidence in a case result without generating prose."""

    if not isinstance(result, RetrievalEvaluationCaseResult):
        raise TypeError("result must be a RetrievalEvaluationCaseResult")
    if type(smaller_k) is not int or smaller_k <= 0:
        raise ValueError("smaller_k must be a positive integer")
    if result.first_relevant_rank is None:
        return RetrievalFailureKind.MISSED
    if result.recall_at_k < 1.0:
        return RetrievalFailureKind.PARTIAL_RECALL
    if result.first_relevant_rank > smaller_k:
        return RetrievalFailureKind.RANKED_BELOW_CUTOFF
    return None


def evaluate_retrieval_multi_k(
    collection: KnowledgeCollection,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> tuple[RetrievalEvaluationReport, ...]:
    _validate_cases(cases)
    ordered_ks = _validate_ks(ks)
    _snapshot_and_validate_relevance(collection, cases)
    rankings: list[tuple[object, ...]] = []
    max_k = max(ordered_ks)
    for case in cases:
        try:
            search_results = collection.search(case.query, limit=max_k)
        except ChunkIndexLimitError as exc:
            raise RetrievalEvaluationError(
                "k exceeds retrieval backend result limit"
            ) from exc
        if type(search_results) is not tuple:
            raise RetrievalEvaluationError("retrieval backend must return a tuple")
        rankings.append(search_results)
    frozen_rankings = tuple(rankings)
    return tuple(
        _report_from_rankings(cases, frozen_rankings, k=k) for k in ordered_ks
    )


def evaluate_retrieval(
    collection: KnowledgeCollection,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    k: int = 5,
) -> RetrievalEvaluationReport:
    if type(k) is not int or k <= 0:
        raise RetrievalEvaluationError("k must be a positive integer")
    return evaluate_retrieval_multi_k(collection, cases, ks=(k,))[0]


__all__ = [
    "RetrievalCategory",
    "RetrievalCategoryReport",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationCaseResult",
    "RetrievalEvaluationDatasetError",
    "RetrievalEvaluationError",
    "RetrievalEvaluationReport",
    "RetrievalFailureKind",
    "RetrievalTarget",
    "MAX_EVALUATION_K",
    "MAX_EVALUATION_K_VALUES",
    "evaluate_retrieval",
    "evaluate_retrieval_multi_k",
    "classify_retrieval_failure",
    "load_retrieval_evaluation_cases",
]
