from dataclasses import dataclass
from math import isfinite


SEARCH_CANDIDATE_MULTIPLIER = 4
MAX_SEARCH_CANDIDATES = 100
PREFERRED_RESULTS_PER_DOCUMENT = 2


@dataclass(frozen=True, slots=True)
class RankedDocumentCandidate:
    document_id: str
    score: float

    def __post_init__(self) -> None:
        if type(self.document_id) is not str or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if type(self.score) is not float or not isfinite(self.score):
            raise ValueError("score must be a finite float")


def search_candidate_depth(limit: int, *, backend_capacity: object = None) -> int:
    _validate_limit(limit)
    if (
        type(backend_capacity) is not int
        or backend_capacity <= 0
        or limit > backend_capacity
    ):
        return limit
    return min(
        max(
            limit,
            min(limit * SEARCH_CANDIDATE_MULTIPLIER, MAX_SEARCH_CANDIDATES),
        ),
        backend_capacity,
    )


def select_document_aware_indices(
    candidates: tuple[RankedDocumentCandidate, ...], *, limit: int
) -> tuple[int, ...]:
    _validate_limit(limit)
    if type(candidates) is not tuple:
        raise TypeError("candidates must be a tuple")
    if any(type(item) is not RankedDocumentCandidate for item in candidates):
        raise TypeError("candidates must contain RankedDocumentCandidate values")
    if not candidates:
        return ()

    raw_top_k = candidates[:limit]
    scores = tuple(item.score for item in raw_top_k)
    center = _lower_median(scores)
    robust_span = _lower_median(tuple(abs(score - center) for score in scores))
    worst = min(scores)
    relevance_floor = worst - robust_span

    selected: list[int] = []
    document_counts: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        if len(selected) == limit:
            break
        if (
            document_counts.get(candidate.document_id, 0)
            >= PREFERRED_RESULTS_PER_DOCUMENT
        ):
            continue
        if index >= limit and candidate.score < relevance_floor:
            continue
        selected.append(index)
        document_counts[candidate.document_id] = (
            document_counts.get(candidate.document_id, 0) + 1
        )

    selected_set = set(selected)
    for index in range(len(candidates)):
        if len(selected) == limit:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    return tuple(sorted(selected))


def _lower_median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _validate_limit(limit: int) -> None:
    if type(limit) is not int:
        raise TypeError("limit must be an integer")
    if limit <= 0:
        raise ValueError("limit must be positive")
