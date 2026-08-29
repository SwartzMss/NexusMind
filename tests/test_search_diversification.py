from math import inf, nan

import pytest

from nexusmind.search_diversification import (
    RankedDocumentCandidate,
    search_candidate_depth,
    select_document_aware_indices,
)


def candidates(*values: tuple[str, float]) -> tuple[RankedDocumentCandidate, ...]:
    return tuple(
        RankedDocumentCandidate(document_id, score) for document_id, score in values
    )


def test_diversifies_inside_query_relative_score_window_and_preserves_raw_indices() -> None:
    ranked = candidates(
        ("a", 10.0),
        ("a", 9.0),
        ("a", 8.0),
        ("a", 7.0),
        ("a", 6.0),
        ("b", 5.5),
        ("c", 5.0),
    )

    assert select_document_aware_indices(ranked, limit=5) == (0, 1, 2, 5, 6)


def test_weak_cross_document_candidate_does_not_displace_strong_chunk() -> None:
    ranked = candidates(
        ("a", 10.0),
        ("a", 9.0),
        ("a", 8.0),
        ("a", 7.0),
        ("a", 6.0),
        ("b", 1.0),
    )

    assert select_document_aware_indices(ranked, limit=5) == (0, 1, 2, 3, 4)


def test_high_score_outlier_does_not_admit_near_zero_cross_document_candidates() -> None:
    ranked = candidates(
        ("a", 1000.0),
        ("a", 1.0),
        ("a", 1.0),
        ("a", 1.0),
        ("a", 1.0),
        ("b", 0.001),
        ("c", 0.0005),
        ("d", 0.0001),
    )

    assert select_document_aware_indices(ranked, limit=5) == (0, 1, 2, 3, 4)


def test_same_document_backfills_all_slots() -> None:
    ranked = candidates(("a", 4.0), ("a", 3.0), ("a", 2.0), ("a", 1.0))

    assert select_document_aware_indices(ranked, limit=4) == (0, 1, 2, 3)


def test_fewer_candidates_than_limit_returns_every_raw_index() -> None:
    ranked = candidates(("a", 3.0), ("b", 2.0))

    assert select_document_aware_indices(ranked, limit=5) == (0, 1)


def test_empty_candidates_return_no_indices() -> None:
    assert select_document_aware_indices((), limit=3) == ()


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [
        (candidates(("a", 1.0), ("a", 1.0), ("a", 1.0), ("b", 1.0)), (0, 1, 3)),
        (candidates(("a", -1.0), ("a", -2.0), ("a", -3.0), ("b", -3.5)), (0, 1, 3)),
    ],
)
def test_selection_supports_equal_and_negative_scores(
    ranked: tuple[RankedDocumentCandidate, ...], expected: tuple[int, ...]
) -> None:
    assert select_document_aware_indices(ranked, limit=3) == expected


def test_positive_affine_score_transform_preserves_selection() -> None:
    ranked = candidates(
        ("a", 10.0),
        ("a", 9.0),
        ("a", 8.0),
        ("a", 7.0),
        ("b", 5.5),
        ("c", 5.0),
    )
    transformed = tuple(
        RankedDocumentCandidate(item.document_id, item.score * 3.0 + 17.0)
        for item in ranked
    )

    assert select_document_aware_indices(ranked, limit=4) == (
        select_document_aware_indices(transformed, limit=4)
    )


def test_selection_is_deterministic_and_bounded_by_limit() -> None:
    ranked = candidates(*((str(index % 3), float(20 - index)) for index in range(20)))

    first = select_document_aware_indices(ranked, limit=7)

    assert first == select_document_aware_indices(ranked, limit=7)
    assert len(first) == 7
    assert first == tuple(sorted(first))


@pytest.mark.parametrize(
    ("limit", "capacity", "depth"),
    [
        (1, 100, 4),
        (5, 10, 10),
        (10, 100, 40),
        (25, 100, 100),
        (100, 100, 100),
        (101, None, 101),
        (150, 200, 150),
        (5, None, 5),
        (5, True, 5),
        (5, 5.0, 5),
        (5, 0, 5),
        (5, 4, 5),
    ],
)
def test_candidate_depth_respects_optional_backend_capacity(
    limit: int, capacity: object, depth: int
) -> None:
    assert search_candidate_depth(limit, backend_capacity=capacity) == depth


@pytest.mark.parametrize("document_id", ["", None, 1, True])
def test_candidate_rejects_invalid_document_id(document_id: object) -> None:
    with pytest.raises(ValueError, match="document_id"):
        RankedDocumentCandidate(document_id, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("score", [1, True, nan, inf, -inf, "1.0"])
def test_candidate_rejects_invalid_score(score: object) -> None:
    with pytest.raises(ValueError, match="score"):
        RankedDocumentCandidate("document", score)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, "1", None])
def test_depth_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises((TypeError, ValueError), match="limit"):
        search_candidate_depth(limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [0, -1, True, 1.0, "1", None])
def test_selection_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises((TypeError, ValueError), match="limit"):
        select_document_aware_indices((), limit=limit)  # type: ignore[arg-type]


def test_selection_rejects_non_tuple_candidates() -> None:
    with pytest.raises(TypeError, match="tuple"):
        select_document_aware_indices([], limit=1)  # type: ignore[arg-type]


def test_selection_rejects_malformed_tuple_item() -> None:
    with pytest.raises(TypeError, match="RankedDocumentCandidate"):
        select_document_aware_indices((object(),), limit=1)  # type: ignore[arg-type]
