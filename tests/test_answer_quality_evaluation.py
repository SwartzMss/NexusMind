from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexusmind.answer_quality_evaluation import (
    AnswerQualityEvaluationDatasetError,
    load_answer_quality_cases,
)


def _case_payload(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "case_id": "binder",
        "question": "Why can Binder oneway calling PID be zero?",
        "required_facts": [
            {
                "fact_id": "pid-zero",
                "answer": "A zero PID can represent an oneway Binder call.",
                "match_phrases": ["zero PID", "oneway Binder call"],
                "required_evidence": [
                    {"source_id": "docs", "logical_path": "binder.md"}
                ],
            }
        ],
        "forbidden_claims": ["zero PID always means an anonymous caller"],
        "required_evidence": [
            {"source_id": "docs", "logical_path": "binder.md"}
        ],
        "allow_insufficient_evidence": False,
        "answer_profile": "grounded",
    }
    case.update(overrides)
    return case


def _write_dataset(tmp_path: Path, *cases: dict[str, object]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"version": 1, "cases": list(cases)}),
        encoding="utf-8",
    )
    return path


def test_load_cases_requires_version_and_unique_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")

    with pytest.raises(AnswerQualityEvaluationDatasetError, match="at least one case"):
        load_answer_quality_cases(path)


def test_load_case_preserves_required_fact_and_evidence_contract(tmp_path: Path) -> None:
    cases = load_answer_quality_cases(_write_dataset(tmp_path, _case_payload()))

    assert len(cases) == 1
    assert cases[0].required_facts[0].fact_id == "pid-zero"
    assert cases[0].required_facts[0].required_evidence[0].logical_path == "binder.md"
    assert cases[0].required_evidence[0].source_id == "docs"


@pytest.mark.parametrize(
    ("root", "message"),
    [
        ({"version": 2, "cases": [_case_payload()]}, "version"),
        ({"version": 1, "cases": [_case_payload(), _case_payload()]}, "duplicate"),
        ({"version": 1, "cases": [_case_payload(extra="invalid")]}, "fields"),
    ],
)
def test_load_cases_rejects_invalid_root_or_case_shape(
    tmp_path: Path,
    root: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(root), encoding="utf-8")

    with pytest.raises(AnswerQualityEvaluationDatasetError, match=message):
        load_answer_quality_cases(path)


@pytest.mark.parametrize(
    "change",
    [
        {"case_id": " "},
        {"question": " "},
        {"required_facts": []},
        {"forbidden_claims": [""]},
        {"required_evidence": []},
        {"allow_insufficient_evidence": "false"},
        {"answer_profile": "model"},
    ],
)
def test_load_cases_rejects_invalid_case_values(
    tmp_path: Path, change: dict[str, object]
) -> None:
    case = _case_payload(**change)

    with pytest.raises(AnswerQualityEvaluationDatasetError):
        load_answer_quality_cases(_write_dataset(tmp_path, case))


def test_load_cases_rejects_duplicate_fact_ids_and_invalid_evidence_fields(
    tmp_path: Path,
) -> None:
    duplicate_fact = _case_payload(
        required_facts=[
            {
                "fact_id": "same",
                "answer": "first",
                "match_phrases": ["first"],
                "required_evidence": [{"source_id": "docs", "logical_path": "binder.md"}],
            },
            {
                "fact_id": "same",
                "answer": "second",
                "match_phrases": ["second"],
                "required_evidence": [{"source_id": "docs", "logical_path": "binder.md"}],
            },
        ]
    )
    with pytest.raises(AnswerQualityEvaluationDatasetError, match="fact"):
        load_answer_quality_cases(_write_dataset(tmp_path, duplicate_fact))

    invalid_evidence = _case_payload(
        required_evidence=[{"source_id": "docs", "logical_path": "binder.md", "extra": 1}]
    )
    with pytest.raises(AnswerQualityEvaluationDatasetError, match="evidence"):
        load_answer_quality_cases(_write_dataset(tmp_path, invalid_evidence))
