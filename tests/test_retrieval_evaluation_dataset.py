from __future__ import annotations

import json

import pytest

from nexusmind import (
    RetrievalEvaluationCase,
    RetrievalEvaluationDatasetError,
    RetrievalTarget,
    load_retrieval_evaluation_cases,
)


def _valid_data() -> dict:
    return {
        "cases": [
            {
                "case_id": "case-1",
                "query": "secure world",
                "relevant_documents": [
                    {"source_id": "eval-corpus", "logical_path": "trustzone.md"}
                ],
            }
        ]
    }


def test_loader_reads_valid_utf8_dataset(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(_valid_data(), ensure_ascii=False), encoding="utf-8")

    assert load_retrieval_evaluation_cases(path) == (
        RetrievalEvaluationCase(
            case_id="case-1",
            query="secure world",
            relevant_documents=(RetrievalTarget("eval-corpus", "trustzone.md"),),
        ),
    )


@pytest.mark.parametrize(
    "data, message",
    [
        ([], "root"),
        ({}, "root fields"),
        ({"cases": [], "extra": True}, "root fields"),
        ({"cases": "bad"}, "cases must be an array"),
        ({"cases": []}, "at least one case"),
        ({"cases": ["bad"]}, "case must be an object"),
        (
            {"cases": [{"case_id": "case", "query": "q"}]},
            "case fields",
        ),
        (
            {
                "cases": [
                    {
                        "case_id": "case",
                        "query": "q",
                        "relevant_documents": [],
                        "extra": True,
                    }
                ]
            },
            "case fields",
        ),
        (
            {
                "cases": [
                    {
                        "case_id": "case",
                        "query": "q",
                        "relevant_documents": "bad",
                    }
                ]
            },
            "relevant_documents must be an array",
        ),
        (
            {
                "cases": [
                    {
                        "case_id": "case",
                        "query": "q",
                        "relevant_documents": ["bad"],
                    }
                ]
            },
            "target must be an object",
        ),
        (
            {
                "cases": [
                    {
                        "case_id": "case",
                        "query": "q",
                        "relevant_documents": [
                            {"source_id": "source", "logical_path": "a.md", "extra": 1}
                        ],
                    }
                ]
            },
            "target fields",
        ),
    ],
)
def test_loader_rejects_wrong_shapes_and_extra_fields(
    data: object, message: str, tmp_path
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RetrievalEvaluationDatasetError, match=message):
        load_retrieval_evaluation_cases(path)


def test_loader_rejects_invalid_json_and_missing_file(tmp_path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(RetrievalEvaluationDatasetError, match="valid JSON"):
        load_retrieval_evaluation_cases(invalid)
    with pytest.raises(RetrievalEvaluationDatasetError, match="read dataset"):
        load_retrieval_evaluation_cases(tmp_path / "missing.json")


def test_loader_rejects_duplicate_case_ids_and_targets(tmp_path) -> None:
    duplicate_cases = _valid_data()
    duplicate_cases["cases"] = duplicate_cases["cases"] * 2
    path = tmp_path / "duplicate-cases.json"
    path.write_text(json.dumps(duplicate_cases), encoding="utf-8")
    with pytest.raises(RetrievalEvaluationDatasetError, match="duplicate case_id"):
        load_retrieval_evaluation_cases(path)

    duplicate_targets = _valid_data()
    targets = duplicate_targets["cases"][0]["relevant_documents"]
    duplicate_targets["cases"][0]["relevant_documents"] = targets * 2
    path = tmp_path / "duplicate-targets.json"
    path.write_text(json.dumps(duplicate_targets), encoding="utf-8")
    with pytest.raises(RetrievalEvaluationDatasetError, match="duplicate targets"):
        load_retrieval_evaluation_cases(path)


@pytest.mark.parametrize(
    "field, value",
    [("case_id", ""), ("query", " "), ("case_id", 1), ("query", None)],
)
def test_loader_wraps_invalid_case_values(field: str, value: object, tmp_path) -> None:
    data = _valid_data()
    data["cases"][0][field] = value
    path = tmp_path / "invalid-case.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RetrievalEvaluationDatasetError, match=field):
        load_retrieval_evaluation_cases(path)
