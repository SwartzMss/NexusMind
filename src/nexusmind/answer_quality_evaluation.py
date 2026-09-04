"""Deterministic end-to-end answer quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any


class AnswerQualityEvaluationError(Exception):
    """Base class for answer-quality evaluation failures."""


class AnswerQualityEvaluationDatasetError(AnswerQualityEvaluationError):
    """Authored answer-quality cases are unreadable or invalid."""


class AnswerQualityStatus(str, Enum):
    """Stable final-answer quality classification."""

    FULLY_CORRECT = "fully_correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNSUPPORTED = "unsupported"


_ANSWER_PROFILES = frozenset({"grounded", "partial", "unsupported", "insufficient"})


@dataclass(frozen=True, slots=True)
class AnswerQualityEvidenceTarget:
    """Canonical evidence location authored for one answer-quality case."""

    source_id: str
    logical_path: str
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.logical_path, "logical_path")
        if self.chunk_id is not None:
            _require_text(self.chunk_id, "chunk_id")


@dataclass(frozen=True, slots=True)
class RequiredAnswerFact:
    """One authored fact and the canonical evidence that should support it."""

    fact_id: str
    answer: str
    match_phrases: tuple[str, ...]
    required_evidence: tuple[AnswerQualityEvidenceTarget, ...]

    def __post_init__(self) -> None:
        _require_text(self.fact_id, "fact_id")
        _require_text(self.answer, "answer")
        if type(self.match_phrases) is not tuple or not self.match_phrases:
            raise ValueError("match_phrases must be a non-empty tuple")
        if any(type(item) is not str or not item.strip() for item in self.match_phrases):
            raise ValueError("match_phrases must contain non-empty strings")
        if len(set(self.match_phrases)) != len(self.match_phrases):
            raise ValueError("match_phrases must not contain duplicates")
        if type(self.required_evidence) is not tuple or not self.required_evidence:
            raise ValueError("required_evidence must be a non-empty tuple")
        if any(
            type(item) is not AnswerQualityEvidenceTarget
            for item in self.required_evidence
        ):
            raise TypeError("required_evidence must contain evidence targets")
        if len(set(self.required_evidence)) != len(self.required_evidence):
            raise ValueError("required_evidence must not contain duplicates")


@dataclass(frozen=True, slots=True)
class AnswerQualityCase:
    """One human-authored end-to-end answer expectation."""

    case_id: str
    question: str
    required_facts: tuple[RequiredAnswerFact, ...]
    forbidden_claims: tuple[str, ...]
    required_evidence: tuple[AnswerQualityEvidenceTarget, ...]
    allow_insufficient_evidence: bool
    answer_profile: str = "grounded"

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case_id")
        _require_text(self.question, "question")
        if type(self.required_facts) is not tuple or not self.required_facts:
            raise ValueError("required_facts must be a non-empty tuple")
        if any(type(item) is not RequiredAnswerFact for item in self.required_facts):
            raise TypeError("required_facts must contain RequiredAnswerFact values")
        fact_ids = [item.fact_id for item in self.required_facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("required_facts must have unique fact IDs")
        if type(self.forbidden_claims) is not tuple:
            raise TypeError("forbidden_claims must be a tuple")
        if any(type(item) is not str or not item.strip() for item in self.forbidden_claims):
            raise ValueError("forbidden_claims must contain non-empty strings")
        if len(set(self.forbidden_claims)) != len(self.forbidden_claims):
            raise ValueError("forbidden_claims must not contain duplicates")
        if type(self.required_evidence) is not tuple or not self.required_evidence:
            raise ValueError("required_evidence must be a non-empty tuple")
        if any(
            type(item) is not AnswerQualityEvidenceTarget
            for item in self.required_evidence
        ):
            raise TypeError("required_evidence must contain evidence targets")
        if len(set(self.required_evidence)) != len(self.required_evidence):
            raise ValueError("required_evidence must not contain duplicates")
        if type(self.allow_insufficient_evidence) is not bool:
            raise TypeError("allow_insufficient_evidence must be a boolean")
        if self.answer_profile not in _ANSWER_PROFILES:
            raise ValueError("answer_profile is invalid")


def load_answer_quality_cases(path: str | Path) -> tuple[AnswerQualityCase, ...]:
    """Load and strictly validate a version-one answer-quality dataset."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise AnswerQualityEvaluationDatasetError("failed to read dataset") from exc
    if type(payload) is not dict or set(payload) != {"version", "cases"}:
        raise AnswerQualityEvaluationDatasetError(
            "dataset root fields must be exactly: version, cases"
        )
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise AnswerQualityEvaluationDatasetError("dataset version must be 1")
    raw_cases = payload["cases"]
    if type(raw_cases) is not list or not raw_cases:
        raise AnswerQualityEvaluationDatasetError("dataset must contain at least one case")

    cases: list[AnswerQualityCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in raw_cases:
        try:
            case = _parse_case(raw_case)
        except (TypeError, ValueError, KeyError) as exc:
            raise AnswerQualityEvaluationDatasetError(str(exc)) from exc
        if case.case_id in seen_case_ids:
            raise AnswerQualityEvaluationDatasetError("dataset contains duplicate case IDs")
        seen_case_ids.add(case.case_id)
        cases.append(case)
    return tuple(sorted(cases, key=lambda item: item.case_id))


def _parse_case(raw_case: Any) -> AnswerQualityCase:
    if type(raw_case) is not dict:
        raise TypeError("each case must be an object")
    expected = {
        "case_id",
        "question",
        "required_facts",
        "forbidden_claims",
        "required_evidence",
        "allow_insufficient_evidence",
        "answer_profile",
    }
    if set(raw_case) != expected:
        raise ValueError("case fields are invalid")
    raw_facts = raw_case["required_facts"]
    if type(raw_facts) is not list or not raw_facts:
        raise ValueError("required_facts must be a non-empty array")
    facts = tuple(_parse_fact(item) for item in raw_facts)
    return AnswerQualityCase(
        case_id=raw_case["case_id"],
        question=raw_case["question"],
        required_facts=facts,
        forbidden_claims=_parse_text_list(raw_case["forbidden_claims"], "forbidden_claims"),
        required_evidence=_parse_evidence_list(raw_case["required_evidence"]),
        allow_insufficient_evidence=raw_case["allow_insufficient_evidence"],
        answer_profile=raw_case["answer_profile"],
    )


def _parse_fact(raw_fact: Any) -> RequiredAnswerFact:
    if type(raw_fact) is not dict:
        raise TypeError("each required fact must be an object")
    if set(raw_fact) != {"fact_id", "answer", "match_phrases", "required_evidence"}:
        raise ValueError("required fact fields are invalid")
    raw_phrases = raw_fact["match_phrases"]
    if type(raw_phrases) is not list:
        raise TypeError("match_phrases must be an array")
    return RequiredAnswerFact(
        fact_id=raw_fact["fact_id"],
        answer=raw_fact["answer"],
        match_phrases=tuple(raw_phrases),
        required_evidence=_parse_evidence_list(raw_fact["required_evidence"]),
    )


def _parse_text_list(value: Any, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an array")
    return tuple(value)


def _parse_evidence_list(value: Any) -> tuple[AnswerQualityEvidenceTarget, ...]:
    if type(value) is not list or not value:
        raise ValueError("required_evidence must be a non-empty array")
    targets: list[AnswerQualityEvidenceTarget] = []
    for raw_target in value:
        if type(raw_target) is not dict:
            raise TypeError("each evidence target must be an object")
        if set(raw_target) not in (
            {"source_id", "logical_path"},
            {"source_id", "logical_path", "chunk_id"},
        ):
            raise ValueError("evidence target fields are invalid")
        targets.append(AnswerQualityEvidenceTarget(**raw_target))
    return tuple(targets)


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


__all__ = [
    "AnswerQualityCase",
    "AnswerQualityEvaluationDatasetError",
    "AnswerQualityEvaluationError",
    "AnswerQualityEvidenceTarget",
    "AnswerQualityStatus",
    "RequiredAnswerFact",
    "load_answer_quality_cases",
]
