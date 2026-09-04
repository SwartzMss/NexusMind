"""Deterministic end-to-end answer quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .knowledge import Document, KnowledgeSource
from .knowledge_answer import AnswerGenerator, GeneratedAnswer, KnowledgeAnswerError
from .knowledge_base import KnowledgeBase
from .knowledge_collection import KnowledgeSearchResult
from .knowledge_query import KnowledgeQueryOptions, KnowledgeQueryResult


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


@dataclass(frozen=True, slots=True)
class AnswerQualityRunConfiguration:
    """One named production-query configuration used by the benchmark."""

    name: str
    expand_context: bool

    def __post_init__(self) -> None:
        _require_text(self.name, "configuration name")
        if type(self.expand_context) is not bool:
            raise TypeError("expand_context must be a boolean")


DEFAULT_ANSWER_QUALITY_CONFIGURATIONS = (
    AnswerQualityRunConfiguration("expand_context=false", False),
    AnswerQualityRunConfiguration("expand_context=true", True),
)


@dataclass(frozen=True, slots=True)
class AnswerQualityQueryRun:
    """One deterministic query execution retained for later scoring."""

    case_id: str
    configuration: str
    query_result: KnowledgeQueryResult | None
    error: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.case_id, "case_id")
        _require_text(self.configuration, "configuration")
        if self.query_result is not None and type(self.query_result) is not KnowledgeQueryResult:
            raise TypeError("query_result must be a KnowledgeQueryResult or None")
        if self.error is not None:
            _require_text(self.error, "error")
        if self.query_result is not None and self.error is not None:
            raise ValueError("query run cannot contain both a result and an error")


class _FixtureSourceAdapter:
    def __init__(self, source: KnowledgeSource, document: Document) -> None:
        self._source = source
        self._document = document

    def source(self) -> KnowledgeSource:
        return self._source

    def load_documents(self) -> tuple[Document, ...]:
        return (self._document,)


class _FixtureAnswerGenerator:
    def __init__(self, case: AnswerQualityCase) -> None:
        self._case = case

    @property
    def config_identity(self) -> str:
        return f"answer-quality-fixture-v1/{self._case.case_id}/{self._case.answer_profile}"

    def generate(self, question, context, *, model_context, limits):
        available: list[tuple[RequiredAnswerFact, str]] = []
        for fact in self._case.required_facts:
            for passage in model_context.passages:
                if not _fact_evidence_matches(fact, passage) or not any(
                    _contains(passage.content, phrase) for phrase in fact.match_phrases
                ):
                    continue
                available.append((fact, passage.citation_id))
                break
        if self._case.answer_profile == "partial" and available:
            available = available[:-1]
        if self._case.answer_profile == "insufficient":
            return GeneratedAnswer(
                "Insufficient evidence to answer this question confidently."
                + _citation_suffix(tuple(item[1] for item in available)),
                tuple(item[1] for item in available),
            )
        parts = [f"{fact.answer} [{citation_id}]" for fact, citation_id in available]
        if self._case.answer_profile == "unsupported":
            claim = self._case.forbidden_claims[0] if self._case.forbidden_claims else "This unsupported claim is true."
            citation_id = available[0][1] if available else "K1"
            parts.append(f"{claim} [{citation_id}]")
        if not parts:
            parts.append("The supplied evidence does not establish an answer. [K1]")
            return GeneratedAnswer(parts[0], ("K1",))
        citations = tuple(dict.fromkeys(item[1] for item in available))
        return GeneratedAnswer(" ".join(parts), citations)


def run_answer_quality_queries(
    cases_path: str | Path,
    *,
    corpus_dir: str | Path,
    configurations: tuple[AnswerQualityRunConfiguration, ...] = DEFAULT_ANSWER_QUALITY_CONFIGURATIONS,
) -> tuple[AnswerQualityQueryRun, ...]:
    """Run authored cases through the real query pipeline offline."""

    cases = load_answer_quality_cases(cases_path)
    if type(configurations) is not tuple or not configurations:
        raise TypeError("configurations must be a non-empty tuple")
    if any(type(item) is not AnswerQualityRunConfiguration for item in configurations):
        raise TypeError("configurations must contain AnswerQualityRunConfiguration values")
    if len({item.name for item in configurations}) != len(configurations):
        raise ValueError("configurations must have unique names")
    root = Path(corpus_dir)
    try:
        paths = tuple(sorted(path for path in root.rglob("*.md") if path.is_file()))
    except OSError as exc:
        raise AnswerQualityEvaluationDatasetError("failed to read corpus") from exc
    if not paths:
        raise AnswerQualityEvaluationDatasetError("corpus must contain Markdown files")
    documents = tuple(_fixture_document(path, root) for path in paths)

    runs: list[AnswerQualityQueryRun] = []
    for case in cases:
        for configuration in sorted(configurations, key=lambda item: item.name):
            generator = _FixtureAnswerGenerator(case)
            with TemporaryDirectory(prefix="nexusmind-answer-quality-") as temp_root:
                knowledge_base = KnowledgeBase.create(
                    temp_root,
                    knowledge_base_id=f"answer-quality-{case.case_id}",
                    answer_generator=generator,
                )
                try:
                    for document in documents:
                        source = KnowledgeSource(
                            source_id=document.source_id,
                            source_type="answer_quality_fixture",
                            display_name=document.logical_path,
                        )
                        knowledge_base._collection.sync(  # noqa: SLF001
                            _FixtureSourceAdapter(source, document)
                        )
                    try:
                        result = knowledge_base.query(
                            case.question,
                            options=KnowledgeQueryOptions(generator=generator),
                            expand_context=configuration.expand_context,
                        )
                    except KnowledgeAnswerError as exc:
                        runs.append(
                            AnswerQualityQueryRun(
                                case.case_id,
                                configuration.name,
                                None,
                                type(exc).__name__,
                            )
                        )
                    else:
                        runs.append(
                            AnswerQualityQueryRun(
                                case.case_id,
                                configuration.name,
                                result,
                            )
                        )
                finally:
                    knowledge_base.close()
    return tuple(runs)


def _fixture_document(path: Path, root: Path) -> Document:
    try:
        logical_path = path.relative_to(root).as_posix()
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        raise AnswerQualityEvaluationDatasetError("failed to read corpus document") from exc
    source_id = Path(logical_path).with_suffix("").as_posix()
    return Document(source_id, logical_path, content)


def _fact_evidence_matches(
    fact: RequiredAnswerFact,
    passage: Any,
) -> bool:
    return any(
        _target_matches(
            target,
            source_id=passage.source_id,
            logical_path=passage.logical_path,
            chunk_id=passage.chunk_id,
        )
        for target in fact.required_evidence
    )


def _target_matches(
    target: AnswerQualityEvidenceTarget,
    *,
    source_id: str,
    logical_path: str,
    chunk_id: str,
) -> bool:
    return (
        target.source_id == source_id
        and target.logical_path == logical_path
        and (target.chunk_id is None or target.chunk_id == chunk_id)
    )


def _contains(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def _citation_suffix(citation_ids: tuple[str, ...]) -> str:
    return "" if not citation_ids else " " + " ".join(f"[{item}]" for item in citation_ids)


__all__ = [
    "AnswerQualityCase",
    "AnswerQualityEvaluationDatasetError",
    "AnswerQualityEvaluationError",
    "AnswerQualityEvidenceTarget",
    "AnswerQualityQueryRun",
    "AnswerQualityRunConfiguration",
    "AnswerQualityStatus",
    "DEFAULT_ANSWER_QUALITY_CONFIGURATIONS",
    "RequiredAnswerFact",
    "load_answer_quality_cases",
    "run_answer_quality_queries",
]
