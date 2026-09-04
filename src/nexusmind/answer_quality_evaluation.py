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
from .knowledge_base_manifest import KnowledgeBaseError
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
class AnswerQualityCaseResult:
    """Inspectable quality metrics for one case/configuration run."""

    case_id: str
    configuration: str
    status: AnswerQualityStatus
    answer: str
    error: str | None
    citations: tuple[Any, ...]
    citation_validity: bool
    required_fact_coverage: float
    citation_coverage: float
    citation_support_precision: float
    groundedness: float
    unsupported_claim_rate: float
    insufficient_evidence_expected: bool
    insufficient_evidence_success: bool
    satisfied_fact_ids: tuple[str, ...]
    missed_fact_ids: tuple[str, ...]
    unsupported_fact_ids: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    retrieval_queries: tuple[str, ...]
    fused_chunk_ids: tuple[str, ...]
    passage_chunk_ids: tuple[str, ...]
    context_expansion_enabled: bool | None
    expanded_passage_count: int
    expanded_document_count: int
    retrieval_backend: str | None
    generator_identity: str | None


def evaluate_answer_quality_case(
    case: AnswerQualityCase,
    result: KnowledgeQueryResult | None,
    *,
    configuration: str = "direct",
    error: str | None = None,
) -> AnswerQualityCaseResult:
    """Score one validated query result against one authored case."""

    if type(case) is not AnswerQualityCase:
        raise TypeError("case must be an AnswerQualityCase")
    _require_text(configuration, "configuration")
    if error is not None:
        _require_text(error, "error")
    if result is not None and type(result) is not KnowledgeQueryResult:
        raise TypeError("result must be a KnowledgeQueryResult or None")
    if result is None:
        return AnswerQualityCaseResult(
            case_id=case.case_id,
            configuration=configuration,
            status=AnswerQualityStatus.INCORRECT,
            answer="",
            error=error or "query_failed",
            citations=(),
            citation_validity=False,
            required_fact_coverage=0.0,
            citation_coverage=0.0,
            citation_support_precision=0.0,
            groundedness=0.0,
            unsupported_claim_rate=0.0,
            insufficient_evidence_expected=case.allow_insufficient_evidence,
            insufficient_evidence_success=False,
            satisfied_fact_ids=(),
            missed_fact_ids=tuple(item.fact_id for item in case.required_facts),
            unsupported_fact_ids=(),
            forbidden_claims=(),
            retrieval_queries=(),
            fused_chunk_ids=(),
            passage_chunk_ids=(),
            context_expansion_enabled=None,
            expanded_passage_count=0,
            expanded_document_count=0,
            retrieval_backend=None,
            generator_identity=None,
        )

    answer = result.answer
    text = answer.text
    satisfied: list[str] = []
    missed: list[str] = []
    unsupported_facts: list[str] = []
    citation_by_id = {item.citation_id: item for item in answer.citations}
    context_by_id = {item.citation_id: item for item in answer.model_context.passages}
    allowed_targets = set(case.required_evidence)
    citation_validity = True
    for citation in answer.citations:
        passage = context_by_id.get(citation.citation_id)
        if passage is None or not any(
            _target_matches(
                target,
                source_id=citation.source_id,
                logical_path=citation.logical_path,
                chunk_id=citation.chunk_id,
            )
            for target in allowed_targets
        ):
            citation_validity = False

    fact_support: dict[str, set[str]] = {}
    for fact in case.required_facts:
        if any(_contains(text, phrase) for phrase in fact.match_phrases):
            satisfied.append(fact.fact_id)
            supported_ids = {
                citation_id
                for citation_id, citation in citation_by_id.items()
                if any(
                    _target_matches(
                        target,
                        source_id=citation.source_id,
                        logical_path=citation.logical_path,
                        chunk_id=citation.chunk_id,
                    )
                    for target in fact.required_evidence
                )
            }
            fact_support[fact.fact_id] = supported_ids
            if not supported_ids:
                unsupported_facts.append(fact.fact_id)
        else:
            missed.append(fact.fact_id)

    forbidden = tuple(
        claim for claim in case.forbidden_claims if _contains(text, claim)
    )
    satisfied_set = set(satisfied)
    supported_fact_ids = {
        fact_id for fact_id, citation_ids in fact_support.items() if citation_ids
    }
    required_count = len(case.required_facts)
    fact_coverage = len(satisfied) / required_count
    citation_coverage = len(supported_fact_ids) / required_count
    supported_citation_ids = set().union(*fact_support.values()) if fact_support else set()
    citation_support_precision = len(
        supported_citation_ids & set(citation_by_id)
    ) / max(1, len(citation_by_id))
    unsupported_claim_rate = (
        len(forbidden) + len(unsupported_facts)
    ) / max(1, len(satisfied) + len(forbidden))
    insufficient_success = _is_insufficient_answer(text) and case.allow_insufficient_evidence
    if case.allow_insufficient_evidence:
        insufficient_success = insufficient_success and not forbidden
    else:
        insufficient_success = False

    if forbidden or unsupported_facts or not citation_validity:
        status = AnswerQualityStatus.UNSUPPORTED
    elif case.allow_insufficient_evidence and insufficient_success:
        status = AnswerQualityStatus.FULLY_CORRECT
    elif (
        not case.allow_insufficient_evidence
        and len(satisfied) == required_count
        and citation_coverage == 1.0
        and citation_validity
    ):
        status = AnswerQualityStatus.FULLY_CORRECT
    elif satisfied:
        status = AnswerQualityStatus.PARTIALLY_CORRECT
    else:
        status = AnswerQualityStatus.INCORRECT

    trace = result.trace
    return AnswerQualityCaseResult(
        case_id=case.case_id,
        configuration=configuration,
        status=status,
        answer=text,
        error=error,
        citations=answer.citations,
        citation_validity=citation_validity,
        required_fact_coverage=fact_coverage,
        citation_coverage=citation_coverage,
        citation_support_precision=citation_support_precision,
        groundedness=1.0 - unsupported_claim_rate,
        unsupported_claim_rate=unsupported_claim_rate,
        insufficient_evidence_expected=case.allow_insufficient_evidence,
        insufficient_evidence_success=insufficient_success,
        satisfied_fact_ids=tuple(satisfied),
        missed_fact_ids=tuple(missed),
        unsupported_fact_ids=tuple(unsupported_facts),
        forbidden_claims=forbidden,
        retrieval_queries=trace.retrieval_queries,
        fused_chunk_ids=tuple(item[0] for item in trace.fused_result_provenance),
        passage_chunk_ids=tuple(item.chunk_id for item in trace.passages),
        context_expansion_enabled=trace.context_expansion_enabled,
        expanded_passage_count=trace.expanded_passage_count,
        expanded_document_count=trace.expanded_document_count,
        retrieval_backend=trace.retrieval_backend,
        generator_identity=answer.model_context.generator_config_identity,
    )


def _is_insufficient_answer(text: str) -> bool:
    return any(
        _contains(text, phrase)
        for phrase in (
            "insufficient evidence",
            "not enough evidence",
            "cannot answer",
            "unable to determine",
            "does not establish an answer",
        )
    )


@dataclass(frozen=True, slots=True)
class AnswerQualityAggregate:
    """Aggregate quality metrics for one configuration or all runs."""

    configuration: str
    case_count: int
    answer_pass_rate: float
    required_fact_coverage: float
    citation_coverage: float
    citation_support_precision: float
    groundedness: float
    unsupported_claim_rate: float
    insufficient_evidence_success_rate: float


@dataclass(frozen=True, slots=True)
class AnswerQualityBenchmarkReport:
    """Stable per-case and aggregate answer-quality benchmark output."""

    case_results: tuple[AnswerQualityCaseResult, ...]
    aggregates: tuple[AnswerQualityAggregate, ...]

    @property
    def aggregate(self) -> AnswerQualityAggregate:
        """Return the aggregate over every case/configuration result."""

        return _aggregate_results(self.case_results, configuration="all", cases_by_id={})


def run_answer_quality_benchmark(
    cases_path: str | Path,
    *,
    corpus_dir: str | Path,
    configurations: tuple[AnswerQualityRunConfiguration, ...] = DEFAULT_ANSWER_QUALITY_CONFIGURATIONS,
) -> AnswerQualityBenchmarkReport:
    """Run and score the deterministic end-to-end answer benchmark."""

    cases = load_answer_quality_cases(cases_path)
    cases_by_id = {case.case_id: case for case in cases}
    query_runs = run_answer_quality_queries(
        cases_path,
        corpus_dir=corpus_dir,
        configurations=configurations,
    )
    results = tuple(
        evaluate_answer_quality_case(
            cases_by_id[run.case_id],
            run.query_result,
            configuration=run.configuration,
            error=run.error,
        )
        for run in query_runs
    )
    aggregates = tuple(
        _aggregate_results(
            tuple(item for item in results if item.configuration == configuration.name),
            configuration=configuration.name,
            cases_by_id=cases_by_id,
        )
        for configuration in sorted(configurations, key=lambda item: item.name)
    )
    return AnswerQualityBenchmarkReport(results, aggregates)


def render_answer_quality_report(report: AnswerQualityBenchmarkReport) -> str:
    """Render a deterministic Markdown report with aggregate diagnostics."""

    if type(report) is not AnswerQualityBenchmarkReport:
        raise TypeError("report must be an AnswerQualityBenchmarkReport")
    lines = [
        "# End-to-End RAG Answer Quality Benchmark",
        "",
        "All cases use the same authored corpus and are evaluated with deterministic offline generators.",
        "",
        "## Aggregate metrics",
        "",
        "| Configuration | Answer pass rate | Required fact coverage | Citation coverage | citation support precision | Groundedness | Unsupported claim rate | Insufficient-evidence success rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for aggregate in report.aggregates:
        lines.append(
            "| "
            + " | ".join(
                (
                    aggregate.configuration,
                    f"{aggregate.answer_pass_rate:.6f}",
                    f"{aggregate.required_fact_coverage:.6f}",
                    f"{aggregate.citation_coverage:.6f}",
                    f"{aggregate.citation_support_precision:.6f}",
                    f"{aggregate.groundedness:.6f}",
                    f"{aggregate.unsupported_claim_rate:.6f}",
                    f"{aggregate.insufficient_evidence_success_rate:.6f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-case diagnostics",
            "",
            "| Case | Configuration | Expansion | Status | Fact coverage | Citation coverage | Support precision | Final passages | Satisfied facts | Missed facts |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for item in report.case_results:
        lines.append(
            f"| {item.case_id} | {item.configuration} | {item.context_expansion_enabled} | {item.status.value} | "
            f"{item.required_fact_coverage:.6f} | {item.citation_coverage:.6f} | "
            f"{item.citation_support_precision:.6f} | "
            f"{','.join(item.passage_chunk_ids)} | {','.join(item.satisfied_fact_ids)} | "
            f"{','.join(item.missed_fact_ids)} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "    PYTHONPATH=src python -m nexusmind.answer_quality_evaluation --cases evals/knowledge/answer_quality/cases.json --corpus evals/knowledge/answer_quality/corpus --write evals/knowledge/answer_quality/baseline.md",
            "",
            "Case order, configuration order, metrics, and float formatting are deterministic.",
        ]
    )
    return "\n".join(lines) + "\n"


def _aggregate_results(
    results: tuple[AnswerQualityCaseResult, ...],
    *,
    configuration: str,
    cases_by_id: dict[str, AnswerQualityCase],
) -> AnswerQualityAggregate:
    count = len(results)
    if count == 0:
        return AnswerQualityAggregate(configuration, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected_insufficient = sum(item.insufficient_evidence_expected for item in results)
    successful_insufficient = sum(
        item.insufficient_evidence_success
        for item in results
        if item.insufficient_evidence_expected
    )
    return AnswerQualityAggregate(
        configuration=configuration,
        case_count=count,
        answer_pass_rate=sum(item.status is AnswerQualityStatus.FULLY_CORRECT for item in results) / count,
        required_fact_coverage=sum(item.required_fact_coverage for item in results) / count,
        citation_coverage=sum(item.citation_coverage for item in results) / count,
        citation_support_precision=sum(item.citation_support_precision for item in results) / count,
        groundedness=sum(item.groundedness for item in results) / count,
        unsupported_claim_rate=sum(item.unsupported_claim_rate for item in results) / count,
        insufficient_evidence_success_rate=successful_insufficient / max(1, expected_insufficient),
    )


def main(argv: tuple[str, ...] | None = None) -> int:
    """Generate the checked-in answer-quality Markdown report."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate the answer quality benchmark")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_answer_quality_benchmark(args.cases, corpus_dir=args.corpus)
    args.write.write_text(render_answer_quality_report(report), encoding="utf-8")
    return 0


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
            citations = tuple(item[1] for item in available)
            if not citations and model_context.passages:
                citations = (model_context.passages[0].citation_id,)
            return GeneratedAnswer(
                "Insufficient evidence to answer this question confidently."
                + _citation_suffix(citations),
                citations,
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
                    except (KnowledgeAnswerError, KnowledgeBaseError) as exc:
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
    "AnswerQualityAggregate",
    "AnswerQualityBenchmarkReport",
    "AnswerQualityCase",
    "AnswerQualityCaseResult",
    "AnswerQualityEvaluationDatasetError",
    "AnswerQualityEvaluationError",
    "AnswerQualityEvidenceTarget",
    "AnswerQualityQueryRun",
    "AnswerQualityRunConfiguration",
    "AnswerQualityStatus",
    "DEFAULT_ANSWER_QUALITY_CONFIGURATIONS",
    "RequiredAnswerFact",
    "load_answer_quality_cases",
    "evaluate_answer_quality_case",
    "main",
    "render_answer_quality_report",
    "run_answer_quality_benchmark",
    "run_answer_quality_queries",
]


if __name__ == "__main__":
    raise SystemExit(main())
