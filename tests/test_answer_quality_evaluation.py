from __future__ import annotations

import json
from pathlib import Path

import pytest
import nexusmind

from nexusmind.answer_quality_evaluation import (
    AnswerQualityCase,
    AnswerQualityEvaluationDatasetError,
    AnswerQualityEvidenceTarget,
    AnswerQualityStatus,
    RequiredAnswerFact,
    evaluate_answer_quality_case,
    load_answer_quality_cases,
    main,
    render_answer_quality_report,
    run_answer_quality_benchmark,
    run_answer_quality_queries,
)
from nexusmind import (
    AnswerGenerationLimits,
    AnswerGenerator,
    ContextPassage,
    GeneratedAnswer,
    KnowledgeAnswer,
    KnowledgeCitation,
    KnowledgeQueryResult,
    KnowledgeQueryTrace,
    ModelContextPassage,
    ModelContextRecord,
    SearchHit,
    assemble_context,
    generate_knowledge_answer,
)
from nexusmind.knowledge import Document, KnowledgeSource
from nexusmind.knowledge_chunking import Chunk
from nexusmind.knowledge_collection import KnowledgeSearchResult


def _case_payload(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "case_id": "binder",
        "question": "Why can Binder oneway calling PID be zero?",
        "required_facts": [
            {
                "fact_id": "pid-zero",
                "answer": "A zero PID can represent an oneway Binder call.",
                "match_phrases": ["zero PID", "oneway Binder call"],
                "evidence_match_phrases": ["zero PID", "oneway Binder call"],
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
                "evidence_match_phrases": ["first"],
                "required_evidence": [{"source_id": "docs", "logical_path": "binder.md"}],
            },
            {
                "fact_id": "same",
                "answer": "second",
                "match_phrases": ["second"],
                "evidence_match_phrases": ["second"],
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


def test_fixture_runner_executes_both_context_configurations(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "binder.md").write_text(
        "# Binder\n\n"
        "A zero PID can represent an oneway Binder call.\n\n"
        + ("Padding keeps this section large enough for structural chunking. " * 80)
        + "\n\nThe nearby caveat says the call still carries caller credentials.",
        encoding="utf-8",
    )
    case_path = _write_dataset(
        tmp_path,
        _case_payload(
            question="zero PID",
            required_facts=[
                {
                    "fact_id": "pid-zero",
                    "answer": "A zero PID can represent an oneway Binder call.",
                    "match_phrases": ["zero PID", "oneway Binder call"],
                    "evidence_match_phrases": ["zero PID", "oneway Binder call"],
                    "required_evidence": [
                        {"source_id": "binder", "logical_path": "binder.md"}
                    ],
                }
            ],
            required_evidence=[
                {"source_id": "binder", "logical_path": "binder.md"}
            ],
        ),
    )

    runs = run_answer_quality_queries(case_path, corpus_dir=corpus_dir)

    assert {(item.case_id, item.configuration) for item in runs} == {
        ("binder", "expand_context=false"),
        ("binder", "expand_context=true"),
    }
    assert all(item.query_result is not None for item in runs)
    assert all(item.query_result.trace.retrieval_queries == ("zero PID",) for item in runs)
    assert {
        item.query_result.trace.context_expansion_enabled
        for item in runs
    } == {False, True}


def test_fixture_runner_rejects_evidence_targets_missing_from_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "binder.md").write_text("# Binder\n\nEvidence.", encoding="utf-8")
    cases_path = _write_dataset(
        tmp_path,
        _case_payload(
            required_evidence=[{"source_id": "missing", "logical_path": "missing.md"}]
        ),
    )

    with pytest.raises(AnswerQualityEvaluationDatasetError, match="not present"):
        run_answer_quality_queries(cases_path, corpus_dir=corpus_dir)


def _two_fact_case() -> AnswerQualityCase:
    evidence = (
        AnswerQualityEvidenceTarget("docs", "doc.md", "one"),
        AnswerQualityEvidenceTarget("docs", "doc.md", "two"),
    )
    return AnswerQualityCase(
        case_id="facts",
        question="facts",
        required_facts=(
            RequiredAnswerFact("one", "fact one", ("fact one",), ("fact one",), (evidence[0],)),
            RequiredAnswerFact("two", "fact two", ("fact two",), ("fact two",), (evidence[1],)),
        ),
        forbidden_claims=("forbidden claim",),
        required_evidence=evidence,
        allow_insufficient_evidence=False,
    )


class _ResultGenerator(AnswerGenerator):
    def __init__(self, generated: GeneratedAnswer) -> None:
        self.generated = generated

    @property
    def config_identity(self) -> str:
        return "test/result-generator"

    def generate(self, question, context, *, model_context, limits):
        return self.generated


def _query_result(text: str, citation_ids: tuple[str, ...]) -> KnowledgeQueryResult:
    document = Document("docs", "doc.md", "fact one\nfact two")
    source = KnowledgeSource(source_id="docs", source_type="test", display_name="Docs")
    chunks = (
        Chunk(document.document_id, "one", "fact one", 0, 8, (), "", ""),
        Chunk(document.document_id, "two", "fact two", 9, 17, (), "", ""),
    )
    results = tuple(
        KnowledgeSearchResult(source, document, SearchHit(chunk, 1.0, ("facts",)))
        for chunk in chunks
    )
    context = assemble_context("facts", results, max_passages=2, max_candidates=2)
    answer = generate_knowledge_answer(
        "facts",
        context,
        _ResultGenerator(GeneratedAnswer(text, citation_ids)),
    )
    config = answer.model_context.context_config
    trace = KnowledgeQueryTrace(
        retrieval_backend="test",
        passages=answer.model_context.passages,
        candidate_count=config.candidate_count,
        context_character_count=config.character_count,
        context_estimated_token_count=config.estimated_token_count,
        retrieval_queries=("facts",),
        fused_result_provenance=(("one", ((0, 1),)), ("two", ((0, 2),))),
    )
    return KnowledgeQueryResult(answer, answer.citations, "trace", trace)


def test_evaluator_distinguishes_complete_partial_incorrect_and_unsupported() -> None:
    case = _two_fact_case()
    complete = evaluate_answer_quality_case(
        case, _query_result("fact one [K1] fact two [K2]", ("K1", "K2"))
    )
    partial = evaluate_answer_quality_case(
        case, _query_result("fact one [K1]", ("K1",))
    )
    incorrect = evaluate_answer_quality_case(
        case, _query_result("unrelated [K1]", ("K1",))
    )
    unsupported = evaluate_answer_quality_case(
        case,
        _query_result("fact one [K1] fact two [K2] forbidden claim [K1]", ("K1", "K2")),
    )

    assert complete.status is AnswerQualityStatus.FULLY_CORRECT
    assert complete.required_fact_coverage == 1.0
    assert partial.status is AnswerQualityStatus.PARTIALLY_CORRECT
    assert partial.required_fact_coverage == 0.5
    assert incorrect.status is AnswerQualityStatus.INCORRECT
    assert unsupported.status is AnswerQualityStatus.UNSUPPORTED


def test_evaluator_reports_citation_validity_coverage_and_support_precision() -> None:
    case = _two_fact_case()
    result = evaluate_answer_quality_case(
        case, _query_result("fact one [K1] fact two [K2]", ("K1", "K2"))
    )

    assert result.citation_validity is True
    assert result.citation_coverage == 1.0
    assert result.citation_support_precision == 1.0
    assert result.satisfied_fact_ids == ("one", "two")
    assert result.missed_fact_ids == ()


def test_fact_support_requires_the_cited_passage_content() -> None:
    evidence = AnswerQualityEvidenceTarget("docs", "doc.md")
    case = AnswerQualityCase(
        case_id="same-document-wrong-section",
        question="facts",
        required_facts=(
            RequiredAnswerFact(
                "one",
                "fact one",
                ("fact one",),
                ("fact one",),
                (evidence,),
            ),
        ),
        forbidden_claims=(),
        required_evidence=(evidence,),
        allow_insufficient_evidence=False,
    )

    result = evaluate_answer_quality_case(
        case, _query_result("fact one [K2]", ("K2",))
    )

    assert result.citation_validity is True
    assert result.citation_coverage == 0.0
    assert result.citation_support_precision == 0.0
    assert result.status is AnswerQualityStatus.UNSUPPORTED


def test_citation_support_precision_is_bounded_for_shared_evidence() -> None:
    evidence = AnswerQualityEvidenceTarget("docs", "doc.md", "one")
    case = AnswerQualityCase(
        case_id="shared",
        question="facts",
        required_facts=(
            RequiredAnswerFact("one", "fact one", ("fact one",), ("fact one",), (evidence,)),
            RequiredAnswerFact("two", "fact two", ("fact two",), ("fact two",), (evidence,)),
        ),
        forbidden_claims=(),
        required_evidence=(evidence,),
        allow_insufficient_evidence=False,
    )

    result = evaluate_answer_quality_case(
        case, _query_result("fact one fact two [K1]", ("K1",))
    )

    assert result.citation_support_precision == 1.0


def test_evaluator_scores_expected_insufficient_evidence_behavior() -> None:
    case = AnswerQualityCase(
        case_id="missing",
        question="missing",
        required_facts=(
            RequiredAnswerFact(
                "missing-fact",
                "missing fact",
                ("missing fact",),
                ("missing fact",),
                (AnswerQualityEvidenceTarget("docs", "doc.md"),),
            ),
        ),
        forbidden_claims=(),
        required_evidence=(AnswerQualityEvidenceTarget("docs", "doc.md"),),
        allow_insufficient_evidence=True,
        answer_profile="insufficient",
    )
    result = evaluate_answer_quality_case(
        case,
        _query_result("Insufficient evidence to answer confidently. [K1]", ("K1",)),
    )

    assert result.insufficient_evidence_success is True
    assert result.status is AnswerQualityStatus.FULLY_CORRECT


def test_report_aggregates_metrics_and_renders_byte_stably(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "binder.md").write_text(
        "# Binder\n\nA zero PID can represent an oneway Binder call.",
        encoding="utf-8",
    )
    cases_path = _write_dataset(
        tmp_path,
        _case_payload(
            question="zero PID",
            required_facts=[
                {
                    "fact_id": "pid-zero",
                    "answer": "A zero PID can represent an oneway Binder call.",
                    "match_phrases": ["zero PID", "oneway Binder call"],
                    "evidence_match_phrases": ["zero PID", "oneway Binder call"],
                    "required_evidence": [
                        {"source_id": "binder", "logical_path": "binder.md"}
                    ],
                }
            ],
            required_evidence=[
                {"source_id": "binder", "logical_path": "binder.md"}
            ],
        ),
    )

    report = run_answer_quality_benchmark(cases_path, corpus_dir=corpus_dir)
    rendered_once = render_answer_quality_report(report)
    rendered_twice = render_answer_quality_report(report)

    assert rendered_once == rendered_twice
    assert 0.0 <= report.aggregate.answer_pass_rate <= 1.0
    assert "citation support precision" in rendered_once
    assert "Answer and evidence details" in rendered_once
    assert "expand_context=true" in rendered_once

    output_path = tmp_path / "report.md"
    assert (
        main(
            (
                "--cases",
                str(cases_path),
                "--corpus",
                str(corpus_dir),
                "--write",
                str(output_path),
            )
        )
        == 0
    )
    assert output_path.read_text(encoding="utf-8") == rendered_once
    assert nexusmind.AnswerQualityCase is AnswerQualityCase
    assert nexusmind.run_answer_quality_benchmark is run_answer_quality_benchmark
