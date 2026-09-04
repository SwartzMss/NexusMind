from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import pytest

from nexusmind import (
    AnswerGenerationLimitError,
    AnswerGenerationLimits,
    AnswerGenerator,
    AnswerGeneratorError,
    CitationValidationError,
    ContextPackage,
    GeneratedAnswer,
    KnowledgeAnswer,
    KnowledgeBase,
    KnowledgeBaseSourceError,
    KnowledgeQueryOptions,
    LocalFileSourceConfig,
    generate_knowledge_answer,
    render_model_context,
)


@dataclass
class FakeGenerator:
    result: GeneratedAnswer = GeneratedAnswer("Supported answer [K1]", ("K1",))
    error: Exception | None = None
    identity: str = "fake/offline-v1"

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, ContextPackage, object, AnswerGenerationLimits]] = []

    @property
    def config_identity(self) -> str:
        return self.identity

    def generate(self, question, context, *, model_context, limits):
        self.calls.append((question, context, model_context, limits))
        if self.error is not None:
            raise self.error
        return self.result


def _knowledge_base(tmp_path: Path, generator: AnswerGenerator | None = None) -> KnowledgeBase:
    source_file = tmp_path / "binder.md"
    source_file.write_text(
        "Binder authenticates callers using kernel-provided credentials.",
        encoding="utf-8",
    )
    kb = KnowledgeBase.create(
        str(tmp_path / "kb"),
        knowledge_base_id="test-kb",
        answer_generator=generator,
    )
    registered = kb.add_source(LocalFileSourceConfig(path=str(source_file)))
    kb.sync_source(registered.source_id)
    return kb


def _query_answer(
    kb: KnowledgeBase,
    question: str,
    *,
    generator: AnswerGenerator | None = None,
    retrieval_limit: int = 8,
    limits: AnswerGenerationLimits | None = None,
) -> KnowledgeAnswer:
    return kb.query(
        question,
        options=KnowledgeQueryOptions(
            generator=generator,
            retrieval_limit=retrieval_limit,
            limits=AnswerGenerationLimits() if limits is None else limits,
        ),
    ).answer


def test_answer_generator_is_runtime_checkable_and_injected_into_knowledge_base(
    tmp_path: Path,
) -> None:
    generator = FakeGenerator()
    assert isinstance(generator, AnswerGenerator)
    kb = _knowledge_base(tmp_path, generator)

    answer = _query_answer(kb, "How does Binder authenticate callers?")

    assert isinstance(answer, KnowledgeAnswer)
    assert answer.text == "Supported answer [K1]"
    assert [citation.citation_id for citation in answer.citations] == ["K1"]
    assert len(generator.calls) == 1
    question, context, record, limits = generator.calls[0]
    assert question == "How does Binder authenticate callers?"
    assert context.passages[0].source_id == kb.list_sources()[0].source_id
    assert record == answer.model_context
    assert limits == AnswerGenerationLimits()


def test_model_context_rendering_is_deterministic_and_replayable(tmp_path: Path) -> None:
    kb = _knowledge_base(tmp_path)
    context = kb._collection.build_context(  # noqa: SLF001
        "Binder credentials?", retrieval_limit=3, max_passages=3
    )
    limits = AnswerGenerationLimits()

    first = render_model_context(
        context,
        question="Binder credentials?",
        limits=limits,
        generator_config_identity="fake/v1",
    )
    second = render_model_context(
        context,
        question="Binder credentials?",
        limits=limits,
        generator_config_identity="fake/v1",
    )

    assert first == second
    assert first.rendered_context.startswith(
        f"[K1]\nsource: {kb.list_sources()[0].source_id}"
    )
    assert first.passages[0].content in first.rendered_context
    assert first.passages[0].document_content_hash
    assert first.passages[0].start_offset == 0
    assert first.passages[0].end_offset > first.passages[0].start_offset
    assert first.context_config.query == first.question
    assert first.context_config.candidate_count == 1
    assert first.context_config.passage_count == len(first.passages)
    assert first.context_config.max_candidates == 3


def test_knowledge_answer_and_nested_records_are_frozen(tmp_path: Path) -> None:
    answer = _query_answer(_knowledge_base(tmp_path, FakeGenerator()), "Binder credentials?")

    with pytest.raises(FrozenInstanceError):
        answer.text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        answer.citations[0].chunk_id = "forged"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        answer.model_context.rendered_context = "changed"  # type: ignore[misc]


def test_public_values_reject_forged_citation_or_replay_provenance(tmp_path: Path) -> None:
    answer = _query_answer(_knowledge_base(tmp_path, FakeGenerator()), "Binder credentials?")

    forged = replace(answer.citations[0], chunk_id="forged-chunk")
    with pytest.raises(ValueError, match="allowed model-context"):
        KnowledgeAnswer(answer.text, (forged,), answer.model_context)

    with pytest.raises(ValueError, match="rendered_context"):
        replace(answer.model_context, rendered_context="[K1]\nforged")


@pytest.mark.parametrize(
    "citation_ids",
    [
        ("K2",),
        ("K1", "K1"),
        ("not-a-handle",),
        (),
    ],
)
def test_citations_fail_closed_unless_they_are_unique_allowed_handles(
    tmp_path: Path, citation_ids: tuple[str, ...]
) -> None:
    generator = FakeGenerator(GeneratedAnswer("answer", citation_ids))
    kb = _knowledge_base(tmp_path, generator)

    with pytest.raises(CitationValidationError):
        _query_answer(kb, "Binder credentials?")


def test_validated_citation_matches_supplied_canonical_provenance(tmp_path: Path) -> None:
    answer = _query_answer(_knowledge_base(tmp_path, FakeGenerator()), "Binder credentials?")
    passage = answer.model_context.passages[0]
    citation = answer.citations[0]

    assert citation.source_id == passage.source_id
    assert citation.document_id == passage.document_id
    assert citation.logical_path == passage.logical_path
    assert citation.document_content_hash == passage.document_content_hash
    assert citation.chunk_id == passage.chunk_id
    assert (citation.start_offset, citation.end_offset) == (
        passage.start_offset,
        passage.end_offset,
    )


@pytest.mark.parametrize("text", ["invented [K2]", "malformed [Kx]", "wrong [k1]"])
def test_answer_text_cannot_smuggle_invented_or_malformed_handles(
    tmp_path: Path, text: str
) -> None:
    generator = FakeGenerator(GeneratedAnswer(text, ("K1",)))

    with pytest.raises(CitationValidationError):
        _query_answer(_knowledge_base(tmp_path, generator), "Binder credentials?")


def test_provider_failure_is_redacted_and_returns_no_partial_answer(tmp_path: Path) -> None:
    generator = FakeGenerator(error=RuntimeError("secret-key private document"))
    kb = _knowledge_base(tmp_path, generator)

    with pytest.raises(AnswerGeneratorError) as caught:
        _query_answer(kb, "Binder credentials?")

    assert str(caught.value) == "answer generator failed"
    assert "secret-key" not in str(caught.value)


def test_retrieval_or_context_failure_prevents_generator_invocation(tmp_path: Path) -> None:
    generator = FakeGenerator()
    kb = _knowledge_base(tmp_path, generator)

    with pytest.raises(KnowledgeBaseSourceError):
        _query_answer(kb, "no matching vocabulary here")
    assert generator.calls == []

    limits = AnswerGenerationLimits(max_context_chars=1)
    with pytest.raises(AnswerGenerationLimitError):
        _query_answer(kb, "Binder credentials?", limits=limits)
    assert generator.calls == []


def test_question_answer_and_citation_limits_are_enforced(tmp_path: Path) -> None:
    generator = FakeGenerator(GeneratedAnswer("one two three", ("K1",)))
    kb = _knowledge_base(tmp_path, generator)

    with pytest.raises(AnswerGenerationLimitError, match="question"):
        _query_answer(kb, "long question", limits=AnswerGenerationLimits(max_question_chars=4))
    assert generator.calls == []

    with pytest.raises(AnswerGenerationLimitError, match="max_answer_tokens"):
        _query_answer(kb, "Binder?", limits=AnswerGenerationLimits(max_answer_tokens=2))

    generator.result = GeneratedAnswer("answer", ("K1", "K2"))
    with pytest.raises(AnswerGenerationLimitError, match="max_citations"):
        _query_answer(kb, "Binder?", limits=AnswerGenerationLimits(max_citations=1))


def test_answer_generation_does_not_change_canonical_snapshot_or_files(
    tmp_path: Path,
) -> None:
    kb = _knowledge_base(tmp_path, FakeGenerator())
    before_documents = kb.list_documents()
    manifest = tmp_path / "kb" / "manifest.json"
    database = tmp_path / "kb" / "knowledge.db"
    before_files = (manifest.read_bytes(), database.read_bytes())

    _query_answer(kb, "Binder credentials?")

    assert kb.list_documents() == before_documents
    assert (manifest.read_bytes(), database.read_bytes()) == before_files


def test_answer_generator_configuration_is_runtime_only(tmp_path: Path) -> None:
    kb = _knowledge_base(tmp_path, FakeGenerator(identity="private-provider-config"))
    root = tmp_path / "kb"
    manifest_before = root.joinpath("manifest.json").read_bytes()
    kb.close()

    reopened = KnowledgeBase.open(str(root))
    with pytest.raises(AnswerGeneratorError, match="no answer generator"):
        _query_answer(reopened, "Binder credentials?")
    reopened.close()

    configured = KnowledgeBase.open(str(root), answer_generator=FakeGenerator())
    assert _query_answer(configured, "Binder credentials?").citations[0].citation_id == "K1"
    assert root.joinpath("manifest.json").read_bytes() == manifest_before


def test_direct_generation_rejects_invalid_generator_output(tmp_path: Path) -> None:
    kb = _knowledge_base(tmp_path)
    context = kb._collection.build_context(  # noqa: SLF001
        "Binder?", retrieval_limit=1, max_passages=1
    )

    class InvalidGenerator(FakeGenerator):
        def generate(self, question, context, *, model_context, limits):
            return object()

    with pytest.raises(AnswerGeneratorError, match="invalid output"):
        generate_knowledge_answer("Binder?", context, InvalidGenerator())


@pytest.mark.parametrize("field", AnswerGenerationLimits.__dataclass_fields__)
def test_answer_limits_require_positive_plain_integers(field: str) -> None:
    with pytest.raises(TypeError):
        AnswerGenerationLimits(**{field: True})
    with pytest.raises(ValueError):
        AnswerGenerationLimits(**{field: 0})
