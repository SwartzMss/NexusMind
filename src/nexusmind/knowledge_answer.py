"""Bounded, provenance-preserving one-shot knowledge answer generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Protocol, runtime_checkable

from .context_assembly import ContextPackage, ContextPassage, estimate_token_count


class KnowledgeAnswerError(Exception):
    """Base class for controlled knowledge-answer failures."""


class AnswerGenerationLimitError(KnowledgeAnswerError):
    """A question, context, answer, or citation bound was exceeded."""


class AnswerGeneratorError(KnowledgeAnswerError):
    """The configured answer generator failed without exposing provider details."""


class CitationValidationError(KnowledgeAnswerError):
    """Generator output did not reference only allowed context evidence."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AnswerGenerationLimits:
    """Explicit bounds for one knowledge-answer operation."""

    max_question_chars: int = 4_000
    max_context_chars: int = 12_000
    max_context_tokens: int = 4_000
    max_passages: int = 10
    max_answer_chars: int = 20_000
    max_answer_tokens: int = 4_000
    max_citations: int = 10

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Untrusted provider output awaiting NexusMind citation validation."""

    text: str
    citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be a string")
        if type(self.citation_ids) is not tuple:
            raise TypeError("citation_ids must be a tuple")
        if any(type(item) is not str for item in self.citation_ids):
            raise TypeError("citation_ids must contain only strings")


@dataclass(frozen=True, slots=True)
class ModelContextPassage:
    """One stable citation handle and the exact evidence rendered to the model."""

    citation_id: str
    source_id: str
    document_id: str
    logical_path: str
    document_content_hash: str
    chunk_id: str
    start_offset: int
    end_offset: int
    content: str

    def __post_init__(self) -> None:
        _require_citation_id(self.citation_id)
        for name in (
            "source_id",
            "document_id",
            "logical_path",
            "document_content_hash",
            "chunk_id",
        ):
            _require_non_empty_text(getattr(self, name), name)
        _require_sha256(self.document_content_hash, "document_content_hash")
        _require_offsets(self.start_offset, self.end_offset)
        if type(self.content) is not str:
            raise TypeError("content must be a string")
        if len(self.content) != self.end_offset - self.start_offset:
            raise ValueError("content length must match passage offsets")


@dataclass(frozen=True, slots=True)
class ContextConfigurationRecord:
    """Replayable retrieval/context selection configuration and outcome."""

    query: str
    candidate_count: int
    passage_count: int
    character_count: int
    estimated_token_count: int
    max_passages: int
    max_candidates: int
    max_chars: int | None
    max_tokens: int | None
    duplicates_removed: int
    overlap_characters_removed: int
    limited: bool
    token_count_method: str

    def __post_init__(self) -> None:
        _require_non_empty_text(self.query, "query")
        for name in (
            "candidate_count",
            "passage_count",
            "character_count",
            "estimated_token_count",
            "duplicates_removed",
            "overlap_characters_removed",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("max_passages", "max_candidates"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in ("max_chars", "max_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.limited) is not bool:
            raise TypeError("limited must be a boolean")
        _require_non_empty_text(self.token_count_method, "token_count_method")


@dataclass(frozen=True, slots=True)
class ModelContextRecord:
    """Inspectable replay record for the knowledge input visible to a generator."""

    question: str
    passages: tuple[ModelContextPassage, ...]
    rendered_context: str
    context_config: ContextConfigurationRecord
    limits: AnswerGenerationLimits
    generator_config_identity: str

    def __post_init__(self) -> None:
        _require_non_empty_text(self.question, "question")
        if type(self.passages) is not tuple or any(
            type(item) is not ModelContextPassage for item in self.passages
        ):
            raise TypeError("passages must be a tuple of ModelContextPassage values")
        expected_ids = tuple(f"K{index}" for index in range(1, len(self.passages) + 1))
        if tuple(item.citation_id for item in self.passages) != expected_ids:
            raise ValueError("passages must use stable sequential citation IDs")
        if type(self.rendered_context) is not str:
            raise TypeError("rendered_context must be a string")
        if self.rendered_context != "\n\n".join(
            _render_passage(passage) for passage in self.passages
        ):
            raise ValueError("rendered_context must match recorded passages")
        if type(self.limits) is not AnswerGenerationLimits:
            raise TypeError("limits must be AnswerGenerationLimits")
        if type(self.context_config) is not ContextConfigurationRecord:
            raise TypeError("context_config must be a ContextConfigurationRecord")
        if self.context_config.query != self.question:
            raise ValueError("context_config query must match question")
        if self.context_config.passage_count != len(self.passages):
            raise ValueError("context_config passage_count must match passages")
        _require_non_empty_text(
            self.generator_config_identity, "generator_config_identity"
        )


@dataclass(frozen=True, slots=True)
class KnowledgeCitation:
    """One validated citation derived only from an allowed context passage."""

    citation_id: str
    source_id: str
    document_id: str
    logical_path: str
    document_content_hash: str
    chunk_id: str
    start_offset: int
    end_offset: int
    content_hash: str

    def __post_init__(self) -> None:
        _require_citation_id(self.citation_id)
        for name in (
            "source_id",
            "document_id",
            "logical_path",
            "document_content_hash",
            "chunk_id",
            "content_hash",
        ):
            _require_non_empty_text(getattr(self, name), name)
        _require_sha256(self.document_content_hash, "document_content_hash")
        _require_sha256(self.content_hash, "content_hash")
        _require_offsets(self.start_offset, self.end_offset)


@dataclass(frozen=True, slots=True)
class KnowledgeAnswer:
    """Trusted answer text, validated citations, and replayable model context."""

    text: str
    citations: tuple[KnowledgeCitation, ...]
    model_context: ModelContextRecord

    def __post_init__(self) -> None:
        _require_non_empty_text(self.text, "text")
        if type(self.citations) is not tuple or any(
            type(item) is not KnowledgeCitation for item in self.citations
        ):
            raise TypeError("citations must be a tuple of KnowledgeCitation values")
        if not self.citations:
            raise ValueError("citations must not be empty")
        if len({item.citation_id for item in self.citations}) != len(self.citations):
            raise ValueError("citations must not contain duplicate IDs")
        if type(self.model_context) is not ModelContextRecord:
            raise TypeError("model_context must be a ModelContextRecord")
        allowed = {item.citation_id: item for item in self.model_context.passages}
        for citation in self.citations:
            passage = allowed.get(citation.citation_id)
            if passage is None or not _citation_matches_passage(citation, passage):
                raise ValueError("citation must match an allowed model-context passage")


@runtime_checkable
class AnswerGenerator(Protocol):
    """Knowledge-domain answer capability; retrieval is deliberately excluded."""

    @property
    def config_identity(self) -> str: ...

    def generate(
        self,
        question: str,
        context: ContextPackage,
        *,
        model_context: ModelContextRecord,
        limits: AnswerGenerationLimits,
    ) -> GeneratedAnswer: ...


def render_model_context(
    context: ContextPackage,
    *,
    question: str,
    limits: AnswerGenerationLimits,
    generator_config_identity: str,
) -> ModelContextRecord:
    """Render stable `[K1]` evidence blocks and return their replay record."""

    if type(context) is not ContextPackage:
        raise TypeError("context must be a ContextPackage")
    _validate_question(question, limits)
    if context.query != question:
        raise CitationValidationError("context query does not match answer question")
    if not context.passages:
        raise CitationValidationError("model context contains no evidence passages")
    if len(context.passages) > limits.max_passages:
        raise AnswerGenerationLimitError("context exceeds max_passages")
    if type(generator_config_identity) is not str or not generator_config_identity.strip():
        raise AnswerGeneratorError("answer generator configuration is invalid")
    if len(generator_config_identity) > 200:
        raise AnswerGeneratorError("answer generator configuration is invalid")

    records = tuple(
        _record_passage(index, passage)
        for index, passage in enumerate(context.passages, start=1)
    )
    rendered = "\n\n".join(_render_passage(passage) for passage in records)
    if len(rendered) > limits.max_context_chars:
        raise AnswerGenerationLimitError("rendered context exceeds max_context_chars")
    if estimate_token_count(rendered) > limits.max_context_tokens:
        raise AnswerGenerationLimitError("rendered context exceeds max_context_tokens")
    return ModelContextRecord(
        question=question,
        passages=records,
        rendered_context=rendered,
        context_config=_context_configuration(context),
        limits=limits,
        generator_config_identity=generator_config_identity,
    )


def generate_knowledge_answer(
    question: str,
    context: ContextPackage,
    generator: AnswerGenerator,
    *,
    limits: AnswerGenerationLimits | None = None,
) -> KnowledgeAnswer:
    """Invoke a generator once and fail closed on any invalid provider output."""

    active_limits = AnswerGenerationLimits() if limits is None else limits
    if type(active_limits) is not AnswerGenerationLimits:
        raise TypeError("limits must be AnswerGenerationLimits")
    if not isinstance(generator, AnswerGenerator):
        raise TypeError("generator must implement AnswerGenerator")
    try:
        identity = generator.config_identity
    except Exception:
        raise AnswerGeneratorError("unable to inspect answer generator") from None
    model_context = render_model_context(
        context,
        question=question,
        limits=active_limits,
        generator_config_identity=identity,
    )
    try:
        generated = generator.generate(
            question,
            context,
            model_context=model_context,
            limits=active_limits,
        )
    except Exception:
        raise AnswerGeneratorError("answer generator failed") from None
    if type(generated) is not GeneratedAnswer:
        raise AnswerGeneratorError("answer generator returned invalid output")
    _validate_answer_text(generated.text, active_limits)
    _validate_answer_handles(generated.text, generated.citation_ids)
    citations = _validate_citations(generated.citation_ids, model_context, active_limits)
    return KnowledgeAnswer(generated.text, citations, model_context)


def _validate_question(question: str, limits: AnswerGenerationLimits) -> None:
    if type(question) is not str:
        raise TypeError("question must be a string")
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    if len(question) > limits.max_question_chars:
        raise AnswerGenerationLimitError("question exceeds max_question_chars")


def _context_configuration(context: ContextPackage) -> ContextConfigurationRecord:
    metadata = context.metadata
    try:
        return ContextConfigurationRecord(
            query=context.query,
            candidate_count=metadata["candidate_count"],
            passage_count=metadata["passage_count"],
            character_count=metadata["character_count"],
            estimated_token_count=metadata["estimated_token_count"],
            max_passages=metadata["max_passages"],
            max_candidates=metadata["max_candidates"],
            max_chars=metadata["max_chars"],
            max_tokens=metadata["max_tokens"],
            duplicates_removed=metadata["duplicates_removed"],
            overlap_characters_removed=metadata["overlap_characters_removed"],
            limited=metadata["limited"],
            token_count_method=metadata["token_count_method"],
        )
    except (KeyError, TypeError, ValueError):
        raise CitationValidationError("context assembly metadata is invalid") from None


def _require_non_empty_text(value: object, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_citation_id(value: object) -> None:
    if type(value) is not str or re.fullmatch(r"K[1-9][0-9]*", value) is None:
        raise ValueError("citation_id must be a valid knowledge handle")


def _require_sha256(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_offsets(start_offset: object, end_offset: object) -> None:
    if type(start_offset) is not int or type(end_offset) is not int:
        raise TypeError("citation offsets must be integers")
    if start_offset < 0 or end_offset <= start_offset:
        raise ValueError("citation offsets must identify a non-empty range")


def _citation_matches_passage(
    citation: KnowledgeCitation, passage: ModelContextPassage
) -> bool:
    return (
        citation.source_id == passage.source_id
        and citation.document_id == passage.document_id
        and citation.logical_path == passage.logical_path
        and citation.document_content_hash == passage.document_content_hash
        and citation.chunk_id == passage.chunk_id
        and citation.start_offset == passage.start_offset
        and citation.end_offset == passage.end_offset
        and citation.content_hash
        == hashlib.sha256(passage.content.encode("utf-8")).hexdigest()
    )


def _validate_answer_text(text: str, limits: AnswerGenerationLimits) -> None:
    if not text.strip():
        raise AnswerGeneratorError("answer generator returned invalid output")
    if len(text) > limits.max_answer_chars:
        raise AnswerGenerationLimitError("answer exceeds max_answer_chars")
    if estimate_token_count(text) > limits.max_answer_tokens:
        raise AnswerGenerationLimitError("answer exceeds max_answer_tokens")


def _validate_answer_handles(text: str, citation_ids: tuple[str, ...]) -> None:
    for handle in re.findall(r"\[([Kk][^\]\s]*)\]", text):
        if re.fullmatch(r"K[1-9][0-9]*", handle) is None:
            raise CitationValidationError("answer text contains a malformed citation")
        if handle not in citation_ids:
            raise CitationValidationError("answer text cites undeclared evidence")


def _record_passage(index: int, passage: ContextPassage) -> ModelContextPassage:
    if type(passage) is not ContextPassage:
        raise TypeError("context contains an invalid passage")
    return ModelContextPassage(
        citation_id=f"K{index}",
        source_id=passage.source_id,
        document_id=passage.document_id,
        logical_path=passage.logical_path,
        document_content_hash=passage.document.content_hash,
        chunk_id=passage.chunk_id,
        start_offset=passage.start_offset,
        end_offset=passage.end_offset,
        content=passage.content,
    )


def _render_passage(passage: ModelContextPassage) -> str:
    return "\n".join(
        (
            f"[{passage.citation_id}]",
            f"source: {passage.source_id}",
            f"document: {passage.document_id}",
            f"path: {passage.logical_path}",
            f"chunk: {passage.chunk_id}",
            f"offsets: {passage.start_offset}-{passage.end_offset}",
            f"content: {passage.content}",
        )
    )


def _validate_citations(
    citation_ids: tuple[str, ...],
    model_context: ModelContextRecord,
    limits: AnswerGenerationLimits,
) -> tuple[KnowledgeCitation, ...]:
    if len(citation_ids) > limits.max_citations:
        raise AnswerGenerationLimitError("citations exceed max_citations")
    if not citation_ids:
        raise CitationValidationError("answer must cite supplied evidence")
    if len(citation_ids) != len(set(citation_ids)):
        raise CitationValidationError("answer contains duplicate citations")
    allowed = {passage.citation_id: passage for passage in model_context.passages}
    citations: list[KnowledgeCitation] = []
    for citation_id in citation_ids:
        if re.fullmatch(r"K[1-9][0-9]*", citation_id) is None:
            raise CitationValidationError("answer contains a malformed citation")
        passage = allowed.get(citation_id)
        if passage is None:
            raise CitationValidationError("answer cites evidence outside model context")
        citations.append(
            KnowledgeCitation(
                citation_id=passage.citation_id,
                source_id=passage.source_id,
                document_id=passage.document_id,
                logical_path=passage.logical_path,
                document_content_hash=passage.document_content_hash,
                chunk_id=passage.chunk_id,
                start_offset=passage.start_offset,
                end_offset=passage.end_offset,
                content_hash=hashlib.sha256(passage.content.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(citations)


__all__ = [
    "AnswerGenerationLimitError",
    "AnswerGenerationLimits",
    "AnswerGenerator",
    "AnswerGeneratorError",
    "CitationValidationError",
    "ContextConfigurationRecord",
    "GeneratedAnswer",
    "KnowledgeAnswer",
    "KnowledgeAnswerError",
    "KnowledgeCitation",
    "ModelContextPassage",
    "ModelContextRecord",
    "generate_knowledge_answer",
    "render_model_context",
]
