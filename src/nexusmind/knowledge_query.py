"""Unified, inspectable orchestration values for one knowledge query."""

from __future__ import annotations

from dataclasses import dataclass
from .knowledge_answer import (
    AnswerGenerationLimits,
    AnswerGenerator,
    KnowledgeAnswer,
    KnowledgeCitation,
    ModelContextPassage,
)

@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeQueryOptions:
    """Runtime-only controls for the existing knowledge-answer pipeline."""

    retrieval_limit: int = 8
    limits: AnswerGenerationLimits = AnswerGenerationLimits()
    generator: AnswerGenerator | None = None

    def __post_init__(self) -> None:
        if type(self.retrieval_limit) is not int:
            raise TypeError("retrieval_limit must be an integer")
        if self.retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than zero")
        if type(self.limits) is not AnswerGenerationLimits:
            raise TypeError("limits must be AnswerGenerationLimits")
        if self.generator is not None and not isinstance(self.generator, AnswerGenerator):
            raise TypeError("generator must implement AnswerGenerator")


@dataclass(frozen=True, slots=True)
class KnowledgeQueryTrace:
    """Bounded debug summary derived from the validated model-context record."""

    retrieval_backend: str
    passages: tuple[ModelContextPassage, ...]
    candidate_count: int
    context_character_count: int
    context_estimated_token_count: int

    def __post_init__(self) -> None:
        if type(self.retrieval_backend) is not str or not self.retrieval_backend.strip():
            raise ValueError("retrieval_backend must be a non-empty string")
        if type(self.passages) is not tuple or any(
            type(item) is not ModelContextPassage for item in self.passages
        ):
            raise TypeError("passages must be a tuple of ModelContextPassage values")
        for name in (
            "candidate_count",
            "context_character_count",
            "context_estimated_token_count",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class KnowledgeQueryResult:
    """Trusted answer and bounded trace from one unified query operation."""

    answer: KnowledgeAnswer
    citations: tuple[KnowledgeCitation, ...]
    trace_id: str | None
    trace: KnowledgeQueryTrace

    def __post_init__(self) -> None:
        if type(self.answer) is not KnowledgeAnswer:
            raise TypeError("answer must be a KnowledgeAnswer")
        if self.citations != self.answer.citations:
            raise ValueError("citations must be the validated answer citations")
        if self.trace_id is not None and (
            type(self.trace_id) is not str or not self.trace_id.strip()
        ):
            raise ValueError("trace_id must be a non-empty string or None")
        if type(self.trace) is not KnowledgeQueryTrace:
            raise TypeError("trace must be a KnowledgeQueryTrace")
        if self.trace.passages != self.answer.model_context.passages:
            raise ValueError("trace passages must match the validated model context")


def knowledge_query_result_dict(
    result: KnowledgeQueryResult, *, include_debug: bool = False
) -> dict[str, object]:
    """Return the stable JSON-ready public representation of a query result."""

    if type(result) is not KnowledgeQueryResult:
        raise TypeError("result must be a KnowledgeQueryResult")
    payload: dict[str, object] = {
        "answer": result.answer.text,
        "citations": [
            {
                "citation_id": item.citation_id,
                "source_id": item.source_id,
                "document_id": item.document_id,
                "logical_path": item.logical_path,
                "document_content_hash": item.document_content_hash,
                "chunk_id": item.chunk_id,
                "start_offset": item.start_offset,
                "end_offset": item.end_offset,
                "content_hash": item.content_hash,
            }
            for item in result.citations
        ],
        "trace_id": result.trace_id,
    }
    if include_debug:
        payload["debug"] = {
            "retrieval_backend": result.trace.retrieval_backend,
            "candidate_count": result.trace.candidate_count,
            "passage_count": len(result.trace.passages),
            "context_character_count": result.trace.context_character_count,
            "context_estimated_token_count": result.trace.context_estimated_token_count,
            "passages": [
                {
                    "citation_id": item.citation_id,
                    "logical_path": item.logical_path,
                    "chunk_id": item.chunk_id,
                }
                for item in result.trace.passages
            ],
        }
    return payload


__all__ = [
    "KnowledgeQueryOptions",
    "KnowledgeQueryResult",
    "KnowledgeQueryTrace",
    "knowledge_query_result_dict",
]
