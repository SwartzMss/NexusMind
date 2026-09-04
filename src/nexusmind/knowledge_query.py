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
from .query_expansion import QueryExpander

@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeQueryOptions:
    """Runtime-only controls for the existing knowledge-answer pipeline."""

    retrieval_limit: int = 8
    limits: AnswerGenerationLimits = AnswerGenerationLimits()
    generator: AnswerGenerator | None = None
    query_expander: QueryExpander | None = None

    def __post_init__(self) -> None:
        if type(self.retrieval_limit) is not int:
            raise TypeError("retrieval_limit must be an integer")
        if self.retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than zero")
        if type(self.limits) is not AnswerGenerationLimits:
            raise TypeError("limits must be AnswerGenerationLimits")
        if self.generator is not None and not isinstance(self.generator, AnswerGenerator):
            raise TypeError("generator must implement AnswerGenerator")
        if self.query_expander is not None and not isinstance(self.query_expander, QueryExpander):
            raise TypeError("query_expander must implement QueryExpander")


@dataclass(frozen=True, slots=True)
class KnowledgeQueryTrace:
    """Bounded debug summary derived from the validated model-context record."""

    retrieval_backend: str
    passages: tuple[ModelContextPassage, ...]
    candidate_count: int
    context_character_count: int
    context_estimated_token_count: int
    retrieval_queries: tuple[str, ...] = ()
    query_expansion_error: str | None = None
    fused_result_provenance: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = ()
    context_expansion_enabled: bool = False
    anchor_passage_count: int = 0
    expanded_passage_count: int = 0
    expanded_document_count: int = 0
    section_boundary_skips: int = 0

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
        if type(self.retrieval_queries) is not tuple or any(
            type(query) is not str or not query.strip() for query in self.retrieval_queries
        ):
            raise TypeError("retrieval_queries must be a tuple of non-empty strings")
        if self.query_expansion_error is not None and (
            type(self.query_expansion_error) is not str or not self.query_expansion_error.strip()
        ):
            raise ValueError("query_expansion_error must be a non-empty string or None")
        if type(self.fused_result_provenance) is not tuple:
            raise TypeError("fused_result_provenance must be a tuple")
        chunk_ids: set[str] = set()
        for item in self.fused_result_provenance:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("fused result provenance entries must be pairs")
            chunk_id, ranks = item
            if type(chunk_id) is not str or not chunk_id.strip() or chunk_id in chunk_ids:
                raise ValueError("fused result chunk ids must be non-empty and unique")
            chunk_ids.add(chunk_id)
            if type(ranks) is not tuple or not ranks:
                raise ValueError("fused result ranks must be a non-empty tuple")
            for rank in ranks:
                if (
                    type(rank) is not tuple
                    or len(rank) != 2
                    or type(rank[0]) is not int
                    or rank[0] < 0
                    or type(rank[1]) is not int
                    or rank[1] <= 0
                ):
                    raise ValueError("fused result ranks must contain query-index/rank pairs")
        if type(self.context_expansion_enabled) is not bool:
            raise TypeError("context_expansion_enabled must be a boolean")
        for name in (
            "anchor_passage_count",
            "expanded_passage_count",
            "expanded_document_count",
            "section_boundary_skips",
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
            "retrieval_queries": list(result.trace.retrieval_queries),
            "query_expansion_error": result.trace.query_expansion_error,
            "context_expansion_enabled": result.trace.context_expansion_enabled,
            "anchor_passage_count": result.trace.anchor_passage_count,
            "expanded_passage_count": result.trace.expanded_passage_count,
            "expanded_document_count": result.trace.expanded_document_count,
            "section_boundary_skips": result.trace.section_boundary_skips,
            "fused_results": [
                {"chunk_id": chunk_id, "ranks": [[query_index, rank] for query_index, rank in ranks]}
                for chunk_id, ranks in result.trace.fused_result_provenance
            ],
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
