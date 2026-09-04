from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json
from pathlib import Path

import pytest

from nexusmind import (
    AnswerGenerationLimits,
    Chunk,
    ContextPackage,
    Document,
    GeneratedAnswer,
    KnowledgeBase,
    KnowledgeCollection,
    KnowledgeQueryOptions,
    KnowledgeQueryResult,
    KnowledgeSource,
    LocalFileSourceConfig,
    SearchHit,
    knowledge_query_result_dict,
)


@dataclass
class FakeGenerator:
    def __post_init__(self) -> None:
        self.calls: list[tuple[object, object, object, object]] = []

    @property
    def config_identity(self) -> str:
        return "fake/query-v1"

    def generate(self, question, context, *, model_context, limits):
        self.calls.append((question, context, model_context, limits))
        return GeneratedAnswer("Binder uses kernel credentials [K1]", ("K1",))


@dataclass
class _ChunkAdapter:
    source_value: KnowledgeSource
    documents: tuple[Document, ...]

    def source(self) -> KnowledgeSource:
        return self.source_value

    def load_documents(self) -> tuple[Document, ...]:
        return self.documents


class _PipeChunker:
    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        offset = 0
        for ordinal, content in enumerate(document.content.split("|")):
            end = offset + len(content)
            heading_path = ("Binder",) if ordinal < 3 else ("SELinux",)
            chunks.append(
                Chunk(
                    document.document_id,
                    ("definition", "anchor", "caveat", "sibling")[ordinal],
                    content,
                    offset,
                    end,
                    heading_path,
                    heading_path[-1],
                    f"guide.md:L{ordinal + 1}",
                )
            )
            offset = end + 1
        return tuple(chunks)


def _multi_chunk_knowledge_base(
    tmp_path: Path,
) -> tuple[KnowledgeBase, FakeGenerator]:
    generator = FakeGenerator()
    knowledge = KnowledgeBase.create(
        str(tmp_path / "kb"),
        knowledge_base_id="query-expansion-test",
        answer_generator=generator,
    )
    source = KnowledgeSource(
        source_id="docs",
        source_type="test",
        display_name="Docs",
    )
    document = Document(
        source_id="docs",
        logical_path="guide.md",
        content="definition|anchor|caveat|sibling",
    )
    knowledge._collection = KnowledgeCollection(chunker=_PipeChunker())  # noqa: SLF001
    knowledge._collection.sync(_ChunkAdapter(source, (document,)))  # noqa: SLF001
    return knowledge, generator


def _knowledge_base(tmp_path: Path) -> KnowledgeBase:
    document = tmp_path / "binder.md"
    document.write_text(
        "Binder authenticates callers using kernel-provided credentials.",
        encoding="utf-8",
    )
    knowledge = KnowledgeBase.create(
        str(tmp_path / "kb"),
        knowledge_base_id="query-test",
        answer_generator=FakeGenerator(),
    )
    knowledge.add_source(LocalFileSourceConfig(path=str(document)))
    knowledge.sync()
    return knowledge


def test_query_runs_existing_pipeline_and_returns_validated_debug_trace(
    tmp_path: Path,
) -> None:
    knowledge = _knowledge_base(tmp_path)

    result = knowledge.query("How does Binder authenticate callers?")

    assert isinstance(result, KnowledgeQueryResult)
    assert result.answer.text == "Binder uses kernel credentials [K1]"
    assert result.citations == result.answer.citations
    assert result.citations[0].logical_path == "binder.md"
    assert result.trace_id
    assert result.trace.retrieval_backend == "InMemoryChunkIndex"
    assert result.trace.passages == result.answer.model_context.passages
    assert result.trace.candidate_count == 1
    assert result.trace.context_character_count > 0


def test_knowledge_base_exposes_only_the_query_answer_api(tmp_path: Path) -> None:
    knowledge = _knowledge_base(tmp_path)

    assert not hasattr(knowledge, "answer")
    assert knowledge.query("Binder credentials?").answer.text


def test_query_result_has_stable_json_schema_and_is_frozen(tmp_path: Path) -> None:
    result = _knowledge_base(tmp_path).query("Binder credentials?")

    payload = knowledge_query_result_dict(result, include_debug=True)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)

    assert list(payload) == ["answer", "citations", "trace_id", "debug"]
    assert json.loads(encoded)["citations"][0]["citation_id"] == "K1"
    assert json.loads(encoded)["debug"]["passage_count"] == 1
    with pytest.raises(FrozenInstanceError):
        result.trace_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("retrieval_limit", [True, 0])
def test_query_options_reject_invalid_retrieval_limits(retrieval_limit: object) -> None:
    error = TypeError if retrieval_limit is True else ValueError
    with pytest.raises(error):
        KnowledgeQueryOptions(retrieval_limit=retrieval_limit)  # type: ignore[arg-type]


def test_query_expands_same_section_by_default(tmp_path: Path) -> None:
    knowledge, generator = _multi_chunk_knowledge_base(tmp_path)

    result = knowledge.query(
        "anchor",
        options=KnowledgeQueryOptions(generator=generator),
    )

    assert [item.chunk_id for item in generator.calls[0][2].passages] == [
        "anchor",
        "definition",
        "caveat",
    ]
    assert result.trace.context_expansion_enabled is True
    assert result.trace.anchor_passage_count == 1
    assert result.trace.expanded_passage_count == 2
    assert result.trace.expanded_document_count == 1
    assert result.trace.section_boundary_skips == 0


def test_query_expand_context_false_reproduces_old_context(tmp_path: Path) -> None:
    knowledge, generator = _multi_chunk_knowledge_base(tmp_path)

    result = knowledge.query(
        "anchor",
        expand_context=False,
        options=KnowledgeQueryOptions(generator=generator),
    )

    assert [item.chunk_id for item in generator.calls[0][2].passages] == ["anchor"]
    assert result.trace.context_expansion_enabled is False
    assert result.trace.anchor_passage_count == 1
    assert result.trace.expanded_passage_count == 0
    assert result.trace.expanded_document_count == 0


def test_query_rejects_non_boolean_expand_context(tmp_path: Path) -> None:
    knowledge, _ = _multi_chunk_knowledge_base(tmp_path)

    with pytest.raises(TypeError, match="expand_context"):
        knowledge.query("anchor", expand_context=1)  # type: ignore[arg-type]


def test_query_debug_json_exposes_context_expansion_metadata(tmp_path: Path) -> None:
    knowledge, generator = _multi_chunk_knowledge_base(tmp_path)

    result = knowledge.query(
        "anchor",
        options=KnowledgeQueryOptions(generator=generator),
    )
    debug = knowledge_query_result_dict(result, include_debug=True)["debug"]

    assert debug["context_expansion_enabled"] is True
    assert debug["anchor_passage_count"] == 1
    assert debug["expanded_passage_count"] == 2
    assert debug["expanded_document_count"] == 1
    assert debug["section_boundary_skips"] == 0
