from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from nexusmind import (
    Chunk,
    ContextAssemblyLimitError,
    ContextPackage,
    Document,
    KnowledgeCollection,
    KnowledgeSource,
    SearchHit,
    assemble_context,
    estimate_token_count,
)


@dataclass
class FakeAdapter:
    source_value: KnowledgeSource
    documents: tuple[Document, ...]

    def source(self) -> KnowledgeSource:
        return self.source_value

    def load_documents(self) -> tuple[Document, ...]:
        return self.documents


@dataclass
class Result:
    source: KnowledgeSource
    document: Document
    hit: SearchHit


def _result(
    document: Document,
    *,
    chunk_id: str,
    start: int,
    end: int,
    score: float,
) -> Result:
    source = KnowledgeSource(
        source_id=document.source_id,
        source_type="test",
        display_name="Test source",
    )
    return Result(
        source,
        document,
        SearchHit(
            Chunk(
                document.document_id,
                chunk_id,
                document.content[start:end],
                start,
                end,
            ),
            score,
            ("term",),
        ),
    )


def test_collection_build_context_preserves_complete_provenance() -> None:
    document = Document("docs", "binder.md", "Binder caller UID is preserved.")
    source = KnowledgeSource(
        source_id="docs",
        source_type="test",
        display_name="Android docs",
        logical_location="docs/",
        metadata={"owner": "platform"},
    )
    collection = KnowledgeCollection()
    collection.sync(FakeAdapter(source, (document,)))

    context = collection.build_context("Binder caller UID", limit=5, max_chars=8000)

    assert isinstance(context, ContextPackage)
    assert context.query == "Binder caller UID"
    assert len(context.passages) == 1
    passage = context.passages[0]
    assert passage.source_id == source.source_id
    assert passage.document_id == document.document_id
    assert passage.logical_path == "binder.md"
    assert passage.chunk_id
    assert (passage.start_offset, passage.end_offset) == (0, len(document.content))
    assert passage.content == document.content
    assert passage.document_content_hash == document.content_hash
    assert passage.chunk_start_offset == 0
    assert passage.chunk_end_offset == len(document.content)
    assert context.metadata["character_count"] == len(document.content)


def test_collection_build_context_separates_retrieval_depth_from_passage_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = Document("docs", "a.txt", "abcdefghij")
    candidates = (
        _result(document, chunk_id="best", start=0, end=5, score=10.0),
        *(
            _result(
                document,
                chunk_id=f"duplicate-{index}",
                start=0,
                end=5,
                score=float(9 - index),
            )
            for index in range(1, 5)
        ),
        _result(document, chunk_id="later", start=5, end=10, score=1.0),
    )
    collection = KnowledgeCollection()
    requested_limits: list[int] = []

    def search(query: str, *, limit: int = 10):
        requested_limits.append(limit)
        return candidates[:limit]

    monkeypatch.setattr(collection, "search", search)

    context = collection.build_context(
        "term", retrieval_limit=6, max_passages=2
    )

    assert requested_limits == [6]
    assert [passage.chunk_id for passage in context.passages] == ["best", "later"]
    assert context.metadata["candidate_count"] == 6
    assert context.metadata["max_candidates"] == 6
    assert context.metadata["max_passages"] == 2


def test_context_passage_is_a_compact_immutable_provenance_reference() -> None:
    document = Document(
        "docs",
        "large.txt",
        "search term " + ("x" * 1_000_000),
        metadata={"tag": "original"},
    )
    results = tuple(
        _result(
            document,
            chunk_id=f"chunk-{index}",
            start=index * 20,
            end=(index * 20) + 10,
            score=float(5 - index),
        )
        for index in range(5)
    )
    context = assemble_context(
        "term", results, max_passages=5, max_chars=50, max_candidates=5
    )
    passage = context.passages[0]

    assert [item.name for item in fields(type(passage))] == [
        "source_id",
        "document_id",
        "logical_path",
        "document_content_hash",
        "chunk_id",
        "chunk_start_offset",
        "chunk_end_offset",
        "start_offset",
        "end_offset",
        "content",
        "score",
        "matched_terms",
    ]
    assert not hasattr(passage, "source")
    assert not hasattr(passage, "document")
    assert not hasattr(passage, "chunk")
    assert len(context.passages) == 5
    assert sum(len(item.content) for item in context.passages) == 50
    assert passage.document_content_hash == document.content_hash


def test_assembly_removes_duplicates_and_preserves_non_overlapping_content() -> None:
    document = Document("docs", "a.txt", "abcdefghij")
    results = (
        _result(document, chunk_id="best", start=0, end=6, score=3.0),
        _result(document, chunk_id="duplicate-id", start=0, end=6, score=2.0),
        _result(document, chunk_id="overlap", start=5, end=9, score=1.0),
        _result(document, chunk_id="separate", start=6, end=10, score=0.5),
    )

    context = assemble_context("term", results, max_passages=4)

    assert [(passage.chunk_id, passage.content) for passage in context.passages] == [
        ("best", "abcdef"),
        ("overlap", "ghi"),
        ("separate", "j"),
    ]
    assert context.metadata["duplicates_removed"] == 1
    assert context.metadata["overlap_characters_removed"] == 4


def test_default_chunk_overlap_keeps_the_new_suffix() -> None:
    document = Document("docs", "a.txt", "x" * 1900)
    results = (
        _result(document, chunk_id="first", start=0, end=1000, score=2.0),
        _result(document, chunk_id="second", start=900, end=1900, score=1.0),
    )

    context = assemble_context("term", results, max_passages=2, max_chars=1900)

    assert [(p.start_offset, p.end_offset) for p in context.passages] == [
        (0, 1000),
        (1000, 1900),
    ]
    assert context.metadata["character_count"] == 1900


def test_token_estimator_counts_each_cjk_character_conservatively() -> None:
    assert estimate_token_count("knowledge retrieval") == 2
    assert estimate_token_count("知识库安全检索") == 7
    assert estimate_token_count("知识库, retrieval!") == 6


def test_assembly_rejects_unbounded_or_excess_candidate_inputs() -> None:
    def forever():
        while True:
            yield object()

    with pytest.raises(TypeError, match="tuple"):
        assemble_context("term", forever(), max_passages=1)  # type: ignore[arg-type]
    with pytest.raises(ContextAssemblyLimitError, match="candidates"):
        assemble_context(
            "term", (object(), object()), max_passages=1, max_candidates=1
        )


def test_assembly_applies_character_and_token_limits_without_truncating_provenance() -> None:
    document = Document("docs", "a.txt", "one two three four")
    results = (
        _result(document, chunk_id="one", start=0, end=7, score=2.0),
        _result(document, chunk_id="two", start=8, end=18, score=1.0),
    )

    by_chars = assemble_context("term", results, max_passages=2, max_chars=7)
    by_tokens = assemble_context("term", results, max_passages=2, max_tokens=2)

    assert [passage.content for passage in by_chars.passages] == ["one two"]
    assert [passage.content for passage in by_tokens.passages] == ["one two"]
    assert by_chars.metadata["limited"] is True
    assert by_tokens.metadata["estimated_token_count"] == 2


@pytest.mark.parametrize(
    "name", ["max_passages", "max_candidates", "max_chars", "max_tokens"]
)
def test_assembly_limits_require_positive_plain_integers(name: str) -> None:
    kwargs = {"max_passages": 1, name: True}
    with pytest.raises(TypeError):
        assemble_context("query", (), **kwargs)

    kwargs[name] = 0
    with pytest.raises(ValueError):
        assemble_context("query", (), **kwargs)
