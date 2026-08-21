from __future__ import annotations

import math

import pytest

from nexusmind import (
    Chunk,
    ChunkIndex,
    EmbeddingVector,
    InMemorySemanticChunkIndex,
    RetrievalStage,
    SearchHit,
    SemanticChunkIndexLimitError,
    SemanticChunkIndexLimits,
    SemanticDimensionError,
    SemanticEmbeddingError,
)


def _chunk(chunk_id: str, content: str, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


class _RecordingProvider:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        self.document_calls.append(texts)
        return tuple(EmbeddingVector(self.vectors[text]) for text in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        self.query_calls.append(text)
        return EmbeddingVector(self.vectors[text])


def test_semantic_contracts_are_public_and_implement_chunk_index() -> None:
    provider = _RecordingProvider({})
    index: ChunkIndex = InMemorySemanticChunkIndex(embedding_provider=provider)

    assert index.search("query") == ()
    assert provider.query_calls == []


@pytest.mark.parametrize(
    "field",
    [
        "max_chunks",
        "max_total_chars",
        "max_total_vector_values",
        "max_dimensions",
        "max_chunks_per_document",
        "max_embedding_batch_size",
        "max_query_chars",
        "max_results",
    ],
)
@pytest.mark.parametrize("value", [True, 0, -1])
def test_semantic_limits_require_positive_plain_integers(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SemanticChunkIndexLimits(**{field: value})


def test_semantic_search_skips_embedding_for_empty_index_and_blank_query() -> None:
    provider = _RecordingProvider({"document": (1.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)

    assert index.search("not configured") == ()
    index.add((_chunk("one", "document"),))
    assert index.search(" \t\n") == ()
    assert provider.query_calls == []


def test_semantic_add_batches_and_search_ranks_all_cosine_scores() -> None:
    provider = _RecordingProvider(
        {
            "alpha": (1.0, 0.0),
            "diagonal": (1.0, 1.0),
            "opposite": (-1.0, 0.0),
            "query": (1.0, 0.0),
        }
    )
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    chunks = (
        _chunk("z-alpha", "alpha"),
        _chunk("middle", "diagonal"),
        _chunk("negative", "opposite"),
    )

    index.add(chunks)
    hits = index.search("query")

    assert provider.document_calls == [("alpha", "diagonal", "opposite")]
    assert provider.query_calls == ["query"]
    assert [hit.chunk.chunk_id for hit in hits] == ["z-alpha", "middle", "negative"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 1 / math.sqrt(2), -1.0])
    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
    assert all(hit.matched_terms == () for hit in hits)


def test_semantic_add_splits_provider_requests_at_batch_limit() -> None:
    provider = _RecordingProvider(
        {f"text-{index}": (float(index + 1),) for index in range(5)}
    )
    index = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(max_embedding_batch_size=2),
    )

    index.add(
        tuple(_chunk(f"chunk-{number}", f"text-{number}") for number in range(5))
    )

    assert provider.document_calls == [
        ("text-0", "text-1"),
        ("text-2", "text-3"),
        ("text-4",),
    ]


def test_semantic_ties_use_chunk_id_and_result_limit() -> None:
    provider = _RecordingProvider({"one": (1.0, 0.0), "two": (2.0, 0.0), "q": (1.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("z", "one"), _chunk("a", "two")))

    hits = index.search("q", limit=1)

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]


def test_semantic_cosine_remains_finite_for_extreme_finite_values() -> None:
    provider = _RecordingProvider(
        {"huge": (1e308, 1e308), "tiny": (5e-324, 0.0), "q": (1e308, 1e308)}
    )
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("huge", "huge"), _chunk("tiny", "tiny")))

    hits = index.search("q")

    assert all(math.isfinite(hit.score) for hit in hits)
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize("method", ["search", "diagnose"])
def test_semantic_search_and_diagnose_validate_before_provider_call(
    method: str,
) -> None:
    provider = _RecordingProvider({"one": (1.0,), "q": (1.0,)})
    index = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(max_query_chars=1, max_results=1),
    )
    index.add((_chunk("one", "one"),))

    operation = getattr(index, method)
    with pytest.raises(TypeError, match="query"):
        operation(1)  # type: ignore[arg-type]
    with pytest.raises(SemanticChunkIndexLimitError, match="max_query_chars"):
        operation("too long")
    with pytest.raises(TypeError, match="limit"):
        operation("q", limit=True)
    with pytest.raises(ValueError, match="greater than zero"):
        operation("q", limit=0)
    with pytest.raises(SemanticChunkIndexLimitError, match="max_results"):
        operation("q", limit=2)
    assert provider.query_calls == []


def test_search_hit_defaults_to_empty_backend_diagnostics() -> None:
    hit = SearchHit(chunk=_chunk("one", "one"), score=0.5)

    assert hit.matched_terms == ()


def test_semantic_diagnose_matches_search_and_embeds_query_once() -> None:
    provider = _RecordingProvider(
        {
            "alpha": (1.0, 0.0),
            "beta": (0.0, 1.0),
            "query": (1.0, 0.0),
        }
    )
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("alpha", "alpha"), _chunk("beta", "beta")))

    before_diagnose = len(provider.query_calls)
    diagnostics = index.diagnose("query", limit=2)
    assert len(provider.query_calls) == before_diagnose + 1
    ordinary_hits = index.clone().search("query", limit=2)

    assert diagnostics.hits == ordinary_hits
    assert len(provider.query_calls) == before_diagnose + 2
    assert [candidate.stage for candidate in diagnostics.candidates] == [
        RetrievalStage.SEMANTIC,
        RetrievalStage.SEMANTIC,
    ]
    assert [candidate.rank for candidate in diagnostics.candidates] == [1, 2]
    assert [candidate.chunk for candidate in diagnostics.candidates] == [
        hit.chunk for hit in diagnostics.hits
    ]
    assert [candidate.score for candidate in diagnostics.candidates] == [
        hit.score for hit in diagnostics.hits
    ]
    assert [candidate.matched_terms for candidate in diagnostics.candidates] == [(), ()]
    assert [candidate.rrf_contribution for candidate in diagnostics.candidates] == [
        None,
        None,
    ]
    assert [candidate.selected for candidate in diagnostics.candidates] == [True, True]


def test_semantic_diagnose_returns_empty_diagnostics_without_a_corpus() -> None:
    provider = _RecordingProvider({})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)

    diagnostics = index.diagnose("query")

    assert diagnostics.hits == ()
    assert diagnostics.candidates == ()
    assert provider.query_calls == []


def test_semantic_diagnose_skips_embedding_for_a_blank_query() -> None:
    provider = _RecordingProvider({"document": (1.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("one", "document"),))

    diagnostics = index.diagnose(" \t\n")

    assert diagnostics.hits == ()
    assert diagnostics.candidates == ()
    assert provider.query_calls == []


def test_replace_document_batches_only_new_chunks_and_removes_stale_chunks() -> None:
    provider = _RecordingProvider(
        {"old": (1.0, 0.0), "kept": (0.0, 1.0), "new": (1.0, 1.0), "q": (1.0, 0.0)}
    )
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    kept = _chunk("kept", "kept")
    index.add((_chunk("old", "old"), kept))

    index.replace_document("doc-1", (kept, _chunk("new", "new")))

    assert provider.document_calls == [("old", "kept"), ("new",)]
    assert {hit.chunk.chunk_id for hit in index.search("q")} == {"kept", "new"}


def test_remove_document_does_not_call_provider_and_empty_state_resets_dimension() -> None:
    provider = _RecordingProvider({"old": (1.0, 0.0), "new": (1.0, 0.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("old", "old"),))

    index.remove_document("doc-1")
    index.add((_chunk("new", "new", "doc-2"),))

    assert provider.document_calls == [("old",), ("new",)]


def test_replacing_only_document_preserves_committed_dimension_until_empty() -> None:
    provider = _RecordingProvider({"old": (1.0, 0.0), "different": (1.0, 0.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("old", "old"),))

    with pytest.raises(SemanticDimensionError, match="dimension"):
        index.replace_document("doc-1", (_chunk("different", "different"),))

    index.replace_document("doc-1", ())
    index.add((_chunk("different", "different"),))


def test_clone_shares_falsey_provider_but_copies_mutable_state() -> None:
    class FalseyProvider(_RecordingProvider):
        def __bool__(self) -> bool:
            return False

    provider = FalseyProvider({"one": (1.0,), "two": (1.0,), "q": (1.0,)})
    original = InMemorySemanticChunkIndex(embedding_provider=provider)
    original.add((_chunk("one", "one"),))

    clone = original.clone()
    clone.add((_chunk("two", "two", "doc-2"),))

    assert clone._provider is provider
    assert [hit.chunk.chunk_id for hit in original.search("q")] == ["one"]
    assert [hit.chunk.chunk_id for hit in clone.search("q")] == ["one", "two"]


def test_dimension_and_vector_value_limits_fail_before_commit() -> None:
    provider = _RecordingProvider({"wide": (1.0, 1.0, 1.0), "one": (1.0, 0.0), "two": (0.0, 1.0)})
    dimension_limited = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(max_dimensions=2),
    )
    with pytest.raises(SemanticChunkIndexLimitError, match="max_dimensions"):
        dimension_limited.add((_chunk("wide", "wide"),))
    assert dimension_limited.search("anything") == ()

    value_limited = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(max_total_vector_values=3),
    )
    with pytest.raises(SemanticChunkIndexLimitError, match="max_total_vector_values"):
        value_limited.add((_chunk("one", "one"), _chunk("two", "two")))
    assert value_limited.search("anything") == ()


def test_query_dimension_mismatch_is_controlled() -> None:
    provider = _RecordingProvider({"one": (1.0, 0.0), "q": (1.0,)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("one", "one"),))

    with pytest.raises(SemanticDimensionError, match="dimension"):
        index.search("q")


def test_failed_replace_keeps_exact_previous_results() -> None:
    class FailingProvider(_RecordingProvider):
        fail = False

        def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
            if self.fail:
                raise RuntimeError("sentinel private document")
            return super().embed_documents(texts)

    provider = FailingProvider({"old": (1.0, 0.0), "q": (1.0, 0.0)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("old", "old"),))
    before = index.search("q")
    provider.fail = True

    with pytest.raises(SemanticEmbeddingError) as caught:
        index.replace_document("doc-1", (_chunk("new", "private replacement"),))

    provider.fail = False
    assert str(caught.value) == "document embedding failed"
    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is not None
    assert index.search("q") == before


def test_failed_later_embedding_batch_keeps_previous_state() -> None:
    class FailingSecondBatchProvider(_RecordingProvider):
        fail_on_call: int | None = None

        def embed_documents(
            self, texts: tuple[str, ...]
        ) -> tuple[EmbeddingVector, ...]:
            if self.fail_on_call == len(self.document_calls) + 1:
                self.document_calls.append(texts)
                raise RuntimeError("sentinel private batch")
            return super().embed_documents(texts)

    provider = FailingSecondBatchProvider(
        {
            "old": (1.0,),
            "new-0": (1.0,),
            "new-1": (1.0,),
            "new-2": (1.0,),
            "q": (1.0,),
        }
    )
    index = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(max_embedding_batch_size=2),
    )
    index.add((_chunk("old", "old"),))
    before = index.search("q")
    provider.fail_on_call = 3

    replacement = tuple(
        _chunk(f"new-{number}", f"new-{number}") for number in range(3)
    )
    with pytest.raises(SemanticEmbeddingError, match="document embedding failed"):
        index.replace_document("doc-1", replacement)

    provider.fail_on_call = None
    assert index.search("q") == before


def test_document_dimension_limit_stops_before_later_batches() -> None:
    provider = _RecordingProvider(
        {
            "wide": (1.0, 1.0, 1.0),
            "unrequested": (1.0,),
        }
    )
    index = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(
            max_dimensions=2,
            max_embedding_batch_size=1,
        ),
    )

    with pytest.raises(SemanticChunkIndexLimitError, match="max_dimensions"):
        index.add(
            (
                _chunk("wide", "wide"),
                _chunk("unrequested", "unrequested"),
            )
        )

    assert provider.document_calls == [("wide",)]
    assert index.search("anything") == ()


def test_document_vector_value_limit_uses_retained_candidate_and_stops_early() -> None:
    provider = _RecordingProvider(
        {
            "retained": (1.0, 0.0),
            "new": (0.0, 1.0),
            "unrequested": (1.0, 1.0),
            "q": (1.0, 0.0),
        }
    )
    index = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(
            max_total_vector_values=3,
            max_embedding_batch_size=1,
        ),
    )
    index.add((_chunk("retained", "retained"),))
    before = index.search("q")

    with pytest.raises(
        SemanticChunkIndexLimitError,
        match="max_total_vector_values",
    ):
        index.add(
            (
                _chunk("new", "new", "doc-2"),
                _chunk("unrequested", "unrequested", "doc-2"),
            )
        )

    assert provider.document_calls == [("retained",), ("new",)]
    assert index.search("q") == before


def test_replace_vector_value_limit_excludes_removed_document_vectors() -> None:
    provider = _RecordingProvider(
        {
            "old": (1.0, 0.0),
            "retained": (0.0, 1.0),
            "replacement": (1.0, 1.0),
        }
    )
    index = InMemorySemanticChunkIndex(
        embedding_provider=provider,
        limits=SemanticChunkIndexLimits(max_total_vector_values=4),
    )
    index.add(
        (
            _chunk("old", "old"),
            _chunk("retained", "retained", "doc-2"),
        )
    )

    index.replace_document(
        "doc-1",
        (_chunk("replacement", "replacement"),),
    )

    assert provider.document_calls == [
        ("old", "retained"),
        ("replacement",),
    ]


@pytest.mark.parametrize(
    "bad_result",
    [
        [],
        (),
        (object(),),
    ],
)
def test_invalid_provider_batch_output_is_controlled_and_atomic(bad_result: object) -> None:
    class HostileProvider(_RecordingProvider):
        hostile = False

        def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
            if self.hostile:
                return bad_result  # type: ignore[return-value]
            return super().embed_documents(texts)

    provider = HostileProvider({"old": (1.0,), "q": (1.0,)})
    index = InMemorySemanticChunkIndex(embedding_provider=provider)
    index.add((_chunk("old", "old"),))
    before = index.search("q")
    provider.hostile = True

    with pytest.raises(SemanticEmbeddingError, match="document embedding failed"):
        index.add((_chunk("new", "new", "doc-2"),))

    provider.hostile = False
    assert index.search("q") == before
