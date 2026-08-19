from __future__ import annotations

import math

import pytest

from nexusmind import (
    Chunk,
    ChunkIdentityConflictError,
    ChunkIndex,
    ChunkIndexLimitError,
    ChunkIndexLimits,
    DocumentReplacementError,
    InMemoryChunkIndex,
    LexicalAnalyzer,
    SearchHit,
    WhitespaceLexicalAnalyzer,
)


def _chunk(chunk_id: str, content: str, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


def test_retrieval_contracts_are_available_from_package_root() -> None:
    index: ChunkIndex = InMemoryChunkIndex()
    index.add((_chunk("chunk-1", "checkpoint resume"),))

    hit = index.search("checkpoint")[0]
    assert hit.chunk == _chunk("chunk-1", "checkpoint resume")
    assert hit.score == pytest.approx(math.log(4 / 3))
    assert hit.matched_terms == ("checkpoint",)


def test_empty_index_non_match_and_blank_query_return_no_hits() -> None:
    index = InMemoryChunkIndex()
    assert index.search("anything") == ()
    index.add((_chunk("chunk-1", "checkpoint"),))
    assert index.search("missing") == ()
    assert index.search(" \t\n") == ()


def test_bm25_single_term_formula_and_score_type() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("only", "term"),))

    hit = index.search("term")[0]

    assert type(hit.score) is float
    assert math.isfinite(hit.score)
    assert hit.score == pytest.approx(math.log(4 / 3))


def test_matching_does_not_match_substrings() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("substring", "concatenate"), _chunk("token", "cat")))

    assert [hit.chunk.chunk_id for hit in index.search("cat")] == ["token"]


def test_default_analyzer_matches_across_punctuation_and_han_bigrams() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("mixed", "Android Binder，提供安全检索"),))

    hit = index.search("Binder 安全检索")[0]

    assert hit.chunk.chunk_id == "mixed"
    assert hit.matched_terms == ("binder", "安全", "全检", "检索")


def test_explicit_whitespace_analyzer_preserves_legacy_matching() -> None:
    index = InMemoryChunkIndex(analyzer=WhitespaceLexicalAnalyzer())
    index.add((_chunk("punctuated", "alpha,beta"), _chunk("spaced", "alpha beta")))

    assert [hit.chunk.chunk_id for hit in index.search("beta")] == ["spaced"]


def test_same_configured_analyzer_is_used_once_per_corpus_text_and_query() -> None:
    class MappingAnalyzer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __bool__(self) -> bool:
            return False

        def analyze(self, text: str) -> tuple[str, ...]:
            self.calls.append(text)
            return {
                "source one": ("shared", "shared"),
                "source two": ("other",),
                "lookup": ("shared", "shared"),
            }[text]

    analyzer: LexicalAnalyzer = MappingAnalyzer()
    index = InMemoryChunkIndex(analyzer=analyzer)
    index.add((_chunk("one", "source one"), _chunk("two", "source two")))

    hit = index.search("lookup")[0]

    assert hit.chunk.chunk_id == "one"
    assert hit.matched_terms == ("shared",)
    assert analyzer.calls == ["source one", "source two", "lookup"]


def test_explicit_analyzer_requires_callable_analyze() -> None:
    with pytest.raises(TypeError, match="callable analyze"):
        InMemoryChunkIndex(analyzer=object())  # type: ignore[arg-type]


class _StrSubclass(str):
    pass


@pytest.mark.parametrize(
    "malformed",
    [
        ["term"],
        "term",
        ("",),
        (1,),
        (_StrSubclass("term"),),
    ],
)
@pytest.mark.parametrize("path", ["corpus", "query"])
def test_malformed_analyzer_output_is_rejected_at_each_use_path(
    malformed: object, path: str
) -> None:
    class MalformedAnalyzer:
        def analyze(self, text: str) -> tuple[str, ...]:
            return malformed  # type: ignore[return-value]

    index = InMemoryChunkIndex(analyzer=MalformedAnalyzer())

    with pytest.raises(TypeError, match="analyzer must return"):
        if path == "corpus":
            index.add((_chunk("one", "source"),))
        else:
            index.search("query")


def test_repeated_term_frequency_increases_score_at_equal_length() -> None:
    index = InMemoryChunkIndex()
    index.add(
        (
            _chunk("repeated", "term term filler"),
            _chunk("single", "term filler filler"),
        )
    )

    hits = {hit.chunk.chunk_id: hit for hit in index.search("term")}

    assert hits["repeated"].score > hits["single"].score


def test_rarer_term_has_higher_idf_contribution() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("one", "rare common"), _chunk("two", "other common")))

    rare_score = index.search("rare")[0].score
    common_score = index.search("common")[0].score

    assert rare_score > common_score


def test_shorter_chunk_scores_higher_at_equal_term_frequency() -> None:
    index = InMemoryChunkIndex()
    index.add(
        (
            _chunk("short", "term filler"),
            _chunk("long", "term filler filler filler"),
        )
    )

    hits = {hit.chunk.chunk_id: hit for hit in index.search("term")}

    assert hits["short"].score > hits["long"].score


def test_duplicate_query_terms_do_not_multiply_bm25_weight() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("one", "term term"),))

    assert index.search("TERM term term") == index.search("term")


def test_multi_term_scoring_casefold_and_duplicate_terms() -> None:
    index = InMemoryChunkIndex()
    index.add(
        (
            _chunk("chunk-b", "Resume from CHECKPOINT"),
            _chunk("chunk-a", "checkpoint only"),
        )
    )

    hits = index.search("CHECKPOINT checkpoint resume")

    assert [(hit.chunk.chunk_id, hit.matched_terms) for hit in hits] == [
        ("chunk-b", ("checkpoint", "resume")),
        ("chunk-a", ("checkpoint",)),
    ]
    assert all(type(hit.score) is float for hit in hits)
    assert hits[0].score > hits[1].score


def test_ties_use_chunk_id_and_limit_is_applied() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("chunk-z", "term"), _chunk("chunk-a", "term")))

    assert [hit.chunk.chunk_id for hit in index.search("term", limit=1)] == ["chunk-a"]


def test_unicode_casefold_is_deterministic() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("chunk-1", "STRASSE 中文"), _chunk("chunk-2", "Straße")))

    first = index.search("straße 中文")

    assert [hit.chunk.chunk_id for hit in first] == ["chunk-1", "chunk-2"]
    assert first[0].score > first[1].score
    assert index.search("straße 中文") == first


def test_add_updates_bm25_corpus_statistics() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("term", "term", "doc-1"),))
    score_before = index.search("term")[0].score

    index.add((_chunk("other", "other", "doc-2"),))

    assert index.search("term")[0].score > score_before


def test_remove_document_updates_bm25_corpus_statistics() -> None:
    index = InMemoryChunkIndex()
    index.add(
        (
            _chunk("term", "term", "doc-1"),
            _chunk("other", "other", "doc-2"),
        )
    )
    score_before = index.search("term")[0].score

    index.remove_document("doc-2")

    score_after = index.search("term")[0].score
    assert score_after < score_before
    assert score_after == pytest.approx(math.log(4 / 3))


def test_exact_duplicate_add_is_idempotent() -> None:
    chunk = _chunk("chunk-1", "term")
    index = InMemoryChunkIndex()
    index.add((chunk, chunk))
    index.add((chunk,))

    assert len(index.search("term")) == 1


def test_clone_has_independent_mutable_state() -> None:
    original = InMemoryChunkIndex()
    original.add((_chunk("chunk-1", "original"),))

    clone = original.clone()
    assert clone.search("original") == original.search("original")
    clone.replace_document("doc-1", (_chunk("chunk-2", "changed"),))

    assert original.search("original")
    assert original.search("changed") == ()
    assert clone.search("original") == ()
    assert clone.search("changed")


def test_conflicting_chunk_id_is_rejected_without_mutation() -> None:
    original = _chunk("chunk-1", "original")
    index = InMemoryChunkIndex()
    index.add((original,))
    before = index.search("original changed")

    with pytest.raises(ChunkIdentityConflictError):
        index.add((_chunk("chunk-1", "changed"),))

    assert index.search("original changed") == before
    assert index.search("original")[0].chunk == original
    assert index.search("changed") == ()


def test_remove_document_does_not_affect_another_document() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("chunk-1", "shared", "doc-1"), _chunk("chunk-2", "shared", "doc-2")))

    index.remove_document("doc-1")

    assert [hit.chunk.document_id for hit in index.search("shared")] == ["doc-2"]


def test_replace_document_removes_stale_chunks_and_empty_replacement_removes_all() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("old-a", "stale", "doc-1"), _chunk("old-b", "stale", "doc-1")))

    index.replace_document("doc-1", (_chunk("new-a", "fresh", "doc-1"),))

    assert index.search("stale") == ()
    assert [hit.chunk.chunk_id for hit in index.search("fresh")] == ["new-a"]
    index.replace_document("doc-1", ())
    assert index.search("fresh") == ()


def test_replace_document_rebuilds_statistics_equivalent_to_fresh_index() -> None:
    replacement = _chunk("replacement", "other", "doc-1")
    retained = _chunk("retained", "term", "doc-2")
    index = InMemoryChunkIndex()
    index.add((_chunk("old", "term term term term", "doc-1"), retained))

    index.replace_document("doc-1", (replacement,))

    fresh = InMemoryChunkIndex()
    fresh.add((replacement, retained))
    assert index.search("term other") == fresh.search("term other")


def test_failed_replacement_is_atomic() -> None:
    old = _chunk("old", "old searchable", "doc-1")
    limits = ChunkIndexLimits(max_total_chars=len(old.content), max_chunks=5, max_chunks_per_document=5)
    index = InMemoryChunkIndex(limits=limits)
    index.add((old,))
    before = index.search("old long")

    with pytest.raises(ChunkIndexLimitError):
        index.replace_document("doc-1", (_chunk("new", "content is too long", "doc-1"),))

    assert index.search("old long") == before
    assert index.search("old")[0].chunk == old
    assert index.search("long") == ()


def test_replacement_rejects_reused_chunk_id_with_different_data() -> None:
    old = _chunk("same-id", "old", "doc-1")
    changed = _chunk("same-id", "new", "doc-1")
    index = InMemoryChunkIndex()
    index.add((old,))

    with pytest.raises(ChunkIdentityConflictError):
        index.replace_document("doc-1", (changed,))

    assert index.search("old")[0].chunk == old
    assert index.search("new") == ()


def test_replacement_rejects_mixed_documents_and_duplicate_ids() -> None:
    index = InMemoryChunkIndex()
    with pytest.raises(DocumentReplacementError, match="belong"):
        index.replace_document("doc-1", (_chunk("a", "a", "doc-1"), _chunk("b", "b", "doc-2")))
    duplicate = _chunk("a", "a", "doc-1")
    with pytest.raises(DocumentReplacementError, match="duplicate"):
        index.replace_document("doc-1", (duplicate, duplicate))


def test_query_and_result_limits_are_enforced() -> None:
    limits = ChunkIndexLimits(max_query_chars=10, max_query_terms=2, max_results=1)
    index = InMemoryChunkIndex(limits=limits)
    with pytest.raises(ChunkIndexLimitError, match="max_query_chars"):
        index.search("12345678901")
    with pytest.raises(ChunkIndexLimitError, match="max_query_terms"):
        index.search("a b c")
    with pytest.raises(ChunkIndexLimitError, match="max_results"):
        index.search("a", limit=2)


def test_query_term_limit_counts_analyzed_repetitions_before_deduplication() -> None:
    index = InMemoryChunkIndex(limits=ChunkIndexLimits(max_query_terms=2))

    with pytest.raises(ChunkIndexLimitError, match="max_query_terms"):
        index.search("term term term")


def test_query_term_limit_counts_default_han_bigram_amplification() -> None:
    index = InMemoryChunkIndex(limits=ChunkIndexLimits(max_query_terms=2))

    with pytest.raises(ChunkIndexLimitError, match="max_query_terms"):
        index.search("安全检索")


def test_index_count_content_and_per_document_limits_are_atomic() -> None:
    index = InMemoryChunkIndex(
        limits=ChunkIndexLimits(max_chunks=2, max_total_chars=4, max_chunks_per_document=1)
    )
    index.add((_chunk("a", "ab", "doc-1"),))
    before_failures = index.search("ab")
    with pytest.raises(ChunkIndexLimitError, match="max_chunks_per_document"):
        index.add((_chunk("b", "c", "doc-1"),))
    with pytest.raises(ChunkIndexLimitError, match="max_total_chars"):
        index.add((_chunk("b", "cde", "doc-2"),))
    assert index.search("ab") == before_failures
    index.add((_chunk("b", "cd", "doc-2"),))
    with pytest.raises(ChunkIndexLimitError, match="max_chunks"):
        index.add((_chunk("c", "", "doc-3"),))
    assert [hit.chunk.chunk_id for hit in index.search("ab cd", limit=1)] == ["a"]


@pytest.mark.parametrize("field", ChunkIndexLimits.__dataclass_fields__)
def test_limits_require_positive_plain_integers(field: str) -> None:
    with pytest.raises(TypeError):
        ChunkIndexLimits(**{field: True})
    with pytest.raises(ValueError):
        ChunkIndexLimits(**{field: 0})


def test_invalid_search_limit_is_controlled() -> None:
    index = InMemoryChunkIndex()
    with pytest.raises(TypeError):
        index.search("term", limit=True)
    with pytest.raises(ValueError):
        index.search("term", limit=0)
