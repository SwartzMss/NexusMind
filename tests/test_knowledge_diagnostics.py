from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from nexusmind import (
    Chunk,
    Document,
    InMemoryChunkIndex,
    KnowledgeCollection,
    KnowledgeRetrievalCandidateDiagnostic,
    KnowledgeRetrievalDiagnostics,
    KnowledgeSearchResolutionError,
    KnowledgeSource,
    RetrievalCandidateDiagnostic,
    RetrievalDiagnostics,
    RetrievalStage,
    RerankedChunkIndex,
    SearchHit,
)


@dataclass
class _Adapter:
    source_id: str
    documents: tuple[Document, ...]

    def source(self) -> KnowledgeSource:
        return KnowledgeSource(
            source_id=self.source_id,
            source_type="test",
            display_name=self.source_id,
            metadata={"owner": "canonical"},
        )

    def load_documents(self) -> tuple[Document, ...]:
        return self.documents


class _OneChunker:
    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        if not document.content:
            return ()
        return (
            Chunk(
                document.document_id,
                f"chunk:{document.document_id}",
                document.content,
                0,
                len(document.content),
            ),
        )


@dataclass
class _IndexState:
    chunks: dict[str, Chunk]
    trace: object | None = None
    diagnose_calls: int = 0
    search_calls: int = 0
    fail: bool = False


class _DiagnosticIndex:
    def __init__(self, state: _IndexState) -> None:
        self.state = state

    def add(self, chunks: tuple[Chunk, ...]) -> None:
        for chunk in chunks:
            self.state.chunks[chunk.chunk_id] = chunk

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self.remove_document(document_id)
        self.add(chunks)

    def remove_document(self, document_id: str) -> None:
        self.state.chunks = {
            key: chunk
            for key, chunk in self.state.chunks.items()
            if chunk.document_id != document_id
        }

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        self.state.search_calls += 1
        if type(self.state.trace) is RetrievalDiagnostics:
            return self.state.trace.hits
        return ()

    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics:
        self.state.diagnose_calls += 1
        if self.state.fail:
            raise RuntimeError("private diagnostic failure")
        return self.state.trace  # type: ignore[return-value]

    def clone(self) -> _DiagnosticIndex:
        return _DiagnosticIndex(self.state)


class _SearchOnlyIndex(_DiagnosticIndex):
    diagnose = None  # type: ignore[assignment]

    def clone(self) -> _SearchOnlyIndex:
        return _SearchOnlyIndex(self.state)


def _document(source_id: str, path: str, content: str) -> Document:
    return Document(
        source_id=source_id,
        logical_path=path,
        content=content,
        metadata={"tag": "canonical"},
    )


def _setup(
    index_type: type[_DiagnosticIndex] = _DiagnosticIndex,
) -> tuple[KnowledgeCollection, _IndexState, Document, Document]:
    state = _IndexState({})
    collection = KnowledgeCollection(
        chunker=_OneChunker(), index_factory=lambda: index_type(state)
    )
    one = _document("docs", "one.txt", "alpha")
    two = _document("docs", "two.txt", "beta")
    collection.sync(_Adapter("docs", (one, two)))
    return collection, state, one, two


def _row(
    chunk: Chunk,
    *,
    stage: RetrievalStage = RetrievalStage.LEXICAL,
    rank: int = 1,
    score: float = 1.0,
    terms: tuple[str, ...] = (),
    selected: bool = True,
) -> RetrievalCandidateDiagnostic:
    return RetrievalCandidateDiagnostic(
        stage=stage,
        rank=rank,
        chunk=chunk,
        score=score,
        matched_terms=terms,
        selected=selected,
    )


def test_diagnostic_models_are_frozen_slotted_and_require_exact_tuples_and_types() -> None:
    collection, state, one, _ = _setup()
    chunk = state.chunks[f"chunk:{one.document_id}"]
    hit = SearchHit(chunk, 1.0)
    state.trace = RetrievalDiagnostics((hit,), (_row(chunk),))
    result = collection.search("alpha")[0]
    source = result.source
    candidate = KnowledgeRetrievalCandidateDiagnostic(source, one, _row(chunk))
    diagnostics = KnowledgeRetrievalDiagnostics("alpha", (result,), (candidate,))

    assert not hasattr(candidate, "__dict__")
    assert not hasattr(diagnostics, "__dict__")
    with pytest.raises(TypeError, match="source"):
        KnowledgeRetrievalCandidateDiagnostic(object(), one, _row(chunk))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document"):
        KnowledgeRetrievalCandidateDiagnostic(source, object(), _row(chunk))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="diagnostic"):
        KnowledgeRetrievalCandidateDiagnostic(source, one, object())  # type: ignore[arg-type]
    forged = _row(chunk)
    object.__setattr__(forged, "rank", True)
    with pytest.raises((TypeError, ValueError), match="rank"):
        KnowledgeRetrievalCandidateDiagnostic(source, one, forged)
    with pytest.raises(TypeError, match="query"):
        KnowledgeRetrievalDiagnostics(1, (), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="results"):
        KnowledgeRetrievalDiagnostics("q", [], ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KnowledgeSearchResult"):
        KnowledgeRetrievalDiagnostics("q", (object(),), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="candidates"):
        KnowledgeRetrievalDiagnostics("q", (), [])  # type: ignore[arg-type]


def test_candidate_provenance_requires_exact_source_and_document_types() -> None:
    class _SourceSubclass(KnowledgeSource):
        pass

    class _DocumentSubclass(Document):
        pass

    collection, state, one, _ = _setup()
    chunk = state.chunks[f"chunk:{one.document_id}"]
    source = KnowledgeSource(
        source_id="docs", source_type="test", display_name="Docs"
    )
    source_subclass = _SourceSubclass(
        source_id="docs", source_type="test", display_name="Docs"
    )
    document_subclass = _DocumentSubclass(
        source_id="docs", logical_path="one.txt", content="alpha"
    )

    with pytest.raises(TypeError, match="source"):
        KnowledgeRetrievalCandidateDiagnostic(source_subclass, one, _row(chunk))
    with pytest.raises(TypeError, match="document"):
        KnowledgeRetrievalCandidateDiagnostic(source, document_subclass, _row(chunk))


def test_diagnose_search_calls_backend_once_and_resolves_all_provenance() -> None:
    collection, state, one, two = _setup()
    first = state.chunks[f"chunk:{one.document_id}"]
    second = state.chunks[f"chunk:{two.document_id}"]
    hits = (SearchHit(second, 2.0, ("beta",)), SearchHit(first, 1.0, ("alpha",)))
    rows = (
        _row(first, rank=1, score=3.0, terms=("alpha",), selected=True),
        _row(second, rank=2, score=2.0, terms=("beta",), selected=True),
        _row(
            second,
            stage=RetrievalStage.RERANKER,
            rank=1,
            score=2.0,
            terms=("beta",),
        ),
        _row(
            first,
            stage=RetrievalStage.RERANKER,
            rank=2,
            score=1.0,
            terms=("alpha",),
        ),
    )
    state.trace = RetrievalDiagnostics(hits, rows)

    diagnostics = collection.diagnose_search("alpha beta", limit=2)

    assert diagnostics.query == "alpha beta"
    assert state.diagnose_calls == 1
    assert state.search_calls == 0
    assert tuple(result.hit for result in diagnostics.results) == hits
    assert [item.document.document_id for item in diagnostics.candidates] == [
        one.document_id,
        two.document_id,
        two.document_id,
        one.document_id,
    ]
    assert tuple(item.diagnostic for item in diagnostics.candidates) == rows
    assert all(item.source.source_id == "docs" for item in diagnostics.candidates)

    diagnostics.results[0].source.metadata["owner"] = "external"
    diagnostics.candidates[0].document.metadata["tag"] = "external"
    again = collection.diagnose_search("alpha beta", limit=2)
    assert again.results[0].source.metadata == {"owner": "canonical"}
    assert again.candidates[0].document.metadata == {"tag": "canonical"}


def test_diagnose_search_normalizes_subclasses_without_changing_ordinary_search() -> None:
    class _SourceSubclass(KnowledgeSource):
        pass

    class _DocumentSubclass(Document):
        pass

    class _SubclassAdapter(_Adapter):
        def source(self) -> KnowledgeSource:
            return _SourceSubclass(
                source_id=self.source_id,
                source_type="test",
                display_name="Subclass source",
                logical_location="memory://docs",
                metadata={"nested": {"owner": "canonical"}},
            )

    state = _IndexState({})
    collection = KnowledgeCollection(
        chunker=_OneChunker(), index_factory=lambda: _DiagnosticIndex(state)
    )
    document = _DocumentSubclass(
        source_id="docs",
        logical_path="subclass.txt",
        content="alpha",
        content_type="text/custom",
        metadata={"nested": {"tag": "canonical"}},
    )
    collection.sync(_SubclassAdapter("docs", (document,)))
    chunk = state.chunks[f"chunk:{document.document_id}"]
    hit = SearchHit(chunk, 1.0, ("alpha",))
    state.trace = RetrievalDiagnostics(
        (hit,), (_row(chunk, terms=("alpha",)),)
    )

    ordinary = collection.search("alpha")
    diagnostics = collection.diagnose_search("alpha")

    assert type(ordinary[0].source) is _SourceSubclass
    assert type(ordinary[0].document) is _DocumentSubclass
    assert type(diagnostics.results[0].source) is KnowledgeSource
    assert type(diagnostics.results[0].document) is Document
    assert type(diagnostics.candidates[0].source) is KnowledgeSource
    assert type(diagnostics.candidates[0].document) is Document
    assert diagnostics.results[0].document.document_id == document.document_id
    assert diagnostics.results[0].document.content_hash == document.content_hash
    assert diagnostics.results[0].document.content_type == "text/custom"
    assert diagnostics.results[0].document.metadata == {
        "nested": {"tag": "canonical"}
    }
    assert diagnostics.results[0].source.logical_location == "memory://docs"
    assert diagnostics.results[0].source.metadata == {
        "nested": {"owner": "canonical"}
    }


def test_diagnose_search_accepts_real_reranker_with_empty_final_output() -> None:
    class _EmptyReranker:
        def __init__(self) -> None:
            self.calls = 0

        def rerank(
            self,
            query: str,
            candidates: tuple[SearchHit, ...],
            *,
            limit: int,
        ) -> tuple[SearchHit, ...]:
            self.calls += 1
            assert candidates
            return ()

    reranker = _EmptyReranker()
    collection = KnowledgeCollection(
        chunker=_OneChunker(),
        index_factory=lambda: RerankedChunkIndex(
            base_index_factory=InMemoryChunkIndex,
            reranker=reranker,
            candidate_depth=2,
        ),
    )
    document = _document("docs", "empty-rerank.txt", "alpha")
    collection.sync(_Adapter("docs", (document,)))

    diagnostics = collection.diagnose_search("alpha", limit=1)

    assert diagnostics.results == ()
    assert diagnostics.candidates
    assert all(not item.diagnostic.selected for item in diagnostics.candidates)
    assert reranker.calls == 1


def test_diagnose_search_rejects_selected_row_when_final_output_is_empty() -> None:
    collection, state, one, _ = _setup()
    chunk = state.chunks[f"chunk:{one.document_id}"]
    state.trace = RetrievalDiagnostics((), (_row(chunk, selected=True),))

    with pytest.raises(KnowledgeSearchResolutionError, match="selection"):
        collection.diagnose_search("alpha")


def test_diagnose_search_rejects_valid_slice_with_ghost_chunk_id() -> None:
    collection, state, one, _ = _setup()
    canonical = state.chunks[f"chunk:{one.document_id}"]
    ghost = Chunk(
        canonical.document_id,
        "ghost-id",
        canonical.content,
        canonical.start_offset,
        canonical.end_offset,
    )
    state.trace = RetrievalDiagnostics(
        (SearchHit(ghost, 1.0),), (_row(ghost),)
    )

    with pytest.raises(KnowledgeSearchResolutionError, match="derived"):
        collection.diagnose_search("alpha")


def test_diagnose_search_accepts_real_builtin_trace_with_derived_chunks() -> None:
    collection = KnowledgeCollection(chunker=_OneChunker())
    document = _document("docs", "builtin.txt", "alpha")
    collection.sync(_Adapter("docs", (document,)))

    diagnostics = collection.diagnose_search("alpha")

    assert len(diagnostics.results) == 1
    assert len(diagnostics.candidates) == 1
    assert diagnostics.results[0].hit.chunk == diagnostics.candidates[0].diagnostic.chunk


def test_diagnose_search_rejects_nondeterministic_chunk_derivation() -> None:
    class _AlternatingChunker:
        def __init__(self) -> None:
            self.calls = 0

        def chunk(self, document: Document) -> tuple[Chunk, ...]:
            self.calls += 1
            return (
                Chunk(
                    document.document_id,
                    f"chunk-{1 if self.calls % 2 else 2}",
                    document.content,
                    0,
                    len(document.content),
                ),
            )

    chunker = _AlternatingChunker()
    collection = KnowledgeCollection(chunker=chunker)
    document = _document("docs", "alternating.txt", "alpha")
    collection.sync(_Adapter("docs", (document,)))

    with pytest.raises(KnowledgeSearchResolutionError, match="deterministic"):
        collection.diagnose_search("alpha")

    assert chunker.calls == 3


def test_diagnose_search_looks_up_once_and_detaches_once_per_output() -> None:
    class _DeepcopyProbe:
        def __init__(self, calls: list[int]) -> None:
            self.calls = calls

        def __deepcopy__(self, memo: dict[int, object]) -> _DeepcopyProbe:
            self.calls[0] += 1
            return _DeepcopyProbe(self.calls)

    class _CountingCollection(KnowledgeCollection):
        def __init__(self, **kwargs: object) -> None:
            self.lookup_calls = 0
            super().__init__(**kwargs)  # type: ignore[arg-type]

        def _find_document(self, document_id: str) -> Document | None:
            self.lookup_calls += 1
            return super()._find_document(document_id)

    class _CountingChunker(_OneChunker):
        def __init__(self) -> None:
            self.calls = 0

        def chunk(self, document: Document) -> tuple[Chunk, ...]:
            self.calls += 1
            return super().chunk(document)

    class _ProbeAdapter(_Adapter):
        def __init__(
            self,
            source_id: str,
            documents: tuple[Document, ...],
            source_probe: _DeepcopyProbe,
        ) -> None:
            super().__init__(source_id, documents)
            self.source_probe = source_probe

        def source(self) -> KnowledgeSource:
            return KnowledgeSource(
                source_id=self.source_id,
                source_type="test",
                display_name="Probe source",
                metadata={"probe": self.source_probe},
            )

    source_copies = [0]
    document_copies = [0]
    state = _IndexState({})
    chunker = _CountingChunker()
    collection = _CountingCollection(
        chunker=chunker, index_factory=lambda: _DiagnosticIndex(state)
    )
    document = Document(
        source_id="docs",
        logical_path="probe.txt",
        content="alpha",
        metadata={"probe": _DeepcopyProbe(document_copies)},
    )
    collection.sync(
        _ProbeAdapter(
            "docs",
            (document,),
            _DeepcopyProbe(source_copies),
        )
    )
    chunk = state.chunks[f"chunk:{document.document_id}"]
    state.trace = RetrievalDiagnostics(
        (SearchHit(chunk, 2.0),),
        (
            _row(chunk, score=1.0),
            _row(chunk, stage=RetrievalStage.FUSION, score=2.0),
        ),
    )
    source_copies[0] = 0
    document_copies[0] = 0
    collection.lookup_calls = 0
    chunker.calls = 0

    diagnostics = collection.diagnose_search("alpha")

    assert len(diagnostics.results) == 1
    assert len(diagnostics.candidates) == 2
    assert collection.lookup_calls == 1
    assert chunker.calls == 2
    assert source_copies[0] == 3
    assert document_copies[0] == 5  # two derivations plus three detached outputs


def test_diagnose_search_accepts_hybrid_terms_and_repeated_reranker_blocks() -> None:
    collection, state, one, two = _setup()
    first = state.chunks[f"chunk:{one.document_id}"]
    second = state.chunks[f"chunk:{two.document_id}"]
    hits = (SearchHit(first, 9.0, ("fusion",)),)
    rows = (
        _row(first, rank=1, score=3.0, terms=("lexical",), selected=True),
        _row(second, rank=2, score=2.0, terms=("other",), selected=False),
        _row(
            first,
            stage=RetrievalStage.SEMANTIC,
            rank=1,
            score=0.8,
            terms=(),
            selected=True,
        ),
        _row(
            first,
            stage=RetrievalStage.FUSION,
            rank=1,
            score=0.6,
            terms=("fusion",),
            selected=True,
        ),
        _row(
            first,
            stage=RetrievalStage.RERANKER,
            rank=1,
            score=0.7,
            terms=("fusion",),
            selected=True,
        ),
        _row(
            first,
            stage=RetrievalStage.RERANKER,
            rank=1,
            score=9.0,
            terms=("fusion",),
            selected=True,
        ),
    )
    state.trace = RetrievalDiagnostics(hits, rows)

    result = collection.diagnose_search("q", limit=1)

    assert result.results[0].hit == hits[0]
    assert tuple(item.diagnostic for item in result.candidates) == rows


@pytest.mark.parametrize("case", ["early_final_unselected", "terminal_extra"])
def test_diagnose_search_requires_per_row_selection_and_exact_terminal_block(
    case: str,
) -> None:
    collection, state, one, two = _setup()
    first = state.chunks[f"chunk:{one.document_id}"]
    second = state.chunks[f"chunk:{two.document_id}"]
    hit = SearchHit(first, 2.0)
    if case == "early_final_unselected":
        rows = (
            _row(first, selected=False),
            _row(
                first,
                stage=RetrievalStage.FUSION,
                score=2.0,
                selected=True,
            ),
        )
    else:
        rows = (
            _row(first, score=2.0, selected=True),
            _row(second, rank=2, score=1.0, selected=False),
        )
    state.trace = RetrievalDiagnostics((hit,), rows)

    with pytest.raises(KnowledgeSearchResolutionError):
        collection.diagnose_search("q")


@pytest.mark.parametrize(
    "score",
    [True, 1, math.nan, math.inf],
)
def test_diagnose_search_requires_exact_finite_float_final_scores(score: object) -> None:
    collection, state, one, _ = _setup()
    chunk = state.chunks[f"chunk:{one.document_id}"]
    state.trace = RetrievalDiagnostics(
        (SearchHit(chunk, score),),  # type: ignore[arg-type]
        (_row(chunk, score=1.0),),
    )

    with pytest.raises(KnowledgeSearchResolutionError, match="final hit"):
        collection.diagnose_search("q")


def test_diagnose_search_requires_exact_final_terms_and_nonempty_exact_chunks() -> None:
    class _Text(str):
        pass

    collection, state, one, _ = _setup()
    canonical = state.chunks[f"chunk:{one.document_id}"]
    malformed_values = (
        SearchHit(canonical, 1.0, (_Text("alpha"),)),
        SearchHit(
            Chunk(canonical.document_id, canonical.chunk_id, "", 0, 0),
            1.0,
        ),
        SearchHit(
            Chunk(canonical.document_id, canonical.chunk_id, _Text("alpha"), 0, 5),
            1.0,
        ),
    )
    for hit in malformed_values:
        row = _row(
            hit.chunk,
            score=1.0,
            terms=tuple(str(term) for term in hit.matched_terms),
        )
        state.trace = RetrievalDiagnostics((hit,), (row,))
        with pytest.raises(KnowledgeSearchResolutionError):
            collection.diagnose_search("q")


def test_diagnose_search_requires_exact_hit_chunk_and_row_shapes() -> None:
    class _HitSubclass(SearchHit):
        pass

    class _ChunkSubclass(Chunk):
        pass

    collection, state, one, _ = _setup()
    canonical = state.chunks[f"chunk:{one.document_id}"]
    chunk_subclass = _ChunkSubclass(
        canonical.document_id,
        canonical.chunk_id,
        canonical.content,
        canonical.start_offset,
        canonical.end_offset,
    )
    hit_subclass_trace = RetrievalDiagnostics(
        (SearchHit(canonical, 1.0),), (_row(canonical),)
    )
    object.__setattr__(
        hit_subclass_trace, "hits", (_HitSubclass(canonical, 1.0),)
    )
    malformed_traces = (
        hit_subclass_trace,
        RetrievalDiagnostics(
            (SearchHit(chunk_subclass, 1.0),),
            (_row(chunk_subclass),),
        ),
        RetrievalDiagnostics(
            (SearchHit(canonical, 1.0),),
            (
                _row(chunk_subclass),
                _row(canonical, stage=RetrievalStage.FUSION),
            ),
        ),
    )
    for trace in malformed_traces:
        state.trace = trace
        with pytest.raises(KnowledgeSearchResolutionError):
            collection.diagnose_search("q")

    forged_row = _row(canonical)
    object.__setattr__(forged_row, "score", True)
    state.trace = RetrievalDiagnostics((SearchHit(canonical, 1.0),), (forged_row,))
    with pytest.raises(KnowledgeSearchResolutionError, match="invalid"):
        collection.diagnose_search("q")


def test_ordinary_search_still_accepts_equal_string_subclass_chunk_content() -> None:
    class _Text(str):
        pass

    collection, state, one, _ = _setup(_SearchOnlyIndex)
    canonical = state.chunks[f"chunk:{one.document_id}"]
    compatible = Chunk(
        canonical.document_id,
        canonical.chunk_id,
        _Text(canonical.content),
        canonical.start_offset,
        canonical.end_offset,
    )
    state.trace = RetrievalDiagnostics((SearchHit(compatible, 1.0),), ())

    result = collection.search("alpha")

    assert result[0].document == one
    assert result[0].hit.chunk.content == "alpha"


def test_diagnostics_unsupported_is_controlled_and_ordinary_search_still_works() -> None:
    collection, state, one, _ = _setup(_SearchOnlyIndex)
    chunk = state.chunks[f"chunk:{one.document_id}"]
    state.trace = RetrievalDiagnostics((SearchHit(chunk, 1.0),), (_row(chunk),))

    assert collection.search("alpha")[0].document == one
    with pytest.raises(KnowledgeSearchResolutionError, match="does not support diagnostics"):
        collection.diagnose_search("alpha")


def test_diagnose_failure_is_redacted_and_returns_no_partial_value() -> None:
    collection, state, _, _ = _setup()
    state.fail = True

    with pytest.raises(KnowledgeSearchResolutionError, match="index diagnose failed") as caught:
        collection.diagnose_search("secret query")
    assert "private" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert state.diagnose_calls == 1


def _valid_single_trace(
    state: _IndexState, document: Document
) -> tuple[SearchHit, RetrievalCandidateDiagnostic]:
    chunk = state.chunks[f"chunk:{document.document_id}"]
    return SearchHit(chunk, 1.0), _row(chunk)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_trace_type",
        "ghost",
        "stale",
        "bad_offset",
        "bad_document_id",
        "bad_chunk_id",
        "content_conflict",
        "duplicate_hit",
        "duplicate_block",
        "bad_rank",
        "bad_stage_order",
        "selected_mismatch",
        "terminal_order",
        "reranker_lineage",
    ],
)
def test_diagnose_search_rejects_malformed_or_incoherent_complete_traces(case: str) -> None:
    collection, state, one, two = _setup()
    hit, row = _valid_single_trace(state, one)
    other_chunk = state.chunks[f"chunk:{two.document_id}"]
    trace: object = RetrievalDiagnostics((hit,), (row,))

    if case == "wrong_trace_type":
        trace = object()
    elif case == "ghost":
        ghost = Chunk("missing", "ghost", "alpha", 0, 5)
        trace = RetrievalDiagnostics((SearchHit(ghost, 1.0),), (_row(ghost),))
    elif case == "stale":
        stale = Chunk(hit.chunk.document_id, hit.chunk.chunk_id, "old", 0, 3)
        trace = RetrievalDiagnostics((SearchHit(stale, 1.0),), (_row(stale),))
    elif case == "bad_offset":
        bad = Chunk(hit.chunk.document_id, hit.chunk.chunk_id, "alpha", -1, 5)
        trace = RetrievalDiagnostics((SearchHit(bad, 1.0),), (_row(bad),))
    elif case == "bad_document_id":
        bad = Chunk(hit.chunk.document_id, hit.chunk.chunk_id, "alpha", 0, 5)
        object.__setattr__(bad, "document_id", [])
        trace = RetrievalDiagnostics((SearchHit(bad, 1.0),), (_row(bad),))
    elif case == "bad_chunk_id":
        bad = Chunk(hit.chunk.document_id, hit.chunk.chunk_id, "alpha", 0, 5)
        object.__setattr__(bad, "chunk_id", [])
        trace = RetrievalDiagnostics((SearchHit(bad, 1.0),), (_row(bad),))
    elif case == "content_conflict":
        conflict = Chunk(hit.chunk.document_id, hit.chunk.chunk_id, "wrong", 0, 5)
        trace = RetrievalDiagnostics((SearchHit(conflict, 1.0),), (_row(conflict),))
    elif case == "duplicate_hit":
        trace = RetrievalDiagnostics((hit, hit), (row,))
    elif case == "duplicate_block":
        trace = RetrievalDiagnostics((hit,), (row, _row(hit.chunk, rank=2)))
    elif case == "bad_rank":
        trace = RetrievalDiagnostics((hit,), (_row(hit.chunk, rank=2),))
    elif case == "bad_stage_order":
        trace = RetrievalDiagnostics(
            (hit,),
            (
                _row(hit.chunk, stage=RetrievalStage.SEMANTIC),
                _row(hit.chunk, stage=RetrievalStage.LEXICAL),
            ),
        )
    elif case == "selected_mismatch":
        trace = RetrievalDiagnostics((hit,), (_row(hit.chunk, selected=False),))
    elif case == "terminal_order":
        other_hit = SearchHit(other_chunk, 0.5)
        trace = RetrievalDiagnostics(
            (hit, other_hit),
            (_row(other_chunk, rank=1, score=0.5), _row(hit.chunk, rank=2)),
        )
    elif case == "reranker_lineage":
        trace = RetrievalDiagnostics(
            (SearchHit(other_chunk, 1.0),),
            (
                _row(hit.chunk, selected=False),
                _row(other_chunk, stage=RetrievalStage.RERANKER),
            ),
        )
    state.trace = trace

    with pytest.raises(KnowledgeSearchResolutionError):
        collection.diagnose_search("q")
    assert state.diagnose_calls == 1


def test_diagnose_search_revalidates_forged_candidate_public_fields() -> None:
    collection, state, one, _ = _setup()
    hit, row = _valid_single_trace(state, one)
    object.__setattr__(row, "rank", True)
    state.trace = RetrievalDiagnostics((hit,), (row,))

    with pytest.raises(KnowledgeSearchResolutionError, match="invalid"):
        collection.diagnose_search("q")
