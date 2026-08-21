from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexusmind import (
    Chunk,
    Document,
    KnowledgeCollection,
    KnowledgeRetrievalCandidateDiagnostic,
    KnowledgeRetrievalDiagnostics,
    KnowledgeSearchResolutionError,
    KnowledgeSource,
    RetrievalCandidateDiagnostic,
    RetrievalDiagnostics,
    RetrievalStage,
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
