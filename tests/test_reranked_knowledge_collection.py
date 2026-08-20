from __future__ import annotations

from pathlib import Path

from nexusmind import (
    Document,
    InMemoryChunkIndex,
    KnowledgeCollection,
    KnowledgeSource,
    RerankedChunkIndex,
    SQLiteKnowledgeSnapshotStore,
    SearchHit,
)


class _Adapter:
    def source(self) -> KnowledgeSource:
        return KnowledgeSource(
            source_id="docs", source_type="fixture", display_name="Reranked fixture"
        )

    def load_documents(self) -> tuple[Document, ...]:
        return (
            Document(
                source_id="docs",
                logical_path="notes.md",
                content="bounded reranking preserves canonical provenance",
            ),
        )


class _IdentityReranker:
    def rerank(
        self, query: str, candidates: tuple[SearchHit, ...], *, limit: int
    ) -> tuple[SearchHit, ...]:
        return tuple(
            SearchHit(hit.chunk, float(limit - rank), hit.matched_terms)
            for rank, hit in enumerate(candidates[:limit])
        )


def _collection() -> KnowledgeCollection:
    return KnowledgeCollection(
        index_factory=lambda: RerankedChunkIndex(
            base_index_factory=InMemoryChunkIndex,
            reranker=_IdentityReranker(),
            candidate_depth=10,
        )
    )


def test_reranked_collection_preserves_provenance_and_restore_state() -> None:
    original = _collection()
    original.sync(_Adapter())
    snapshot = original.snapshot()

    result = original.search("canonical", limit=1)[0]

    assert result.source.source_id == "docs"
    assert result.document == snapshot.documents[0]
    assert result.hit.chunk.content == snapshot.documents[0].content
    restored = _collection()
    restored.restore(snapshot)
    assert restored.snapshot() == snapshot
    assert restored.search("canonical", limit=1)[0].document == snapshot.documents[0]


def test_sqlite_persists_only_canonical_state_for_reranked_collection(
    tmp_path: Path,
) -> None:
    original = _collection()
    original.sync(_Adapter())
    path = tmp_path / "knowledge.db"
    SQLiteKnowledgeSnapshotStore(path).save(original.snapshot())

    restarted = _collection()
    restarted.restore(SQLiteKnowledgeSnapshotStore(path).load())

    assert restarted.snapshot() == original.snapshot()
    assert restarted.search("bounded", limit=1)[0].document.logical_path == "notes.md"
