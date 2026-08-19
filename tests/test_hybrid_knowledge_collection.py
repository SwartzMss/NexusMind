from __future__ import annotations

from pathlib import Path

import pytest

from nexusmind import (
    Document,
    EmbeddingVector,
    HybridChunkIndex,
    HybridChunkIndexError,
    InMemoryChunkIndex,
    InMemorySemanticChunkIndex,
    KnowledgeCollection,
    KnowledgeSource,
    SQLiteKnowledgeSnapshotStore,
)


class _Adapter:
    def __init__(self, document: Document) -> None:
        self.document = document

    def source(self) -> KnowledgeSource:
        return KnowledgeSource(
            source_id=self.document.source_id,
            source_type="fixture",
            display_name="Hybrid fixture",
        )

    def load_documents(self) -> tuple[Document, ...]:
        return (self.document,)


class _Provider:
    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []
        self.fail = False

    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        self.document_calls.append(texts)
        if self.fail:
            raise RuntimeError("private provider failure")
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector((1.0, 0.0))


def _document(content: str) -> Document:
    return Document(source_id="docs", logical_path="a.md", content=content)


def _collection(provider: _Provider) -> KnowledgeCollection:
    return KnowledgeCollection(
        index_factory=lambda: HybridChunkIndex(
            lexical_index_factory=InMemoryChunkIndex,
            semantic_index_factory=lambda: InMemorySemanticChunkIndex(
                embedding_provider=provider
            ),
        )
    )


def test_hybrid_collection_sync_search_and_restore_preserve_provenance() -> None:
    provider = _Provider()
    collection = _collection(provider)
    document = _document("exact hybrid evidence")
    collection.sync(_Adapter(document))
    snapshot = collection.snapshot()

    result = collection.search("exact", limit=1)[0]

    assert result.document == document
    assert result.source.source_id == "docs"
    assert result.hit.matched_terms == ("exact",)
    assert result.hit.score > 0.0

    restored_provider = _Provider()
    restored = _collection(restored_provider)
    restored.restore(snapshot)

    assert restored.snapshot() == snapshot
    assert restored.search("exact", limit=1)[0].document == document
    assert restored_provider.document_calls


def test_failed_hybrid_sync_preserves_canonical_and_search_state() -> None:
    provider = _Provider()
    collection = _collection(provider)
    collection.sync(_Adapter(_document("old searchable evidence")))
    before_snapshot = collection.snapshot()
    before_results = collection.search("old")
    provider.fail = True

    with pytest.raises(HybridChunkIndexError, match="mutation failed"):
        collection.sync(_Adapter(_document("new private evidence")))

    provider.fail = False
    assert collection.snapshot() == before_snapshot
    assert collection.search("old") == before_results
    assert collection.search("new")[0].document.content == "old searchable evidence"


def test_sqlite_restart_rebuilds_hybrid_children_from_canonical_state(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    original = _collection(provider)
    original.sync(_Adapter(_document("durable hybrid evidence")))
    path = tmp_path / "knowledge.db"
    SQLiteKnowledgeSnapshotStore(path).save(original.snapshot())

    restarted_provider = _Provider()
    restarted = _collection(restarted_provider)
    restarted.restore(SQLiteKnowledgeSnapshotStore(path).load())

    result = restarted.search("durable", limit=1)[0]
    assert result.document.content == "durable hybrid evidence"
    assert result.hit.matched_terms == ("durable",)
    assert restarted_provider.document_calls
