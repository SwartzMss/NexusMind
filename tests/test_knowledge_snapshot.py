from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexusmind import (
    Chunk,
    ChunkIdentityConflictError,
    ChunkIndexLimitError,
    ChunkIndexLimits,
    Document,
    EmbeddingVector,
    InMemoryChunkIndex,
    InMemorySemanticChunkIndex,
    KnowledgeCollection,
    KnowledgeCollectionLimitError,
    KnowledgeCollectionLimits,
    KnowledgeRestoreError,
    KnowledgeRestoreResult,
    KnowledgeSnapshot,
    KnowledgeSnapshotError,
    KnowledgeSource,
    TextChunker,
    WhitespaceLexicalAnalyzer,
)


class _SnapshotSemanticProvider:
    def __init__(self, document_vector: tuple[float, ...]) -> None:
        self.document_vector = document_vector
        self.document_calls: list[tuple[str, ...]] = []

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        self.document_calls.append(texts)
        return tuple(EmbeddingVector(self.document_vector) for _ in texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector(self.document_vector)


def _document(source_id: str, logical_path: str, content: str) -> Document:
    return Document(source_id=source_id, logical_path=logical_path, content=content)


def _source(source_id: str) -> KnowledgeSource:
    return KnowledgeSource(source_id=source_id, source_type="fake", display_name=source_id)


@dataclass
class FakeAdapter:
    source_id: str
    documents: tuple[Document, ...]

    def source(self) -> KnowledgeSource:
        return _source(self.source_id)

    def load_documents(self) -> tuple[Document, ...]:
        return self.documents


class RecordingChunker:
    def __init__(self, *, fail_content: str | None = None) -> None:
        self.document_ids: list[str] = []
        self.fail_content = fail_content
        self.delegate = TextChunker(chunk_size=100, overlap=0)

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        self.document_ids.append(document.document_id)
        if document.content == self.fail_content:
            raise RuntimeError("chunking failed")
        return self.delegate.chunk(document)


class ConflictingChunker:
    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        return (
            Chunk(
                document_id=document.document_id,
                chunk_id="reused-id",
                content=document.content,
                start_offset=0,
                end_offset=len(document.content),
            ),
        )


class MutatingMetadataChunker:
    def __init__(self) -> None:
        self.delegate = TextChunker(chunk_size=100, overlap=0)

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        document.metadata["tag"] = "changed"
        return self.delegate.chunk(document)


def test_empty_collection_snapshot_is_stable_and_restorable() -> None:
    collection = KnowledgeCollection()

    snapshot = collection.snapshot()

    assert snapshot == KnowledgeSnapshot(sources=(), documents=())
    assert collection.snapshot() == snapshot
    assert collection.restore(snapshot) == KnowledgeRestoreResult(0, 0, 0)


def test_semantic_restore_reembeds_with_current_runtime_provider() -> None:
    snapshot = KnowledgeSnapshot(
        sources=(_source("docs"),),
        documents=(_document("docs", "semantic.md", "canonical semantic content"),),
    )
    provider = _SnapshotSemanticProvider((0.0, 1.0))
    collection = KnowledgeCollection(
        index_factory=lambda: InMemorySemanticChunkIndex(embedding_provider=provider)
    )

    result = collection.restore(snapshot)

    assert result == KnowledgeRestoreResult(1, 1, 1)
    assert provider.document_calls == [("canonical semantic content",)]
    hit = collection.search("concept")[0]
    assert hit.document == snapshot.documents[0]
    assert hit.hit.chunk.content == snapshot.documents[0].content


def test_snapshot_exports_multiple_sources_and_documents_in_identity_order() -> None:
    collection = KnowledgeCollection()
    z_document = _document("z-source", "z.txt", "z content")
    a_second = _document("a-source", "b.txt", "b content")
    a_first = _document("a-source", "a.txt", "a content")
    collection.sync(FakeAdapter("z-source", (z_document,)))
    collection.sync(FakeAdapter("a-source", (a_second, a_first)))

    snapshot = collection.snapshot()

    assert [source.source_id for source in snapshot.sources] == ["a-source", "z-source"]
    assert [document.document_id for document in snapshot.documents] == sorted(
        document.document_id for document in (z_document, a_second, a_first)
    )
    assert collection.snapshot() == snapshot


def test_snapshot_does_not_expose_internal_mutable_metadata() -> None:
    source = KnowledgeSource(
        source_id="docs", source_type="fake", display_name="Docs", metadata={"owner": "original"}
    )
    document = Document(
        source_id="docs", logical_path="a.txt", content="content", metadata={"kind": "original"}
    )

    class MetadataAdapter(FakeAdapter):
        def source(self) -> KnowledgeSource:
            return source

    collection = KnowledgeCollection()
    collection.sync(MetadataAdapter("docs", (document,)))
    exported = collection.snapshot()
    exported.sources[0].metadata["owner"] = "changed"
    exported.documents[0].metadata["kind"] = "changed"

    fresh = collection.snapshot()
    assert fresh.sources[0].metadata == {"owner": "original"}
    assert fresh.documents[0].metadata == {"kind": "original"}


def test_restore_replaces_existing_state_and_rebuilds_searchable_chunks() -> None:
    collection = KnowledgeCollection()
    collection.sync(FakeAdapter("old", (_document("old", "old.txt", "old content"),)))
    restored = _document("docs", "new.txt", "new searchable")

    result = collection.restore(KnowledgeSnapshot((_source("docs"),), (restored,)))

    assert result == KnowledgeRestoreResult(1, 1, 1)
    assert collection.search("old") == ()
    search_result = collection.search("searchable")[0]
    assert search_result.source == _source("docs")
    assert search_result.document == restored
    assert search_result.hit.chunk.document_id == restored.document_id
    restored_snapshot = collection.snapshot()
    assert restored_snapshot.sources == (_source("docs"),)
    assert restored_snapshot.documents == (restored,)
    assert len(restored_snapshot.document_versions) == 1
    assert restored_snapshot.document_versions[0].content == restored.content


def test_restore_rebuilds_chinese_search_state_with_current_index_analyzer() -> None:
    source = _source("docs")
    document = _document("docs", "guide.txt", "知识图谱支持语义检索")
    snapshot = KnowledgeSnapshot((source,), (document,))

    default_collection = KnowledgeCollection()
    default_collection.restore(snapshot)
    default_result = default_collection.search("语义检索")[0]

    assert default_result.source == source
    assert default_result.document == document
    assert default_result.hit.chunk.document_id == document.document_id
    assert default_result.hit.chunk.content == document.content
    assert default_result.hit.matched_terms == ("语义", "义检", "检索")

    whitespace_collection = KnowledgeCollection(
        index_factory=lambda: InMemoryChunkIndex(
            analyzer=WhitespaceLexicalAnalyzer()
        )
    )
    whitespace_collection.restore(snapshot)

    assert whitespace_collection.search("语义检索") == ()


def test_restore_empty_snapshot_replaces_non_empty_collection() -> None:
    collection = KnowledgeCollection()
    collection.sync(FakeAdapter("docs", (_document("docs", "a.txt", "existing"),)))

    collection.restore(KnowledgeSnapshot((), ()))

    assert collection.snapshot() == KnowledgeSnapshot((), ())
    assert collection.search("existing") == ()


def test_round_trip_multiple_sources_preserves_canonical_state_and_isolation() -> None:
    original = KnowledgeCollection()
    original.sync(FakeAdapter("one", (_document("one", "a.txt", "first unique"),)))
    original.sync(FakeAdapter("two", (_document("two", "b.txt", "second unique"),)))
    original_results = original.search("first")
    snapshot = original.snapshot()
    restored = KnowledgeCollection()

    restored.restore(snapshot)

    assert restored.snapshot() == snapshot
    first_result = restored.search("first")[0]
    assert first_result.source.source_id == "one"
    assert first_result.document == snapshot.documents[0]
    assert [result.hit.chunk.document_id for result in restored.search("first")] == [
        snapshot.documents[0].document_id
    ]
    restored_results = restored.search("first")
    assert [result.hit.chunk.chunk_id for result in restored_results] == [
        result.hit.chunk.chunk_id for result in original_results
    ]
    assert [result.hit.score for result in restored_results] == pytest.approx(
        [result.hit.score for result in original_results]
    )
    assert len(restored.search("second")) == 1


def test_restore_rechunks_canonical_documents_in_deterministic_order() -> None:
    documents = (
        _document("docs", "z.txt", "z"),
        _document("docs", "a.txt", "a"),
    )
    chunker = RecordingChunker()
    collection = KnowledgeCollection(chunker=chunker)

    collection.restore(KnowledgeSnapshot((_source("docs"),), documents))

    assert chunker.document_ids == sorted(document.document_id for document in documents)
    assert not hasattr(collection.snapshot(), "chunks")
    assert not hasattr(collection.snapshot(), "index")


def test_restore_chunker_cannot_mutate_snapshot_or_canonical_state() -> None:
    document = Document(
        source_id="docs",
        logical_path="a.txt",
        content="hello",
        metadata={"tag": "original"},
    )
    snapshot = KnowledgeSnapshot((_source("docs"),), (document,))
    collection = KnowledgeCollection(chunker=MutatingMetadataChunker())

    collection.restore(snapshot)

    assert snapshot.documents[0].metadata == {"tag": "original"}
    assert collection.snapshot().documents[0].metadata == {"tag": "original"}


def test_restore_then_sync_detects_unchanged_changed_added_and_removed() -> None:
    stable = _document("docs", "stable.txt", "stable")
    changing = _document("docs", "changing.txt", "old")
    removed = _document("docs", "removed.txt", "removed")
    collection = KnowledgeCollection()
    collection.restore(
        KnowledgeSnapshot((_source("docs"),), (stable, changing, removed))
    )

    unchanged = collection.sync(FakeAdapter("docs", (stable, changing, removed)))
    assert unchanged.documents_unchanged == 3
    changed = collection.sync(
        FakeAdapter(
            "docs",
            (stable, _document("docs", "changing.txt", "new"), _document("docs", "added.txt", "added")),
        )
    )

    assert (changed.documents_added, changed.documents_updated, changed.documents_unchanged, changed.documents_removed) == (1, 1, 1, 1)
    assert collection.search("removed") == ()
    assert collection.search("new")


@pytest.mark.parametrize(
    "snapshot, message",
    [
        ("invalid", "KnowledgeSnapshot"),
        (KnowledgeSnapshot([], ()), "sources must be a tuple"),  # type: ignore[arg-type]
        (KnowledgeSnapshot((), []), "documents must be a tuple"),  # type: ignore[arg-type]
        (KnowledgeSnapshot(("bad",), ()), "KnowledgeSource"),  # type: ignore[arg-type]
        (KnowledgeSnapshot((), ("bad",)), "Document"),  # type: ignore[arg-type]
        (KnowledgeSnapshot((_source("docs"), _source("docs")), ()), "duplicate source_id"),
        (
            KnowledgeSnapshot((_source("docs"),), (_document("docs", "a.txt", "a"),) * 2),
            "duplicate document_id",
        ),
        (
            KnowledgeSnapshot((_source("docs"),), (_document("missing", "a.txt", "a"),)),
            "missing source_id",
        ),
    ],
)
def test_invalid_snapshots_are_rejected_without_changing_existing_state(
    snapshot: object, message: str
) -> None:
    collection = KnowledgeCollection()
    collection.sync(FakeAdapter("old", (_document("old", "old.txt", "preserved"),)))

    with pytest.raises(KnowledgeSnapshotError, match=message):
        collection.restore(snapshot)  # type: ignore[arg-type]

    assert collection.search("preserved")


@pytest.mark.parametrize("field", ["document_id", "content_hash"])
def test_restore_rejects_forged_document_identity(field: str) -> None:
    document = _document("docs", "a.txt", "content")
    object.__setattr__(document, field, "forged")
    collection = KnowledgeCollection()

    with pytest.raises(KnowledgeSnapshotError, match="incoherent"):
        collection.restore(KnowledgeSnapshot((_source("docs"),), (document,)))


def test_restore_collection_limit_failure_preserves_previous_state() -> None:
    collection = KnowledgeCollection(
        limits=KnowledgeCollectionLimits(max_sources=1, max_documents=1)
    )
    collection.sync(FakeAdapter("old", (_document("old", "old.txt", "preserved"),)))
    oversized = KnowledgeSnapshot(
        (_source("one"), _source("two")),
        (_document("one", "a.txt", "one"), _document("two", "b.txt", "two")),
    )

    with pytest.raises(KnowledgeCollectionLimitError):
        collection.restore(oversized)

    assert collection.search("preserved")


def test_chunking_and_index_failures_during_restore_are_atomic() -> None:
    chunker = RecordingChunker(fail_content="broken")
    collection = KnowledgeCollection(chunker=chunker)
    collection.sync(FakeAdapter("old", (_document("old", "old.txt", "preserved"),)))
    with pytest.raises(RuntimeError, match="chunking failed"):
        collection.restore(
            KnowledgeSnapshot((_source("docs"),), (_document("docs", "a.txt", "broken"),))
        )
    assert collection.search("preserved")

    limited = KnowledgeCollection(
        index_factory=lambda: InMemoryChunkIndex(limits=ChunkIndexLimits(max_total_chars=4))
    )
    limited.sync(FakeAdapter("old", (_document("old", "old.txt", "old"),)))
    with pytest.raises(ChunkIndexLimitError):
        limited.restore(
            KnowledgeSnapshot((_source("docs"),), (_document("docs", "a.txt", "too long"),))
        )
    assert limited.search("old")


def test_analyzed_character_limit_failure_during_restore_is_atomic() -> None:
    class ExpandingAnalyzer:
        def analyze(self, text: str) -> tuple[str, ...]:
            return (text if text == "old" else "x" * 6,)

    collection = KnowledgeCollection(
        index_factory=lambda: InMemoryChunkIndex(
            analyzer=ExpandingAnalyzer(),
            limits=ChunkIndexLimits(max_total_analyzed_token_chars=5),
        )
    )
    collection.sync(FakeAdapter("old", (_document("old", "old.txt", "old"),)))
    before = collection.search("old")

    with pytest.raises(ChunkIndexLimitError, match="max_total_analyzed_token_chars"):
        collection.restore(
            KnowledgeSnapshot((_source("docs"),), (_document("docs", "a.txt", "attack"),))
        )

    assert collection.search("old") == before


def test_chunk_identity_conflict_during_restore_is_atomic() -> None:
    collection = KnowledgeCollection(chunker=ConflictingChunker())
    collection.sync(FakeAdapter("old", (_document("old", "old.txt", "preserved"),)))
    snapshot = KnowledgeSnapshot(
        (_source("docs"),),
        (_document("docs", "a.txt", "one"), _document("docs", "b.txt", "two")),
    )

    with pytest.raises(ChunkIdentityConflictError):
        collection.restore(snapshot)

    assert collection.search("preserved")
    assert collection.search("one two") == ()


def test_invalid_fresh_index_during_restore_preserves_previous_state() -> None:
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return InMemoryChunkIndex() if calls == 1 else object()

    collection = KnowledgeCollection(index_factory=factory)
    collection.sync(FakeAdapter("old", (_document("old", "old.txt", "preserved"),)))

    with pytest.raises(KnowledgeRestoreError, match="invalid index"):
        collection.restore(KnowledgeSnapshot((), ()))

    assert collection.search("preserved")
