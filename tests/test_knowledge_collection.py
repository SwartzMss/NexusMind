from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexusmind import (
    Chunk,
    ChunkIdentityConflictError,
    ChunkIndex,
    ChunkIndexLimitError,
    ChunkIndexLimits,
    Document,
    InMemoryChunkIndex,
    KnowledgeCollection,
    KnowledgeCollectionLimitError,
    KnowledgeCollectionLimits,
    KnowledgeSearchResult,
    KnowledgeSnapshotError,
    KnowledgeSource,
    KnowledgeSyncResult,
    TextChunker,
)


def _document(source_id: str, logical_path: str, content: str) -> Document:
    return Document(source_id=source_id, logical_path=logical_path, content=content)


@dataclass
class FakeAdapter:
    source_id: str
    documents: tuple[Document, ...]
    display_name: str = "Fake source"

    def source(self) -> KnowledgeSource:
        return KnowledgeSource(
            source_id=self.source_id,
            source_type="fake",
            display_name=self.display_name,
        )

    def load_documents(self) -> tuple[Document, ...]:
        return self.documents


class CountingChunker:
    def __init__(self, *, fail_content: str | None = None) -> None:
        self.calls: list[str] = []
        self._delegate = TextChunker(chunk_size=100, overlap=0)
        self._fail_content = fail_content

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        self.calls.append(document.document_id)
        if document.content == self._fail_content:
            raise RuntimeError("chunking failed")
        return self._delegate.chunk(document)


class ReusingIdentityChunker:
    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        return (
            Chunk(
                document_id=document.document_id,
                chunk_id=f"fixed-{document.document_id}",
                content=document.content,
                start_offset=0,
                end_offset=len(document.content),
            ),
        )


class MutatingMetadataChunker:
    def __init__(self) -> None:
        self._delegate = TextChunker(chunk_size=100, overlap=0)

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        document.metadata["tag"] = "modified-by-chunker"
        return self._delegate.chunk(document)


def test_first_sync_indexes_one_document_and_returns_summary() -> None:
    document = _document("docs", "a.txt", "checkpoint resume")
    collection = KnowledgeCollection()

    result = collection.sync(FakeAdapter("docs", (document,)))

    assert result == KnowledgeSyncResult("docs", 1, 0, 0, 0, 1)
    assert collection.search("checkpoint")[0].hit.chunk.document_id == document.document_id


def test_search_resolves_ordered_hits_to_canonical_source_and_document() -> None:
    collection = KnowledgeCollection()
    one = _document("one", "a.txt", "checkpoint resume")
    two = _document("two", "b.txt", "checkpoint")
    collection.sync(FakeAdapter("one", (one,), display_name="Source one"))
    collection.sync(FakeAdapter("two", (two,), display_name="Source two"))

    results = collection.search("checkpoint resume")

    assert isinstance(results[0], KnowledgeSearchResult)
    assert [result.source.source_id for result in results] == ["one", "two"]
    assert results[0].source.display_name == "Source one"
    assert results[0].document == one
    assert results[0].document.logical_path == "a.txt"
    assert results[0].hit.chunk.document_id == one.document_id
    assert results[0].hit.chunk.content == "checkpoint resume"
    assert (results[0].hit.chunk.start_offset, results[0].hit.chunk.end_offset) == (
        0,
        len(one.content),
    )
    assert results[0].hit.score == 2
    assert results[0].hit.matched_terms == ("checkpoint", "resume")


def test_resolved_search_returns_empty_tuple_for_no_hits() -> None:
    assert KnowledgeCollection().search("missing") == ()


def test_sync_detaches_committed_state_from_adapter_objects() -> None:
    source = KnowledgeSource(
        source_id="docs",
        source_type="fake",
        display_name="Docs",
        metadata={"owner": "original"},
    )
    document = Document(
        source_id="docs",
        logical_path="a.txt",
        content="content",
        metadata={"tag": "original"},
    )

    class OwnedObjectAdapter(FakeAdapter):
        def source(self) -> KnowledgeSource:
            return source

    collection = KnowledgeCollection()
    collection.sync(OwnedObjectAdapter("docs", (document,)))
    source.metadata["owner"] = "external"
    document.metadata["tag"] = "external"

    snapshot = collection.snapshot()
    assert snapshot.sources[0].metadata == {"owner": "original"}
    assert snapshot.documents[0].metadata == {"tag": "original"}


def test_sync_chunker_cannot_mutate_adapter_or_canonical_document() -> None:
    document = Document(
        source_id="docs",
        logical_path="a.txt",
        content="content",
        metadata={"tag": "original"},
    )
    collection = KnowledgeCollection(chunker=MutatingMetadataChunker())

    collection.sync(FakeAdapter("docs", (document,)))

    assert document.metadata == {"tag": "original"}
    assert collection.snapshot().documents[0].metadata == {"tag": "original"}


def test_sync_rejects_forged_document_id_without_mutation() -> None:
    collection = KnowledgeCollection()
    old = _document("docs", "a.txt", "preserved")
    collection.sync(FakeAdapter("docs", (old,)))
    forged = _document("docs", "b.txt", "forged content")
    object.__setattr__(forged, "document_id", "forged-id")

    with pytest.raises(KnowledgeSnapshotError, match="incoherent document_id"):
        collection.sync(FakeAdapter("docs", (forged,)))

    assert collection.search("preserved")
    assert collection.search("forged") == ()


def test_sync_rejects_forged_content_hash_without_mutation() -> None:
    collection = KnowledgeCollection()
    old = _document("docs", "a.txt", "old searchable")
    collection.sync(FakeAdapter("docs", (old,)))
    forged = _document("docs", "a.txt", "new hidden")
    object.__setattr__(forged, "content_hash", old.content_hash)

    with pytest.raises(KnowledgeSnapshotError, match="incoherent content_hash"):
        collection.sync(FakeAdapter("docs", (forged,)))

    assert collection.search("old")
    assert collection.search("new") == ()


def test_successful_sync_snapshot_is_self_restorable() -> None:
    collection = KnowledgeCollection()
    collection.sync(
        FakeAdapter(
            "docs",
            (_document("docs", "a.txt", "alpha"), _document("docs", "b.txt", "beta")),
        )
    )
    snapshot = collection.snapshot()
    restored = KnowledgeCollection()

    restored.restore(snapshot)

    assert restored.snapshot() == snapshot


def test_base_chunk_index_contract_does_not_require_collection_staging() -> None:
    assert "clone" not in ChunkIndex.__dict__


def test_index_factory_creates_collection_owned_empty_state() -> None:
    calls = 0

    def factory() -> InMemoryChunkIndex:
        nonlocal calls
        calls += 1
        return InMemoryChunkIndex()

    collection = KnowledgeCollection(index_factory=factory)
    assert collection.search("anything") == ()

    collection.sync(FakeAdapter("docs", (_document("docs", "a.txt", "searchable"),)))

    assert calls == 1
    assert collection.search("searchable")


def test_first_multi_document_sync_and_search_ranking() -> None:
    collection = KnowledgeCollection()
    collection.sync(
        FakeAdapter(
            "docs",
            (
                _document("docs", "b.txt", "checkpoint"),
                _document("docs", "a.txt", "checkpoint resume"),
            ),
        )
    )

    hits = collection.search("checkpoint resume", limit=1)

    assert len(hits) == 1
    assert hits[0].hit.score == 2
    assert hits[0].hit.chunk.content == "checkpoint resume"


def test_identical_sync_is_unchanged_and_does_not_rechunk() -> None:
    document = _document("docs", "a.txt", "stable")
    chunker = CountingChunker()
    collection = KnowledgeCollection(chunker=chunker)
    collection.sync(FakeAdapter("docs", (document,)))

    result = collection.sync(FakeAdapter("docs", (document,), display_name="Renamed"))

    assert result == KnowledgeSyncResult("docs", 0, 0, 1, 0, 0)
    assert chunker.calls == [document.document_id]
    assert collection.search("stable")


def test_refresh_adds_changes_preserves_and_removes_documents() -> None:
    old_changed = _document("docs", "changed.txt", "old value")
    unchanged = _document("docs", "same.txt", "same value")
    removed = _document("docs", "removed.txt", "removed value")
    collection = KnowledgeCollection()
    collection.sync(FakeAdapter("docs", (old_changed, unchanged, removed)))
    new_changed = _document("docs", "changed.txt", "new value")
    added = _document("docs", "added.txt", "added value")

    result = collection.sync(FakeAdapter("docs", (new_changed, unchanged, added)))

    assert result == KnowledgeSyncResult("docs", 1, 1, 1, 1, 2)
    assert collection.search("old") == ()
    assert collection.search("removed") == ()
    assert collection.search("new")
    assert collection.search("added")
    assert collection.search("same")


def test_refresh_and_remove_source_do_not_affect_unrelated_source() -> None:
    collection = KnowledgeCollection()
    collection.sync(FakeAdapter("one", (_document("one", "a.txt", "first unique"),)))
    collection.sync(FakeAdapter("two", (_document("two", "a.txt", "second unique"),)))

    collection.sync(FakeAdapter("one", ()))
    assert collection.search("first") == ()
    assert collection.search("second")

    collection.remove_source("two")
    assert collection.search("second") == ()
    collection.remove_source("unknown")


def test_duplicate_and_mixed_source_snapshots_fail_closed() -> None:
    document = _document("docs", "a.txt", "original")
    collection = KnowledgeCollection()
    collection.sync(FakeAdapter("docs", (document,)))

    with pytest.raises(KnowledgeSnapshotError, match="duplicate"):
        collection.sync(FakeAdapter("docs", (document, document)))
    with pytest.raises(KnowledgeSnapshotError, match="source_id"):
        collection.sync(FakeAdapter("docs", (_document("other", "a.txt", "other"),)))

    assert collection.search("original")
    assert collection.search("other") == ()


def test_chunking_failure_preserves_previous_snapshot_and_index() -> None:
    old = _document("docs", "a.txt", "old")
    chunker = CountingChunker(fail_content="broken")
    collection = KnowledgeCollection(chunker=chunker)
    collection.sync(FakeAdapter("docs", (old,)))

    with pytest.raises(RuntimeError, match="chunking failed"):
        collection.sync(FakeAdapter("docs", (_document("docs", "a.txt", "broken"),)))

    assert collection.search("old")
    assert collection.search("broken") == ()


def test_index_limit_failure_preserves_previous_snapshot_and_index() -> None:
    collection = KnowledgeCollection(
        index_factory=lambda: InMemoryChunkIndex(
            limits=ChunkIndexLimits(max_total_chars=4)
        )
    )
    old = _document("docs", "a.txt", "old")
    collection.sync(FakeAdapter("docs", (old,)))

    with pytest.raises(ChunkIndexLimitError):
        collection.sync(FakeAdapter("docs", (_document("docs", "a.txt", "too long"),)))

    assert collection.search("old")
    assert collection.search("long") == ()


def test_chunk_identity_conflict_preserves_previous_snapshot_and_index() -> None:
    collection = KnowledgeCollection(chunker=ReusingIdentityChunker())
    old = _document("docs", "a.txt", "old")
    collection.sync(FakeAdapter("docs", (old,)))

    with pytest.raises(ChunkIdentityConflictError):
        collection.sync(FakeAdapter("docs", (_document("docs", "a.txt", "new"),)))

    assert collection.search("old")
    assert collection.search("new") == ()


def test_failed_first_sync_does_not_create_source_bookkeeping() -> None:
    limits = ChunkIndexLimits(max_total_chars=1)
    collection = KnowledgeCollection(
        index_factory=lambda: InMemoryChunkIndex(limits=limits)
    )
    with pytest.raises(ChunkIndexLimitError):
        collection.sync(FakeAdapter("docs", (_document("docs", "a.txt", "too long"),)))

    result = collection.sync(FakeAdapter("docs", (_document("docs", "a.txt", "x"),)))

    assert result.documents_added == 1
    assert result.documents_unchanged == 0


def test_collection_limits_are_preflighted_without_affecting_other_sources() -> None:
    collection = KnowledgeCollection(
        limits=KnowledgeCollectionLimits(max_sources=1, max_documents=1)
    )
    collection.sync(FakeAdapter("one", (_document("one", "a.txt", "one"),)))

    with pytest.raises(KnowledgeCollectionLimitError, match="max_sources"):
        collection.sync(FakeAdapter("two", (_document("two", "a.txt", "two"),)))
    with pytest.raises(KnowledgeCollectionLimitError, match="max_documents"):
        collection.sync(
            FakeAdapter(
                "one",
                (_document("one", "a.txt", "one"), _document("one", "b.txt", "extra")),
            )
        )

    assert collection.search("one")
    assert collection.search("two extra") == ()


def test_search_delegates_index_query_and_result_limits() -> None:
    collection = KnowledgeCollection(
        index_factory=lambda: InMemoryChunkIndex(
            limits=ChunkIndexLimits(max_results=1, max_query_chars=3)
        )
    )
    with pytest.raises(ChunkIndexLimitError, match="max_query_chars"):
        collection.search("long")
    with pytest.raises(ChunkIndexLimitError, match="max_results"):
        collection.search("x", limit=2)


@pytest.mark.parametrize("field", KnowledgeCollectionLimits.__dataclass_fields__)
def test_collection_limits_require_positive_plain_integers(field: str) -> None:
    with pytest.raises(TypeError):
        KnowledgeCollectionLimits(**{field: True})
    with pytest.raises(ValueError):
        KnowledgeCollectionLimits(**{field: 0})
