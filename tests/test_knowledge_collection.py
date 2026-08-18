from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexusmind import (
    Chunk,
    ChunkIdentityConflictError,
    ChunkIndexLimitError,
    ChunkIndexLimits,
    Document,
    InMemoryChunkIndex,
    KnowledgeCollection,
    KnowledgeCollectionLimitError,
    KnowledgeCollectionLimits,
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


def test_first_sync_indexes_one_document_and_returns_summary() -> None:
    document = _document("docs", "a.txt", "checkpoint resume")
    collection = KnowledgeCollection()

    result = collection.sync(FakeAdapter("docs", (document,)))

    assert result == KnowledgeSyncResult("docs", 1, 0, 0, 0, 1)
    assert collection.search("checkpoint")[0].chunk.document_id == document.document_id


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
    assert hits[0].score == 2
    assert hits[0].chunk.content == "checkpoint resume"


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
    index = InMemoryChunkIndex(limits=ChunkIndexLimits(max_total_chars=4))
    collection = KnowledgeCollection(index=index)
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
    collection = KnowledgeCollection(index=InMemoryChunkIndex(limits=limits))
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
        index=InMemoryChunkIndex(limits=ChunkIndexLimits(max_results=1, max_query_chars=3))
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
