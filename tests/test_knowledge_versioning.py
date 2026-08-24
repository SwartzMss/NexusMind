from __future__ import annotations

import pytest
from datetime import datetime, timezone

from nexusmind import (
    Document,
    DocumentVersion,
    KnowledgeCollection,
    KnowledgeCollectionLimits,
    KnowledgeSnapshot,
    KnowledgeSnapshotError,
    KnowledgeSource,
    compute_content_hash,
    stable_document_version_id,
)


def _document(source_id: str = "docs", logical_path: str = "guide.md", content: str = "first") -> Document:
    return Document(source_id=source_id, logical_path=logical_path, content=content)


class Adapter:
    def __init__(self, documents: tuple[Document, ...]) -> None:
        self._documents = documents

    def source(self) -> KnowledgeSource:
        return KnowledgeSource(
            source_id="docs", source_type="fake", display_name="Docs"
        )

    def load_documents(self) -> tuple[Document, ...]:
        return self._documents


def _fixed_clock(minute: int = 0):
    return lambda: datetime(2026, 8, 24, 2, minute, tzinfo=timezone.utc)


def test_document_version_captures_identity_and_provenance() -> None:
    document = _document()

    version = DocumentVersion.from_document(
        document,
        created_at="2026-08-24T02:00:00.000000Z",
        sync_context="sync-fixed",
    )

    assert version.document_id == document.document_id
    assert version.source_id == "docs"
    assert version.logical_path == "guide.md"
    assert version.content == "first"
    assert version.content_hash == compute_content_hash("first")
    assert version.previous_version_id is None
    assert version.version_id == stable_document_version_id(
        document.document_id, document.content_hash, None
    )


def test_document_version_links_to_predecessor_deterministically() -> None:
    document = _document(content="second")

    version = DocumentVersion.from_document(
        document,
        created_at="2026-08-24T02:01:00.000000Z",
        sync_context="sync-next",
        previous_version_id="version-" + "1" * 64,
    )

    assert version.previous_version_id == "version-" + "1" * 64
    assert version.version_id == stable_document_version_id(
        document.document_id,
        document.content_hash,
        "version-" + "1" * 64,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_id", "", "source_id"),
        ("logical_path", "", "logical_path"),
        ("content_hash", "bad", "content_hash"),
        ("created_at", "2026-08-24T02:00:00Z", "created_at"),
        ("created_at", "2026-08-24T02:00:00.000000+00:00", "created_at"),
        ("sync_context", "", "sync_context"),
        ("previous_version_id", "bad", "previous_version_id"),
        ("version_id", "bad", "version_id"),
    ],
)
def test_document_version_rejects_malformed_fields(
    field: str, value: str, message: str
) -> None:
    document = _document()
    values = {
        "version_id": stable_document_version_id(
            document.document_id, document.content_hash, None
        ),
        "document_id": document.document_id,
        "source_id": document.source_id,
        "logical_path": document.logical_path,
        "content": document.content,
        "content_hash": document.content_hash,
        "created_at": "2026-08-24T02:00:00.000000Z",
        "previous_version_id": None,
        "sync_context": "sync-fixed",
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        DocumentVersion(**values)


def test_document_version_rejects_incoherent_content_and_document_identity() -> None:
    document = _document()
    base = DocumentVersion.from_document(
        document,
        created_at="2026-08-24T02:00:00.000000Z",
        sync_context="sync-fixed",
    )

    with pytest.raises(ValueError, match="content_hash"):
        DocumentVersion(
            version_id=base.version_id,
            document_id=base.document_id,
            source_id=base.source_id,
            logical_path=base.logical_path,
            content="forged",
            content_hash=base.content_hash,
            created_at=base.created_at,
            previous_version_id=None,
            sync_context=base.sync_context,
        )

    with pytest.raises(ValueError, match="document_id"):
        DocumentVersion(
            version_id=base.version_id,
            document_id="document-" + "2" * 64,
            source_id=base.source_id,
            logical_path=base.logical_path,
            content=base.content,
            content_hash=base.content_hash,
            created_at=base.created_at,
            previous_version_id=None,
            sync_context=base.sync_context,
        )


def test_sync_creates_root_version_and_unchanged_sync_does_not_append() -> None:
    document = _document()
    collection = KnowledgeCollection(clock=_fixed_clock())

    first = collection.sync(Adapter((document,)))
    second = collection.sync(Adapter((_document(),)))

    versions = collection.snapshot().document_versions
    assert first.documents_added == 1
    assert second.documents_unchanged == 1
    assert len(versions) == 1
    assert versions[0].content == "first"
    assert versions[0].previous_version_id is None
    assert versions[0].created_at == "2026-08-24T02:00:00.000000Z"


def test_changed_sync_appends_version_and_searches_only_current_content() -> None:
    times = iter(
        (
            datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 24, 2, 1, tzinfo=timezone.utc),
        )
    )
    collection = KnowledgeCollection(clock=lambda: next(times))
    collection.sync(Adapter((_document(content="obsolete"),)))

    result = collection.sync(Adapter((_document(content="current-term"),)))

    versions = collection.snapshot().document_versions
    assert result.documents_updated == 1
    assert [version.content for version in versions] == ["obsolete", "current-term"]
    assert versions[1].previous_version_id == versions[0].version_id
    assert versions[1].sync_context != versions[0].sync_context
    assert collection.search("obsolete") == ()
    assert collection.search("current-term")[0].document.content == "current-term"


def test_delete_preserves_history_and_same_content_reappearance_appends() -> None:
    times = iter(
        (
            datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 24, 2, 1, tzinfo=timezone.utc),
        )
    )
    collection = KnowledgeCollection(clock=lambda: next(times))
    collection.sync(Adapter((_document(content="returning-term"),)))

    collection.sync(Adapter(()))
    history_after_delete = collection.snapshot().document_versions
    assert collection.search("returning-term") == ()

    collection.sync(Adapter((_document(content="returning-term"),)))
    versions = collection.snapshot().document_versions
    assert history_after_delete == versions[:1]
    assert len(versions) == 2
    assert versions[1].previous_version_id == versions[0].version_id
    assert versions[1].version_id != versions[0].version_id


def test_one_sync_uses_one_context_for_all_new_versions() -> None:
    collection = KnowledgeCollection(clock=_fixed_clock())

    collection.sync(
        Adapter(
            (
                _document(logical_path="a.md", content="a"),
                _document(logical_path="b.md", content="b"),
            )
        )
    )

    contexts = {version.sync_context for version in collection.snapshot().document_versions}
    assert len(contexts) == 1


def test_version_limit_rejects_sync_without_changing_state() -> None:
    collection = KnowledgeCollection(
        clock=_fixed_clock(),
        limits=KnowledgeCollectionLimits(max_document_versions=1),
    )
    collection.sync(Adapter((_document(content="one"),)))
    before = collection.snapshot()

    with pytest.raises(Exception, match="max_document_versions"):
        collection.sync(Adapter((_document(content="two"),)))

    assert collection.snapshot() == before
    assert collection.search("one")


def test_restore_round_trips_history_and_indexes_only_current_document() -> None:
    times = iter(
        (
            datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 24, 2, 1, tzinfo=timezone.utc),
        )
    )
    original = KnowledgeCollection(clock=lambda: next(times))
    original.sync(Adapter((_document(content="obsolete"),)))
    original.sync(Adapter((_document(content="current-term"),)))
    snapshot = original.snapshot()
    restored = KnowledgeCollection()

    restored.restore(snapshot)

    assert restored.snapshot() == snapshot
    assert restored.search("obsolete") == ()
    assert restored.search("current-term")


def test_restore_legacy_snapshot_synthesizes_root_versions() -> None:
    document = _document(content="legacy")
    snapshot = KnowledgeSnapshot(
        sources=(Adapter((document,)).source(),), documents=(document,)
    )
    restored = KnowledgeCollection(clock=_fixed_clock(3))

    restored.restore(snapshot)

    versions = restored.snapshot().document_versions
    assert len(versions) == 1
    assert versions[0].content == "legacy"
    assert versions[0].previous_version_id is None
    assert versions[0].created_at == "2026-08-24T02:03:00.000000Z"


@pytest.mark.parametrize("mutation", ["missing_predecessor", "stale_tip"])
def test_restore_rejects_incoherent_history_without_changing_state(
    mutation: str,
) -> None:
    times = iter(
        (
            datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 24, 2, 1, tzinfo=timezone.utc),
        )
    )
    source = KnowledgeCollection(clock=lambda: next(times))
    source.sync(Adapter((_document(content="one"),)))
    source.sync(Adapter((_document(content="two"),)))
    snapshot = source.snapshot()
    versions = list(snapshot.document_versions)
    if mutation == "missing_predecessor":
        object.__setattr__(versions[1], "previous_version_id", "version-" + "9" * 64)
    else:
        versions = versions[:1]
    forged = KnowledgeSnapshot(snapshot.sources, snapshot.documents, tuple(versions))
    target = KnowledgeCollection(clock=_fixed_clock())
    target.sync(Adapter((_document(content="preserved"),)))
    before = target.snapshot()

    with pytest.raises(KnowledgeSnapshotError, match="version"):
        target.restore(forged)

    assert target.snapshot() == before
    assert target.search("preserved")
