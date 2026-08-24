from __future__ import annotations

import pytest

from nexusmind import (
    Document,
    DocumentVersion,
    compute_content_hash,
    stable_document_version_id,
)


def _document(source_id: str = "docs", logical_path: str = "guide.md", content: str = "first") -> Document:
    return Document(source_id=source_id, logical_path=logical_path, content=content)


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
