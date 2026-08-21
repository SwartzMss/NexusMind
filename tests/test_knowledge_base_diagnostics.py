from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import traceback

import pytest
import nexusmind
import nexusmind.knowledge_base as knowledge_base_module
import nexusmind.knowledge_inspection as inspection_module

from nexusmind import (
    Document,
    InMemoryChunkIndex,
    KnowledgeBase,
    KnowledgeBaseClosedError,
    KnowledgeBaseStatus,
    KnowledgeBaseSourceError,
    KnowledgeDocumentInspection,
    KnowledgeRetrievalDiagnostics,
    KnowledgeSnapshot,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    SQLiteKnowledgeSnapshotStore,
)
from nexusmind.knowledge_base import KnowledgeBaseStatus as CompatibleKnowledgeBaseStatus


class _TextSubclass(str):
    pass


class _SearchOnlyIndex:
    """Cloneable delegating index that intentionally has no diagnose method."""

    def __init__(self, delegate: InMemoryChunkIndex | None = None) -> None:
        self._delegate = InMemoryChunkIndex() if delegate is None else delegate

    def add(self, chunks) -> None:
        self._delegate.add(chunks)

    def replace_document(self, document_id, chunks) -> None:
        self._delegate.replace_document(document_id, chunks)

    def remove_document(self, document_id) -> None:
        self._delegate.remove_document(document_id)

    def search(self, query: str, *, limit: int = 10):
        return self._delegate.search(query, limit=limit)

    def clone(self) -> _SearchOnlyIndex:
        return _SearchOnlyIndex(self._delegate.clone())


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type"),
    (
        ("knowledge_base_id", None, TypeError),
        ("knowledge_base_id", 1, TypeError),
        ("knowledge_base_id", _TextSubclass("kb"), TypeError),
        ("knowledge_base_id", "", ValueError),
        ("knowledge_base_id", "  ", ValueError),
        ("display_name", 1, TypeError),
        ("display_name", _TextSubclass("Docs"), TypeError),
        ("display_name", "", ValueError),
        ("display_name", "\t", ValueError),
        ("registered_source_count", True, TypeError),
        ("registered_source_count", 1.0, TypeError),
        ("registered_source_count", -1, ValueError),
        ("canonical_source_count", True, TypeError),
        ("canonical_source_count", 1.0, TypeError),
        ("canonical_source_count", -1, ValueError),
        ("document_count", True, TypeError),
        ("document_count", 1.0, TypeError),
        ("document_count", -1, ValueError),
    ),
)
def test_knowledge_base_status_rejects_invalid_exact_public_fields(
    field_name: str, invalid_value: object, error_type: type[Exception]
) -> None:
    values: dict[str, object] = {
        "knowledge_base_id": "kb",
        "display_name": None,
        "registered_source_count": 0,
        "canonical_source_count": 0,
        "document_count": 0,
    }
    values[field_name] = invalid_value

    with pytest.raises(error_type, match=field_name):
        KnowledgeBaseStatus(**values)  # type: ignore[arg-type]


def test_public_knowledge_base_inspection_values_are_strict_frozen_and_slotted(
    tmp_path: Path,
) -> None:
    assert hasattr(nexusmind, "KnowledgeBaseInspection")
    assert hasattr(nexusmind, "KnowledgeDocumentSummary")
    assert hasattr(nexusmind, "KnowledgeSourceInspection")
    assert hasattr(nexusmind, "KnowledgeSourceSyncStatus")
    assert hasattr(inspection_module, "KnowledgeBaseStatus")
    KnowledgeBaseInspection = nexusmind.KnowledgeBaseInspection
    KnowledgeDocumentSummary = nexusmind.KnowledgeDocumentSummary
    KnowledgeSourceInspection = nexusmind.KnowledgeSourceInspection
    KnowledgeSourceSyncStatus = nexusmind.KnowledgeSourceSyncStatus
    InspectionKnowledgeBaseStatus = inspection_module.KnowledgeBaseStatus
    config = LocalFileSourceConfig(source_id="docs", path=str(tmp_path / "docs.txt"))
    status = KnowledgeBaseStatus("kb", "Knowledge", 1, 1, 1)
    source = KnowledgeSourceInspection(
        config, KnowledgeSourceSyncStatus.SYNCED, 1, 2
    )
    metadata = {"nested": {"owner": "canonical"}}
    document = KnowledgeDocumentSummary(
        "docs", "document-1", "docs.txt", "text/plain", "sha256", metadata, 12, 2
    )
    inspection = KnowledgeBaseInspection(status, (source,), (document,))

    assert CompatibleKnowledgeBaseStatus is InspectionKnowledgeBaseStatus
    assert KnowledgeBaseStatus is InspectionKnowledgeBaseStatus
    assert KnowledgeSourceSyncStatus.REGISTERED.value == "registered"
    assert KnowledgeSourceSyncStatus.SYNCED.value == "synced"
    assert not hasattr(status, "__dict__")
    assert not hasattr(source, "__dict__")
    assert not hasattr(document, "__dict__")
    assert not hasattr(inspection, "__dict__")
    with pytest.raises(FrozenInstanceError):
        inspection.status = status  # type: ignore[misc]

    metadata["nested"]["owner"] = "external"
    assert document.metadata == {"nested": {"owner": "canonical"}}
    document.metadata["nested"]["owner"] = "consumer"
    assert metadata == {"nested": {"owner": "external"}}

    for values in (
        (object(), KnowledgeSourceSyncStatus.SYNCED, 1, 2),
        (config, "synced", 1, 2),
        (config, KnowledgeSourceSyncStatus.SYNCED, True, 2),
        (config, KnowledgeSourceSyncStatus.SYNCED, -1, 2),
        (config, KnowledgeSourceSyncStatus.SYNCED, 1, -1),
    ):
        with pytest.raises((TypeError, ValueError)):
            KnowledgeSourceInspection(*values)  # type: ignore[arg-type]

    for field, value in (
        (0, ""),
        (1, ""),
        (2, ""),
        (3, ""),
        (4, ""),
        (5, []),
        (6, True),
        (6, -1),
        (7, -1),
    ):
        values: list[object] = [
            "docs",
            "document-1",
            "docs.txt",
            "text/plain",
            "sha256",
            {},
            12,
            2,
        ]
        values[field] = value
        with pytest.raises((TypeError, ValueError)):
            KnowledgeDocumentSummary(*values)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="status"):
        KnowledgeBaseInspection(object(), (), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sources"):
        KnowledgeBaseInspection(status, [], ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KnowledgeSourceInspection"):
        KnowledgeBaseInspection(status, (object(),), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="documents"):
        KnowledgeBaseInspection(status, (), [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KnowledgeDocumentSummary"):
        KnowledgeBaseInspection(status, (), (object(),))  # type: ignore[arg-type]


def test_source_inspection_accepts_both_exact_local_config_types(tmp_path: Path) -> None:
    KnowledgeSourceInspection = nexusmind.KnowledgeSourceInspection
    KnowledgeSourceSyncStatus = nexusmind.KnowledgeSourceSyncStatus
    file_config = LocalFileSourceConfig(source_id="file", path=str(tmp_path / "one.txt"))
    directory_config = LocalDirectorySourceConfig(
        source_id="directory", path=str(tmp_path / "directory")
    )

    assert type(
        KnowledgeSourceInspection(
            file_config, KnowledgeSourceSyncStatus.REGISTERED, 0, 0
        ).config
    ) is LocalFileSourceConfig
    assert type(
        KnowledgeSourceInspection(
            directory_config, KnowledgeSourceSyncStatus.SYNCED, 0, 0
        ).config
    ) is LocalDirectorySourceConfig


def _create_synced_file_base(
    tmp_path: Path, *, index_factory=None
) -> tuple[KnowledgeBase, Document]:
    source_path = tmp_path / "source.txt"
    source_path.write_text("alpha beta gamma delta", encoding="utf-8")
    kb = KnowledgeBase.create(
        str(tmp_path / "kb"),
        knowledge_base_id="kb",
        index_factory=index_factory,
    )
    kb.add_source(LocalFileSourceConfig(source_id="docs", path=str(source_path)))
    kb.sync_source("docs")
    return kb, kb.list_documents()[0]


def test_inspect_reports_pending_nonempty_and_zero_document_sources_coherently(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    docs.joinpath("z.txt").write_text("zulu knowledge", encoding="utf-8")
    docs.joinpath("a.md").write_text("alpha knowledge", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    root = tmp_path / "kb"
    kb = KnowledgeBase.create(str(root), knowledge_base_id="kb", display_name="Docs")
    kb.add_source(
        LocalFileSourceConfig(source_id="a-pending", path=str(tmp_path / "missing.txt"))
    )
    kb.add_source(LocalDirectorySourceConfig(source_id="b-docs", path=str(docs)))
    kb.add_source(LocalDirectorySourceConfig(source_id="z-empty", path=str(empty)))
    kb.sync_source("b-docs")
    kb.sync_source("z-empty")

    snapshot = SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load()
    enriched_documents = tuple(
        Document(
            source_id=document.source_id,
            logical_path=document.logical_path,
            content=document.content,
            content_type=document.content_type,
            metadata={"nested": {"logical_path": document.logical_path}},
        )
        for document in snapshot.documents
    )
    SQLiteKnowledgeSnapshotStore(root / "knowledge.db").save(
        KnowledgeSnapshot(snapshot.sources, enriched_documents)
    )
    kb.close()
    kb = KnowledgeBase.open(str(root))

    inspection = kb.inspect()

    assert type(inspection) is nexusmind.KnowledgeBaseInspection
    assert inspection.status == kb.status() == KnowledgeBaseStatus("kb", "Docs", 3, 2, 2)
    assert tuple(item.config.source_id for item in inspection.sources) == (
        "a-pending",
        "b-docs",
        "z-empty",
    )
    assert tuple(item.sync_status for item in inspection.sources) == (
        nexusmind.KnowledgeSourceSyncStatus.REGISTERED,
        nexusmind.KnowledgeSourceSyncStatus.SYNCED,
        nexusmind.KnowledgeSourceSyncStatus.SYNCED,
    )
    assert tuple(
        (item.document_count, item.chunk_count) for item in inspection.sources
    ) == ((0, 0), (2, 2), (0, 0))
    assert tuple(
        (item.source_id, item.logical_path) for item in inspection.documents
    ) == (("b-docs", "a.md"), ("b-docs", "z.txt"))
    assert tuple(item.character_count for item in inspection.documents) == (
        len("alpha knowledge"),
        len("zulu knowledge"),
    )
    assert tuple(item.chunk_count for item in inspection.documents) == (1, 1)
    assert all(not hasattr(item, "content") for item in inspection.documents)
    assert inspection.sources[0].config is not kb.list_sources()[0]

    pristine = kb.inspect()
    inspection.documents[0].metadata["nested"]["logical_path"] = "consumer"
    assert kb.inspect() == pristine
    kb.close()
    assert KnowledgeBase.open(str(root)).inspect() == pristine


def test_inspect_document_wraps_collection_inspection_and_validates_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, document = _create_synced_file_base(tmp_path)

    inspection = kb.inspect_document(document.document_id, preview_chars=5)

    assert type(inspection) is KnowledgeDocumentInspection
    assert type(inspection.source) is nexusmind.KnowledgeSource
    assert type(inspection.document) is Document
    assert inspection.document == document
    assert inspection.chunks
    assert all(len(chunk.preview) <= 5 for chunk in inspection.chunks)

    for document_id in (None, 1, True):
        with pytest.raises(TypeError, match="document_id"):
            kb.inspect_document(document_id)  # type: ignore[arg-type]
    for document_id in ("", " "):
        with pytest.raises(ValueError, match="document_id"):
            kb.inspect_document(document_id)
    for preview_chars in (None, 1.0, True):
        with pytest.raises(TypeError, match="preview_chars"):
            kb.inspect_document(document.document_id, preview_chars=preview_chars)  # type: ignore[arg-type]
    for preview_chars in (0, -1):
        with pytest.raises(ValueError, match="preview_chars"):
            kb.inspect_document(document.document_id, preview_chars=preview_chars)

    def fail_inspection(*args, **kwargs):
        raise RuntimeError("private path /srv/secret/source.txt")

    monkeypatch.setattr(kb._collection, "inspect_document", fail_inspection)
    with pytest.raises(KnowledgeBaseSourceError) as caught:
        kb.inspect_document(document.document_id)
    assert str(caught.value) == "unable to inspect knowledge base"
    formatted = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert "unable to inspect knowledge base" in formatted
    assert "secret" not in formatted
    assert "/srv" not in formatted

    monkeypatch.setattr(kb._collection, "inspect_document", lambda *args, **kwargs: object())
    with pytest.raises(KnowledgeBaseSourceError, match="^unable to inspect knowledge base$"):
        kb.inspect_document(document.document_id)


def test_product_diagnose_search_supports_bm25_reopen_and_stable_errors(
    tmp_path: Path,
) -> None:
    factory_calls: list[InMemoryChunkIndex] = []

    def factory() -> InMemoryChunkIndex:
        index = InMemoryChunkIndex()
        factory_calls.append(index)
        return index

    kb, _ = _create_synced_file_base(tmp_path, index_factory=factory)

    diagnostics = kb.diagnose_search("alpha", limit=3)

    assert type(diagnostics) is KnowledgeRetrievalDiagnostics
    assert diagnostics.query == "alpha"
    assert diagnostics.results
    assert diagnostics.candidates
    assert all(type(item.source) is nexusmind.KnowledgeSource for item in diagnostics.results)
    assert all(type(item.document) is Document for item in diagnostics.results)
    kb.close()
    reopened = KnowledgeBase.open(str(tmp_path / "kb"), index_factory=factory)
    assert reopened.diagnose_search("alpha", limit=3) == diagnostics
    assert len(factory_calls) >= 4

    for query in (None, 1, True):
        with pytest.raises(TypeError, match="query"):
            reopened.diagnose_search(query)  # type: ignore[arg-type]
    for limit in (None, 1.0, True):
        with pytest.raises(TypeError, match="limit"):
            reopened.diagnose_search("alpha", limit=limit)  # type: ignore[arg-type]
    for limit in (0, -1):
        with pytest.raises(ValueError, match="limit"):
            reopened.diagnose_search("alpha", limit=limit)

    unsupported_root = tmp_path / "unsupported-case"
    unsupported_root.mkdir()
    unsupported, _ = _create_synced_file_base(
        unsupported_root, index_factory=_SearchOnlyIndex
    )
    assert unsupported.search("alpha")
    unsupported.close()
    unsupported = KnowledgeBase.open(
        str(unsupported_root / "kb"), index_factory=_SearchOnlyIndex
    )
    assert unsupported.search("alpha")
    with pytest.raises(
        KnowledgeBaseSourceError, match="^unable to diagnose knowledge search$"
    ):
        unsupported.diagnose_search("alpha")

    class ExplodingIndex(InMemoryChunkIndex):
        def diagnose(self, query: str, *, limit: int = 10):
            raise RuntimeError(f"provider /private/index leaked query {query}")

    exploding = KnowledgeBase.create(
        str(tmp_path / "exploding"),
        knowledge_base_id="exploding",
        index_factory=ExplodingIndex,
    )
    sensitive_query = "secret-query"
    with pytest.raises(KnowledgeBaseSourceError) as caught:
        exploding.diagnose_search(sensitive_query)
    assert str(caught.value) == "unable to diagnose knowledge search"
    formatted = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert "unable to diagnose knowledge search" in formatted
    assert "private" not in formatted
    assert sensitive_query not in formatted


def test_diagnostic_methods_are_read_only_and_closed_handles_fail_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, document = _create_synced_file_base(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only diagnostics touched persistence or source IO")

    monkeypatch.setattr(knowledge_base_module, "SQLiteKnowledgeSnapshotStore", forbidden)
    monkeypatch.setattr(knowledge_base_module, "write_manifest", forbidden)
    monkeypatch.setattr(knowledge_base_module, "LocalFileAdapter", forbidden)
    monkeypatch.setattr(knowledge_base_module, "LocalDirectoryAdapter", forbidden)
    kb.inspect()
    kb.inspect_document(document.document_id)
    kb.diagnose_search("alpha")

    kb.close()
    for call in (
        lambda: kb.inspect(),
        lambda: kb.inspect_document(document.document_id),
        lambda: kb.inspect_document("", preview_chars=0),
        lambda: kb.diagnose_search("alpha"),
        lambda: kb.diagnose_search(1, limit=0),
    ):
        with pytest.raises(KnowledgeBaseClosedError):
            call()


def test_wrappers_redact_unknown_and_malformed_lower_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kb, document = _create_synced_file_base(tmp_path)

    with pytest.raises(
        KnowledgeBaseSourceError, match="^unable to inspect knowledge base$"
    ):
        kb.inspect_document("missing-private-document")

    def fail_inspect_documents(*args, **kwargs):
        raise RuntimeError("private source /srv/secret/whole-base.txt")

    monkeypatch.setattr(kb._collection, "inspect_documents", fail_inspect_documents)
    with pytest.raises(KnowledgeBaseSourceError) as inspection_error:
        kb.inspect()
    assert str(inspection_error.value) == "unable to inspect knowledge base"
    formatted_inspection = "".join(
        traceback.format_exception(
            type(inspection_error.value),
            inspection_error.value,
            inspection_error.value.__traceback__,
        )
    )
    assert "unable to inspect knowledge base" in formatted_inspection
    assert "secret" not in formatted_inspection
    assert "/srv" not in formatted_inspection

    def fail_diagnose(*args, **kwargs):
        raise RuntimeError("provider failure for private query and /private/path")

    monkeypatch.setattr(kb._collection, "diagnose_search", fail_diagnose)
    sensitive_query = "private query"
    with pytest.raises(KnowledgeBaseSourceError) as caught:
        kb.diagnose_search(sensitive_query)
    assert str(caught.value) == "unable to diagnose knowledge search"
    formatted_diagnosis = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    assert "unable to diagnose knowledge search" in formatted_diagnosis
    assert sensitive_query not in formatted_diagnosis
    assert "/private/path" not in formatted_diagnosis

    monkeypatch.setattr(kb._collection, "diagnose_search", lambda *args, **kwargs: object())
    with pytest.raises(
        KnowledgeBaseSourceError, match="^unable to diagnose knowledge search$"
    ):
        kb.diagnose_search("alpha")
