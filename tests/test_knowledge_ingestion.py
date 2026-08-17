from __future__ import annotations

import os
from pathlib import Path

import pytest

from nexusmind import (
    DocumentCountLimitError,
    FileTooLargeError,
    InvalidTextEncodingError,
    KnowledgeIngestionError,
    KnowledgeSourceAdapter,
    LocalDirectoryAdapter,
    LocalFileAdapter,
    LocalIngestionLimits,
    SourceNotFoundError,
    SymlinkSourceError,
    TotalBytesLimitError,
    UnsupportedFileTypeError,
)


def test_local_file_adapter_maps_one_text_file_without_absolute_identity(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# NexusMind", encoding="utf-8")
    adapter = LocalFileAdapter(path, source_id="docs")

    assert isinstance(adapter, KnowledgeSourceAdapter)
    assert adapter.source().source_type == "local_file"
    assert adapter.source().logical_location == "README.md"
    document = adapter.load_documents()[0]

    assert document.source_id == "docs"
    assert document.logical_path == "README.md"
    assert str(tmp_path) not in document.logical_path
    assert document.content == "# NexusMind"
    assert document.content_type == "text/markdown"


def test_local_directory_adapter_recurses_and_orders_by_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    (root / "z").mkdir(parents=True)
    (root / "a.md").write_text("a", encoding="utf-8")
    (root / "z" / "nested.txt").write_text("nested", encoding="utf-8")
    (root / "ignored.py").write_text("ignored", encoding="utf-8")

    documents = LocalDirectoryAdapter(root, source_id="docs").load_documents()

    assert [document.logical_path for document in documents] == ["a.md", "z/nested.txt"]
    assert all(not Path(document.logical_path).is_absolute() for document in documents)
    assert all(str(root) not in document.logical_path for document in documents)


def test_same_source_and_logical_path_preserve_identity_across_hosts(tmp_path: Path) -> None:
    first = tmp_path / "first" / "notes.txt"
    second = tmp_path / "second" / "notes.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    original = LocalFileAdapter(first, source_id="notes").load_documents()[0]
    changed = LocalFileAdapter(second, source_id="notes").load_documents()[0]

    assert original.document_id == changed.document_id
    assert original.content_hash != changed.content_hash


def test_local_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        LocalFileAdapter(path, source_id="json").load_documents()


def test_local_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "broken.txt"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(InvalidTextEncodingError) as error:
        LocalFileAdapter(path, source_id="broken").load_documents()
    assert str(tmp_path) not in str(error.value)


def test_local_file_rejects_missing_and_oversized_sources(tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError):
        LocalFileAdapter(tmp_path / "missing.txt", source_id="missing").load_documents()

    path = tmp_path / "large.txt"
    path.write_text("12345", encoding="utf-8")
    limits = LocalIngestionLimits(max_file_bytes=4, max_documents=10, max_total_bytes=20)
    with pytest.raises(FileTooLargeError):
        LocalFileAdapter(path, source_id="large", limits=limits).load_documents()


def test_directory_enforces_document_and_total_byte_limits(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("123", encoding="utf-8")
    (tmp_path / "b.txt").write_text("456", encoding="utf-8")

    with pytest.raises(DocumentCountLimitError):
        LocalDirectoryAdapter(
            tmp_path,
            source_id="limited-count",
            limits=LocalIngestionLimits(max_documents=1, max_file_bytes=10, max_total_bytes=20),
        ).load_documents()
    with pytest.raises(TotalBytesLimitError):
        LocalDirectoryAdapter(
            tmp_path,
            source_id="limited-total",
            limits=LocalIngestionLimits(max_documents=10, max_file_bytes=10, max_total_bytes=5),
        ).load_documents()


def test_directory_skips_unsupported_files_but_rejects_invalid_supported_files(tmp_path: Path) -> None:
    (tmp_path / "ignored.bin").write_bytes(b"\xff")
    (tmp_path / "broken.md").write_bytes(b"\xff")

    with pytest.raises(InvalidTextEncodingError):
        LocalDirectoryAdapter(tmp_path, source_id="docs").load_documents()

    (tmp_path / "broken.md").unlink()
    assert LocalDirectoryAdapter(tmp_path, source_id="docs").load_documents() == ()


def test_root_symlink_is_rejected_and_nested_symlinks_are_skipped(tmp_path: Path) -> None:
    real_file = tmp_path / "real.txt"
    real_file.write_text("content", encoding="utf-8")
    root_link = tmp_path / "root-link.txt"
    nested_link = tmp_path / "nested-link.txt"
    try:
        root_link.symlink_to(real_file)
        nested_link.symlink_to(real_file)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable in this Windows environment")

    with pytest.raises(SymlinkSourceError):
        LocalFileAdapter(root_link, source_id="link").load_documents()
    assert LocalDirectoryAdapter(tmp_path, source_id="docs").load_documents()[0].logical_path == "real.txt"


def test_adapter_errors_are_controlled_and_do_not_expose_host_paths(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    with pytest.raises(KnowledgeIngestionError) as error:
        LocalFileAdapter(path, source_id="missing").source()
    assert str(tmp_path) not in str(error.value)


def test_relative_file_paths_are_normalized_for_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "docs" / "notes.txt"
    path.parent.mkdir()
    path.write_text("notes", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    document = LocalFileAdapter(os.path.join("docs", ".", "notes.txt"), source_id="docs").load_documents()[0]

    assert document.logical_path == "notes.txt"
