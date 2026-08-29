from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import nexusmind.knowledge_ingestion as knowledge_ingestion
from nexusmind import (
    DEFAULT_DOCUMENT_EXTRACTORS,
    DirectoryDepthLimitError,
    DocumentCountLimitError,
    DocumentExtractor,
    DocumentExtractorNotFoundError,
    EntryScanLimitError,
    FileIdentityChangedError,
    FileTooLargeError,
    ExtractedDocument,
    InvalidTextEncodingError,
    KnowledgeIngestionError,
    KnowledgeSourceAdapter,
    LocalDirectoryAdapter,
    LocalFileAdapter,
    LocalIngestionLimits,
    PlainTextDocumentExtractor,
    SourceNotFoundError,
    SymlinkSourceError,
    TotalBytesLimitError,
    UnsupportedFileTypeError,
    select_document_extractor,
)


def test_plain_text_extractor_selection_is_case_insensitive() -> None:
    markdown = select_document_extractor("guides/README.MD")
    text = select_document_extractor("notes.txt")

    assert isinstance(markdown, DocumentExtractor)
    assert markdown is DEFAULT_DOCUMENT_EXTRACTORS[".md"]
    assert text is DEFAULT_DOCUMENT_EXTRACTORS[".txt"]
    with pytest.raises(DocumentExtractorNotFoundError):
        select_document_extractor("data.json")


@pytest.mark.parametrize(
    ("logical_path", "content_type"),
    [
        ("notes.txt", "text/plain"),
        ("README.md", "text/markdown"),
        ("guide.markdown", "text/markdown"),
    ],
)
def test_plain_text_extractor_returns_canonical_text_and_content_type(
    logical_path: str,
    content_type: str,
) -> None:
    extracted = PlainTextDocumentExtractor().extract(
        "NexusMind 文档".encode(),
        logical_path=logical_path,
    )

    assert extracted == ExtractedDocument(
        content="NexusMind 文档",
        content_type=content_type,
        metadata={},
    )


def test_plain_text_extractor_normalizes_strict_utf8_failures() -> None:
    with pytest.raises(InvalidTextEncodingError) as error:
        PlainTextDocumentExtractor().extract(b"\xff", logical_path="broken.txt")

    assert "broken.txt" in str(error.value)


def test_adapter_propagates_extracted_content_type_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(b"verified bytes")

    class RecordingExtractor:
        def extract(self, content: bytes, *, logical_path: str) -> ExtractedDocument:
            assert content == b"verified bytes"
            assert logical_path == "notes.txt"
            return ExtractedDocument(
                content="canonical text",
                content_type="text/x-test",
                metadata={"extractor": "recording"},
            )

    monkeypatch.setattr(
        knowledge_ingestion,
        "select_document_extractor",
        lambda logical_path, **kwargs: RecordingExtractor(),
    )

    document = LocalFileAdapter(path, source_id="docs").load_documents()[0]

    assert document.content == "canonical text"
    assert document.content_type == "text/x-test"
    assert document.metadata == {"extractor": "recording"}


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
    (root / "a").mkdir(parents=True)
    (root / "z").mkdir(parents=True)
    (root / "a.md").write_text("a", encoding="utf-8")
    (root / "a" / "x.txt").write_text("nested prefix", encoding="utf-8")
    (root / "z" / "nested.txt").write_text("nested", encoding="utf-8")
    (root / "ignored.py").write_text("ignored", encoding="utf-8")

    documents = LocalDirectoryAdapter(root, source_id="docs").load_documents()

    assert [document.logical_path for document in documents] == ["a.md", "a/x.txt", "z/nested.txt"]
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


def test_custom_supported_text_extension_preserves_plain_text_behavior(tmp_path: Path) -> None:
    path = tmp_path / "notes.text"
    path.write_text("custom extension", encoding="utf-8")

    document = LocalFileAdapter(
        path,
        source_id="custom",
        supported_extensions={".text"},
    ).load_documents()[0]

    assert document.content == "custom extension"
    assert document.content_type == "text/plain"


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


def test_directory_bounds_unsupported_entry_scanning(tmp_path: Path) -> None:
    for name in ("a.py", "b.bin", "c.json"):
        (tmp_path / name).write_text("ignored", encoding="utf-8")

    with pytest.raises(EntryScanLimitError):
        LocalDirectoryAdapter(
            tmp_path,
            source_id="bounded-scan",
            limits=LocalIngestionLimits(
                max_entries_scanned=2,
                max_documents=10,
                max_file_bytes=100,
                max_total_bytes=1_000,
            ),
        ).load_documents()


def test_directory_bounds_nesting_depth_without_python_recursion(tmp_path: Path) -> None:
    nested = tmp_path / "level-one"
    (nested / "level-two").mkdir(parents=True)
    (nested / "level-two" / "notes.txt").write_text("notes", encoding="utf-8")

    with pytest.raises(DirectoryDepthLimitError):
        LocalDirectoryAdapter(
            tmp_path,
            source_id="bounded-depth",
            limits=LocalIngestionLimits(max_directory_depth=0),
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
    real_directory = tmp_path / "real-directory"
    real_directory.mkdir()
    (real_directory / "inside.txt").write_text("inside", encoding="utf-8")
    root_link = tmp_path / "root-link.txt"
    nested_link = tmp_path / "nested-link.txt"
    root_directory_link = tmp_path / "root-directory-link"
    nested_directory_link = tmp_path / "nested-directory-link"
    outside_file = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside_file.write_text("outside", encoding="utf-8")
    outside_link = tmp_path / "outside-link.txt"
    try:
        root_link.symlink_to(real_file)
        nested_link.symlink_to(real_file)
        root_directory_link.symlink_to(real_directory, target_is_directory=True)
        nested_directory_link.symlink_to(real_directory, target_is_directory=True)
        outside_link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable in this Windows environment")

    with pytest.raises(SymlinkSourceError):
        LocalFileAdapter(root_link, source_id="link").load_documents()
    with pytest.raises(SymlinkSourceError):
        LocalDirectoryAdapter(root_directory_link, source_id="link").load_documents()

    logical_paths = {
        document.logical_path for document in LocalDirectoryAdapter(tmp_path, source_id="docs").load_documents()
    }
    assert "real.txt" in logical_paths
    assert "real-directory/inside.txt" in logical_paths
    assert "nested-link.txt" not in logical_paths
    assert "nested-directory-link/inside.txt" not in logical_paths
    assert "outside-link.txt" not in logical_paths
    assert not any(path.startswith("..") for path in logical_paths)

    outside_file.unlink()


def test_read_rejects_path_identity_change_after_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "notes.txt"
    other = tmp_path / "other.txt"
    path.write_text("notes", encoding="utf-8")
    other.write_text("other", encoding="utf-8")
    expected_identity = knowledge_ingestion._capture_file_identity(path)
    real_stat = knowledge_ingestion.os.stat

    def mismatching_stat(candidate, *args, **kwargs):
        if Path(candidate) == path and kwargs.get("follow_symlinks") is False:
            return real_stat(other, *args, **kwargs)
        return real_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(knowledge_ingestion.os, "stat", mismatching_stat)
    with pytest.raises(FileIdentityChangedError):
        knowledge_ingestion._read_verified_bytes(
            path,
            root=tmp_path,
            expected_identity=expected_identity,
            logical_path="notes.txt",
            limits=LocalIngestionLimits(),
            total_bytes_before=0,
        )


@pytest.mark.parametrize("adapter_kind", ["file", "directory"])
def test_adapter_rejects_file_replaced_between_discovery_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter_kind: str,
) -> None:
    path = tmp_path / "notes.txt"
    replacement = tmp_path / "replacement.bin"
    path.write_text("original", encoding="utf-8")
    replacement.write_text("replacement", encoding="utf-8")
    real_read_document = knowledge_ingestion._read_document

    def replace_before_open(candidate: Path, **kwargs):
        if candidate == path and replacement.exists():
            replacement.replace(path)
        return real_read_document(candidate, **kwargs)

    monkeypatch.setattr(knowledge_ingestion, "_read_document", replace_before_open)
    adapter = (
        LocalFileAdapter(path, source_id="docs")
        if adapter_kind == "file"
        else LocalDirectoryAdapter(tmp_path, source_id="docs")
    )

    with pytest.raises(FileIdentityChangedError):
        adapter.load_documents()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction policy")
def test_windows_junction_root_is_rejected_and_nested_junction_is_skipped(tmp_path: Path) -> None:
    source_root = tmp_path / "docs"
    target = tmp_path / "junction-target"
    source_root.mkdir()
    target.mkdir()
    (target / "outside.txt").write_text("outside", encoding="utf-8")
    junction = source_root / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", os.fspath(junction), os.fspath(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("junction creation is unavailable in this Windows environment")

    try:
        assert knowledge_ingestion._is_symlink_or_reparse_point(junction)
        assert LocalDirectoryAdapter(source_root, source_id="docs").load_documents() == ()
        with pytest.raises(SymlinkSourceError):
            LocalDirectoryAdapter(junction, source_id="junction").load_documents()
    finally:
        junction.rmdir()


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
