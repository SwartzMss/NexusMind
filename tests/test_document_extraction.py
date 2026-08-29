from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from nexusmind import (
    ANYDOC_CONTENT_TYPES,
    DEFAULT_SUPPORTED_EXTENSIONS,
    AnyDocExtractor,
    DocumentExtractionError,
    KnowledgeCollection,
    LocalDirectoryAdapter,
    PlainTextDocumentExtractor,
    UnsupportedDocumentFormatError,
    select_document_extractor,
)


FIXTURES = Path(__file__).parent / "fixtures" / "structured"
STRUCTURED_EXTENSIONS = {
    ".doc",
    ".docx",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
    ".epub",
    ".odt",
}


def test_default_selection_routes_only_enabled_structured_formats_to_anydoc() -> None:
    assert set(ANYDOC_CONTENT_TYPES) == STRUCTURED_EXTENSIONS
    assert STRUCTURED_EXTENSIONS <= DEFAULT_SUPPORTED_EXTENSIONS
    assert isinstance(select_document_extractor("notes.txt"), PlainTextDocumentExtractor)

    for extension in STRUCTURED_EXTENSIONS:
        assert isinstance(select_document_extractor(f"document{extension}"), AnyDocExtractor)

    for extension in (".xls", ".xlsx", ".xlsm", ".xlsb", ".csv"):
        assert extension not in DEFAULT_SUPPORTED_EXTENSIONS
        with pytest.raises(DocumentExtractionError):
            select_document_extractor(f"table{extension}")


def test_anydoc_extractor_consumes_bytes_and_propagates_original_format() -> None:
    class RecordingBackend:
        def to_markdown_bytes(self, data: bytes) -> str:
            assert data == b"verified structured bytes"
            return "# Extracted\n\nmarker"

    extracted = AnyDocExtractor(RecordingBackend()).extract(
        b"verified structured bytes",
        logical_path="architecture.DOCX",
    )

    assert extracted.content == "# Extracted\n\nmarker"
    assert extracted.content_type == ANYDOC_CONTENT_TYPES[".docx"]
    assert extracted.metadata == {"extractor": "anydoc", "source_format": "docx"}


def test_anydoc_extractor_normalizes_backend_and_format_failures() -> None:
    class FailingBackend:
        def to_markdown_bytes(self, data: bytes) -> str:
            raise RuntimeError("private PyO3 failure")

    with pytest.raises(DocumentExtractionError) as backend_error:
        AnyDocExtractor(FailingBackend()).extract(b"content", logical_path="broken.pdf")
    assert isinstance(backend_error.value.__cause__, RuntimeError)
    assert "private PyO3 failure" not in str(backend_error.value)

    with pytest.raises(UnsupportedDocumentFormatError):
        AnyDocExtractor(FailingBackend()).extract(b"content", logical_path="table.xlsx")

    with pytest.raises(UnsupportedDocumentFormatError):
        AnyDocExtractor().extract(b"not a document", logical_path="broken.pdf")


def test_mixed_directory_structured_markers_are_searchable_after_sync(tmp_path: Path) -> None:
    shutil.copyfile(FIXTURES / "marker.docx", tmp_path / "architecture.docx")
    shutil.copyfile(FIXTURES / "marker.pdf", tmp_path / "boundary.pdf")
    (tmp_path / "notes.txt").write_text("NEXUSMIND_TEXT_MARKER plain text", encoding="utf-8")

    adapter = LocalDirectoryAdapter(tmp_path, source_id="mixed-docs")
    documents = adapter.load_documents()

    assert [document.logical_path for document in documents] == [
        "architecture.docx",
        "boundary.pdf",
        "notes.txt",
    ]
    assert "NEXUSMIND_DOCX_MARKER" in documents[0].content
    assert "NEXUSMIND_PDF_MARKER" in documents[1].content
    assert documents[0].metadata == {"extractor": "anydoc", "source_format": "docx"}
    assert documents[1].content_type == "application/pdf"

    collection = KnowledgeCollection()
    collection.sync(adapter)

    assert collection.search("architecture boundary")
    assert collection.search("verified text layer")
    assert collection.search("plain text")
