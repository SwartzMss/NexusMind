"""Provider-neutral extraction of verified document bytes into canonical text."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Mapping, Protocol, runtime_checkable

import anydoc


class KnowledgeIngestionError(Exception):
    """Base class for normalized ingestion failures."""


class DocumentExtractionError(KnowledgeIngestionError):
    """Base class for normalized document extraction failures."""


class InvalidTextEncodingError(DocumentExtractionError):
    """A plain-text document is not valid strict UTF-8 text."""


class DocumentExtractorNotFoundError(DocumentExtractionError):
    """No configured extractor accepts a document's logical path."""


class UnsupportedDocumentFormatError(DocumentExtractionError):
    """A structured document or its required conversion mode is unsupported."""


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Canonical text and format-neutral attributes returned by an extractor."""

    content: str
    content_type: str
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class DocumentExtractor(Protocol):
    """Extract canonical text from bytes already verified by a source adapter."""

    def extract(self, content: bytes, *, logical_path: str) -> ExtractedDocument:
        """Extract one document without reopening its source path."""


class PlainTextDocumentExtractor:
    """Extract strict UTF-8 text and distinguish plain text from Markdown."""

    def extract(self, content: bytes, *, logical_path: str) -> ExtractedDocument:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidTextEncodingError(
                f"document is not valid UTF-8: {logical_path}"
            ) from exc

        extension = PurePath(logical_path).suffix.lower()
        content_type = "text/markdown" if extension in {".md", ".markdown"} else "text/plain"
        return ExtractedDocument(content=text, content_type=content_type)


ANYDOC_CONTENT_TYPES: Mapping[str, str] = {
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".rtf": "application/rtf",
    ".epub": "application/epub+zip",
    ".odt": "application/vnd.oasis.opendocument.text",
}


class _AnyDocBackend(Protocol):
    def to_markdown_bytes(self, data: bytes) -> str: ...


class AnyDocExtractor:
    """Convert enabled structured formats from verified bytes to Markdown."""

    def __init__(self, backend: _AnyDocBackend = anydoc) -> None:
        self._backend = backend

    def extract(self, content: bytes, *, logical_path: str) -> ExtractedDocument:
        extension = PurePath(logical_path).suffix.lower()
        try:
            content_type = ANYDOC_CONTENT_TYPES[extension]
        except KeyError as exc:
            raise UnsupportedDocumentFormatError(
                f"unsupported structured document extension: {extension or '<none>'}"
            ) from exc

        try:
            markdown = self._backend.to_markdown_bytes(content)
        except (anydoc.UnsupportedError, anydoc.NeedsOcrError) as exc:
            raise UnsupportedDocumentFormatError(
                f"structured document format is unsupported: {logical_path}"
            ) from exc
        except Exception as exc:
            raise DocumentExtractionError(
                f"structured document extraction failed: {logical_path}"
            ) from exc

        return ExtractedDocument(
            content=markdown,
            content_type=content_type,
            metadata={"extractor": "anydoc", "source_format": extension.removeprefix(".")},
        )


_PLAIN_TEXT_EXTRACTOR = PlainTextDocumentExtractor()
_ANYDOC_EXTRACTOR = AnyDocExtractor()
DEFAULT_DOCUMENT_EXTRACTORS: Mapping[str, DocumentExtractor] = {
    ".md": _PLAIN_TEXT_EXTRACTOR,
    ".markdown": _PLAIN_TEXT_EXTRACTOR,
    ".txt": _PLAIN_TEXT_EXTRACTOR,
    **{extension: _ANYDOC_EXTRACTOR for extension in ANYDOC_CONTENT_TYPES},
}


def select_document_extractor(
    logical_path: str,
    *,
    extractors: Mapping[str, DocumentExtractor] = DEFAULT_DOCUMENT_EXTRACTORS,
    fallback: DocumentExtractor | None = None,
) -> DocumentExtractor:
    """Select an extractor by a logical path's case-insensitive extension."""

    extension = PurePath(logical_path).suffix.lower()
    try:
        return extractors[extension]
    except KeyError as exc:
        if fallback is not None:
            return fallback
        raise DocumentExtractorNotFoundError(
            f"no document extractor for extension: {extension or '<none>'}"
        ) from exc


__all__ = [
    "ANYDOC_CONTENT_TYPES",
    "AnyDocExtractor",
    "DEFAULT_DOCUMENT_EXTRACTORS",
    "DocumentExtractionError",
    "DocumentExtractor",
    "DocumentExtractorNotFoundError",
    "ExtractedDocument",
    "InvalidTextEncodingError",
    "KnowledgeIngestionError",
    "PlainTextDocumentExtractor",
    "UnsupportedDocumentFormatError",
    "select_document_extractor",
]
