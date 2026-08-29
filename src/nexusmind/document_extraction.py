"""Provider-neutral extraction of verified document bytes into canonical text."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Mapping, Protocol, runtime_checkable


class DocumentExtractionError(Exception):
    """Base class for normalized document extraction failures."""


class KnowledgeIngestionError(DocumentExtractionError):
    """Base class for normalized ingestion failures, including extraction."""


class InvalidTextEncodingError(KnowledgeIngestionError):
    """A plain-text document is not valid strict UTF-8 text."""


class DocumentExtractorNotFoundError(DocumentExtractionError):
    """No configured extractor accepts a document's logical path."""


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


_PLAIN_TEXT_EXTRACTOR = PlainTextDocumentExtractor()
DEFAULT_DOCUMENT_EXTRACTORS: Mapping[str, DocumentExtractor] = {
    ".md": _PLAIN_TEXT_EXTRACTOR,
    ".markdown": _PLAIN_TEXT_EXTRACTOR,
    ".txt": _PLAIN_TEXT_EXTRACTOR,
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
    "DEFAULT_DOCUMENT_EXTRACTORS",
    "DocumentExtractionError",
    "DocumentExtractor",
    "DocumentExtractorNotFoundError",
    "ExtractedDocument",
    "InvalidTextEncodingError",
    "KnowledgeIngestionError",
    "PlainTextDocumentExtractor",
    "select_document_extractor",
]
