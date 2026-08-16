"""Provider-neutral contracts for the first Knowledge Runtime boundary.

This module deliberately models knowledge as ``source -> document``.  It does
not know how a source is crawled, stored, chunked, embedded, or retrieved.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
from typing import Any, Mapping


class KnowledgeSourceType(str, Enum):
    """Common source categories without coupling the model to a provider."""

    LOCAL = "local"
    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"
    WEB = "web"
    MCP = "mcp"
    GITHUB = "github"
    GOOGLE_DRIVE = "google_drive"
    DATABASE = "database"
    OTHER = "other"


# ``SourceType`` is a short, convenient name for callers that do not need the
# longer domain-specific enum name.
SourceType = KnowledgeSourceType


def compute_content_hash(content: str | bytes) -> str:
    """Return the deterministic SHA-256 hash used for document change checks.

    Text is encoded as UTF-8.  Bytes are hashed as-is so future source
    adapters can represent non-text content without changing this contract.
    """

    if isinstance(content, str):
        raw = content.encode("utf-8")
    elif isinstance(content, bytes):
        raw = content
    else:
        raise TypeError("Document content must be str or bytes")
    return hashlib.sha256(raw).hexdigest()


def _require_text(value: str, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _stable_id(prefix: str, *parts: str) -> str:
    canonical = "\x00".join(parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(canonical).hexdigest()}"


def stable_source_id(source_type: KnowledgeSourceType | str, logical_location: str) -> str:
    """Create a stable source ID from source-neutral identifying values."""

    source_value = source_type.value if isinstance(source_type, KnowledgeSourceType) else str(source_type)
    return _stable_id("source", source_value, _require_text(logical_location, "logical_location"))


def stable_document_id(source_id: str, logical_path: str) -> str:
    """Create a stable document ID without using an absolute host path."""

    return _stable_id("document", _require_text(source_id, "source_id"), _require_text(logical_path, "logical_path"))


@dataclass(frozen=True, slots=True, init=False)
class KnowledgeSource:
    """The origin of one or more knowledge documents.

    ``source_id`` is preferred when a host already owns a durable identifier.
    When omitted, it is derived from ``source_type`` and the logical location
    (or display name).  The location is intentionally metadata rather than a
    ``Path`` so the core contract remains usable for non-filesystem sources.
    """

    source_id: str
    source_type: KnowledgeSourceType | str
    display_name: str
    logical_location: str | None
    metadata: dict[str, Any] = field(repr=False)

    def __init__(
        self,
        source_id: str | None = None,
        source_type: KnowledgeSourceType | str = KnowledgeSourceType.OTHER,
        display_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        id: str | None = None,
        logical_location: str | None = None,
        name: str | None = None,
        location: str | None = None,
    ) -> None:
        if id is not None:
            if source_id is not None and source_id != id:
                raise ValueError("source_id and id must match when both are provided")
            source_id = id
        if name is not None:
            if display_name is not None and display_name != name:
                raise ValueError("display_name and name must match when both are provided")
            display_name = name
        if location is not None:
            if logical_location is not None and logical_location != location:
                raise ValueError("logical_location and location must match when both are provided")
            logical_location = location

        source_value = source_type.value if isinstance(source_type, KnowledgeSourceType) else _require_text(source_type, "source_type")
        display_value = display_name or logical_location or source_id
        display_value = _require_text(display_value, "display_name")
        if source_id is None:
            source_id = stable_source_id(source_value, logical_location or display_value)
        source_id = _require_text(source_id, "source_id")
        if logical_location is not None:
            logical_location = _require_text(logical_location, "logical_location")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("KnowledgeSource metadata must be a mapping")

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "display_name", display_value)
        object.__setattr__(self, "logical_location", logical_location)
        object.__setattr__(self, "metadata", deepcopy(dict(metadata)))

    @property
    def id(self) -> str:
        return self.source_id

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def location(self) -> str | None:
        return self.logical_location


@dataclass(frozen=True, slots=True, init=False)
class Document:
    """One source-neutral knowledge document.

    The identity is the pair ``(source_id, logical_path)``.  ``document_id``
    is derived from that pair by default, so moving a local source to another
    machine does not change document identity merely because its absolute path
    changed.
    """

    document_id: str
    source_id: str
    logical_path: str
    content: str | bytes
    content_type: str
    metadata: dict[str, Any] = field(repr=False)
    content_hash: str
    imported_at: datetime | None
    updated_at: datetime | None

    def __init__(
        self,
        source_id: str | KnowledgeSource | None = None,
        logical_path: str | None = None,
        content: str | bytes = "",
        *,
        document_id: str | None = None,
        id: str | None = None,
        source: str | KnowledgeSource | None = None,
        path: str | None = None,
        content_type: str = "text/plain",
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        content_hash: str | None = None,
        imported_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        if source is not None:
            source_value = source.source_id if isinstance(source, KnowledgeSource) else source
            if source_id is not None and source_id != source_value:
                raise ValueError("source_id and source must identify the same source")
            source_id = source_value
        if path is not None:
            if logical_path is not None and logical_path != path:
                raise ValueError("logical_path and path must match when both are provided")
            logical_path = path
        if id is not None:
            if document_id is not None and document_id != id:
                raise ValueError("document_id and id must match when both are provided")
            document_id = id
        if mime_type is not None:
            if content_type != "text/plain" and content_type != mime_type:
                raise ValueError("content_type and mime_type must match when both are provided")
            content_type = mime_type

        if isinstance(source_id, KnowledgeSource):
            source_id = source_id.source_id
        source_id = _require_text(source_id, "source_id")
        logical_path = _require_text(logical_path, "logical_path")
        if not isinstance(content, (str, bytes)):
            raise TypeError("Document content must be str or bytes")
        content_type = _require_text(content_type, "content_type")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("Document metadata must be a mapping")

        calculated_hash = compute_content_hash(content)
        if content_hash is not None and content_hash != calculated_hash:
            raise ValueError("content_hash does not match content")
        resolved_id = document_id or stable_document_id(source_id, logical_path)
        resolved_id = _require_text(resolved_id, "document_id")

        object.__setattr__(self, "document_id", resolved_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "logical_path", logical_path)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "metadata", deepcopy(dict(metadata)))
        object.__setattr__(self, "content_hash", calculated_hash)
        object.__setattr__(self, "imported_at", imported_at)
        object.__setattr__(self, "updated_at", updated_at)

    @property
    def id(self) -> str:
        return self.document_id

    @property
    def path(self) -> str:
        return self.logical_path

    @property
    def source(self) -> str:
        return self.source_id

    @property
    def mime_type(self) -> str:
        return self.content_type

    @property
    def content_sha256(self) -> str:
        return self.content_hash

    @property
    def identity(self) -> tuple[str, str]:
        return (self.source_id, self.logical_path)

    def has_same_identity(self, other: Document) -> bool:
        return isinstance(other, Document) and self.identity == other.identity

    def has_content_changed(self, other: Document) -> bool:
        """Return whether another representation of this logical document changed."""

        if not isinstance(other, Document):
            raise TypeError("other must be a Document")
        if not self.has_same_identity(other):
            raise ValueError("content changes can only be compared for the same logical document")
        return self.content_hash != other.content_hash

    def content_changed_from(self, other: Document) -> bool:
        return self.has_content_changed(other)

    def has_changed(self, other: Document) -> bool:
        return self.has_content_changed(other)

    def is_content_changed(self, other: Document) -> bool:
        return self.has_content_changed(other)


KnowledgeDocument = Document


__all__ = [
    "Document",
    "KnowledgeDocument",
    "KnowledgeSource",
    "KnowledgeSourceType",
    "SourceType",
    "compute_content_hash",
    "stable_document_id",
    "stable_source_id",
]
