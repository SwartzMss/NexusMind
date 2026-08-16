"""Provider-neutral contracts for the first Knowledge Runtime boundary.

This module deliberately models knowledge as ``source -> document``. It does
not know how a source is crawled, stored, chunked, embedded, or retrieved.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class KnowledgeSourceType(str, Enum):
    """Source types supported by the first local validation scenario."""

    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"


def compute_content_hash(content: str) -> str:
    """Return the deterministic UTF-8 SHA-256 hash for document content."""

    if type(content) is not str:
        raise TypeError("Document content must be a string")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _require_text(value: str | None, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _stable_id(prefix: str, *parts: str) -> str:
    # JSON array encoding preserves part boundaries, including when a part
    # contains NUL or other separators.
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(canonical).hexdigest()}"


def stable_source_id(source_type: KnowledgeSourceType | str, logical_location: str) -> str:
    """Create a stable source ID from source type and logical location."""

    source_value = source_type.value if isinstance(source_type, KnowledgeSourceType) else _require_text(source_type, "source_type")
    return _stable_id("source", source_value, _require_text(logical_location, "logical_location"))


def stable_document_id(source_id: str, logical_path: str) -> str:
    """Create a stable document ID from source and source-relative identity."""

    return _stable_id("document", _require_text(source_id, "source_id"), _require_text(logical_path, "logical_path"))


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeSource:
    """The origin of one or more knowledge documents."""

    source_type: KnowledgeSourceType | str
    display_name: str
    logical_location: str | None = None
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        source_value = self.source_type.value if isinstance(self.source_type, KnowledgeSourceType) else _require_text(self.source_type, "source_type")
        display_name = _require_text(self.display_name, "display_name")
        source_id = self.source_id
        if source_id is None:
            logical_location = _require_text(self.logical_location, "logical_location")
            source_id = stable_source_id(source_value, logical_location)
        else:
            source_id = _require_text(source_id, "source_id")
            logical_location = self.logical_location
            if logical_location is not None:
                logical_location = _require_text(logical_location, "logical_location")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("KnowledgeSource metadata must be a mapping")

        object.__setattr__(self, "source_type", self.source_type)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "logical_location", logical_location)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Document:
    """One source-neutral text document."""

    source_id: str
    logical_path: str
    content: str
    content_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    document_id: str | None = None
    content_hash: str | None = None
    imported_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        source_id = _require_text(self.source_id, "source_id")
        logical_path = _require_text(self.logical_path, "logical_path")
        content_type = _require_text(self.content_type, "content_type")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Document metadata must be a mapping")

        calculated_hash = compute_content_hash(self.content)
        if self.content_hash is not None and self.content_hash != calculated_hash:
            raise ValueError("content_hash does not match content")
        document_id = self.document_id or stable_document_id(source_id, logical_path)
        document_id = _require_text(document_id, "document_id")

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "logical_path", logical_path)
        object.__setattr__(self, "content_type", content_type)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "content_hash", calculated_hash)

    @property
    def identity(self) -> tuple[str, str]:
        return (self.source_id, self.logical_path)

    def has_content_changed(self, other: "Document") -> bool:
        """Return whether another representation of this logical document changed."""

        if not isinstance(other, Document):
            raise TypeError("other must be a Document")
        if self.identity != other.identity:
            raise ValueError("content changes can only be compared for the same logical document")
        return self.content_hash != other.content_hash


__all__ = [
    "Document",
    "KnowledgeSource",
    "KnowledgeSourceType",
    "compute_content_hash",
    "stable_document_id",
    "stable_source_id",
]
