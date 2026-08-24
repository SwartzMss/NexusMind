"""Provider-neutral contracts for the first Knowledge Runtime boundary.

This module deliberately models knowledge as ``source -> document``. It does
not know how a source is crawled, stored, chunked, embedded, or retrieved.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
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


def stable_document_id(source_id: str, logical_path: str) -> str:
    """Create a stable document ID from source and source-relative identity."""

    return _stable_id("document", _require_text(source_id, "source_id"), _require_text(logical_path, "logical_path"))


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_ID_PATTERN = re.compile(r"version-[0-9a-f]{64}\Z")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z"
)


def stable_document_version_id(
    document_id: str,
    content_hash: str,
    previous_version_id: str | None,
) -> str:
    """Create a stable identity for one linked document version."""

    document_id = _require_text(document_id, "document_id")
    if _SHA256_PATTERN.fullmatch(content_hash) is None:
        raise ValueError("content_hash must be a lowercase SHA-256 digest")
    if previous_version_id is not None:
        if type(previous_version_id) is not str or _VERSION_ID_PATTERN.fullmatch(
            previous_version_id
        ) is None:
            raise ValueError("previous_version_id must be a version ID or None")
    predecessor = "" if previous_version_id is None else previous_version_id
    return _stable_id("version", document_id, content_hash, predecessor)


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeSource:
    """The origin of one or more knowledge documents."""

    source_id: str
    source_type: KnowledgeSourceType | str
    display_name: str
    logical_location: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        source_id = _require_text(self.source_id, "source_id")
        if isinstance(self.source_type, KnowledgeSourceType):
            source_type = self.source_type
        else:
            source_type = _require_text(self.source_type, "source_type")
        display_name = _require_text(self.display_name, "display_name")
        logical_location = self.logical_location
        if logical_location is not None:
            logical_location = _require_text(logical_location, "logical_location")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("KnowledgeSource metadata must be a mapping")

        object.__setattr__(self, "source_type", source_type)
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
    document_id: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_id = _require_text(self.source_id, "source_id")
        logical_path = _require_text(self.logical_path, "logical_path")
        content_type = _require_text(self.content_type, "content_type")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Document metadata must be a mapping")

        calculated_hash = compute_content_hash(self.content)
        document_id = stable_document_id(source_id, logical_path)

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


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """One immutable content snapshot in a document's internal version chain."""

    version_id: str
    document_id: str
    source_id: str
    logical_path: str
    content: str
    content_hash: str
    created_at: str
    previous_version_id: str | None
    sync_context: str

    def __post_init__(self) -> None:
        source_id = _require_text(self.source_id, "source_id")
        logical_path = _require_text(self.logical_path, "logical_path")
        document_id = _require_text(self.document_id, "document_id")
        sync_context = _require_text(self.sync_context, "sync_context")
        if type(self.content) is not str:
            raise TypeError("content must be a string")
        if type(self.created_at) is not str or _UTC_TIMESTAMP_PATTERN.fullmatch(
            self.created_at
        ) is None:
            raise ValueError("created_at must be a canonical UTC timestamp")
        if compute_content_hash(self.content) != self.content_hash:
            raise ValueError("content_hash does not match content")
        if stable_document_id(source_id, logical_path) != document_id:
            raise ValueError("document_id does not match provenance")
        expected_version_id = stable_document_version_id(
            document_id, self.content_hash, self.previous_version_id
        )
        if self.version_id != expected_version_id:
            raise ValueError("version_id does not match version identity")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "logical_path", logical_path)
        object.__setattr__(self, "document_id", document_id)
        object.__setattr__(self, "sync_context", sync_context)

    @classmethod
    def from_document(
        cls,
        document: Document,
        *,
        created_at: str,
        sync_context: str,
        previous_version_id: str | None = None,
    ) -> "DocumentVersion":
        if not isinstance(document, Document):
            raise TypeError("document must be a Document")
        return cls(
            version_id=stable_document_version_id(
                document.document_id,
                document.content_hash,
                previous_version_id,
            ),
            document_id=document.document_id,
            source_id=document.source_id,
            logical_path=document.logical_path,
            content=document.content,
            content_hash=document.content_hash,
            created_at=created_at,
            previous_version_id=previous_version_id,
            sync_context=sync_context,
        )


__all__ = [
    "Document",
    "DocumentVersion",
    "KnowledgeSource",
    "KnowledgeSourceType",
    "compute_content_hash",
    "stable_document_id",
    "stable_document_version_id",
]
