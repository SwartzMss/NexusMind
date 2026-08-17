"""Bounded local adapters for the provider-neutral Knowledge contracts.

This module owns filesystem discovery and text decoding.  The Knowledge Core
only receives ``KnowledgeSource`` and ``Document`` objects, so future adapters
can use the same boundary without adding source-specific fields to those
contracts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from .knowledge import Document, KnowledgeSource, KnowledgeSourceType


DEFAULT_SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt"})
DEFAULT_MAX_FILE_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_DOCUMENTS = 1_000
DEFAULT_MAX_TOTAL_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_ENTRIES_SCANNED = 10_000
DEFAULT_MAX_DIRECTORY_DEPTH = 32


class KnowledgeIngestionError(Exception):
    """Base class for normalized local ingestion failures."""


class SourceNotFoundError(KnowledgeIngestionError):
    """The configured local source does not exist."""


class SourceTypeError(KnowledgeIngestionError):
    """The configured path is not the source type required by an adapter."""


class SymlinkSourceError(KnowledgeIngestionError):
    """A configured source or path component is a symbolic link."""


class UnsupportedFileTypeError(KnowledgeIngestionError):
    """A local file does not use one of the adapter's supported extensions."""


class InvalidTextEncodingError(KnowledgeIngestionError):
    """A supported local file is not valid strict UTF-8 text."""


class FileTooLargeError(KnowledgeIngestionError):
    """A local file exceeds the configured per-file byte limit."""


class DocumentCountLimitError(KnowledgeIngestionError):
    """A directory contains more accepted documents than allowed."""


class TotalBytesLimitError(KnowledgeIngestionError):
    """Accepted documents exceed the configured total byte limit."""


class EntryScanLimitError(KnowledgeIngestionError):
    """A directory scan encountered more filesystem entries than allowed."""


class DirectoryDepthLimitError(KnowledgeIngestionError):
    """A directory scan exceeded the configured nesting depth."""


class PathEscapeError(KnowledgeIngestionError):
    """A discovered path cannot be proven to remain under the source root."""


@runtime_checkable
class KnowledgeSourceAdapter(Protocol):
    """Convert one external source into Knowledge Core contracts."""

    def source(self) -> KnowledgeSource:
        """Describe the external source without loading its documents."""

    def load_documents(self) -> tuple[Document, ...]:
        """Load documents from the source in a deterministic order."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalIngestionLimits:
    """Resource limits applied by the local adapters.

    ``max_entries_scanned`` counts every child filesystem entry inspected,
    including unsupported files, directories, and symlinks.  Directory depth
    is measured from the configured root, where the root itself is depth 0.
    """

    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_documents: int = DEFAULT_MAX_DOCUMENTS
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_entries_scanned: int = DEFAULT_MAX_ENTRIES_SCANNED
    max_directory_depth: int = DEFAULT_MAX_DIRECTORY_DEPTH

    def __post_init__(self) -> None:
        if type(self.max_file_bytes) is not int or self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        if type(self.max_documents) is not int or self.max_documents <= 0:
            raise ValueError("max_documents must be a positive integer")
        if type(self.max_total_bytes) is not int or self.max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be a positive integer")
        if type(self.max_entries_scanned) is not int or self.max_entries_scanned <= 0:
            raise ValueError("max_entries_scanned must be a positive integer")
        if type(self.max_directory_depth) is not int or self.max_directory_depth < 0:
            raise ValueError("max_directory_depth must be a non-negative integer")


def _normalize_extensions(extensions: Iterable[str] | None) -> frozenset[str]:
    values = DEFAULT_SUPPORTED_EXTENSIONS if extensions is None else extensions
    if isinstance(values, (str, bytes)):
        raise TypeError("supported_extensions must be an iterable of strings")

    normalized: set[str] = set()
    for extension in values:
        if type(extension) is not str or not extension.strip():
            raise ValueError("supported extensions must be non-empty strings")
        extension = extension.strip().lower()
        if not extension.startswith("."):
            raise ValueError("supported extensions must start with a dot")
        normalized.add(extension)
    if not normalized:
        raise ValueError("supported_extensions must not be empty")
    return frozenset(normalized)


def _path_from_input(path: str | os.PathLike[str]) -> Path:
    try:
        raw_path = os.fspath(path)
    except TypeError as exc:
        raise TypeError("path must be a filesystem path") from exc
    if isinstance(raw_path, bytes):
        raise TypeError("path must be a text filesystem path")
    if type(raw_path) is not str or not raw_path.strip():
        raise ValueError("path must be a non-empty filesystem path")
    return Path(raw_path)


def _reject_symlink_components(path: Path) -> None:
    """Reject symlink path components without exposing the absolute path."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = absolute
    while True:
        try:
            is_symlink = current.is_symlink()
        except OSError as exc:
            raise KnowledgeIngestionError("source path could not be inspected") from exc
        if is_symlink:
            raise SymlinkSourceError("symbolic links are not supported")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _validate_root(path: str | os.PathLike[str], expected: str) -> Path:
    candidate = _path_from_input(path)
    _reject_symlink_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SourceNotFoundError("configured source does not exist") from exc
    except OSError as exc:
        raise KnowledgeIngestionError("configured source could not be inspected") from exc

    try:
        is_expected = resolved.is_file() if expected == "file" else resolved.is_dir()
    except OSError as exc:
        raise KnowledgeIngestionError("configured source could not be inspected") from exc
    if not is_expected:
        raise SourceTypeError(f"configured source must be a {expected}")
    return resolved


def _display_name(path: Path, fallback: str) -> str:
    return path.name or fallback


def _content_type(path: Path) -> str:
    if path.suffix.lower() in {".md", ".markdown"}:
        return "text/markdown"
    return "text/plain"


def _read_document(
    path: Path,
    *,
    source_id: str,
    logical_path: str,
    limits: LocalIngestionLimits,
    total_bytes_before: int,
) -> tuple[Document, int]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise KnowledgeIngestionError("document could not be inspected") from exc
    if file_size > limits.max_file_bytes:
        raise FileTooLargeError(f"document exceeds the file-size limit: {logical_path}")
    if total_bytes_before + file_size > limits.max_total_bytes:
        raise TotalBytesLimitError("documents exceed the total-byte limit")

    try:
        content_bytes = path.read_bytes()
    except OSError as exc:
        raise KnowledgeIngestionError("document could not be read") from exc
    if len(content_bytes) > limits.max_file_bytes:
        raise FileTooLargeError(f"document exceeds the file-size limit: {logical_path}")
    if total_bytes_before + len(content_bytes) > limits.max_total_bytes:
        raise TotalBytesLimitError("documents exceed the total-byte limit")
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidTextEncodingError(f"document is not valid UTF-8: {logical_path}") from exc

    return (
        Document(
            source_id=source_id,
            logical_path=logical_path,
            content=content,
            content_type=_content_type(path),
        ),
        len(content_bytes),
    )


class LocalFileAdapter:
    """Load one bounded, strict-UTF-8 local text file as one document."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        source_id: str,
        limits: LocalIngestionLimits | None = None,
        supported_extensions: Iterable[str] | None = None,
    ) -> None:
        self._path = _path_from_input(path)
        self._source_id = source_id
        self._limits = limits or LocalIngestionLimits()
        self._supported_extensions = _normalize_extensions(supported_extensions)

    def source(self) -> KnowledgeSource:
        root = _validate_root(self._path, "file")
        return KnowledgeSource(
            source_id=self._source_id,
            source_type=KnowledgeSourceType.LOCAL_FILE,
            display_name=_display_name(root, "local-file"),
            logical_location=_display_name(root, "local-file"),
        )

    def load_documents(self) -> tuple[Document, ...]:
        root = _validate_root(self._path, "file")
        if root.suffix.lower() not in self._supported_extensions:
            raise UnsupportedFileTypeError(f"unsupported document extension: {root.suffix.lower() or '<none>'}")
        document, _ = _read_document(
            root,
            source_id=self._source_id,
            logical_path=_display_name(root, "document"),
            limits=self._limits,
            total_bytes_before=0,
        )
        return (document,)


class LocalDirectoryAdapter:
    """Recursively load supported local text files in deterministic order."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        source_id: str,
        limits: LocalIngestionLimits | None = None,
        supported_extensions: Iterable[str] | None = None,
    ) -> None:
        self._path = _path_from_input(path)
        self._source_id = source_id
        self._limits = limits or LocalIngestionLimits()
        self._supported_extensions = _normalize_extensions(supported_extensions)

    def source(self) -> KnowledgeSource:
        root = _validate_root(self._path, "directory")
        return KnowledgeSource(
            source_id=self._source_id,
            source_type=KnowledgeSourceType.LOCAL_DIRECTORY,
            display_name=_display_name(root, "local-directory"),
            logical_location=_display_name(root, "local-directory"),
        )

    def load_documents(self) -> tuple[Document, ...]:
        root = _validate_root(self._path, "directory")
        documents: list[Document] = []
        total_bytes = 0
        for path, logical_path in self._discover_supported_files(root):
            if len(documents) >= self._limits.max_documents:
                raise DocumentCountLimitError("documents exceed the document-count limit")
            document, loaded_bytes = _read_document(
                path,
                source_id=self._source_id,
                logical_path=logical_path,
                limits=self._limits,
                total_bytes_before=total_bytes,
            )
            documents.append(document)
            total_bytes += loaded_bytes
        return tuple(documents)

    def _discover_supported_files(self, root: Path) -> list[tuple[Path, str]]:
        """Discover supported files with explicit, bounded traversal state."""

        stack: list[tuple[Path, int]] = [(root, 0)]
        candidates: list[tuple[Path, str]] = []
        entries_scanned = 0

        while stack:
            directory, depth = stack.pop()
            child_directories: list[tuple[Path, int]] = []
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        entries_scanned += 1
                        if entries_scanned > self._limits.max_entries_scanned:
                            raise EntryScanLimitError("source scan exceeds the entry-count limit")

                        # The first version is deliberately fail-closed for the
                        # configured root and skips symlinks discovered below it.
                        entry_path = Path(entry.path)
                        try:
                            if entry.is_symlink():
                                continue
                            resolved = entry_path.resolve(strict=True)
                        except FileNotFoundError as exc:
                            raise KnowledgeIngestionError("source entry disappeared during scan") from exc
                        except OSError as exc:
                            raise KnowledgeIngestionError("source entry could not be inspected") from exc

                        try:
                            relative_path = resolved.relative_to(root).as_posix()
                            is_directory = resolved.is_dir()
                            is_file = resolved.is_file()
                        except ValueError as exc:
                            raise PathEscapeError("discovered path escaped the source root") from exc
                        except OSError as exc:
                            raise KnowledgeIngestionError("source entry could not be inspected") from exc

                        if is_directory:
                            if depth >= self._limits.max_directory_depth:
                                raise DirectoryDepthLimitError("source exceeds the directory-depth limit")
                            child_directories.append((resolved, depth + 1))
                        elif is_file and resolved.suffix.lower() in self._supported_extensions:
                            candidates.append((resolved, relative_path))
                            if len(candidates) > self._limits.max_documents:
                                raise DocumentCountLimitError("documents exceed the document-count limit")
            except EntryScanLimitError:
                raise
            except OSError as exc:
                raise KnowledgeIngestionError("source directory could not be scanned") from exc

            # Keep traversal deterministic while using an explicit stack:
            # the first directory alphabetically is processed first.
            stack.extend(reversed(child_directories))

        return sorted(candidates, key=lambda item: (item[1].casefold(), item[1]))


__all__ = [
    "DEFAULT_MAX_DIRECTORY_DEPTH",
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_ENTRIES_SCANNED",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "DirectoryDepthLimitError",
    "DocumentCountLimitError",
    "EntryScanLimitError",
    "FileTooLargeError",
    "InvalidTextEncodingError",
    "KnowledgeIngestionError",
    "KnowledgeSourceAdapter",
    "LocalDirectoryAdapter",
    "LocalFileAdapter",
    "LocalIngestionLimits",
    "PathEscapeError",
    "SourceNotFoundError",
    "SourceTypeError",
    "SymlinkSourceError",
    "TotalBytesLimitError",
    "UnsupportedFileTypeError",
]
