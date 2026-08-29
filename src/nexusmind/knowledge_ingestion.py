"""Bounded local adapters for the provider-neutral Knowledge contracts.

This module owns filesystem discovery and verified byte reads.  Extractors
turn those bytes into canonical text before the Knowledge Core
only receives ``KnowledgeSource`` and ``Document`` objects, so future adapters
can use the same boundary without adding source-specific fields to those
contracts.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

from .document_extraction import (
    DEFAULT_DOCUMENT_EXTRACTORS,
    InvalidTextEncodingError,
    KnowledgeIngestionError,
    PlainTextDocumentExtractor,
    select_document_extractor,
)
from .knowledge import Document, KnowledgeSource, KnowledgeSourceType


DEFAULT_SUPPORTED_EXTENSIONS = frozenset(DEFAULT_DOCUMENT_EXTRACTORS)
DEFAULT_MAX_FILE_BYTES = 1 * 1024 * 1024
DEFAULT_MAX_DOCUMENTS = 1_000
DEFAULT_MAX_TOTAL_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_ENTRIES_SCANNED = 10_000
DEFAULT_MAX_DIRECTORY_DEPTH = 32
_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_PLAIN_TEXT_EXTRACTOR = PlainTextDocumentExtractor()


class SourceNotFoundError(KnowledgeIngestionError):
    """The configured local source does not exist."""


class SourceTypeError(KnowledgeIngestionError):
    """The configured path is not the source type required by an adapter."""


class SymlinkSourceError(KnowledgeIngestionError):
    """A configured source or path component is a symlink or reparse point."""


class FileIdentityChangedError(KnowledgeIngestionError):
    """A discovered file was replaced before or during ingestion."""


class UnsupportedFileTypeError(KnowledgeIngestionError):
    """A local file does not use one of the adapter's supported extensions."""


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


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _DiscoveredFile:
    path: Path
    logical_path: str
    identity: _FileIdentity


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


def _is_symlink_or_reparse_point(path: Path) -> bool:
    """Return whether a path is a symlink or Windows reparse point."""

    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise KnowledgeIngestionError("source path could not be inspected") from exc
    return _file_stat_is_symlink_or_reparse(file_stat)


def _file_stat_is_symlink_or_reparse(file_stat: os.stat_result) -> bool:
    return stat.S_ISLNK(file_stat.st_mode) or bool(
        getattr(file_stat, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _reject_symlink_components(path: Path) -> None:
    """Reject symlink/reparse path components without exposing absolute paths."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = absolute
    while True:
        if _is_symlink_or_reparse_point(current):
            raise SymlinkSourceError("symbolic links and Windows reparse points are not supported")
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


def _file_identity(file_stat: os.stat_result) -> _FileIdentity:
    return _FileIdentity(device=file_stat.st_dev, inode=file_stat.st_ino)


def _capture_file_identity(path: Path) -> _FileIdentity:
    try:
        file_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise KnowledgeIngestionError("document could not be inspected") from exc
    if _file_stat_is_symlink_or_reparse(file_stat):
        raise SymlinkSourceError("symbolic links and Windows reparse points are not supported")
    if not stat.S_ISREG(file_stat.st_mode):
        raise FileIdentityChangedError("document changed type during discovery")
    return _file_identity(file_stat)


def _read_verified_bytes(
    path: Path,
    *,
    root: Path,
    expected_identity: _FileIdentity,
    logical_path: str,
    limits: LocalIngestionLimits,
    total_bytes_before: int,
) -> bytes:
    """Read from an opened handle after rechecking path and file identity.

    Opening first makes subsequent path changes unable to redirect the bytes
    read from the handle.  The opened identity must match the identity captured
    during discovery, then no-follow path checks ensure the current path still
    names that same regular file before bytes are read.
    """

    file_descriptor: int | None = None
    try:
        try:
            file_descriptor = os.open(
                os.fspath(path),
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
            opened_stat = os.fstat(file_descriptor)
        except OSError as exc:
            raise KnowledgeIngestionError("document could not be opened") from exc

        opened_identity = _file_identity(opened_stat)
        if opened_identity != expected_identity:
            raise FileIdentityChangedError("document identity changed after discovery")
        if opened_stat.st_size > limits.max_file_bytes:
            raise FileTooLargeError(f"document exceeds the file-size limit: {logical_path}")
        remaining_bytes = limits.max_total_bytes - total_bytes_before
        if opened_stat.st_size > remaining_bytes:
            raise TotalBytesLimitError("documents exceed the total-byte limit")

        _reject_symlink_components(path)
        try:
            current_stat = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise KnowledgeIngestionError("document could not be inspected") from exc
        if _file_stat_is_symlink_or_reparse(current_stat):
            raise SymlinkSourceError("document changed to a symlink or reparse point during ingestion")
        if _file_identity(current_stat) != opened_identity:
            raise FileIdentityChangedError("document identity changed during ingestion")

        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except FileNotFoundError as exc:
            raise KnowledgeIngestionError("document disappeared during ingestion") from exc
        except ValueError as exc:
            raise PathEscapeError("document escaped the source root during ingestion") from exc
        except OSError as exc:
            raise KnowledgeIngestionError("document could not be inspected") from exc

        try:
            current_stat_after_resolve = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise KnowledgeIngestionError("document could not be inspected") from exc
        if _file_stat_is_symlink_or_reparse(current_stat_after_resolve):
            raise SymlinkSourceError("document changed to a symlink or reparse point during ingestion")
        if _file_identity(current_stat_after_resolve) != opened_identity:
            raise FileIdentityChangedError("document identity changed during ingestion")

        read_limit = min(limits.max_file_bytes, remaining_bytes)
        try:
            with os.fdopen(file_descriptor, "rb") as handle:
                file_descriptor = None
                content_bytes = handle.read(read_limit + 1)
        except OSError as exc:
            raise KnowledgeIngestionError("document could not be read") from exc
        if len(content_bytes) > limits.max_file_bytes:
            raise FileTooLargeError(f"document exceeds the file-size limit: {logical_path}")
        if total_bytes_before + len(content_bytes) > limits.max_total_bytes:
            raise TotalBytesLimitError("documents exceed the total-byte limit")
        return content_bytes
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass


def _read_document(
    path: Path,
    *,
    root: Path,
    expected_identity: _FileIdentity,
    source_id: str,
    logical_path: str,
    limits: LocalIngestionLimits,
    total_bytes_before: int,
) -> tuple[Document, int]:
    content_bytes = _read_verified_bytes(
        path,
        root=root,
        expected_identity=expected_identity,
        logical_path=logical_path,
        limits=limits,
        total_bytes_before=total_bytes_before,
    )
    extracted = select_document_extractor(
        logical_path,
        fallback=_PLAIN_TEXT_EXTRACTOR,
    ).extract(
        content_bytes,
        logical_path=logical_path,
    )

    return (
        Document(
            source_id=source_id,
            logical_path=logical_path,
            content=extracted.content,
            content_type=extracted.content_type,
            metadata=extracted.metadata,
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
        expected_identity = _capture_file_identity(root)
        document, _ = _read_document(
            root,
            root=root,
            expected_identity=expected_identity,
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
        for candidate in self._discover_supported_files(root):
            if len(documents) >= self._limits.max_documents:
                raise DocumentCountLimitError("documents exceed the document-count limit")
            document, loaded_bytes = _read_document(
                candidate.path,
                root=root,
                expected_identity=candidate.identity,
                source_id=self._source_id,
                logical_path=candidate.logical_path,
                limits=self._limits,
                total_bytes_before=total_bytes,
            )
            documents.append(document)
            total_bytes += loaded_bytes
        return tuple(documents)

    def _discover_supported_files(self, root: Path) -> list[_DiscoveredFile]:
        """Discover supported files with explicit, bounded traversal state."""

        stack: list[tuple[Path, int]] = [(root, 0)]
        candidates: list[_DiscoveredFile] = []
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
                            if _is_symlink_or_reparse_point(entry_path):
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
                            candidates.append(
                                _DiscoveredFile(
                                    path=resolved,
                                    logical_path=relative_path,
                                    identity=_capture_file_identity(resolved),
                                )
                            )
                            if len(candidates) > self._limits.max_documents:
                                raise DocumentCountLimitError("documents exceed the document-count limit")
            except EntryScanLimitError:
                raise
            except OSError as exc:
                raise KnowledgeIngestionError("source directory could not be scanned") from exc

            # Keep traversal deterministic while using an explicit stack:
            # the first directory alphabetically is processed first.
            child_directories.sort(key=lambda item: (item[0].name.casefold(), item[0].name))
            stack.extend(reversed(child_directories))

        return sorted(
            candidates,
            key=lambda candidate: (candidate.logical_path.casefold(), candidate.logical_path),
        )


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
    "FileIdentityChangedError",
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
