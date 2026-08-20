"""Persistent knowledge-base lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Callable

from .knowledge import Document, KnowledgeSourceType
from .knowledge_base_manifest import (
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBaseLimits,
    KnowledgeBaseManifest,
    KnowledgeBasePersistenceError,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    RegisteredSourceConfig,
    read_manifest,
    write_manifest,
)
from .knowledge_chunking import TextChunker
from .knowledge_collection import (
    CloneableChunkIndex,
    KnowledgeCollection,
    KnowledgeSearchResult,
    KnowledgeSnapshot,
)
from .knowledge_retrieval import InMemoryChunkIndex
from .knowledge_store import KnowledgeSnapshotStoreError, SQLiteKnowledgeSnapshotStore
from .lexical_analysis import UnicodeCJKLexicalAnalyzer


_MANIFEST_NAME = "manifest.json"
_DATABASE_NAME = "knowledge.db"
_SQLITE_HEADER = b"SQLite format 3\x00"


@dataclass(frozen=True, slots=True)
class KnowledgeBaseStatus:
    knowledge_base_id: str
    display_name: str | None
    registered_source_count: int
    canonical_source_count: int
    document_count: int


def _limits(value: KnowledgeBaseLimits | None) -> KnowledgeBaseLimits:
    if value is None:
        return KnowledgeBaseLimits()
    if not isinstance(value, KnowledgeBaseLimits):
        raise KnowledgeBaseConfigError("limits must be KnowledgeBaseLimits")
    return value


def _root(path: str) -> Path:
    if type(path) is not str or not path:
        raise KnowledgeBaseConfigError("path must be a non-empty text filesystem path")
    try:
        return Path(path)
    except (OSError, ValueError) as exc:
        raise KnowledgeBaseConfigError("path is not a valid filesystem path") from exc


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _default_index_factory() -> InMemoryChunkIndex:
    return InMemoryChunkIndex(analyzer=UnicodeCJKLexicalAnalyzer())


def _require_sqlite_header(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            header = stream.read(len(_SQLITE_HEADER))
    except OSError as exc:
        raise KnowledgeBasePersistenceError(
            "unable to read canonical knowledge state"
        ) from exc
    if header != _SQLITE_HEADER:
        raise KnowledgeBasePersistenceError("canonical knowledge state is invalid")


class KnowledgeBase:
    """A persistent canonical snapshot plus process-local derived retrieval state."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: KnowledgeBaseManifest,
        collection: KnowledgeCollection,
        index_factory: Callable[[], CloneableChunkIndex],
        limits: KnowledgeBaseLimits,
    ) -> None:
        self._root = root
        self._manifest = manifest
        self._collection = collection
        self._index_factory = index_factory
        self._limits = limits
        self._closed = False

    @classmethod
    def create(
        cls,
        path: str,
        *,
        knowledge_base_id: str,
        display_name: str | None = None,
        index_factory: Callable[[], CloneableChunkIndex] | None = None,
        limits: KnowledgeBaseLimits | None = None,
    ) -> "KnowledgeBase":
        active_limits = _limits(limits)
        root = _root(path)
        manifest = KnowledgeBaseManifest(
            knowledge_base_id=knowledge_base_id,
            display_name=display_name,
            limits=active_limits,
        )
        factory = cls._validate_factory(index_factory)
        collection = cls._new_collection(factory)

        made_root = False
        created: list[Path] = []
        try:
            if os.path.lexists(root):
                if _is_reparse_or_symlink(root) or not root.is_dir():
                    raise KnowledgeBasePersistenceError("knowledge-base root must be a real directory")
                if any(root.iterdir()):
                    raise KnowledgeBasePersistenceError("knowledge-base root must be empty")
            else:
                root.mkdir(parents=False)
                made_root = True

            database_path = root / _DATABASE_NAME
            try:
                SQLiteKnowledgeSnapshotStore(database_path)
            finally:
                for suffix in ("", "-journal", "-wal", "-shm"):
                    artifact = root / f"{_DATABASE_NAME}{suffix}"
                    if os.path.lexists(artifact):
                        created.append(artifact)
            manifest_path = root / _MANIFEST_NAME
            write_manifest(manifest_path, manifest, active_limits)
            created.append(manifest_path)
        except (KnowledgeBaseConfigError, KnowledgeBasePersistenceError):
            cls._rollback_create(created, root, made_root)
            raise
        except (KnowledgeSnapshotStoreError, OSError) as exc:
            cls._rollback_create(created, root, made_root)
            raise KnowledgeBasePersistenceError("unable to create knowledge base") from exc
        except Exception as exc:
            cls._rollback_create(created, root, made_root)
            raise KnowledgeBasePersistenceError("unable to create knowledge base") from exc

        return cls(
            root=root,
            manifest=manifest,
            collection=collection,
            index_factory=factory,
            limits=active_limits,
        )

    @classmethod
    def open(
        cls,
        path: str,
        *,
        index_factory: Callable[[], CloneableChunkIndex] | None = None,
        limits: KnowledgeBaseLimits | None = None,
    ) -> "KnowledgeBase":
        active_limits = _limits(limits)
        root = _root(path)
        factory = cls._validate_factory(index_factory)
        try:
            if (
                not os.path.lexists(root)
                or _is_reparse_or_symlink(root)
                or not root.is_dir()
            ):
                raise KnowledgeBasePersistenceError("knowledge-base root must be a real directory")
            manifest_path = root / _MANIFEST_NAME
            database_path = root / _DATABASE_NAME
            if not manifest_path.is_file() or not database_path.is_file():
                raise KnowledgeBasePersistenceError("knowledge-base layout is incomplete")
            if _is_reparse_or_symlink(manifest_path) or _is_reparse_or_symlink(database_path):
                raise KnowledgeBasePersistenceError("knowledge-base layout is invalid")
        except KnowledgeBasePersistenceError:
            raise
        except OSError as exc:
            raise KnowledgeBasePersistenceError("unable to inspect knowledge-base layout") from exc

        manifest = read_manifest(manifest_path, active_limits)
        _require_sqlite_header(database_path)
        try:
            snapshot = SQLiteKnowledgeSnapshotStore(database_path).load()
        except KnowledgeSnapshotStoreError as exc:
            raise KnowledgeBasePersistenceError("unable to read canonical knowledge state") from exc
        cls._validate_coherence(manifest, snapshot)
        collection = cls._new_collection(factory)
        try:
            collection.restore(snapshot)
        except Exception as exc:
            raise KnowledgeBasePersistenceError("unable to rebuild knowledge retrieval state") from exc
        return cls(
            root=root,
            manifest=manifest,
            collection=collection,
            index_factory=factory,
            limits=active_limits,
        )

    @staticmethod
    def _validate_factory(
        index_factory: Callable[[], CloneableChunkIndex] | None,
    ) -> Callable[[], CloneableChunkIndex]:
        if index_factory is None:
            return _default_index_factory
        if not callable(index_factory):
            raise KnowledgeBaseConfigError("index_factory must be callable")
        return index_factory

    @staticmethod
    def _new_collection(
        factory: Callable[[], CloneableChunkIndex],
    ) -> KnowledgeCollection:
        try:
            return KnowledgeCollection(chunker=TextChunker(), index_factory=factory)
        except Exception as exc:
            raise KnowledgeBaseConfigError("unable to configure knowledge retrieval") from exc

    @staticmethod
    def _validate_coherence(
        manifest: KnowledgeBaseManifest, snapshot: KnowledgeSnapshot
    ) -> None:
        registrations = {item.source_id: item for item in manifest.sources}
        expected_types = {
            LocalFileSourceConfig: KnowledgeSourceType.LOCAL_FILE,
            LocalDirectorySourceConfig: KnowledgeSourceType.LOCAL_DIRECTORY,
        }
        for source in snapshot.sources:
            registration = registrations.get(source.source_id)
            if registration is None:
                raise KnowledgeBasePersistenceError("knowledge-base stores are incoherent")
            expected = expected_types[type(registration)]
            if source.source_type != expected:
                raise KnowledgeBasePersistenceError("knowledge-base stores are incoherent")

    @staticmethod
    def _rollback_create(created: list[Path], root: Path, made_root: bool) -> None:
        for artifact in reversed(created):
            try:
                artifact.unlink()
            except OSError:
                pass
        if made_root:
            try:
                root.rmdir()
            except OSError:
                pass

    def _require_open(self) -> None:
        if self._closed:
            raise KnowledgeBaseClosedError("knowledge base is closed")

    def status(self) -> KnowledgeBaseStatus:
        self._require_open()
        snapshot = self._collection.snapshot()
        return KnowledgeBaseStatus(
            self._manifest.knowledge_base_id,
            self._manifest.display_name,
            len(self._manifest.sources),
            len(snapshot.sources),
            len(snapshot.documents),
        )

    def list_sources(self) -> tuple[RegisteredSourceConfig, ...]:
        self._require_open()
        return self._manifest.sources

    def list_documents(self) -> tuple[Document, ...]:
        self._require_open()
        return self._collection.snapshot().documents

    def search(
        self, query: str, *, limit: int = 10
    ) -> tuple[KnowledgeSearchResult, ...]:
        self._require_open()
        return self._collection.search(query, limit=limit)

    def close(self) -> None:
        self._closed = True


__all__ = ["KnowledgeBase", "KnowledgeBaseStatus"]
