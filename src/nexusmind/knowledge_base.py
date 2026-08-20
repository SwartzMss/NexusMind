"""Persistent knowledge-base lifecycle orchestration."""

from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import sqlite3
import stat
from typing import Callable
from uuid import uuid4

from .knowledge import Document, KnowledgeSourceType
from .knowledge_base_manifest import (
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBaseLimits,
    KnowledgeBaseManifest,
    KnowledgeBasePersistenceError,
    KnowledgeBaseSourceError,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    RegisteredSourceConfig,
    encode_manifest,
    read_manifest,
    write_manifest,
)
from .knowledge_chunking import TextChunker
from .knowledge_collection import (
    CloneableChunkIndex,
    KnowledgeCollection,
    KnowledgeCollectionLimits,
    KnowledgeSearchResult,
    KnowledgeSnapshot,
    KnowledgeSyncResult,
)
from .knowledge_retrieval import InMemoryChunkIndex
from .knowledge_ingestion import LocalDirectoryAdapter, LocalFileAdapter
from .knowledge_store import KnowledgeSnapshotStoreError, SQLiteKnowledgeSnapshotStore
from .lexical_analysis import UnicodeCJKLexicalAnalyzer


_MANIFEST_NAME = "manifest.json"
_DATABASE_NAME = "knowledge.db"
_SQLITE_HEADER = b"SQLite format 3\x00"
_UNSUPPORTED_LINK_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "EPERM", None),
    )
    if value is not None
)
_WINDOWS_ERROR_NOT_SUPPORTED = 50


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


def _require_store_sentinel(path: Path) -> None:
    try:
        uri = f"{path.resolve(strict=True).as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as database:
            table = database.execute(
                "SELECT type FROM sqlite_master "
                "WHERE name = 'knowledge_store_metadata'"
            ).fetchone()
            if table != ("table",):
                raise KnowledgeBasePersistenceError(
                    "canonical knowledge state is invalid"
                )
            version = database.execute(
                "SELECT value FROM knowledge_store_metadata "
                "WHERE key = 'schema_version'"
            ).fetchone()
            if version != ("1",):
                raise KnowledgeBasePersistenceError(
                    "canonical knowledge state is invalid"
                )
    except KnowledgeBasePersistenceError:
        raise
    except (OSError, ValueError, sqlite3.Error) as exc:
        raise KnowledgeBasePersistenceError(
            "unable to read canonical knowledge state"
        ) from exc


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
        collection = cls._new_collection(factory, active_limits)

        made_root = False
        owned: dict[Path, tuple[int, int]] = {}
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
            manifest_path = root / _MANIFEST_NAME
            owned[database_path] = cls._create_database(root, database_path)
            owned[manifest_path] = cls._write_new_manifest(
                manifest_path, encode_manifest(manifest, active_limits)
            )
            cls._fsync_directory(root)
            for artifact, identity in owned.items():
                cls._require_identity(artifact, identity)
        except (KnowledgeBaseConfigError, KnowledgeBasePersistenceError):
            cls._rollback_create(owned, root, made_root)
            raise
        except (KnowledgeSnapshotStoreError, OSError) as exc:
            cls._rollback_create(owned, root, made_root)
            raise KnowledgeBasePersistenceError("unable to create knowledge base") from exc
        except Exception as exc:
            cls._rollback_create(owned, root, made_root)
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
        _require_store_sentinel(database_path)
        try:
            snapshot = SQLiteKnowledgeSnapshotStore(database_path).load()
        except KnowledgeSnapshotStoreError as exc:
            raise KnowledgeBasePersistenceError("unable to read canonical knowledge state") from exc
        cls._validate_coherence(manifest, snapshot)
        collection = cls._new_collection(factory, active_limits)
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
        limits: KnowledgeBaseLimits,
    ) -> KnowledgeCollection:
        try:
            return KnowledgeCollection(
                chunker=TextChunker(),
                index_factory=factory,
                limits=KnowledgeCollectionLimits(max_sources=limits.max_sources),
            )
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
    def _file_identity(path: Path) -> tuple[int, int]:
        info = path.stat(follow_symlinks=False)
        return (info.st_dev, info.st_ino)

    @classmethod
    def _require_identity(cls, path: Path, expected: tuple[int, int]) -> None:
        try:
            if cls._file_identity(path) != expected:
                raise KnowledgeBasePersistenceError(
                    "knowledge-base layout ownership changed"
                )
        except KnowledgeBasePersistenceError:
            raise
        except OSError as exc:
            raise KnowledgeBasePersistenceError(
                "knowledge-base layout ownership changed"
            ) from exc

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    @classmethod
    def _open_exclusive(cls, path: Path) -> tuple[int, tuple[int, int]]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise KnowledgeBasePersistenceError(
                "unable to create knowledge-base artifact"
            ) from exc
        try:
            info = os.fstat(descriptor)
        except OSError as exc:
            os.close(descriptor)
            raise KnowledgeBasePersistenceError(
                "unable to identify knowledge-base artifact"
            ) from exc
        return descriptor, (info.st_dev, info.st_ino)

    @classmethod
    def _write_new_manifest(cls, path: Path, data: bytes) -> tuple[int, int]:
        descriptor, identity = cls._open_exclusive(path)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                view = memoryview(data)
                while view:
                    written = stream.write(view)
                    if written is None or written <= 0:
                        raise OSError("incomplete manifest write")
                    view = view[written:]
                stream.flush()
                os.fsync(stream.fileno())
            if cls._file_identity(path) != identity:
                raise KnowledgeBasePersistenceError(
                    "knowledge-base layout ownership changed"
                )
            return identity
        except KnowledgeBasePersistenceError:
            cls._rollback_owned_files({path: identity})
            raise
        except OSError as exc:
            cls._rollback_owned_files({path: identity})
            raise KnowledgeBasePersistenceError(
                "unable to write knowledge-base manifest"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @classmethod
    def _create_database(cls, root: Path, final: Path) -> tuple[int, int]:
        temporary = root / f".{_DATABASE_NAME}.{uuid4().hex}.tmp"
        descriptor, temporary_identity = cls._open_exclusive(temporary)
        os.close(descriptor)
        published: dict[Path, tuple[int, int]] = {}
        try:
            SQLiteKnowledgeSnapshotStore(temporary)
            if cls._file_identity(temporary) != temporary_identity:
                raise KnowledgeBasePersistenceError(
                    "knowledge-base layout ownership changed"
                )
            try:
                os.link(temporary, final)
            except OSError as exc:
                if not cls._link_is_unsupported(exc):
                    raise
                final_identity = cls._copy_database_no_clobber(temporary, final)
            else:
                published[final] = temporary_identity
                cls._require_identity(final, temporary_identity)
                final_identity = temporary_identity
            published[final] = final_identity
            cls._rollback_owned_files({temporary: temporary_identity})
            return final_identity
        except KnowledgeBasePersistenceError:
            cls._rollback_owned_files(published)
            cls._rollback_owned_files({temporary: temporary_identity})
            raise
        except (KnowledgeSnapshotStoreError, OSError) as exc:
            cls._rollback_owned_files(published)
            cls._rollback_owned_files({temporary: temporary_identity})
            raise KnowledgeBasePersistenceError(
                "unable to create canonical knowledge state"
            ) from exc
        except Exception as exc:
            cls._rollback_owned_files(published)
            cls._rollback_owned_files({temporary: temporary_identity})
            raise KnowledgeBasePersistenceError(
                "unable to create canonical knowledge state"
            ) from exc

    @staticmethod
    def _link_is_unsupported(exc: OSError) -> bool:
        return (
            exc.errno in _UNSUPPORTED_LINK_ERRNOS
            or getattr(exc, "winerror", None) == _WINDOWS_ERROR_NOT_SUPPORTED
        )

    @classmethod
    def _copy_database_no_clobber(
        cls, source: Path, final: Path
    ) -> tuple[int, int]:
        descriptor, identity = cls._open_exclusive(final)
        try:
            with source.open("rb") as input_stream, os.fdopen(
                descriptor, "wb"
            ) as output_stream:
                descriptor = -1
                while True:
                    chunk = input_stream.read(64 * 1024)
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = output_stream.write(view)
                        if written is None or written <= 0:
                            raise OSError("incomplete database write")
                        view = view[written:]
                output_stream.flush()
                os.fsync(output_stream.fileno())
            cls._require_identity(final, identity)
            return identity
        except KnowledgeBasePersistenceError:
            cls._rollback_owned_files({final: identity})
            raise
        except OSError as exc:
            cls._rollback_owned_files({final: identity})
            raise KnowledgeBasePersistenceError(
                "unable to publish canonical knowledge state"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @classmethod
    def _rollback_owned_files(cls, owned: dict[Path, tuple[int, int]]) -> None:
        for artifact, expected in reversed(tuple(owned.items())):
            try:
                if cls._file_identity(artifact) != expected:
                    continue
                artifact.unlink()
            except OSError:
                pass

    @classmethod
    def _rollback_create(
        cls,
        owned: dict[Path, tuple[int, int]],
        root: Path,
        made_root: bool,
    ) -> None:
        cls._rollback_owned_files(owned)
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
        return tuple(deepcopy(item) for item in self._collection.snapshot().documents)

    def search(
        self, query: str, *, limit: int = 10
    ) -> tuple[KnowledgeSearchResult, ...]:
        self._require_open()
        return self._collection.search(query, limit=limit)

    def add_source(self, config: RegisteredSourceConfig) -> None:
        """Persist a source registration without touching the source itself."""
        self._require_open()
        if type(config) not in (LocalFileSourceConfig, LocalDirectorySourceConfig):
            raise KnowledgeBaseConfigError("source config type is unsupported")
        if any(item.source_id == config.source_id for item in self._manifest.sources):
            raise KnowledgeBaseSourceError("source_id is already registered")
        candidate = KnowledgeBaseManifest(
            knowledge_base_id=self._manifest.knowledge_base_id,
            display_name=self._manifest.display_name,
            sources=self._manifest.sources + (config,),
            limits=self._limits,
        )
        self._persist_manifest(candidate)
        self._manifest = candidate

    def unregister_source(self, source_id: str) -> None:
        """Remove an unused registration while preserving canonical content."""
        self._require_open()
        if type(source_id) is not str or not source_id:
            raise KnowledgeBaseConfigError("source_id must be non-empty text")
        registration = next(
            (item for item in self._manifest.sources if item.source_id == source_id), None
        )
        if registration is None:
            raise KnowledgeBaseSourceError("source_id is not registered")
        if any(item.source_id == source_id for item in self._collection.snapshot().sources):
            raise KnowledgeBaseSourceError("synchronized source cannot be unregistered")
        candidate = KnowledgeBaseManifest(
            knowledge_base_id=self._manifest.knowledge_base_id,
            display_name=self._manifest.display_name,
            sources=tuple(
                item for item in self._manifest.sources if item.source_id != source_id
            ),
            limits=self._limits,
        )
        self._persist_manifest(candidate)
        self._manifest = candidate

    def _persist_manifest(self, candidate: KnowledgeBaseManifest) -> None:
        try:
            write_manifest(self._root / _MANIFEST_NAME, candidate, self._limits)
        except KnowledgeBaseConfigError:
            raise
        except Exception as exc:
            raise KnowledgeBasePersistenceError(
                "unable to persist knowledge-base manifest"
            ) from exc

    def sync(self) -> tuple[KnowledgeSyncResult, ...]:
        """Atomically synchronize every registered source in identifier order."""
        self._require_open()
        if not self._manifest.sources:
            return ()
        return self._sync_configs(self._manifest.sources)

    def sync_source(self, source_id: str) -> KnowledgeSyncResult:
        """Atomically synchronize exactly one registered source."""
        self._require_open()
        config = next(
            (item for item in self._manifest.sources if item.source_id == source_id), None
        )
        if config is None:
            raise KnowledgeBaseSourceError("source_id is not registered")
        return self._sync_configs((config,))[0]

    def _sync_configs(
        self, configs: tuple[RegisteredSourceConfig, ...]
    ) -> tuple[KnowledgeSyncResult, ...]:
        live_snapshot = self._collection.snapshot()
        try:
            staging = self._new_collection(self._index_factory, self._limits)
            staging.restore(live_snapshot)
            results: list[KnowledgeSyncResult] = []
            for config in sorted(configs, key=lambda item: item.source_id):
                adapter_class = (
                    LocalFileAdapter
                    if type(config) is LocalFileSourceConfig
                    else LocalDirectoryAdapter
                )
                adapter = adapter_class(config.path, source_id=config.source_id)
                results.append(staging.sync(adapter))
        except Exception as exc:
            raise KnowledgeBaseSourceError("unable to synchronize registered source") from exc
        try:
            SQLiteKnowledgeSnapshotStore(self._root / _DATABASE_NAME).save(
                staging.snapshot()
            )
        except Exception as exc:
            raise KnowledgeBasePersistenceError(
                "unable to persist canonical knowledge state"
            ) from exc
        self._collection = staging
        return tuple(results)

    def close(self) -> None:
        self._closed = True


__all__ = ["KnowledgeBase", "KnowledgeBaseStatus"]
