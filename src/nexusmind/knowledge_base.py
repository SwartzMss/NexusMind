"""Persistent knowledge-base lifecycle orchestration."""

from __future__ import annotations

from contextlib import closing, contextmanager
from copy import deepcopy
import errno
import os
from pathlib import Path
import sqlite3
import stat
from threading import RLock
from typing import Callable
from uuid import uuid4

from .knowledge import Document, KnowledgeSource, KnowledgeSourceType
from .knowledge_answer import (
    AnswerGenerationLimitError,
    AnswerGenerator,
    AnswerGeneratorError,
    generate_knowledge_answer,
)
from .context_assembly import assemble_context
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
    _path_identity,
    encode_manifest,
    read_manifest,
    write_manifest,
)
from .knowledge_chunking import TextChunker
from .knowledge_inspection import (
    KnowledgeBaseInspection,
    KnowledgeBaseStatus,
    KnowledgeChunkInspection,
    KnowledgeDocumentInspection,
    KnowledgeDocumentSummary,
    KnowledgeSourceInspection,
    KnowledgeSourceSyncStatus,
)
from .knowledge_collection import (
    CloneableChunkIndex,
    KnowledgeCollection,
    KnowledgeCollectionLimits,
    KnowledgeRetrievalDiagnostics,
    KnowledgeSearchResult,
    KnowledgeSnapshot,
    KnowledgeSyncResult,
)
from .knowledge_retrieval import InMemoryChunkIndex, SearchHit
from .knowledge_query import (
    KnowledgeQueryOptions,
    KnowledgeQueryResult,
    KnowledgeQueryTrace,
)
from .query_expansion import QueryExpander
from .search_diversification import RankedDocumentCandidate, select_document_aware_indices
from .knowledge_ingestion import LocalDirectoryAdapter, LocalFileAdapter
from .knowledge_store import KnowledgeSnapshotStoreError, SQLiteKnowledgeSnapshotStore
from .lexical_analysis import UnicodeCJKLexicalAnalyzer

if os.name == "nt":
    import msvcrt
else:
    import fcntl


_MANIFEST_NAME = "manifest.json"
_DATABASE_NAME = "knowledge.db"
_MUTATION_LOCK_NAME = ".knowledge-base.lock"
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


def _acquire_advisory_lock(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_advisory_lock(descriptor: int) -> None:
    if os.name == "nt":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


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
        answer_generator: AnswerGenerator | None = None,
        query_expander: QueryExpander | None = None,
    ) -> None:
        self._root = root
        self._manifest = manifest
        self._collection = collection
        self._index_factory = index_factory
        self._limits = limits
        self._answer_generator = answer_generator
        self._query_expander = query_expander
        self._closed = False
        self._mutation_mutex = RLock()

    @classmethod
    def create(
        cls,
        path: str,
        *,
        knowledge_base_id: str | None = None,
        index_factory: Callable[[], CloneableChunkIndex] | None = None,
        limits: KnowledgeBaseLimits | None = None,
        answer_generator: AnswerGenerator | None = None,
        query_expander: QueryExpander | None = None,
    ) -> "KnowledgeBase":
        active_limits = _limits(limits)
        root = _root(path)
        try:
            if os.path.lexists(root) and _is_reparse_or_symlink(root):
                raise KnowledgeBasePersistenceError(
                    "knowledge-base root must be a real directory"
                )
            root = root.resolve(strict=False)
        except KnowledgeBasePersistenceError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise KnowledgeBasePersistenceError(
                "unable to resolve knowledge-base root"
            ) from exc
        manifest = KnowledgeBaseManifest(
            knowledge_base_id=(
                knowledge_base_id if knowledge_base_id is not None else str(uuid4())
            ),
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
            lock_path = root / _MUTATION_LOCK_NAME
            owned[database_path] = cls._create_database(root, database_path)
            owned[lock_path] = cls._write_new_manifest(lock_path, b"\0")
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
            answer_generator=answer_generator,
            query_expander=query_expander,
        )

    @classmethod
    def open(
        cls,
        path: str,
        *,
        index_factory: Callable[[], CloneableChunkIndex] | None = None,
        limits: KnowledgeBaseLimits | None = None,
        answer_generator: AnswerGenerator | None = None,
        query_expander: QueryExpander | None = None,
    ) -> "KnowledgeBase":
        active_limits = _limits(limits)
        root = _root(path)
        factory = cls._validate_factory(index_factory)
        try:
            if os.path.lexists(root) and _is_reparse_or_symlink(root):
                raise KnowledgeBasePersistenceError(
                    "knowledge-base root must be a real directory"
                )
            root = root.resolve(strict=False)
            if (
                not os.path.lexists(root)
                or _is_reparse_or_symlink(root)
                or not root.is_dir()
            ):
                raise KnowledgeBasePersistenceError("knowledge-base root must be a real directory")
            manifest_path = root / _MANIFEST_NAME
            database_path = root / _DATABASE_NAME
            lock_path = root / _MUTATION_LOCK_NAME
            if (
                not manifest_path.is_file()
                or not database_path.is_file()
                or not lock_path.is_file()
            ):
                raise KnowledgeBasePersistenceError("knowledge-base layout is incomplete")
            if (
                _is_reparse_or_symlink(manifest_path)
                or _is_reparse_or_symlink(database_path)
                or _is_reparse_or_symlink(lock_path)
                or not stat.S_ISREG(lock_path.stat(follow_symlinks=False).st_mode)
                or lock_path.stat(follow_symlinks=False).st_size < 1
            ):
                raise KnowledgeBasePersistenceError("knowledge-base layout is invalid")
        except KnowledgeBasePersistenceError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
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
            answer_generator=answer_generator,
            query_expander=query_expander,
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
            len(self._manifest.sources),
            len(snapshot.sources),
            len(snapshot.documents),
        )

    def inspect(self) -> KnowledgeBaseInspection:
        """Summarize one coherent current manifest and canonical collection state."""

        self._require_open()
        try:
            with self._mutation_mutex:
                manifest = self._manifest
                snapshot = self._collection.snapshot()
                document_inspections = self._collection.inspect_documents(
                    preview_chars=1
                )

            canonical_source_ids = {source.source_id for source in snapshot.sources}
            snapshot_document_keys = {
                (
                    document.source_id,
                    document.document_id,
                    document.content_hash,
                )
                for document in snapshot.documents
            }
            inspection_document_keys = {
                (
                    item.document.source_id,
                    item.document.document_id,
                    item.document.content_hash,
                )
                for item in document_inspections
            }
            if (
                len(snapshot.documents) != len(document_inspections)
                or snapshot_document_keys != inspection_document_keys
            ):
                raise ValueError("collection inspection does not match canonical snapshot")
            documents = tuple(
                KnowledgeDocumentSummary(
                    source_id=item.document.source_id,
                    document_id=item.document.document_id,
                    logical_path=item.document.logical_path,
                    content_type=item.document.content_type,
                    content_hash=item.document.content_hash,
                    metadata=item.document.metadata,
                    character_count=len(item.document.content),
                    chunk_count=len(item.chunks),
                )
                for item in sorted(
                    document_inspections,
                    key=lambda value: (
                        value.document.source_id,
                        value.document.logical_path,
                    ),
                )
            )
            document_counts = dict.fromkeys(canonical_source_ids, 0)
            chunk_counts = dict.fromkeys(canonical_source_ids, 0)
            for document in documents:
                document_counts[document.source_id] += 1
                chunk_counts[document.source_id] += document.chunk_count
            sources = tuple(
                KnowledgeSourceInspection(
                    config=config,
                    sync_status=(
                        KnowledgeSourceSyncStatus.SYNCED
                        if config.source_id in canonical_source_ids
                        else KnowledgeSourceSyncStatus.REGISTERED
                    ),
                    document_count=document_counts.get(config.source_id, 0),
                    chunk_count=chunk_counts.get(config.source_id, 0),
                )
                for config in sorted(manifest.sources, key=lambda item: item.source_id)
            )
            status = KnowledgeBaseStatus(
                manifest.knowledge_base_id,
                len(manifest.sources),
                len(snapshot.sources),
                len(document_inspections),
            )
            return KnowledgeBaseInspection(status, sources, documents)
        except Exception:
            raise KnowledgeBaseSourceError("unable to inspect knowledge base") from None

    def inspect_document(
        self, document_id: str, *, preview_chars: int = 160
    ) -> KnowledgeDocumentInspection:
        """Return detached canonical provenance and bounded chunk previews."""

        self._require_open()
        if type(document_id) is not str:
            raise TypeError("document_id must be a string")
        if not document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if type(preview_chars) is not int:
            raise TypeError("preview_chars must be an integer")
        if preview_chars <= 0:
            raise ValueError("preview_chars must be greater than zero")
        try:
            inspection = self._collection.inspect_document(
                document_id, preview_chars=preview_chars
            )
            if type(inspection) is not KnowledgeDocumentInspection:
                raise TypeError("collection returned an invalid document inspection")
            source = KnowledgeSource(
                source_id=inspection.source.source_id,
                source_type=inspection.source.source_type,
                display_name=inspection.source.display_name,
                logical_location=inspection.source.logical_location,
                metadata=inspection.source.metadata,
            )
            document = Document(
                source_id=inspection.document.source_id,
                logical_path=inspection.document.logical_path,
                content=inspection.document.content,
                content_type=inspection.document.content_type,
                metadata=inspection.document.metadata,
            )
            if (
                document.document_id != inspection.document.document_id
                or document.content_hash != inspection.document.content_hash
            ):
                raise ValueError("collection returned incoherent document provenance")
            chunks = tuple(
                KnowledgeChunkInspection(
                    item.ordinal,
                    item.chunk_id,
                    item.start_offset,
                    item.end_offset,
                    item.character_count,
                    item.preview,
                )
                for item in inspection.chunks
            )
            return KnowledgeDocumentInspection(source, document, chunks)
        except Exception:
            raise KnowledgeBaseSourceError("unable to inspect knowledge base") from None

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
        if type(query) is not str:
            raise TypeError("query must be a string")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        try:
            return self._collection.search(query, limit=limit)
        except Exception as exc:
            raise KnowledgeBaseSourceError("unable to search knowledge base") from exc

    def query(
        self,
        question: str,
        *,
        options: KnowledgeQueryOptions | None = None,
    ) -> KnowledgeQueryResult:
        """Run retrieval, context assembly, answer generation, and citation validation."""

        self._require_open()
        if type(question) is not str:
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must be a non-empty string")
        active_options = KnowledgeQueryOptions() if options is None else options
        if type(active_options) is not KnowledgeQueryOptions:
            raise TypeError("options must be KnowledgeQueryOptions")
        active_limits = active_options.limits
        if len(question) > active_limits.max_question_chars:
            raise AnswerGenerationLimitError("question exceeds max_question_chars")
        active_generator = (
            self._answer_generator
            if active_options.generator is None
            else active_options.generator
        )
        if active_generator is None:
            raise AnswerGeneratorError("no answer generator is configured")
        active_expander = (
            self._query_expander
            if active_options.query_expander is None
            else active_options.query_expander
        )
        retrieval_queries = (question,)
        expansion_error: str | None = None
        if active_expander is not None:
            try:
                expansion = active_expander.expand(question)
                if expansion.original_query != question:
                    raise ValueError("query expander changed the original question")
                retrieval_queries += expansion.expanded_queries
            except Exception as exc:
                expansion_error = type(exc).__name__
        try:
            ranked_lists = tuple(
                self._collection.search_backend(
                    retrieval_query, limit=active_options.retrieval_limit
                )
                for retrieval_query in retrieval_queries
            )
            fused, fused_provenance = self._fuse_query_results(
                ranked_lists, limit=active_options.retrieval_limit
            )
            context = assemble_context(
                question,
                fused,
                max_passages=active_limits.max_passages,
                max_candidates=active_options.retrieval_limit,
                max_chars=active_limits.max_context_chars,
                max_tokens=active_limits.max_context_tokens,
            )
        except Exception:
            raise KnowledgeBaseSourceError("unable to assemble knowledge context") from None
        if not context.passages:
            if context.metadata.get("candidate_count") == 0:
                raise KnowledgeBaseSourceError("knowledge retrieval returned no evidence")
            raise AnswerGenerationLimitError("context limits exclude all evidence passages")
        answer = generate_knowledge_answer(
            question,
            context,
            active_generator,
            limits=active_limits,
        )
        config = answer.model_context.context_config
        trace = KnowledgeQueryTrace(
            retrieval_backend=self._collection.retrieval_backend_name,
            passages=answer.model_context.passages,
            candidate_count=config.candidate_count,
            context_character_count=config.character_count,
            context_estimated_token_count=config.estimated_token_count,
            retrieval_queries=retrieval_queries,
            query_expansion_error=expansion_error,
            fused_result_provenance=fused_provenance,
        )
        return KnowledgeQueryResult(
            answer=answer,
            citations=answer.citations,
            trace_id=str(uuid4()),
            trace=trace,
        )

    @staticmethod
    def _fuse_query_results(
        ranked_lists: tuple[tuple[KnowledgeSearchResult, ...], ...], *, limit: int
    ) -> tuple[
        tuple[KnowledgeSearchResult, ...],
        tuple[tuple[str, tuple[tuple[int, int], ...]], ...],
    ]:
        """Fuse backend-final rankings with deterministic query-level RRF."""

        rrf_k = 60
        scores: dict[str, float] = {}
        first_seen: dict[str, tuple[int, int]] = {}
        results: dict[str, KnowledgeSearchResult] = {}
        ranks: dict[str, list[tuple[int, int]]] = {}
        for query_index, ranked in enumerate(ranked_lists):
            seen: set[str] = set()
            for rank, result in enumerate(ranked, start=1):
                chunk_id = result.hit.chunk.chunk_id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                first_seen.setdefault(chunk_id, (query_index, rank))
                results.setdefault(chunk_id, result)
                ranks.setdefault(chunk_id, []).append((query_index, rank))
        ordered_ids = sorted(
            scores, key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id], chunk_id)
        )
        fused = tuple(
            KnowledgeSearchResult(
                results[chunk_id].source,
                results[chunk_id].document,
                SearchHit(
                    results[chunk_id].hit.chunk,
                    float(scores[chunk_id]),
                    results[chunk_id].hit.matched_terms,
                ),
            )
            for chunk_id in ordered_ids
        )
        diversified = select_document_aware_indices(
            tuple(
                RankedDocumentCandidate(item.document.document_id, item.hit.score)
                for item in fused
            ),
            limit=limit,
        )
        selected = tuple(fused[index] for index in diversified)
        provenance = tuple(
            (item.hit.chunk.chunk_id, tuple(ranks[item.hit.chunk.chunk_id]))
            for item in selected
        )
        return selected, provenance

    def diagnose_search(
        self, query: str, *, limit: int = 10
    ) -> KnowledgeRetrievalDiagnostics:
        """Return one provenance-resolved trace from a diagnostic index."""

        self._require_open()
        if type(query) is not str:
            raise TypeError("query must be a string")
        if type(limit) is not int:
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        try:
            diagnostics = self._collection.diagnose_search(query, limit=limit)
            if type(diagnostics) is not KnowledgeRetrievalDiagnostics:
                raise TypeError("collection returned invalid search diagnostics")
            validated = KnowledgeRetrievalDiagnostics(
                diagnostics.query,
                diagnostics.results,
                diagnostics.candidates,
            )
            if validated.query != query:
                raise ValueError("collection returned diagnostics for another query")
            return validated
        except Exception:
            raise KnowledgeBaseSourceError(
                "unable to diagnose knowledge search"
            ) from None

    def add_source(self, config: RegisteredSourceConfig) -> RegisteredSourceConfig:
        """Persist and return a canonical source registration without ingesting it."""
        self._require_open()
        with self._mutation_guard():
            self._refresh_manifest()
            return self._add_source(config)

    def _add_source(self, config: RegisteredSourceConfig) -> RegisteredSourceConfig:
        if type(config) not in (LocalFileSourceConfig, LocalDirectorySourceConfig):
            raise KnowledgeBaseConfigError("source config type is unsupported")
        normalized = KnowledgeBaseManifest(
            knowledge_base_id=self._manifest.knowledge_base_id,
            sources=(config,),
            limits=self._limits,
        ).sources[0]
        normalized_identity = _path_identity(normalized.path)
        if any(
            _path_identity(item.path) == normalized_identity
            for item in self._manifest.sources
        ):
            raise KnowledgeBaseSourceError("source path is already registered")
        if any(item.source_id == normalized.source_id for item in self._manifest.sources):
            raise KnowledgeBaseSourceError("source_id is already registered")
        candidate = KnowledgeBaseManifest(
            knowledge_base_id=self._manifest.knowledge_base_id,
            sources=self._manifest.sources + (normalized,),
            limits=self._limits,
        )
        self._persist_manifest(candidate)
        self._manifest = candidate
        return next(
            item for item in candidate.sources if item.source_id == normalized.source_id
        )

    def unregister_source(self, source_id: str) -> None:
        """Remove an unused registration while preserving canonical content."""
        self._require_open()
        with self._mutation_guard():
            self._refresh_manifest()
            self._refresh_collection()
            self._unregister_source(source_id)

    def remove_source(self, source_id: str) -> None:
        """Atomically remove a registration and all of its canonical content."""
        self._require_open()
        if type(source_id) is not str or not source_id.strip():
            raise KnowledgeBaseConfigError("source_id must be non-empty text")
        with self._mutation_guard():
            self._refresh_manifest()
            self._refresh_collection()
            self._remove_source(source_id)

    def _remove_source(self, source_id: str) -> None:
        registration = next(
            (item for item in self._manifest.sources if item.source_id == source_id), None
        )
        if registration is None:
            raise KnowledgeBaseSourceError("source_id is not registered")

        old_manifest = self._manifest
        old_snapshot = self._collection.snapshot()
        try:
            staging = self._new_collection(self._index_factory, self._limits)
            staging.restore(old_snapshot)
            staging.remove_source(source_id)
            candidate = KnowledgeBaseManifest(
                knowledge_base_id=self._manifest.knowledge_base_id,
                sources=tuple(
                    item for item in self._manifest.sources if item.source_id != source_id
                ),
                limits=self._limits,
            )
            encode_manifest(candidate, self._limits)
        except KnowledgeBaseConfigError:
            raise
        except Exception as exc:
            raise KnowledgeBasePersistenceError(
                "unable to prepare knowledge-source removal"
            ) from exc

        try:
            store = SQLiteKnowledgeSnapshotStore(self._root / _DATABASE_NAME)
        except Exception as exc:
            raise KnowledgeBasePersistenceError(
                "unable to persist canonical knowledge state"
            ) from exc

        try:
            store.save(staging.snapshot())
        except Exception as save_error:
            try:
                store.save(old_snapshot)
            except Exception as recovery_error:
                self._closed = True
                raise KnowledgeBasePersistenceError(
                    "knowledge-source removal recovery failed; knowledge base is unusable"
                ) from recovery_error
            raise KnowledgeBasePersistenceError(
                "unable to persist canonical knowledge state"
            ) from save_error

        try:
            self._persist_manifest(candidate)
        except Exception as manifest_error:
            recovery_failed = False
            try:
                self._persist_manifest(old_manifest)
            except Exception:
                recovery_failed = True
            try:
                store.save(old_snapshot)
            except Exception:
                recovery_failed = True
            if recovery_failed:
                self._closed = True
                raise KnowledgeBasePersistenceError(
                    "knowledge-source removal recovery failed; knowledge base is unusable"
                ) from manifest_error
            raise KnowledgeBasePersistenceError(
                "unable to persist knowledge-source removal"
            ) from manifest_error

        self._collection = staging
        self._manifest = candidate

    def _unregister_source(self, source_id: str) -> None:
        if type(source_id) is not str or not source_id.strip():
            raise KnowledgeBaseConfigError("source_id must be non-empty text")
        registration = next(
            (item for item in self._manifest.sources if item.source_id == source_id), None
        )
        if registration is None:
            raise KnowledgeBaseSourceError("source_id is not registered")
        snapshot = self._collection.snapshot()
        if any(item.source_id == source_id for item in snapshot.sources):
            raise KnowledgeBaseSourceError("synchronized source cannot be unregistered")
        candidate = KnowledgeBaseManifest(
            knowledge_base_id=self._manifest.knowledge_base_id,
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
        with self._mutation_guard():
            self._refresh_manifest()
            if not self._manifest.sources:
                return ()
            self._refresh_collection()
            return self._sync_configs(self._manifest.sources)

    def sync_source(self, source_id: str) -> KnowledgeSyncResult:
        """Atomically synchronize exactly one registered source."""
        self._require_open()
        if type(source_id) is not str or not source_id.strip():
            raise KnowledgeBaseConfigError("source_id must be non-empty text")
        with self._mutation_guard():
            self._refresh_manifest()
            config = next(
                (item for item in self._manifest.sources if item.source_id == source_id),
                None,
            )
            if config is None:
                raise KnowledgeBaseSourceError("source_id is not registered")
            self._refresh_collection()
            return self._sync_configs((config,))[0]

    def _refresh_manifest(self) -> None:
        self._manifest = read_manifest(self._root / _MANIFEST_NAME, self._limits)

    def _refresh_collection(self) -> None:
        try:
            snapshot = SQLiteKnowledgeSnapshotStore(
                self._root / _DATABASE_NAME
            ).load()
            self._validate_coherence(self._manifest, snapshot)
            candidate = self._new_collection(self._index_factory, self._limits)
            candidate.restore(snapshot)
        except KnowledgeBasePersistenceError:
            raise
        except Exception as exc:
            raise KnowledgeBasePersistenceError(
                "unable to refresh canonical knowledge state"
            ) from exc
        self._collection = candidate

    @contextmanager
    def _mutation_guard(self):
        """Serialize mutations locally and fail closed across open handles."""
        with self._mutation_mutex:
            lock_path = self._root / _MUTATION_LOCK_NAME
            flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            flags |= getattr(os, "O_BINARY", 0)
            descriptor = -1
            try:
                if _is_reparse_or_symlink(lock_path):
                    raise KnowledgeBasePersistenceError(
                        "knowledge-base coordination state is invalid"
                    )
                descriptor = os.open(lock_path, flags, 0o600)
                info = os.fstat(descriptor)
                identity = (info.st_dev, info.st_ino)
                if not stat.S_ISREG(info.st_mode) or info.st_size < 1:
                    raise KnowledgeBasePersistenceError(
                        "knowledge-base coordination state is invalid"
                    )
                if self._file_identity(lock_path) != identity:
                    raise KnowledgeBasePersistenceError(
                        "knowledge-base coordination state changed"
                    )
                _acquire_advisory_lock(descriptor)
                if (
                    _is_reparse_or_symlink(lock_path)
                    or self._file_identity(lock_path) != identity
                ):
                    raise KnowledgeBasePersistenceError(
                        "knowledge-base coordination state changed"
                    )
            except KnowledgeBasePersistenceError:
                if descriptor >= 0:
                    os.close(descriptor)
                raise
            except OSError as exc:
                if descriptor >= 0:
                    os.close(descriptor)
                raise KnowledgeBasePersistenceError(
                    "another knowledge-base mutation is active"
                ) from exc
            body_failed = False
            try:
                yield
            except BaseException:
                body_failed = True
                raise
            finally:
                try:
                    _release_advisory_lock(descriptor)
                except OSError:
                    # Closing the descriptor releases the advisory lock on every
                    # supported platform.  An explicit-unlock anomaly therefore
                    # must not reverse a committed mutation or mask its failure.
                    pass
                try:
                    os.close(descriptor)
                except OSError as exc:
                    self._closed = True
                    if not body_failed:
                        raise KnowledgeBasePersistenceError(
                            "knowledge-base mutation cleanup failed; "
                            "knowledge base is unusable"
                        ) from exc

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
        self._save_staging_snapshot(staging.snapshot(), live_snapshot)
        self._collection = staging
        return tuple(results)

    def _save_staging_snapshot(
        self, staged_snapshot: KnowledgeSnapshot, old_snapshot: KnowledgeSnapshot
    ) -> None:
        try:
            store = SQLiteKnowledgeSnapshotStore(self._root / _DATABASE_NAME)
        except Exception as exc:
            raise KnowledgeBasePersistenceError(
                "unable to persist canonical knowledge state"
            ) from exc
        try:
            store.save(staged_snapshot)
        except Exception as original_exc:
            try:
                store.save(old_snapshot)
            except Exception as recovery_exc:
                self._closed = True
                raise KnowledgeBasePersistenceError(
                    "unable to recover canonical knowledge state"
                ) from recovery_exc
            raise KnowledgeBasePersistenceError(
                "unable to persist canonical knowledge state"
            ) from original_exc

    def close(self) -> None:
        self._closed = True


__all__ = ["KnowledgeBase", "KnowledgeBaseStatus"]
