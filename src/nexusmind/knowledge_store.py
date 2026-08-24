"""Persistence contracts for canonical Knowledge snapshots."""

from __future__ import annotations

from contextlib import closing
from enum import Enum
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Protocol

from .knowledge import Document, DocumentVersion, KnowledgeSource
from .knowledge_collection import KnowledgeSnapshot


_SCHEMA_VERSION = "2"


class KnowledgeSnapshotStoreError(RuntimeError):
    """A snapshot could not be encoded, saved, or loaded safely."""


class KnowledgeSnapshotStore(Protocol):
    """Source-neutral persistence boundary for canonical Knowledge state."""

    def save(self, snapshot: KnowledgeSnapshot) -> None: ...

    def load(self) -> KnowledgeSnapshot: ...


class SQLiteKnowledgeSnapshotStore:
    """SQLite v2 storage for canonical snapshots and document history."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        try:
            raw_path = os.fspath(path)
        except TypeError as exc:
            raise TypeError("path must be a filesystem path") from exc
        if isinstance(raw_path, bytes):
            raise TypeError("path must be a text filesystem path")
        if type(raw_path) is not str or not raw_path.strip():
            raise ValueError("path must be a non-empty filesystem path")
        if raw_path == ":memory:":
            raise ValueError("SQLiteKnowledgeSnapshotStore requires a persistent database path")
        self._path = Path(raw_path)
        self._initialize()

    def save(self, snapshot: KnowledgeSnapshot) -> None:
        sources, documents, versions = self._encode_snapshot(snapshot)
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                self._validate_schema(db)
                db.execute("DELETE FROM document_versions")
                db.execute("DELETE FROM documents")
                db.execute("DELETE FROM sources")
                db.executemany(
                    "INSERT INTO sources "
                    "(source_id, source_type, display_name, logical_location, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    sources,
                )
                db.executemany(
                    "INSERT INTO documents "
                    "(document_id, source_id, logical_path, content, content_type, metadata_json, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    documents,
                )
                db.executemany(
                    "INSERT INTO document_versions "
                    "(version_id, document_id, source_id, logical_path, content, content_hash, "
                    "created_at, previous_version_id, sync_context) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    versions,
                )
                db.commit()
        except KnowledgeSnapshotStoreError:
            raise
        except sqlite3.IntegrityError as exc:
            raise KnowledgeSnapshotStoreError("snapshot conflicts with the SQLite schema") from exc
        except sqlite3.Error as exc:
            raise KnowledgeSnapshotStoreError("SQLite snapshot save failed") from exc

    def load(self) -> KnowledgeSnapshot:
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN")
                self._validate_schema(db)
                source_rows = db.execute(
                    "SELECT source_id, source_type, display_name, logical_location, metadata_json "
                    "FROM sources ORDER BY source_id"
                ).fetchall()
                document_rows = db.execute(
                    "SELECT document_id, source_id, logical_path, content, content_type, "
                    "metadata_json, content_hash FROM documents ORDER BY source_id, document_id"
                ).fetchall()
                version_rows = db.execute(
                    "SELECT version_id, document_id, source_id, logical_path, content, "
                    "content_hash, created_at, previous_version_id, sync_context "
                    "FROM document_versions ORDER BY document_id, version_id"
                ).fetchall()
                db.commit()
        except KnowledgeSnapshotStoreError:
            raise
        except sqlite3.Error as exc:
            raise KnowledgeSnapshotStoreError("SQLite snapshot load failed") from exc

        sources: list[KnowledgeSource] = []
        source_ids: set[str] = set()
        for row in source_rows:
            source_id, source_type, display_name, logical_location, metadata_json = row
            try:
                source = KnowledgeSource(
                    source_id=source_id,
                    source_type=source_type,
                    display_name=display_name,
                    logical_location=logical_location,
                    metadata=self._decode_metadata(metadata_json),
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSnapshotStoreError("stored KnowledgeSource is invalid") from exc
            if source.source_id in source_ids:
                raise KnowledgeSnapshotStoreError("stored snapshot contains duplicate source IDs")
            source_ids.add(source.source_id)
            sources.append(source)

        documents: list[Document] = []
        document_ids: set[str] = set()
        for row in document_rows:
            stored_id, source_id, logical_path, content, content_type, metadata_json, stored_hash = row
            if source_id not in source_ids:
                raise KnowledgeSnapshotStoreError("stored Document references a missing source")
            try:
                document = Document(
                    source_id=source_id,
                    logical_path=logical_path,
                    content=content,
                    content_type=content_type,
                    metadata=self._decode_metadata(metadata_json),
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSnapshotStoreError("stored Document is invalid") from exc
            if document.document_id != stored_id or document.content_hash != stored_hash:
                raise KnowledgeSnapshotStoreError("stored Document integrity check failed")
            if document.document_id in document_ids:
                raise KnowledgeSnapshotStoreError("stored snapshot contains duplicate document IDs")
            document_ids.add(document.document_id)
            documents.append(document)
        versions: list[DocumentVersion] = []
        for row in version_rows:
            try:
                versions.append(
                    DocumentVersion(
                        version_id=row[0],
                        document_id=row[1],
                        source_id=row[2],
                        logical_path=row[3],
                        content=row[4],
                        content_hash=row[5],
                        created_at=row[6],
                        previous_version_id=row[7],
                        sync_context=row[8],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSnapshotStoreError(
                    "stored DocumentVersion is invalid"
                ) from exc
        return KnowledgeSnapshot(
            sources=tuple(sources),
            documents=tuple(documents),
            document_versions=self._order_versions(versions),
        )

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                objects = db.execute(
                    "SELECT type, name FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                ).fetchall()
                if not objects:
                    db.execute(
                        "CREATE TABLE knowledge_store_metadata "
                        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE sources ("
                        "source_id TEXT PRIMARY KEY, source_type TEXT NOT NULL, "
                        "display_name TEXT NOT NULL, logical_location TEXT, metadata_json TEXT NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE documents ("
                        "document_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, "
                        "logical_path TEXT NOT NULL, content TEXT NOT NULL, content_type TEXT NOT NULL, "
                        "metadata_json TEXT NOT NULL, content_hash TEXT NOT NULL, "
                        "FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE)"
                    )
                    self._create_versions_table(db)
                    db.execute(
                        "INSERT INTO knowledge_store_metadata (key, value) VALUES ('schema_version', ?)",
                        (_SCHEMA_VERSION,),
                    )
                else:
                    row = db.execute(
                        "SELECT value FROM knowledge_store_metadata WHERE key = 'schema_version'"
                    ).fetchone()
                    if row is not None and row[0] == "1":
                        self._validate_schema_v1(db)
                        self._create_versions_table(db)
                        db.execute(
                            "UPDATE knowledge_store_metadata SET value = ? "
                            "WHERE key = 'schema_version'",
                            (_SCHEMA_VERSION,),
                        )
                self._validate_schema(db)
                db.commit()
        except KnowledgeSnapshotStoreError:
            raise
        except sqlite3.Error as exc:
            raise KnowledgeSnapshotStoreError(
                "knowledge snapshot schema initialization failed"
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def _validate_schema(db: sqlite3.Connection) -> None:
        try:
            row = db.execute(
                "SELECT value FROM knowledge_store_metadata WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.Error as exc:
            raise KnowledgeSnapshotStoreError("knowledge snapshot schema is missing or malformed") from exc
        if row is None or row[0] != _SCHEMA_VERSION:
            raise KnowledgeSnapshotStoreError("unsupported knowledge snapshot schema version")

        expected = {
            "knowledge_store_metadata": {
                "key": ("TEXT", 0, 1),
                "value": ("TEXT", 1, 0),
            },
            "sources": {
                "source_id": ("TEXT", 0, 1),
                "source_type": ("TEXT", 1, 0),
                "display_name": ("TEXT", 1, 0),
                "logical_location": ("TEXT", 0, 0),
                "metadata_json": ("TEXT", 1, 0),
            },
            "documents": {
                "document_id": ("TEXT", 0, 1),
                "source_id": ("TEXT", 1, 0),
                "logical_path": ("TEXT", 1, 0),
                "content": ("TEXT", 1, 0),
                "content_type": ("TEXT", 1, 0),
                "metadata_json": ("TEXT", 1, 0),
                "content_hash": ("TEXT", 1, 0),
            },
            "document_versions": {
                "version_id": ("TEXT", 0, 1),
                "document_id": ("TEXT", 1, 0),
                "source_id": ("TEXT", 1, 0),
                "logical_path": ("TEXT", 1, 0),
                "content": ("TEXT", 1, 0),
                "content_hash": ("TEXT", 1, 0),
                "created_at": ("TEXT", 1, 0),
                "previous_version_id": ("TEXT", 0, 0),
                "sync_context": ("TEXT", 1, 0),
            },
        }
        for table, columns in expected.items():
            actual = {
                info[1]: (info[2].upper(), info[3], info[5])
                for info in db.execute(f"PRAGMA table_info({table})")
            }
            if actual != columns:
                raise KnowledgeSnapshotStoreError("knowledge snapshot schema is incomplete or incompatible")
        foreign_keys = db.execute("PRAGMA foreign_key_list(documents)").fetchall()
        if not any(
            row[2] == "sources"
            and row[3] == "source_id"
            and row[4] == "source_id"
            and row[6].upper() == "CASCADE"
            for row in foreign_keys
        ):
            raise KnowledgeSnapshotStoreError("knowledge snapshot schema has an invalid foreign key")

    @staticmethod
    def _validate_schema_v1(db: sqlite3.Connection) -> None:
        expected_tables = {"knowledge_store_metadata", "sources", "documents"}
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        if tables != expected_tables:
            raise KnowledgeSnapshotStoreError(
                "knowledge snapshot schema is incomplete or incompatible"
            )

    @staticmethod
    def _create_versions_table(db: sqlite3.Connection) -> None:
        db.execute(
            "CREATE TABLE document_versions ("
            "version_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, "
            "source_id TEXT NOT NULL, logical_path TEXT NOT NULL, content TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, created_at TEXT NOT NULL, "
            "previous_version_id TEXT, sync_context TEXT NOT NULL)"
        )
        db.execute(
            "CREATE INDEX document_versions_document_id "
            "ON document_versions(document_id)"
        )

    @classmethod
    def _encode_snapshot(
        cls, snapshot: KnowledgeSnapshot
    ) -> tuple[
        list[tuple[Any, ...]],
        list[tuple[Any, ...]],
        list[tuple[Any, ...]],
    ]:
        if not isinstance(snapshot, KnowledgeSnapshot):
            raise KnowledgeSnapshotStoreError("snapshot must be a KnowledgeSnapshot")
        if (
            type(snapshot.sources) is not tuple
            or type(snapshot.documents) is not tuple
            or type(snapshot.document_versions) is not tuple
        ):
            raise KnowledgeSnapshotStoreError(
                "snapshot sources, documents, and document_versions must be tuples"
            )
        sources: list[tuple[Any, ...]] = []
        source_ids: set[str] = set()
        for source in snapshot.sources:
            if not isinstance(source, KnowledgeSource):
                raise KnowledgeSnapshotStoreError("snapshot contains an invalid KnowledgeSource")
            try:
                validated_source = KnowledgeSource(
                    source_id=source.source_id,
                    source_type=source.source_type,
                    display_name=source.display_name,
                    logical_location=source.logical_location,
                    metadata=source.metadata,
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSnapshotStoreError(
                    "snapshot contains an invalid KnowledgeSource"
                ) from exc
            if validated_source.source_id in source_ids:
                raise KnowledgeSnapshotStoreError("snapshot contains duplicate source IDs")
            source_ids.add(validated_source.source_id)
            source_type = (
                validated_source.source_type.value
                if isinstance(validated_source.source_type, Enum)
                else validated_source.source_type
            )
            sources.append(
                (
                    validated_source.source_id,
                    source_type,
                    validated_source.display_name,
                    validated_source.logical_location,
                    cls._encode_metadata(validated_source.metadata),
                )
            )
        documents: list[tuple[Any, ...]] = []
        document_ids: set[str] = set()
        for document in snapshot.documents:
            if not isinstance(document, Document):
                raise KnowledgeSnapshotStoreError("snapshot contains an invalid Document")
            try:
                validated_document = Document(
                    source_id=document.source_id,
                    logical_path=document.logical_path,
                    content=document.content,
                    content_type=document.content_type,
                    metadata=document.metadata,
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSnapshotStoreError("snapshot contains an invalid Document") from exc
            if validated_document.source_id not in source_ids:
                raise KnowledgeSnapshotStoreError("snapshot Document references a missing source")
            if validated_document.document_id in document_ids:
                raise KnowledgeSnapshotStoreError("snapshot contains duplicate document IDs")
            if document.document_id != validated_document.document_id:
                raise KnowledgeSnapshotStoreError("snapshot Document identity is incoherent")
            if document.content_hash != validated_document.content_hash:
                raise KnowledgeSnapshotStoreError("snapshot Document content hash is incoherent")
            document_ids.add(validated_document.document_id)
            documents.append(
                (
                    validated_document.document_id,
                    validated_document.source_id,
                    validated_document.logical_path,
                    validated_document.content,
                    validated_document.content_type,
                    cls._encode_metadata(validated_document.metadata),
                    validated_document.content_hash,
                )
            )
        sources.sort(key=lambda row: row[0])
        documents.sort(key=lambda row: row[0])
        versions: list[tuple[Any, ...]] = []
        version_ids: set[str] = set()
        for version in snapshot.document_versions:
            if type(version) is not DocumentVersion:
                raise KnowledgeSnapshotStoreError(
                    "snapshot contains an invalid DocumentVersion"
                )
            try:
                validated = DocumentVersion(
                    version_id=version.version_id,
                    document_id=version.document_id,
                    source_id=version.source_id,
                    logical_path=version.logical_path,
                    content=version.content,
                    content_hash=version.content_hash,
                    created_at=version.created_at,
                    previous_version_id=version.previous_version_id,
                    sync_context=version.sync_context,
                )
            except (TypeError, ValueError) as exc:
                raise KnowledgeSnapshotStoreError(
                    "snapshot contains an invalid DocumentVersion"
                ) from exc
            if validated.version_id in version_ids:
                raise KnowledgeSnapshotStoreError(
                    "snapshot contains duplicate DocumentVersion IDs"
                )
            version_ids.add(validated.version_id)
            versions.append(
                (
                    validated.version_id,
                    validated.document_id,
                    validated.source_id,
                    validated.logical_path,
                    validated.content,
                    validated.content_hash,
                    validated.created_at,
                    validated.previous_version_id,
                    validated.sync_context,
                )
            )
        return sources, documents, versions

    @staticmethod
    def _order_versions(
        versions: list[DocumentVersion],
    ) -> tuple[DocumentVersion, ...]:
        grouped: dict[str, list[DocumentVersion]] = {}
        for version in versions:
            grouped.setdefault(version.document_id, []).append(version)
        ordered: list[DocumentVersion] = []
        for document_id in sorted(grouped):
            candidates = grouped[document_id]
            children = {
                version.previous_version_id: version
                for version in candidates
                if version.previous_version_id is not None
            }
            roots = [version for version in candidates if version.previous_version_id is None]
            if len(roots) != 1:
                raise KnowledgeSnapshotStoreError("stored document version chain is invalid")
            current = roots[0]
            chain: list[DocumentVersion] = []
            while current not in chain:
                chain.append(current)
                successor = children.get(current.version_id)
                if successor is None:
                    break
                current = successor
            if len(chain) != len(candidates):
                raise KnowledgeSnapshotStoreError("stored document version chain is invalid")
            ordered.extend(chain)
        return tuple(ordered)

    @classmethod
    def _encode_metadata(cls, metadata: dict[str, Any]) -> str:
        try:
            cls._validate_json_value(metadata)
            return json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise KnowledgeSnapshotStoreError("metadata is not JSON-compatible") from exc

    @classmethod
    def _decode_metadata(cls, payload: Any) -> dict[str, Any]:
        if type(payload) is not str:
            raise KnowledgeSnapshotStoreError("stored metadata has an invalid type")
        try:
            value = json.loads(payload)
            if type(value) is not dict:
                raise ValueError("metadata root must be an object")
            cls._validate_json_value(value)
            return value
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
            raise KnowledgeSnapshotStoreError("stored metadata is invalid") from exc

    @classmethod
    def _validate_json_value(cls, value: Any) -> None:
        if value is None or type(value) in (bool, int, str):
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("non-finite number")
            return
        if type(value) is list:
            for item in value:
                cls._validate_json_value(item)
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("metadata object keys must be strings")
                cls._validate_json_value(item)
            return
        raise TypeError("unsupported metadata value")


__all__ = [
    "KnowledgeSnapshotStore",
    "KnowledgeSnapshotStoreError",
    "SQLiteKnowledgeSnapshotStore",
]
