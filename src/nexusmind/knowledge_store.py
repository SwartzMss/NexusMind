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

from .knowledge import Document, KnowledgeSource, compute_content_hash, stable_document_id
from .knowledge_collection import KnowledgeSnapshot


_SCHEMA_VERSION = "1"


class KnowledgeSnapshotStoreError(RuntimeError):
    """A snapshot could not be encoded, saved, or loaded safely."""


class KnowledgeSnapshotStore(Protocol):
    """Source-neutral persistence boundary for canonical Knowledge state."""

    def save(self, snapshot: KnowledgeSnapshot) -> None: ...

    def load(self) -> KnowledgeSnapshot: ...


class SQLiteKnowledgeSnapshotStore:
    """SQLite v1 storage for canonical Source/Document snapshots only."""

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
        sources, documents = self._encode_snapshot(snapshot)
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                self._validate_schema(db)
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
                self._validate_schema(db)
                source_rows = db.execute(
                    "SELECT source_id, source_type, display_name, logical_location, metadata_json "
                    "FROM sources ORDER BY source_id"
                ).fetchall()
                document_rows = db.execute(
                    "SELECT document_id, source_id, logical_path, content, content_type, "
                    "metadata_json, content_hash FROM documents ORDER BY source_id, document_id"
                ).fetchall()
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
        return KnowledgeSnapshot(sources=tuple(sources), documents=tuple(documents))

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
                    db.execute(
                        "INSERT INTO knowledge_store_metadata (key, value) VALUES ('schema_version', ?)",
                        (_SCHEMA_VERSION,),
                    )
                self._validate_schema(db)
                db.commit()
        except KnowledgeSnapshotStoreError:
            raise
        except sqlite3.Error as exc:
            raise KnowledgeSnapshotStoreError("SQLite snapshot store initialization failed") from exc

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

    @classmethod
    def _encode_snapshot(
        cls, snapshot: KnowledgeSnapshot
    ) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        if not isinstance(snapshot, KnowledgeSnapshot):
            raise KnowledgeSnapshotStoreError("snapshot must be a KnowledgeSnapshot")
        if type(snapshot.sources) is not tuple or type(snapshot.documents) is not tuple:
            raise KnowledgeSnapshotStoreError("snapshot sources and documents must be tuples")
        sources: list[tuple[Any, ...]] = []
        source_ids: set[str] = set()
        for source in snapshot.sources:
            if not isinstance(source, KnowledgeSource):
                raise KnowledgeSnapshotStoreError("snapshot contains an invalid KnowledgeSource")
            if source.source_id in source_ids:
                raise KnowledgeSnapshotStoreError("snapshot contains duplicate source IDs")
            source_ids.add(source.source_id)
            source_type = source.source_type.value if isinstance(source.source_type, Enum) else source.source_type
            if type(source_type) is not str or not source_type.strip():
                raise KnowledgeSnapshotStoreError("snapshot source type is invalid")
            sources.append(
                (
                    source.source_id,
                    source_type,
                    source.display_name,
                    source.logical_location,
                    cls._encode_metadata(source.metadata),
                )
            )
        documents: list[tuple[Any, ...]] = []
        document_ids: set[str] = set()
        for document in snapshot.documents:
            if not isinstance(document, Document):
                raise KnowledgeSnapshotStoreError("snapshot contains an invalid Document")
            if document.source_id not in source_ids:
                raise KnowledgeSnapshotStoreError("snapshot Document references a missing source")
            if document.document_id in document_ids:
                raise KnowledgeSnapshotStoreError("snapshot contains duplicate document IDs")
            if document.document_id != stable_document_id(document.source_id, document.logical_path):
                raise KnowledgeSnapshotStoreError("snapshot Document identity is incoherent")
            if document.content_hash != compute_content_hash(document.content):
                raise KnowledgeSnapshotStoreError("snapshot Document content hash is incoherent")
            document_ids.add(document.document_id)
            documents.append(
                (
                    document.document_id,
                    document.source_id,
                    document.logical_path,
                    document.content,
                    document.content_type,
                    cls._encode_metadata(document.metadata),
                    document.content_hash,
                )
            )
        sources.sort(key=lambda row: row[0])
        documents.sort(key=lambda row: row[0])
        return sources, documents

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
