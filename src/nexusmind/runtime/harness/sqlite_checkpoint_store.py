from __future__ import annotations
import asyncio
import hashlib
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from .checkpoint import HarnessCheckpoint
from .checkpoint_codec import CheckpointDecodeError, checkpoint_from_json, checkpoint_to_json

class CheckpointStoreError(RuntimeError):
    pass

class SQLiteCheckpointStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        if self._path in {"", ":memory:"}:
            raise ValueError("SQLiteCheckpointStore requires a persistent database path")
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize)
        self._initialized = True
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    def _connect(self):
        connection = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self):
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise CheckpointStoreError("Unsupported checkpoint database schema")
                table_exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='harness_checkpoints'").fetchone() is not None
                if version == 0:
                    objects = db.execute("SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
                    if objects:
                        raise CheckpointStoreError("Unversioned database is not empty")
                    db.execute("""CREATE TABLE harness_checkpoints (checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence >= 0), checkpoint_schema_version INTEGER NOT NULL, boundary TEXT NOT NULL, created_at TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, UNIQUE(run_id, sequence))""")
                    db.execute("CREATE INDEX idx_harness_checkpoints_run_sequence ON harness_checkpoints(run_id, sequence DESC)")
                    db.execute("PRAGMA user_version = 1")
                elif not table_exists:
                    raise CheckpointStoreError("Checkpoint schema is missing")
                self._validate_v1_schema(db)
                db.commit()
        except CheckpointStoreError:
            raise
        except sqlite3.Error as exc:
            raise CheckpointStoreError("SQLite checkpoint initialization failed") from exc

    @staticmethod
    def _validate_v1_schema(db) -> None:
        expected = {"checkpoint_id": ("TEXT", 0, 1), "run_id": ("TEXT", 1, 0), "sequence": ("INTEGER", 1, 0), "checkpoint_schema_version": ("INTEGER", 1, 0), "boundary": ("TEXT", 1, 0), "created_at": ("TEXT", 1, 0), "payload_json": ("TEXT", 1, 0), "payload_sha256": ("TEXT", 1, 0)}
        actual = {row[1]: (row[2].upper(), row[3], row[5]) for row in db.execute("PRAGMA table_info(harness_checkpoints)")}
        if actual != expected:
            raise CheckpointStoreError("Checkpoint database schema is incomplete or incompatible")
        indexes = db.execute("PRAGMA index_list(harness_checkpoints)").fetchall()
        unique_run_sequence = False
        query_index = False
        for row in indexes:
            index_name, is_unique, is_partial = row[1], row[2], row[4]
            safe_index_name = index_name.replace('"', '""')
            columns = [item[2] for item in db.execute(f'PRAGMA index_info("{safe_index_name}")')]
            if is_unique == 1 and is_partial == 0 and columns == ["run_id", "sequence"]:
                unique_run_sequence = True
            if index_name == "idx_harness_checkpoints_run_sequence" and is_partial == 0 and columns == ["run_id", "sequence"]:
                query_index = True
        if not unique_run_sequence:
            raise CheckpointStoreError("Checkpoint database is missing the run sequence uniqueness constraint")
        if not query_index:
            raise CheckpointStoreError("Checkpoint sequence index is missing or invalid")

    def _require(self):
        if not self._initialized or self._closed:
            raise CheckpointStoreError("Checkpoint store is not initialized")

    async def save(self, checkpoint: HarnessCheckpoint) -> None:
        self._require(); await asyncio.to_thread(self._save, deepcopy(checkpoint))

    def _save(self, checkpoint):
        checkpoint.validate(); payload = checkpoint_to_json(checkpoint); digest = hashlib.sha256(payload.encode()).hexdigest()
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                latest = db.execute("SELECT checkpoint_id, run_id, sequence, checkpoint_schema_version, boundary, created_at, payload_json, payload_sha256 FROM harness_checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1", (checkpoint.run_id,)).fetchone()
                if latest is not None:
                    latest_checkpoint = self._decode_row(latest)
                    if checkpoint.sequence <= latest_checkpoint.sequence:
                        raise CheckpointStoreError("Checkpoint sequence must increase")
                if type(payload) is not str:
                    raise CheckpointStoreError("Checkpoint payload has an invalid storage type")
                db.execute("INSERT INTO harness_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (checkpoint.checkpoint_id, checkpoint.run_id, checkpoint.sequence, checkpoint.schema_version, checkpoint.boundary.value, checkpoint.created_at, payload, digest))
                db.commit()
        except CheckpointStoreError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CheckpointStoreError("Checkpoint conflicts with existing data") from exc
        except sqlite3.Error as exc:
            raise CheckpointStoreError("SQLite checkpoint operation failed") from exc

    async def load_latest(self, run_id: str) -> HarnessCheckpoint | None:
        self._require(); return await asyncio.to_thread(self._load_latest, run_id)

    def _load_latest(self, run_id):
        try:
            with closing(self._connect()) as db:
                row = db.execute("SELECT checkpoint_id, run_id, sequence, checkpoint_schema_version, boundary, created_at, payload_json, payload_sha256 FROM harness_checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1", (run_id,)).fetchone()
            return self._decode_row(row) if row else None
        except CheckpointStoreError:
            raise
        except sqlite3.Error as exc:
            raise CheckpointStoreError("SQLite checkpoint operation failed") from exc

    async def list(self, run_id: str) -> tuple[HarnessCheckpoint, ...]:
        self._require(); return await asyncio.to_thread(self._list, run_id)

    def _list(self, run_id):
        try:
            with closing(self._connect()) as db:
                rows = db.execute("SELECT checkpoint_id, run_id, sequence, checkpoint_schema_version, boundary, created_at, payload_json, payload_sha256 FROM harness_checkpoints WHERE run_id = ? ORDER BY sequence ASC", (run_id,)).fetchall()
            return tuple(self._decode_row(row) for row in rows)
        except CheckpointStoreError:
            raise
        except sqlite3.Error as exc:
            raise CheckpointStoreError("SQLite checkpoint operation failed") from exc

    @staticmethod
    def _decode_row(row):
        checkpoint_id, run_id, sequence, version, boundary, created_at, payload, digest = row
        if type(payload) is not str:
            raise CheckpointStoreError("Checkpoint payload has an invalid storage type")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
            raise CheckpointStoreError("Checkpoint payload integrity check failed")
        try:
            checkpoint = checkpoint_from_json(payload)
        except Exception as exc:
            raise CheckpointStoreError("Checkpoint payload is invalid") from exc
        if (checkpoint.checkpoint_id, checkpoint.run_id, checkpoint.sequence, checkpoint.schema_version, checkpoint.boundary.value, checkpoint.created_at) != (checkpoint_id, run_id, sequence, version, boundary, created_at):
            raise CheckpointStoreError("Checkpoint envelope does not match payload")
        return checkpoint
