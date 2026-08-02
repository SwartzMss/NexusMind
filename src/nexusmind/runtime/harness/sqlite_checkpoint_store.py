from __future__ import annotations
import asyncio
import hashlib
import sqlite3
from copy import deepcopy
from pathlib import Path
from .checkpoint import HarnessCheckpoint
from .checkpoint_codec import CheckpointDecodeError, checkpoint_from_json, checkpoint_to_json

class CheckpointStoreError(RuntimeError):
    pass

class SQLiteCheckpointStore:
    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
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
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise CheckpointStoreError("Unsupported checkpoint database schema")
            table_exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='harness_checkpoints'").fetchone() is not None
            if version == 0 and table_exists:
                raise CheckpointStoreError("Unversioned checkpoint schema is partially initialized")
            if version == 0:
                db.execute("""CREATE TABLE harness_checkpoints (checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence >= 0), checkpoint_schema_version INTEGER NOT NULL, boundary TEXT NOT NULL, created_at TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, UNIQUE(run_id, sequence))""")
                db.execute("CREATE INDEX idx_harness_checkpoints_run_sequence ON harness_checkpoints(run_id, sequence DESC)")
                db.execute("PRAGMA user_version = 1")
            elif not table_exists:
                raise CheckpointStoreError("Checkpoint schema is missing")
            expected = {"checkpoint_id": ("TEXT", 0, 1), "run_id": ("TEXT", 1, 0), "sequence": ("INTEGER", 1, 0), "checkpoint_schema_version": ("INTEGER", 1, 0), "boundary": ("TEXT", 1, 0), "created_at": ("TEXT", 1, 0), "payload_json": ("TEXT", 1, 0), "payload_sha256": ("TEXT", 1, 0)}
            actual = {row[1]: (row[2].upper(), row[3], row[5]) for row in db.execute("PRAGMA table_info(harness_checkpoints)")}
            if actual != expected:
                raise CheckpointStoreError("Checkpoint database schema is incomplete or incompatible")
            table_sql = (db.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'harness_checkpoints'").fetchone() or ("",))[0] or ""
            if "UNIQUE(run_id,sequence)" not in table_sql.replace(" ", ""):
                raise CheckpointStoreError("Checkpoint database is missing the run sequence uniqueness constraint")
            index = db.execute("PRAGMA index_list(harness_checkpoints)").fetchall()
            index_name = next((row[1] for row in index if row[1] == "idx_harness_checkpoints_run_sequence"), None)
            if index_name is None:
                raise CheckpointStoreError("Checkpoint sequence index is missing")
            columns = [row[2] for row in db.execute(f"PRAGMA index_info({index_name})")]
            if columns != ["run_id", "sequence"]:
                raise CheckpointStoreError("Checkpoint sequence index is invalid")

    def _require(self):
        if not self._initialized or self._closed:
            raise CheckpointStoreError("Checkpoint store is not initialized")

    async def save(self, checkpoint: HarnessCheckpoint) -> None:
        self._require(); await asyncio.to_thread(self._save, deepcopy(checkpoint))

    def _save(self, checkpoint):
        checkpoint.validate(); payload = checkpoint_to_json(checkpoint); digest = hashlib.sha256(payload.encode()).hexdigest()
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                latest = db.execute("SELECT sequence FROM harness_checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1", (checkpoint.run_id,)).fetchone()
                if latest and checkpoint.sequence <= latest[0]: raise CheckpointStoreError("Checkpoint sequence must increase")
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
            with self._connect() as db:
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
            with self._connect() as db:
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
