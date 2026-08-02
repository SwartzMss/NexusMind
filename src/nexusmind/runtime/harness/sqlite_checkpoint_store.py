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

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize)
        self._initialized = True

    def _connect(self):
        connection = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self):
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise CheckpointStoreError("Unsupported checkpoint database schema")
            db.execute("""CREATE TABLE IF NOT EXISTS harness_checkpoints (checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, sequence INTEGER NOT NULL CHECK(sequence >= 0), checkpoint_schema_version INTEGER NOT NULL, boundary TEXT NOT NULL, created_at TEXT NOT NULL, payload_json BLOB NOT NULL, payload_sha256 TEXT NOT NULL, UNIQUE(run_id, sequence))""")
            expected = {"checkpoint_id", "run_id", "sequence", "checkpoint_schema_version", "boundary", "created_at", "payload_json", "payload_sha256"}
            actual = {row[1] for row in db.execute("PRAGMA table_info(harness_checkpoints)")}
            if actual != expected:
                raise CheckpointStoreError("Checkpoint database schema is incomplete or incompatible")
            db.execute("CREATE INDEX IF NOT EXISTS idx_harness_checkpoints_run_sequence ON harness_checkpoints(run_id, sequence DESC)")
            db.execute("PRAGMA user_version = 1")

    def _require(self):
        if not self._initialized:
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
                db.execute("INSERT INTO harness_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (checkpoint.checkpoint_id, checkpoint.run_id, checkpoint.sequence, checkpoint.schema_version, checkpoint.boundary.value, checkpoint.created_at, payload, digest))
                db.commit()
        except sqlite3.IntegrityError as exc:
            raise CheckpointStoreError("Checkpoint conflicts with existing data") from exc

    async def load_latest(self, run_id: str) -> HarnessCheckpoint | None:
        self._require(); return await asyncio.to_thread(self._load_latest, run_id)

    def _load_latest(self, run_id):
        with self._connect() as db:
            row = db.execute("SELECT checkpoint_id, run_id, sequence, checkpoint_schema_version, boundary, created_at, payload_json, payload_sha256 FROM harness_checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1", (run_id,)).fetchone()
        return self._decode_row(row) if row else None

    async def list(self, run_id: str) -> tuple[HarnessCheckpoint, ...]:
        self._require(); return await asyncio.to_thread(self._list, run_id)

    def _list(self, run_id):
        with self._connect() as db:
            rows = db.execute("SELECT checkpoint_id, run_id, sequence, checkpoint_schema_version, boundary, created_at, payload_json, payload_sha256 FROM harness_checkpoints WHERE run_id = ? ORDER BY sequence ASC", (run_id,)).fetchall()
        return tuple(self._decode_row(row) for row in rows)

    @staticmethod
    def _decode_row(row):
        checkpoint_id, run_id, sequence, version, boundary, created_at, payload, digest = row
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise CheckpointStoreError("Checkpoint payload integrity check failed")
        checkpoint = checkpoint_from_json(payload)
        if (checkpoint.checkpoint_id, checkpoint.run_id, checkpoint.sequence, checkpoint.schema_version, checkpoint.boundary.value, checkpoint.created_at) != (checkpoint_id, run_id, sequence, version, boundary, created_at):
            raise CheckpointStoreError("Checkpoint envelope does not match payload")
        return checkpoint
