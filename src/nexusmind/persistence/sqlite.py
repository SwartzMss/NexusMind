from __future__ import annotations
import hashlib, json, sqlite3, uuid
import asyncio
from functools import wraps
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .contracts import RunKind, RunStartContext, RunStatus, RunTraceEvent

MAX_EVENT_BYTES=64*1024; MAX_EVENTS=10000; MAX_FINAL_TEXT=1024*1024
class StateStoreError(RuntimeError): pass
def _async_db_error(fn):
    @wraps(fn)
    async def wrapped(self, *args, **kwargs):
        try: return await fn(self, *args, **kwargs)
        except asyncio.CancelledError:
            try: self.db.rollback()
            except BaseException: pass
            raise
        except (StateStoreError, ValueError):
            try: self.db.rollback()
            except BaseException: pass
            raise
        except Exception as exc:
            try: self.db.rollback()
            except BaseException: pass
            raise StateStoreError("Run store operation failed") from exc
    return wrapped
def _sync_db_error(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        try: return fn(self, *args, **kwargs)
        except asyncio.CancelledError:
            try: self.db.rollback()
            except BaseException: pass
            raise
        except (StateStoreError, ValueError):
            try: self.db.rollback()
            except BaseException: pass
            raise
        except Exception as exc:
            try: self.db.rollback()
            except BaseException: pass
            raise StateStoreError("Run store operation failed") from exc
    return wrapped
def _now(): return datetime.now(timezone.utc).isoformat()
class SQLiteRunStore:
    def __init__(self, path: str | Path, *, execution_id: str | None = None, recover_abandoned: bool = False):
        self.path=Path(path); self.execution_id=execution_id or uuid.uuid4().hex
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db=sqlite3.connect(self.path, timeout=0.5); self.db.execute("PRAGMA busy_timeout=500"); self.db.execute("PRAGMA foreign_keys=ON")
            self._schema()
            if recover_abandoned: self.recover_abandoned()
        except (OSError, sqlite3.Error) as exc: raise StateStoreError("Run store could not be initialized") from exc
    def _schema(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._schema_transaction()
            self.db.commit()
        except BaseException:
            try: self.db.rollback()
            except BaseException: pass
            raise
    def _schema_transaction(self):
        metadata_exists=self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_metadata'").fetchone()
        if metadata_exists:
            row=self.db.execute("SELECT value FROM schema_metadata WHERE key='version'").fetchone()
            if row is None or row[0] != "1": raise StateStoreError("Unsupported state database schema version")
            return
        existing=self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone()
        if existing: raise StateStoreError("Unsupported state database schema version")
        self.db.execute("CREATE TABLE schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE runs(id TEXT PRIMARY KEY,schema_version INTEGER NOT NULL,execution_id TEXT NOT NULL,kind TEXT NOT NULL,status TEXT NOT NULL,skill_name TEXT,model_name TEXT,input_preview TEXT,input_sha256 TEXT,final_text TEXT,error_code TEXT,error_message TEXT,trace_complete INTEGER NOT NULL DEFAULT 1,event_count INTEGER NOT NULL DEFAULT 0,started_at TEXT NOT NULL,updated_at TEXT NOT NULL,finished_at TEXT)")
        self.db.execute("CREATE TABLE run_events(run_id TEXT NOT NULL,sequence INTEGER NOT NULL,event_type TEXT NOT NULL,occurred_at TEXT NOT NULL,payload_json TEXT NOT NULL,payload_bytes INTEGER NOT NULL,PRIMARY KEY(run_id,sequence),FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE)")
        self.db.execute("CREATE INDEX idx_runs_started_at ON runs(started_at DESC)"); self.db.execute("CREATE INDEX idx_runs_status_started ON runs(status,started_at DESC)")
        self.db.execute("INSERT INTO schema_metadata VALUES('version','1')")
    def _recover_abandoned(self):
        now=_now(); rows=self.db.execute("SELECT id FROM runs WHERE status='running' AND execution_id<>?",(self.execution_id,)).fetchall()
        for (run_id,) in rows:
            cur=self.db.execute("UPDATE runs SET status='abandoned',trace_complete=0,error_code='process_abandoned',updated_at=?,finished_at=?,error_message=? WHERE id=? AND status='running'",(now,now,"Previous NexusMind process ended before the run reached a terminal state",run_id))
            if cur.rowcount == 0: continue
            seq=self.db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM run_events WHERE run_id=?",(run_id,)).fetchone()[0]
            payload=json.dumps({"reason":"previous_process"})
            self.db.execute("INSERT INTO run_events VALUES(?,?,?,?,?,?)",(run_id,seq,"run_abandoned",now,payload,len(payload)))
            self.db.execute("UPDATE runs SET event_count=? WHERE id=?",(seq,run_id))
        self.db.commit()
    @_async_db_error
    async def start_run(self, context: RunStartContext):
        run_id=uuid.uuid4().hex; now=_now(); digest=hashlib.sha256((context.input_text or '').encode()).hexdigest()
        preview=(context.input_text or '').encode('utf-8')[:512].decode('utf-8','ignore') if context.record_content else None
        self.db.execute("INSERT INTO runs(id,schema_version,execution_id,kind,status,skill_name,model_name,input_preview,input_sha256,started_at,updated_at,event_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,1)",(run_id,1,self.execution_id,context.kind.value,'running',context.skill_name,context.model_name,preview,digest,now,now))
        payload=json.dumps({"kind":context.kind.value},separators=(',',':')); self.db.execute("INSERT INTO run_events VALUES(?,?,?,?,?,?)",(run_id,1,"run_started",now,payload,len(payload))); self.db.commit(); return run_id
    @_async_db_error
    async def append_event(self, run_id, event: RunTraceEvent):
        self.db.execute("BEGIN IMMEDIATE")
        raw=json.dumps(event.payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>MAX_EVENT_BYTES: raise StateStoreError("Run event payload exceeds limit")
        row=self.db.execute("SELECT status,event_count FROM runs WHERE id=?",(run_id,)).fetchone()
        if not row: raise StateStoreError("Run not found")
        if row[0] != RunStatus.RUNNING.value: raise StateStoreError("Run is not running")
        if row[1]>=MAX_EVENTS-1: raise StateStoreError("Run event limit exceeded")
        seq=row[1]+1; self.db.execute("INSERT INTO run_events VALUES(?,?,?,?,?,?)",(run_id,seq,event.event_type,event.occurred_at.isoformat(),raw.decode(),len(raw))); self.db.execute("UPDATE runs SET event_count=?,updated_at=? WHERE id=?",(seq,_now(),run_id)); self.db.commit()
    @_async_db_error
    async def finish_run(self, run_id, status, *, error_code=None, error_message=None, trace_complete=None, final_text=None):
        if status is RunStatus.RUNNING: raise StateStoreError("Run finish status must be terminal")
        bounded=(final_text or '').encode('utf-8')[:MAX_FINAL_TEXT].decode('utf-8','ignore') or None
        now=_now(); cur=self.db.execute("UPDATE runs SET status=?,error_code=?,error_message=?,trace_complete=COALESCE(?,trace_complete),final_text=COALESCE(?,final_text),updated_at=?,finished_at=? WHERE id=? AND status='running'",(status.value,error_code,(error_message or '')[:1024] or None, None if trace_complete is None else int(trace_complete), bounded,now,now,run_id))
        if cur.rowcount == 0: raise StateStoreError("Run does not exist or is already terminal")
        if cur.rowcount: 
            seq=self.db.execute("SELECT event_count FROM runs WHERE id=?",(run_id,)).fetchone()[0]+1; event_type={RunStatus.COMPLETED:"run_completed",RunStatus.FAILED:"run_failed",RunStatus.CANCELLED:"run_cancelled",RunStatus.ABANDONED:"run_abandoned"}.get(status)
            if event_type:
                payload=json.dumps({"error_code":error_code} if error_code else {},separators=(',',':')); self.db.execute("INSERT INTO run_events VALUES(?,?,?,?,?,?)",(run_id,seq,event_type,now,payload,len(payload))); self.db.execute("UPDATE runs SET event_count=? WHERE id=?",(seq,run_id))
        self.db.commit()
    @_sync_db_error
    def list_runs(self, *, limit=20, status=None, kind=None, skill=None):
        limit=max(1,min(int(limit),200)); q="SELECT id,status,kind,started_at,skill_name FROM runs"; vals=[]; filters=[]
        for col,val in (("status",status),("kind",kind),("skill_name",skill)):
            if val: filters.append(col+"=?"); vals.append(val)
        if filters:q+=" WHERE "+" AND ".join(filters)
        return self.db.execute(q+" ORDER BY started_at DESC LIMIT ?",(*vals,limit)).fetchall()
    @_sync_db_error
    def show_run(self, run_id):
        cur=self.db.execute("SELECT * FROM runs WHERE id=?",(run_id,)); row=cur.fetchone()
        if not row:return None
        run=dict(zip([d[0] for d in cur.description], row)); cur=self.db.execute("SELECT sequence,event_type,occurred_at,payload_json FROM run_events WHERE run_id=? ORDER BY sequence",(run_id,))
        events=[{"sequence":r[0],"event_type":r[1],"occurred_at":r[2],"payload":json.loads(r[3])} for r in cur.fetchall()]; return {"run":run,"events":events}
    @_sync_db_error
    def prune(self, older_than_days):
        if int(older_than_days) < 0: raise ValueError("older-than-days must be non-negative")
        cutoff=(datetime.now(timezone.utc)-timedelta(days=int(older_than_days))).isoformat()
        rows=self.db.execute("SELECT id FROM runs WHERE status<>'running' AND started_at<?",(cutoff,)).fetchall(); self.db.executemany("DELETE FROM runs WHERE id=?",rows); self.db.commit(); return len(rows)
    @_sync_db_error
    def recover_abandoned(self):
        self._recover_abandoned()
    @_sync_db_error
    def close(self): self.db.close()
