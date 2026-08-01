from __future__ import annotations
import hashlib, json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path
from .contracts import RunKind, RunStartContext, RunStatus, RunTraceEvent

MAX_EVENT_BYTES=64*1024; MAX_EVENTS=10000; MAX_FINAL_TEXT=1024*1024
class StateStoreError(RuntimeError): pass
def _now(): return datetime.now(timezone.utc).isoformat()
class SQLiteRunStore:
    def __init__(self, path: str | Path, *, execution_id: str | None = None, recover_abandoned: bool = False):
        self.path=Path(path); self.execution_id=execution_id or uuid.uuid4().hex
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db=sqlite3.connect(self.path); self.db.execute("PRAGMA foreign_keys=ON")
            self._schema()
            if recover_abandoned: self.recover_abandoned()
        except (OSError, sqlite3.Error) as exc: raise StateStoreError("Run store could not be initialized") from exc
    def _schema(self):
        self.db.executescript('''CREATE TABLE IF NOT EXISTS schema_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,schema_version INTEGER NOT NULL,execution_id TEXT NOT NULL,kind TEXT NOT NULL,status TEXT NOT NULL,skill_name TEXT,model_name TEXT,input_preview TEXT,input_sha256 TEXT,final_text TEXT,error_code TEXT,error_message TEXT,trace_complete INTEGER NOT NULL DEFAULT 1,event_count INTEGER NOT NULL DEFAULT 0,started_at TEXT NOT NULL,updated_at TEXT NOT NULL,finished_at TEXT);
CREATE TABLE IF NOT EXISTS run_events(run_id TEXT NOT NULL,sequence INTEGER NOT NULL,event_type TEXT NOT NULL,occurred_at TEXT NOT NULL,payload_json TEXT NOT NULL,payload_bytes INTEGER NOT NULL,PRIMARY KEY(run_id,sequence),FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC); CREATE INDEX IF NOT EXISTS idx_runs_status_started ON runs(status,started_at DESC);''')
        row=self.db.execute("SELECT value FROM schema_metadata WHERE key='version'").fetchone()
        if row is None:
            self.db.execute("INSERT INTO schema_metadata VALUES('version','1')")
        elif row[0] != "1":
            raise StateStoreError("Unsupported state database schema version")
        self.db.commit()
    def recover_abandoned(self):
        now=_now(); rows=self.db.execute("SELECT id FROM runs WHERE status='running' AND execution_id<>?",(self.execution_id,)).fetchall()
        for (run_id,) in rows:
            self.db.execute("UPDATE runs SET status='abandoned',updated_at=?,finished_at=?,error_message=? WHERE id=?",(now,now,"Previous NexusMind process ended before the run reached a terminal state",run_id))
            seq=self.db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM run_events WHERE run_id=?",(run_id,)).fetchone()[0]
            payload=json.dumps({"reason":"previous_process"})
            self.db.execute("INSERT INTO run_events VALUES(?,?,?,?,?,?)",(run_id,seq,"run_abandoned",now,payload,len(payload)))
            self.db.execute("UPDATE runs SET event_count=? WHERE id=?",(seq,run_id))
        self.db.commit()
    async def start_run(self, context: RunStartContext):
        run_id=uuid.uuid4().hex; now=_now(); digest=hashlib.sha256((context.input_text or '').encode()).hexdigest()
        preview=(context.input_text or '')[:512] if context.record_content else None
        self.db.execute("INSERT INTO runs(id,schema_version,execution_id,kind,status,skill_name,model_name,input_preview,input_sha256,started_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(run_id,1,self.execution_id,context.kind.value,'running',context.skill_name,context.model_name,preview,digest,now,now)); self.db.commit(); return run_id
    async def append_event(self, run_id, event: RunTraceEvent):
        raw=json.dumps(event.payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
        if len(raw)>MAX_EVENT_BYTES: raise StateStoreError("Run event payload exceeds limit")
        row=self.db.execute("SELECT event_count FROM runs WHERE id=?",(run_id,)).fetchone()
        if not row or row[0]>=MAX_EVENTS: raise StateStoreError("Run event limit exceeded")
        seq=row[0]+1; self.db.execute("INSERT INTO run_events VALUES(?,?,?,?,?,?)",(run_id,seq,event.event_type,event.occurred_at.isoformat(),raw.decode(),len(raw))); self.db.execute("UPDATE runs SET event_count=?,updated_at=? WHERE id=?",(seq,_now(),run_id)); self.db.commit()
    async def finish_run(self, run_id, status, *, error_code=None, error_message=None, trace_complete=None):
        now=_now(); self.db.execute("UPDATE runs SET status=?,error_code=?,error_message=?,trace_complete=COALESCE(?,trace_complete),updated_at=?,finished_at=? WHERE id=? AND status='running'",(status.value,error_code,(error_message or '')[:1024] or None, None if trace_complete is None else int(trace_complete),now,now,run_id)); self.db.commit()
    def list_runs(self, *, limit=20, status=None, kind=None, skill=None):
        limit=max(1,min(int(limit),200)); q="SELECT id,status,kind,started_at,skill_name FROM runs"; vals=[]; filters=[]
        for col,val in (("status",status),("kind",kind),("skill_name",skill)):
            if val: filters.append(col+"=?"); vals.append(val)
        if filters:q+=" WHERE "+" AND ".join(filters)
        return self.db.execute(q+" ORDER BY started_at DESC LIMIT ?",(*vals,limit)).fetchall()
    def show_run(self, run_id):
        cur=self.db.execute("SELECT * FROM runs WHERE id=?",(run_id,)); row=cur.fetchone()
        if not row:return None
        run=dict(zip([d[0] for d in cur.description], row)); cur=self.db.execute("SELECT sequence,event_type,occurred_at,payload_json FROM run_events WHERE run_id=? ORDER BY sequence",(run_id,))
        events=[{"sequence":r[0],"event_type":r[1],"occurred_at":r[2],"payload":json.loads(r[3])} for r in cur.fetchall()]; return {"run":run,"events":events}
    def prune(self, older_than_days):
        if int(older_than_days) < 0: raise ValueError("older-than-days must be non-negative")
        cutoff=datetime.now(timezone.utc).timestamp()-int(older_than_days)*86400
        rows=self.db.execute("SELECT id FROM runs WHERE status<>'running' AND strftime('%s',started_at)<?",(int(cutoff),)).fetchall(); self.db.executemany("DELETE FROM runs WHERE id=?",rows); self.db.commit(); return len(rows)
    def close(self): self.db.close()
