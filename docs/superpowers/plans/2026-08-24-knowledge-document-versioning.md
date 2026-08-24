# Knowledge Document Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist immutable document provenance and version chains across sync and SQLite restart while keeping queries restricted to current documents.

**Architecture:** Add a focused `DocumentVersion` contract beside `Document`. `KnowledgeCollection` owns separate history and stages it atomically with current documents and the index; `KnowledgeSnapshot` and SQLite schema v2 carry history while legacy history-free snapshots remain restorable.

**Tech Stack:** Python 3.11+, frozen dataclasses, SHA-256/canonical JSON, UTC `datetime`, SQLite transactions, pytest.

---

## File Map

- `src/nexusmind/knowledge.py`: immutable version values and stable IDs.
- `src/nexusmind/knowledge_collection.py`: atomic version lifecycle and validation.
- `src/nexusmind/knowledge_store.py`: SQLite migration and version persistence.
- `src/nexusmind/__init__.py`: serialization type/helper exports.
- `tests/test_knowledge_versioning.py`: focused lifecycle and coherence coverage.
- `tests/test_knowledge_snapshot.py`: legacy and atomic restore coverage.
- `tests/test_knowledge_store.py`: persistence, migration, and rollback coverage.
- `README.md`: lifecycle and current-only search documentation.

### Task 1: Immutable Document Version Contract

**Files:**
- Modify: `src/nexusmind/knowledge.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_knowledge_versioning.py`

- [ ] **Step 1: Write the failing contract test**

```python
def test_document_version_captures_identity_and_provenance() -> None:
    document = _document("docs", "guide.md", "first")
    version = DocumentVersion.from_document(
        document,
        created_at="2026-08-24T02:00:00.000000Z",
        sync_context="sync-fixed",
    )
    assert version.document_id == document.document_id
    assert version.content_hash == compute_content_hash("first")
    assert version.previous_version_id is None
    assert version.version_id == stable_document_version_id(
        document.document_id, document.content_hash, None
    )
```

Add parametrized cases for bad timestamps, hashes, predecessors, IDs, and content/hash disagreement.

- [ ] **Step 2: Run RED**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_versioning.py -q`

Expected: import failure because `DocumentVersion` is absent.

- [ ] **Step 3: Implement the minimal contract**

```python
def stable_document_version_id(
    document_id: str, content_hash: str, previous_version_id: str | None
) -> str:
    predecessor = "" if previous_version_id is None else previous_version_id
    return _stable_id("version", document_id, content_hash, predecessor)

@dataclass(frozen=True, slots=True)
class DocumentVersion:
    version_id: str
    document_id: str
    source_id: str
    logical_path: str
    content: str
    content_hash: str
    created_at: str
    previous_version_id: str | None
    sync_context: str
```

Validate non-empty fields, lowercase SHA-256, canonical UTC timestamp, content hash, stable document ID, deterministic version ID, and optional predecessor. Add `from_document()` and exports.

- [ ] **Step 4: Run GREEN**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_versioning.py -q`

Expected: contract tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/nexusmind/knowledge.py src/nexusmind/__init__.py tests/test_knowledge_versioning.py
git commit -m "feat: add immutable document version contract"
```

### Task 2: Version-Aware Atomic Sync

**Files:**
- Modify: `src/nexusmind/knowledge_collection.py`
- Modify: `tests/test_knowledge_versioning.py`

- [ ] **Step 1: Write failing lifecycle tests**

Cover first sync, unchanged sync, changed content, deletion, same-content reappearance, shared context for two changes, and detached snapshots with a deterministic clock:

```python
times = iter((
    datetime(2026, 8, 24, 2, 0, tzinfo=timezone.utc),
    datetime(2026, 8, 24, 2, 1, tzinfo=timezone.utc),
))
collection = KnowledgeCollection(clock=lambda: next(times))
collection.sync(Adapter((_document("docs", "a.md", "one"),)))
collection.sync(Adapter((_document("docs", "a.md", "two"),)))
versions = collection.snapshot().document_versions
assert [item.content for item in versions] == ["one", "two"]
assert versions[1].previous_version_id == versions[0].version_id
```

- [ ] **Step 2: Run RED**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_versioning.py -q`

Expected: no history/clock support.

- [ ] **Step 3: Implement staged history**

```python
class KnowledgeCollectionLimits:
    max_sources: int = 100
    max_documents: int = 10_000
    max_document_versions: int = 100_000

class KnowledgeSnapshot:
    sources: tuple[KnowledgeSource, ...]
    documents: tuple[Document, ...]
    document_versions: tuple[DocumentVersion, ...] = ()
```

Add `clock: Callable[[], datetime] | None`, `_document_versions`, canonical UTC formatting, one context per version-producing sync, version-limit preflight, and staged commit. Added documents with retained history are reappearances and append even when content equals the tip. `remove_source()` preserves history.

- [ ] **Step 4: Add atomic failure tests**

Make clock, chunker, and index replacement raise; test version-limit overflow. For every failure, assert snapshot and search equal their pre-sync values.

- [ ] **Step 5: Run GREEN**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_versioning.py tests/test_knowledge_collection.py -q`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/nexusmind/knowledge_collection.py tests/test_knowledge_versioning.py
git commit -m "feat: retain document versions during sync"
```

### Task 3: Coherent Restore and Legacy Upgrade

**Files:**
- Modify: `src/nexusmind/knowledge_collection.py`
- Modify: `tests/test_knowledge_versioning.py`
- Modify: `tests/test_knowledge_snapshot.py`

- [ ] **Step 1: Write failing restore tests**

Test multi-version round trip, current-only search, historical-only documents/sources, and legacy two-field snapshots. Parametrize wrong hashes/IDs, missing predecessor, duplicate IDs, two roots, fork, cycle, stale current content, invalid timestamps, and partial history.

```python
restored = KnowledgeCollection(clock=fixed_clock)
restored.restore(original.snapshot())
assert restored.snapshot() == original.snapshot()
assert restored.search("old-only-term") == ()
assert restored.search("current-term")
```

- [ ] **Step 2: Run RED**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_versioning.py tests/test_knowledge_snapshot.py -q`

Expected: malformed histories are accepted or discarded.

- [ ] **Step 3: Validate complete chains before rechunking**

Require tuple/exact types; group by document ID; validate identity; walk predecessors to prove one linear root-to-tip chain; require each current document to match its tip. If history is exactly empty, synthesize one root per current document using one clock reading/context. Partial history fails closed.

- [ ] **Step 4: Stage restore atomically**

Keep candidate history local while rechunking current documents. Swap sources, documents, history, and index only after all operations succeed.

- [ ] **Step 5: Run GREEN**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_versioning.py tests/test_knowledge_snapshot.py tests/test_knowledge_collection.py -q`

Expected: all pass, including two-field snapshot construction.

- [ ] **Step 6: Commit**

```bash
git add src/nexusmind/knowledge_collection.py tests/test_knowledge_versioning.py tests/test_knowledge_snapshot.py
git commit -m "feat: restore coherent document version history"
```

### Task 4: SQLite Schema v2 and Migration

**Files:**
- Modify: `src/nexusmind/knowledge_store.py`
- Modify: `tests/test_knowledge_store.py`

- [ ] **Step 1: Write failing persistence tests**

Round-trip active and historical-only versions, restart and prove current-only search, and extend concurrent loading to include version rows from one point in time.

- [ ] **Step 2: Run RED**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_store.py -q`

Expected: loaded snapshots omit history.

- [ ] **Step 3: Implement schema v2**

Add `document_versions(version_id PRIMARY KEY, document_id, source_id, logical_path, content, content_hash, created_at, previous_version_id, sync_context)` plus a document index. Bump schema metadata to `2`; transactionally replace and load versions with sources/documents:

```python
return KnowledgeSnapshot(
    sources=tuple(sources),
    documents=tuple(documents),
    document_versions=tuple(versions),
)
```

- [ ] **Step 4: Add and implement v1 migration**

Create a former-schema fixture. Opening it must add the v2 table in one transaction without changing sources/documents; load returns empty history for collection legacy upgrade. Unknown versions still fail closed.

- [ ] **Step 5: Test rollback and malformed rows**

Force a version insert failure and assert the prior snapshot remains. Insert malformed version fields and assert `KnowledgeSnapshotStoreError` without content leakage.

- [ ] **Step 6: Run GREEN**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_store.py tests/test_knowledge_versioning.py tests/test_knowledge_snapshot.py -q`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/nexusmind/knowledge_store.py tests/test_knowledge_store.py
git commit -m "feat: persist document version history in sqlite"
```

### Task 5: Documentation and Regression Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document lifecycle boundaries**

Explain append conditions, retention after removal, SQLite persistence, current-only indexing/search, and non-goals: public history retrieval, Git, background sync, and UI timeline.

- [ ] **Step 2: Run focused tests**

Run: `../../.venv/bin/python -m pytest tests/test_knowledge_versioning.py tests/test_knowledge_snapshot.py tests/test_knowledge_store.py tests/test_knowledge_collection.py tests/test_knowledge_base.py -q`

Expected: all pass.

- [ ] **Step 3: Run complete suite**

Run: `../../.venv/bin/python -m pytest`

Expected: all pass with only the established optional skip.

- [ ] **Step 4: Check hygiene and commit**

```bash
git diff --check
git status --short
git add README.md
git commit -m "docs: explain internal knowledge version history"
```

Expected: no whitespace errors and only intended changes.

### Task 6: Review and PR

**Files:**
- Review: every file changed from `origin/main`.

- [ ] **Step 1: Inspect the complete diff**

Run: `git diff --stat origin/main...HEAD && git diff --check origin/main...HEAD`

Expected: only issue #99 design, plan, implementation, tests, and README changes.

- [ ] **Step 2: Request code review**

Invoke `superpowers:requesting-code-review`; check acceptance criteria, API compatibility, atomicity, migration, and current-only retrieval. Fix substantiated findings and rerun affected tests.

- [ ] **Step 3: Verify completion**

Invoke `superpowers:verification-before-completion`; rerun the full suite and diff checks immediately before claiming readiness.

- [ ] **Step 4: Push and create PR**

```bash
git push -u origin agent/issue-99-document-versioning
gh pr create --repo SwartzMss/NexusMind --base main --head agent/issue-99-document-versioning --title "Add knowledge document provenance and version tracking" --body-file /tmp/nexusmind-issue-99-pr.md
```

The PR body summarizes the contract, atomic lifecycle, SQLite migration, current-only retrieval, test evidence, and includes `Closes #99`.
