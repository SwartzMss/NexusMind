# User-Facing KnowledgeBase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent user-facing `KnowledgeBase` API with strict local-source registration, atomic explicit synchronization, canonical SQLite persistence, and deterministic offline retrieval defaults.

**Architecture:** `knowledge_base_manifest.py` owns immutable configuration contracts, strict deterministic JSON encoding/decoding, limits, and atomic manifest replacement. `knowledge_base.py` owns product orchestration over fresh/staging `KnowledgeCollection` instances and the unchanged `SQLiteKnowledgeSnapshotStore`. The manifest and canonical database remain separate stores with explicit commit and compensation rules.

**Tech Stack:** Python 3.11–3.13, frozen dataclasses, JSON, pathlib/os atomic filesystem operations, SQLite snapshot store, pytest, existing Knowledge Runtime.

---

### Task 1: Manifest contracts, strict codec, and atomic writer

**Files:**
- Create: `src/nexusmind/knowledge_base_manifest.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_knowledge_base_manifest.py`

- [ ] **Step 1: Write failing public-contract and limit tests**

Test imports for `KnowledgeBaseError`, `KnowledgeBaseConfigError`, `KnowledgeBaseSourceError`, `KnowledgeBasePersistenceError`, `KnowledgeBaseClosedError`, `KnowledgeBaseLimits`, `KnowledgeBaseManifest`, `LocalFileSourceConfig`, and `LocalDirectorySourceConfig`. Parameterize every limit with `True`, `0`, `-1`, float, and string. Test non-empty IDs, optional non-empty display name, text paths, fixed config/type discriminators, and duplicate source rejection.

```python
@pytest.mark.parametrize("field", KnowledgeBaseLimits.__dataclass_fields__)
@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "1"])
def test_limits_require_positive_plain_integers(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        KnowledgeBaseLimits(**{field: value})
```

- [ ] **Step 2: Run RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base_manifest.py`

Expected: collection fails because the manifest module and exports do not exist.

- [ ] **Step 3: Implement immutable contracts**

Define the five-error hierarchy, positive plain-integer limits, two frozen source config dataclasses, a union alias, and a frozen manifest. Normalize source tuples by ascending ID only after validating exact tuple type, member type, uniqueness, configured field bounds, source count, and absolute normalized paths.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeBaseLimits:
    max_manifest_bytes: int = 1_000_000
    max_sources: int = 1_000
    max_knowledge_base_id_chars: int = 256
    max_display_name_chars: int = 1_024
    max_source_id_chars: int = 256
    max_path_chars: int = 32_768
```

- [ ] **Step 4: Write failing codec tests**

Assert exact UTF-8 bytes for empty and two-source manifests; input source order must not change bytes. Reject invalid UTF-8, oversized bytes before decode, invalid JSON, arrays at the root, unknown/missing keys at both layers, version/type mismatches, duplicate sources, relative persisted paths, and every JSON type mismatch.

```python
expected = (
    b'{"display_name":null,"format_version":"1",'
    b'"knowledge_base_id":"kb","sources":[]}\n'
)
assert encode_manifest(manifest, limits) == expected
assert decode_manifest(expected, limits) == manifest
```

- [ ] **Step 5: Implement strict codec and atomic write**

Implement private exact-key helpers, source-specific decode dispatch, `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"`, pre/post byte bounds, and controlled config errors. Implement a same-directory uniquely named temporary file opened in binary exclusive mode; write all bytes, flush, `os.fsync`, `os.replace`, best-effort directory fsync where supported, and controlled cleanup without exposing paths in public errors.

- [ ] **Step 6: Verify GREEN and commit**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base_manifest.py`

Expected: all manifest tests pass.

```bash
git add src/nexusmind/knowledge_base_manifest.py src/nexusmind/__init__.py tests/test_knowledge_base_manifest.py
git commit -m "feat: add strict knowledge base manifest"
```

### Task 2: Create/open lifecycle and default retrieval profile

**Files:**
- Create: `src/nexusmind/knowledge_base.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_knowledge_base.py`

- [ ] **Step 1: Write failing create/open tests**

Cover create in a new or existing empty directory, reject non-empty/file/symlink roots, initialize exact `manifest.json` and `knowledge.db`, reject missing/corrupt files on open, preserve ID/name across reopen, make close idempotent, and reject every other operation after close. Assert create with no sources causes no adapter/provider work.

```python
kb = KnowledgeBase.create(root, knowledge_base_id="security", display_name="Security")
assert kb.status() == KnowledgeBaseStatus(
    knowledge_base_id="security",
    display_name="Security",
    registered_source_count=0,
    canonical_source_count=0,
    document_count=0,
)
kb.close()
with pytest.raises(KnowledgeBaseClosedError):
    kb.status()
```

- [ ] **Step 2: Run RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base.py -k 'create or open or close'`

Expected: failures identify the missing `KnowledgeBase` orchestration module.

- [ ] **Step 3: Implement lifecycle and collection factory**

Validate root paths without following a symlink/reparse root, create the layout with controlled rollback of files created by the failed operation, and require both files on open before constructing the SQLite store. Store the immutable runtime `index_factory`; build default indexes with `UnicodeCJKLexicalAnalyzer`. On open, load and restore the snapshot, then validate every canonical source has a matching registration and matching `KnowledgeSourceType`. Add frozen `KnowledgeBaseStatus`, detached properties through status, `_require_open()`, and idempotent poison-aware close.

- [ ] **Step 4: Write and pass default/injected profile tests**

Sync fixture state through a directly prepared snapshot, reopen, and prove Unicode/CJK BM25 search works without network. Inject a recording cloneable index factory and prove it rebuilds derived state but never appears in manifest bytes.

- [ ] **Step 5: Verify GREEN and commit**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base.py tests/test_knowledge_base_manifest.py`

Expected: lifecycle/profile tests pass.

```bash
git add src/nexusmind/knowledge_base.py src/nexusmind/__init__.py tests/test_knowledge_base.py
git commit -m "feat: create and open knowledge bases"
```

### Task 3: Registration, inspection, and atomic synchronization

**Files:**
- Modify: `src/nexusmind/knowledge_base.py`
- Modify: `src/nexusmind/knowledge_base_manifest.py`
- Modify: `tests/test_knowledge_base.py`
- Create: `tests/test_knowledge_base_sync.py`

- [ ] **Step 1: Write failing registry tests**

Register LocalFile and LocalDirectory configs from relative paths and assert persisted/reopened absolute paths and sorted frozen tuples. Reject duplicate IDs and registry/manifest bounds. Monkeypatch adapter constructors and use a failing index/provider fixture to prove `add_source()` performs only one manifest replacement and no ingestion/retrieval work. Verify unsynchronized unregister, unknown unregister rejection, and synchronized unregister rejection.

- [ ] **Step 2: Implement registry and detached inspection**

For add/unregister, build a complete candidate manifest, encode/bound it, atomically replace disk, then swap in-memory manifest. Use canonical snapshot source IDs to guard unregister. Implement `list_sources`, deep-detached `list_documents`, `status`, and direct `search` delegation.

- [ ] **Step 3: Run registry GREEN**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base.py -k 'source or list or status or search'`

Expected: all registry/inspection tests pass.

- [ ] **Step 4: Write failing sync tests**

Create two real local sources registered in reverse order; recording adapters or index mutations must prove ascending source ID synchronization and result ordering. Test single-source sync, no-source no-op without SQLite save, reopen preserving documents/search, missing source error redaction, and batch failure after one successful staging source leaving live snapshot and stored snapshot exactly unchanged.

```python
before = kb.list_documents()
with pytest.raises(KnowledgeBaseSourceError):
    kb.sync()
assert kb.list_documents() == before
assert SQLiteKnowledgeSnapshotStore(root / "knowledge.db").load() == before_snapshot
```

- [ ] **Step 5: Implement staging sync**

Create a fresh collection from the stored chunker/index configuration, restore the live snapshot, construct only the selected registered adapters, and delegate to `KnowledgeCollection.sync()`. For batch sync use sorted registrations. On complete staging success, save the one complete snapshot through the store and only then swap `_collection`. Wrap adapter/runtime failures in stable source errors and store failures in persistence errors.

- [ ] **Step 6: Verify GREEN and commit**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base.py tests/test_knowledge_base_sync.py`

Expected: registration, inspection, one-source/batch sync, rollback, reopen, and search tests pass.

```bash
git add src/nexusmind/knowledge_base.py src/nexusmind/knowledge_base_manifest.py tests/test_knowledge_base.py tests/test_knowledge_base_sync.py
git commit -m "feat: synchronize registered knowledge sources"
```

### Task 4: Full removal and two-store recovery

**Files:**
- Modify: `src/nexusmind/knowledge_base.py`
- Create: `tests/test_knowledge_base_atomicity.py`

- [ ] **Step 1: Write failing successful-removal tests**

Register and sync two sources, remove one, and assert its registration, source, documents, and search hits disappear from memory and after reopen while the other source remains. Unknown removal must fail without writes.

- [ ] **Step 2: Write failing failure-injection tests**

Inject/store-spy failures for initial canonical save, manifest replacement after canonical commit, successful old-snapshot compensation, and failed compensation. Assert initial-save failure performs no manifest write; compensated failure leaves old memory/manifest/SQLite state; failed compensation poisons the object and all later methods raise `KnowledgeBaseClosedError`. Verify public messages omit injected private details and document content.

- [ ] **Step 3: Implement removal commit protocol**

Stage a fresh collection restored from the old snapshot and call its existing `remove_source`. Build/encode the candidate manifest before persistence. Save new canonical state, replace manifest, then swap memory. If manifest replacement fails, save the exact old snapshot; on compensation success raise persistence error with live state unchanged, and on failure mark `_poisoned = True` before raising the explicit recovery error.

- [ ] **Step 4: Verify GREEN and commit**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base_atomicity.py tests/test_knowledge_base_sync.py tests/test_knowledge_base.py`

Expected: all success, compensation, poison, and removal-state tests pass.

```bash
git add src/nexusmind/knowledge_base.py tests/test_knowledge_base_atomicity.py
git commit -m "feat: remove knowledge sources atomically"
```

### Task 5: Documentation and complete verification

**Files:**
- Modify: `README.md`
- Add: `docs/superpowers/specs/2026-08-20-knowledge-base-design.md`
- Add: `docs/superpowers/plans/2026-08-20-knowledge-base.md`

- [ ] **Step 1: Add the user-facing example and invariants**

Document create → add LocalDirectory source → sync → search → close → reopen. Explain why this is not Agent Workspace, registration versus canonical source, both removal operations, manifest/database layout, all-or-nothing sync, deterministic offline BM25 default, non-persisted factory injection, synchronous/manual sync, and CLI follow-up.

- [ ] **Step 2: Run focused product/runtime compatibility**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base_manifest.py tests/test_knowledge_base.py tests/test_knowledge_base_sync.py tests/test_knowledge_base_atomicity.py tests/test_knowledge_collection.py tests/test_knowledge_ingestion.py tests/test_knowledge_store.py tests/test_knowledge_retrieval.py tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py tests/test_reranking.py`

Expected: all selected tests pass.

- [ ] **Step 3: Run compile and repository hygiene checks**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m compileall -q src && git diff --check && git status --short`

Expected: exit zero, no whitespace errors, and only intentional #81 files.

- [ ] **Step 4: Run the complete supported suite where available**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q`

Expected: all tests pass. If the workspace only has the known unsupported Python 3.10 venv and the pre-existing checkpoint CLI test hangs, record that exact limitation and rely on the supported Windows 3.11–3.13 PR matrix without claiming a local full-suite pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-20-knowledge-base-design.md docs/superpowers/plans/2026-08-20-knowledge-base.md
git commit -m "docs: describe the KnowledgeBase API"
```

- [ ] **Step 6: Publish one Draft PR**

After explicit publication authorization, inspect status and commit scope, push `agent/issue-81-knowledge-base`, verify no matching PR exists, and create one Draft PR against `main` with `Closes #81`, exact verification evidence, persistence/recovery summary, and pending supported Windows CI checkboxes.
