# First-Release Schema Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove prerelease compatibility paths and publish one strict version-1 contract for manifests, source identities, SQLite snapshots, and the Python/CLI API.

**Architecture:** Source identity is derived only from source type plus the platform-correct normalized path. Persistence formats keep version fields but accept only their complete first-release v1 schemas. Snapshots and public APIs expose only current contracts; no migration, synthesis, tombstone, alias, or repair behavior remains.

**Tech Stack:** Python 3.11+, frozen dataclasses, pathlib, SQLite, argparse, pytest.

---

### Task 1: Make source identity auto-only and collapse the manifest to v1

**Files:**
- Modify: `src/nexusmind/knowledge_base_manifest.py`
- Modify: `src/nexusmind/knowledge_base.py`
- Test: `tests/test_knowledge_base_manifest.py`
- Test: `tests/test_knowledge_base_sync.py`
- Test: all tests constructing `LocalFileSourceConfig` or `LocalDirectorySourceConfig`

- [ ] **Step 1: Write failing first-release source and manifest contract tests**

Require source constructors to reject `source_id`, require encoded manifests to contain only `format_version`, `knowledge_base_id`, `display_name`, and `sources`, and require decoding to reject format v2 and mismatched persisted IDs.

```python
with pytest.raises(TypeError):
    LocalFileSourceConfig(source_id="manual", path="notes.md")

assert set(json.loads(encode_manifest(manifest(), limits))) == {
    "format_version", "knowledge_base_id", "display_name", "sources"
}
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_base_manifest.py tests/test_knowledge_base_sync.py`

Expected: failures from accepted explicit IDs, format v2, and tombstone behavior.

- [ ] **Step 3: Implement the single v1 manifest and auto-only configs**

Make `source_id` an `init=False` field computed in `__post_init__`, remove `_source_id_was_auto`, remove `retired_sources`, and validate persisted IDs by reconstructing the config from `path`.

```python
source = source_class(path=path)
if source.source_id != persisted_source_id:
    raise KnowledgeBaseConfigError("persisted source_id does not match source identity")
return source
```

Delete retired matching and `_requires_retired_identity` from `KnowledgeBase`; add/remove/unregister operate only on active sources. Keep `_path_identity`, canonical `add_source()` return values, duplicate-path rejection, and retained document history.

- [ ] **Step 4: Convert tests from explicit IDs to derived IDs**

Construct configs from paths, capture `config.source_id` or the return from `add_source()`, and use those IDs in assertions. Delete tests whose only purpose is legacy explicit-ID/tombstone behavior.

- [ ] **Step 5: Run focused source and manifest suites**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_base_manifest.py tests/test_knowledge_base_sync.py tests/test_knowledge_base.py tests/test_knowledge_base_atomicity.py tests/test_knowledge_base_ui.py`

Expected: PASS.

### Task 2: Publish the complete SQLite schema as v1

**Files:**
- Modify: `src/nexusmind/knowledge_store.py`
- Test: `tests/test_knowledge_store.py`
- Test: `tests/test_knowledge_base.py`

- [ ] **Step 1: Replace the migration test with a rejection test**

Build a history-free prerelease database marked schema `"0"` or with a missing `document_versions` table and assert `SQLiteKnowledgeSnapshotStore(path)` raises `KnowledgeSnapshotStoreError` without modifying the database.

- [ ] **Step 2: Run the new test and verify RED**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_store.py -k 'schema'`

Expected: current migration behavior or schema version mismatch fails the new contract.

- [ ] **Step 3: Remove migration logic**

Set `_SCHEMA_VERSION = "1"`, describe the store as v1, delete the row-`"1"` migration branch and `_validate_schema_v1`, and create the full four-table/index schema only for an empty database.

- [ ] **Step 4: Verify SQLite persistence**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_store.py tests/test_knowledge_base.py tests/test_knowledge_base_atomicity.py`

Expected: PASS.

### Task 3: Reject history-free non-empty snapshots

**Files:**
- Modify: `src/nexusmind/knowledge_collection.py`
- Test: `tests/test_knowledge_versioning.py`
- Test: `tests/test_knowledge_collection.py`

- [ ] **Step 1: Change the legacy synthesis test to require rejection**

```python
with pytest.raises(KnowledgeSnapshotError):
    restored.restore(KnowledgeSnapshot(sources=(source,), documents=(document,)))
```

Also retain a test proving `KnowledgeSnapshot((), ())` restores successfully.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_versioning.py -k 'snapshot'`

Expected: the current implementation synthesizes a root version instead of raising.

- [ ] **Step 3: Remove root-version synthesis**

In `_validate_restore_versions`, return `{}` only when both documents and versions are empty; otherwise reject documents without versions using `KnowledgeSnapshotError`.

- [ ] **Step 4: Verify collection and versioning suites**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_versioning.py tests/test_knowledge_collection.py tests/test_knowledge_snapshot.py`

Expected: PASS.

### Task 4: Remove legacy CLI repair and the answer alias

**Files:**
- Modify: `src/nexusmind/cli.py`
- Modify: `src/nexusmind/knowledge_base.py`
- Modify: `src/nexusmind/knowledge_base_ui.py` only if protocol annotations require it
- Modify: `tests/test_knowledge_cli.py`
- Modify: `tests/test_knowledge_answer.py`
- Modify: `tests/test_knowledge_query.py`

- [ ] **Step 1: Write failing current-contract tests**

Assert `source remove --id` is rejected by argparse, path removal remains supported, and `KnowledgeBase` has no `answer` attribute. Convert answer behavior tests to call `query(...).answer`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_cli.py tests/test_knowledge_answer.py tests/test_knowledge_query.py`

Expected: failures from `--id` acceptance and the existing `answer()` method.

- [ ] **Step 3: Remove compatibility entrypoints**

Delete `_AmbiguousSourcePathError`, the `--id` selector, its exception branch, and ambiguous legacy repair logic. Make `source remove` require one path. Delete `KnowledgeBase.answer()` and unused imports introduced solely for that method.

- [ ] **Step 4: Verify CLI and answer/query suites**

Run: `../../.venv/bin/python -m pytest -q tests/test_knowledge_cli.py tests/test_knowledge_answer.py tests/test_knowledge_query.py tests/test_knowledge_query_cli.py`

Expected: PASS.

### Task 5: Align public documentation and verify the release baseline

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: current user-facing docs containing migration or legacy claims

- [ ] **Step 1: Remove prerelease compatibility claims**

Document stable automatic source identity, path-only CLI operations, strict manifest/SQLite v1 schemas, required document history, and `query()` as the sole KnowledgeBase answer API. Do not edit archived design records except the new design and plan.

- [ ] **Step 2: Scan for stale current-code compatibility language**

Run: `rg -n "retired_sources|legacy duplicate|旧版 KnowledgeBase|Backward-compatible answer|schema v1 database migrates" README.md docs/architecture.md src tests`

Expected: no current-contract references; unrelated lexical analyzer terminology may remain.

- [ ] **Step 3: Run full verification**

Run: `../../.venv/bin/python -m pytest -q`

Run: `git diff --check`

Expected: full suite passes with only intentional platform skips; diff check exits 0.

- [ ] **Step 4: Review final diff against the design**

Confirm one manifest schema, one SQLite schema, no tombstones, no explicit source IDs, no snapshot synthesis, no CLI repair flag, no `answer()` alias, and no unrelated changes.
