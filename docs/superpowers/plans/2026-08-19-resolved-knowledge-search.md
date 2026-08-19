# Resolved Knowledge Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve collection search hits to detached canonical source and document provenance while preserving retrieval behavior and persistence compatibility.

**Architecture:** Keep `ChunkIndex.search()` unchanged as the retrieval-layer API. Make `KnowledgeCollection.search()` validate each backend hit, resolve it through committed collection-owned dictionaries, and return ordered `KnowledgeSearchResult` values containing deep-copied canonical provenance plus the immutable hit.

**Tech Stack:** Python 3.11+, frozen/slotted dataclasses, pytest, SQLite snapshot store.

---

### Task 1: Define and resolve Knowledge-layer search results

**Files:**
- Modify: `tests/test_knowledge_collection.py`
- Modify: `src/nexusmind/knowledge_collection.py`
- Modify: `src/nexusmind/__init__.py`

- [ ] **Step 1: Write failing public-contract and provenance tests**

Add imports for `KnowledgeSearchResult` and write tests that sync documents from two metadata-bearing sources, call `collection.search()`, and assert the exact ordered tuple shape:

```python
assert isinstance(results[0], KnowledgeSearchResult)
assert results[0].source.source_id == "one"
assert results[0].document.logical_path == "a.txt"
assert results[0].hit.chunk.document_id == results[0].document.document_id
assert results[0].hit.score == 2
assert results[0].hit.matched_terms == ("checkpoint", "resume")
```

Also assert empty search returns `()` and backend ranking order is unchanged.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_collection.py`

Expected: FAIL because `KnowledgeSearchResult` is not exported and collection search still returns `SearchHit`.

- [ ] **Step 3: Add the minimal result contract and resolution path**

In `knowledge_collection.py`, add:

```python
class KnowledgeSearchResolutionError(KnowledgeCollectionError):
    """A retrieval hit cannot be resolved to committed canonical state."""


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    source: KnowledgeSource
    document: Document
    hit: SearchHit
```

Replace collection search with ordered resolution:

```python
def search(self, query: str, *, limit: int = 10) -> tuple[KnowledgeSearchResult, ...]:
    hits = self._index.search(query, limit=limit)
    results: list[KnowledgeSearchResult] = []
    for hit in hits:
        if not isinstance(hit, SearchHit):
            raise KnowledgeSearchResolutionError("index returned a malformed search hit")
        document = next(
            (documents[hit.chunk.document_id] for documents in self._documents.values()
             if hit.chunk.document_id in documents),
            None,
        )
        if document is None:
            raise KnowledgeSearchResolutionError("search hit references an unknown document_id")
        source = self._sources.get(document.source_id)
        if source is None or hit.chunk.document_id != document.document_id:
            raise KnowledgeSearchResolutionError("search hit has incoherent canonical provenance")
        results.append(KnowledgeSearchResult(deepcopy(source), deepcopy(document), hit))
    return tuple(results)
```

Export both contracts from `knowledge_collection.py` and package `__init__.py`. Update existing collection-level assertions from `.chunk`/`.score` to `.hit.chunk`/`.hit.score`.

- [ ] **Step 4: Run collection tests to verify GREEN**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_collection.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the resolved search contract**

```bash
git add src/nexusmind/knowledge_collection.py src/nexusmind/__init__.py tests/test_knowledge_collection.py
git commit -m "feat: resolve knowledge search provenance"
```

### Task 2: Enforce isolation and fail-closed integrity

**Files:**
- Modify: `tests/test_knowledge_collection.py`
- Modify: `src/nexusmind/knowledge_collection.py`

- [ ] **Step 1: Write failing isolation and hostile-index tests**

Add a cloneable fake index that returns configured values. Cover a non-tuple backend return, malformed non-`SearchHit` tuple members, a non-`Chunk` hit payload, and an unknown document ID. Add a metadata test that mutates returned `source.metadata` and `document.metadata`, searches again, and asserts the second result retains original metadata.

```python
with pytest.raises(KnowledgeSearchResolutionError, match="unknown document_id"):
    collection.search("ghost")

first.source.metadata["owner"] = "caller"
first.document.metadata["tag"] = "caller"
second = collection.search("term")[0]
assert second.source.metadata == {"owner": "canonical"}
assert second.document.metadata == {"tag": "canonical"}
```

- [ ] **Step 2: Run focused tests to verify RED**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_collection.py -k 'search and (isolation or malformed or ghost or incoherent)'`

Expected: integrity tests fail because the initial path does not validate the backend tuple and chunk type.

- [ ] **Step 3: Validate the complete hit-to-document relationship**

Require the backend return value to be exactly a tuple. Require each member to be a `SearchHit`, require `hit.chunk` to be a `Chunk`, and require its `document_id` to resolve to one document in `_documents` whose `source_id` resolves in `_sources`. Raise `KnowledgeSearchResolutionError` with stable messages for each failure. Keep resolution read-only and build the full output before returning, so failures expose no partial tuple.

- [ ] **Step 4: Run focused and full collection tests**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_collection.py`

Expected: all tests pass.

- [ ] **Step 5: Commit integrity coverage**

```bash
git add src/nexusmind/knowledge_collection.py tests/test_knowledge_collection.py
git commit -m "test: enforce resolved search integrity"
```

### Task 3: Cover restore and SQLite restart provenance

**Files:**
- Modify: `tests/test_knowledge_snapshot.py`
- Modify: `tests/test_knowledge_store.py`

- [ ] **Step 1: Update restore tests and add provenance assertions**

Change existing collection-search accesses to `result.hit`. Extend round-trip tests to assert restored results expose the matching canonical source and document, including logical path and metadata.

```python
result = restored.search("first")[0]
assert result.source.source_id == "one"
assert result.document == snapshot.documents[0]
assert result.hit.chunk.document_id == result.document.document_id
```

- [ ] **Step 2: Run snapshot tests**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_snapshot.py`

Expected: all tests pass after the deliberate API assertion updates.

- [ ] **Step 3: Extend SQLite restart test**

Assert the restarted collection returns the loaded canonical source and document provenance and preserves score/matched terms:

```python
result = restarted.search("checkpoint resume")[0]
assert result.source.source_id == "docs"
assert result.document.logical_path == "notes.md"
assert result.hit.score == 2
assert result.hit.matched_terms == ("checkpoint", "resume")
```

- [ ] **Step 4: Run persistence tests**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_store.py tests/test_knowledge_snapshot.py`

Expected: all tests pass with no schema changes.

- [ ] **Step 5: Commit persistence compatibility tests**

```bash
git add tests/test_knowledge_snapshot.py tests/test_knowledge_store.py
git commit -m "test: cover resolved search after restore"
```

### Task 4: Document layering and verify the repository

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Knowledge Runtime documentation**

Document this exact boundary and non-goals:

```text
KnowledgeSource -> Document -> Chunk -> ChunkIndex -> SearchHit
                                                   -> KnowledgeCollection resolution
                                                   -> KnowledgeSearchResult
```

Explain that collection results are detached canonical provenance, while ranking, citation formatting, semantic retrieval, persistent indexes, and RAG remain out of scope.

- [ ] **Step 2: Run full verification**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q`

Expected: complete suite passes with zero failures.

Run: `git diff --check`

Expected: exit 0 with no output.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain resolved knowledge search"
```

- [ ] **Step 4: Review the complete branch diff**

Run: `git diff --stat origin/main...HEAD && git diff --check origin/main...HEAD`

Expected: only issue #65 design, plan, source, tests, and README changes; no whitespace errors.
