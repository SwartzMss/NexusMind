# Document Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-neutral `Chunk` contract and a deterministic, bounded character-based `TextChunker` for `Document` objects.

**Architecture:** Keep `knowledge.py` focused on source and document contracts, and add `knowledge_chunking.py` for derived chunk contracts, deterministic IDs, validation, and chunking. Re-export the public API from the package root and document `Document -> Chunk` as a separate layer while leaving Index and Retrieval as future work.

**Tech Stack:** Python 3.11+, frozen/slotted dataclasses, SHA-256 and canonical JSON from the standard library, pytest.

---

### Task 1: Define chunk boundaries and deterministic identity

**Files:**
- Create: `tests/test_knowledge_chunking.py`
- Create: `src/nexusmind/knowledge_chunking.py`

- [ ] **Step 1: Write failing behavioral tests**

Add tests that construct `Document` values and assert short, exact-boundary,
multi-chunk, overlap, empty, Unicode, slice-preservation, repeated-run, and
changed-content behavior. Use these concrete expectations:

```python
def test_multi_chunk_document_preserves_overlap_and_source_slices() -> None:
    document = Document(source_id="docs", logical_path="a.txt", content="abcdefghij")
    chunks = TextChunker(chunk_size=4, overlap=1).chunk(document)
    assert [(chunk.start_offset, chunk.end_offset, chunk.content) for chunk in chunks] == [
        (0, 4, "abcd"), (3, 7, "defg"), (6, 10, "ghij")
    ]
    assert all(chunk.content == document.content[chunk.start_offset:chunk.end_offset] for chunk in chunks)

def test_changed_document_content_does_not_reuse_chunk_identity() -> None:
    first = Document(source_id="docs", logical_path="a.txt", content="abcd")
    changed = Document(source_id="docs", logical_path="a.txt", content="abce")
    chunker = TextChunker(chunk_size=4, overlap=0)
    assert chunker.chunk(first)[0].chunk_id != chunker.chunk(changed)[0].chunk_id
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_knowledge_chunking.py`

Expected: collection fails because `nexusmind.knowledge_chunking` does not exist.

- [ ] **Step 3: Implement the minimal chunk contract and algorithm**

Implement:

```python
class ChunkLimitError(Exception):
    """The requested operation would exceed the configured chunk limit."""

@dataclass(frozen=True, slots=True)
class Chunk:
    document_id: str
    chunk_id: str
    content: str
    start_offset: int
    end_offset: int

@dataclass(frozen=True, slots=True, kw_only=True)
class TextChunker:
    chunk_size: int = 1000
    overlap: int = 100
    max_chunks: int = 10000

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        ...
```

Validate exact integer types, compute `step = chunk_size - overlap`, calculate
the required count before allocation, iterate half-open character slices, and
derive each `chunk_id` from canonical JSON containing `document_id`,
`content_hash`, offsets, `chunk_size`, and `overlap`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_knowledge_chunking.py`

Expected: all chunk boundary and identity tests pass.

- [ ] **Step 5: Commit the contract and core algorithm**

```bash
git add src/nexusmind/knowledge_chunking.py tests/test_knowledge_chunking.py
git commit -m "feat: add deterministic document chunking"
```

### Task 2: Enforce configuration and resource bounds

**Files:**
- Modify: `tests/test_knowledge_chunking.py`
- Modify: `src/nexusmind/knowledge_chunking.py`

- [ ] **Step 1: Write failing validation and limit tests**

Add parametrized tests for `chunk_size` values `0`, `-1`, `True`, and `1.5`;
`overlap` values `-1`, `chunk_size`, `True`, and `1.5`; `max_chunks` values `0`,
`-1`, `True`, and `1.5`; a document that needs more chunks than allowed; and a
non-`Document` argument. Assert `ValueError` for invalid integer ranges,
`TypeError` for wrong types, `ChunkLimitError` for the resource bound, and
`TypeError` for the input contract.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `pytest -q tests/test_knowledge_chunking.py`

Expected: the newly added validation or limit cases fail until all controlled
error behavior is implemented.

- [ ] **Step 3: Complete validation and preflight limit enforcement**

Use exact checks such as:

```python
if type(self.chunk_size) is not int:
    raise TypeError("chunk_size must be an integer")
if self.chunk_size <= 0:
    raise ValueError("chunk_size must be greater than zero")
```

Apply the equivalent rules to `overlap` and `max_chunks`. For non-empty text,
compute required chunks as `1 + max(0, len(content) - chunk_size + step - 1) // step`
and raise `ChunkLimitError` before building any `Chunk` values.

- [ ] **Step 4: Run focused and Knowledge regression tests**

Run: `pytest -q tests/test_knowledge_chunking.py tests/test_knowledge.py tests/test_knowledge_ingestion.py`

Expected: all tests pass.

- [ ] **Step 5: Commit bounded error behavior**

```bash
git add src/nexusmind/knowledge_chunking.py tests/test_knowledge_chunking.py
git commit -m "test: cover document chunking limits"
```

### Task 3: Publish and document the API

**Files:**
- Modify: `src/nexusmind/__init__.py`
- Modify: `tests/test_knowledge_chunking.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing public-export test**

Import `Chunk`, `ChunkLimitError`, and `TextChunker` from `nexusmind` in the test
module and assert that the package-level types perform one short-document
chunking operation.

- [ ] **Step 2: Run the export test and verify RED**

Run: `pytest -q tests/test_knowledge_chunking.py`

Expected: import fails because the new public names are not exported yet.

- [ ] **Step 3: Export names and update README**

Add this package import and corresponding `__all__` entries:

```python
from .knowledge_chunking import Chunk, ChunkLimitError, TextChunker
```

Update the Knowledge Runtime prose and diagram to show ingestion producing
`Document`, chunking producing `Chunk`, character offsets and bounded settings,
with Index and Retrieval explicitly marked future.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_knowledge_chunking.py`

Expected: all focused tests pass.

- [ ] **Step 5: Commit API and documentation**

```bash
git add src/nexusmind/__init__.py tests/test_knowledge_chunking.py README.md
git commit -m "docs: publish document chunking API"
```

### Task 4: Verify the complete milestone

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run repository hygiene checks**

Run: `git diff --check origin/main...HEAD && git status --short`

Expected: no whitespace errors; status contains no uncommitted implementation
files.

- [ ] **Step 3: Review acceptance criteria against the diff**

Run: `git diff --stat origin/main...HEAD && git diff origin/main...HEAD -- src/nexusmind/knowledge_chunking.py src/nexusmind/__init__.py README.md tests/test_knowledge_chunking.py`

Confirm every Issue #54 acceptance criterion is represented by code, tests, or
documentation and no Index/Retrieval scope was added.
