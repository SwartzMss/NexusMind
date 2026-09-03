# Structure-Aware Chunk Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete issue #126 by attaching heading hierarchy to derived chunks, using it in every retrieval backend, and measuring deterministic before/after retrieval quality.

**Architecture:** Keep `Chunk.content` as the exact canonical document slice and add immutable structural metadata plus a derived `retrieval_text` property. Extend the existing `StructureAwareChunker` metadata pass without changing canonical persistence, then route only indexing/reranking inputs through `retrieval_text`; context assembly and provenance continue to use `content` and offsets.

**Tech Stack:** Python 3.11–3.13, dataclasses, pytest, in-memory BM25/semantic/hybrid/reranked indexes, checked-in JSON/Markdown evaluation fixtures.

---

## File map

- Modify `src/nexusmind/knowledge_chunking.py`: add validated `Chunk` metadata, `retrieval_text`, heading parsing state, and metadata-aware structure chunk construction.
- Modify `src/nexusmind/knowledge_retrieval.py`: analyze `chunk.retrieval_text` in BM25.
- Modify `src/nexusmind/semantic_retrieval.py`: embed `chunk.retrieval_text` in add/replace paths.
- Modify `src/nexusmind/reranking.py`: use retrieval text for candidate scoring/bounds while preserving canonical chunks in returned hits.
- Modify `src/nexusmind/retrieval_evaluation.py`: add precision-at-K to case, category, and overall reports with backward-compatible defaults for manually constructed results.
- Modify `src/nexusmind/retrieval_benchmark.py`: render precision and run the before/after structure benchmark alongside the existing backend comparison.
- Modify `tests/test_knowledge_chunking.py`: add tests for metadata, hierarchy, source locations, validation, and retrieval text.
- Modify `tests/test_knowledge_retrieval.py`, `tests/test_semantic_retrieval.py`, `tests/test_reranking.py`: verify heading terms are indexed without changing exact content/offsets.
- Modify `tests/test_retrieval_evaluation.py`, `tests/test_retrieval_evaluation_categories.py`: add precision calculations and constructor compatibility tests.
- Modify `tests/test_structure_chunking_benchmark.py`, `evals/knowledge/chunking/corpus/`, `evals/knowledge/chunking/cases.json`, and `evals/knowledge/chunking.md`: add nested technical cases and deterministic before/after evidence.
- Modify `docs/architecture.md` and `README.md`: document metadata and retrieval-text behavior.

## Task 1: Establish a usable Python baseline

**Files:** none.

- [ ] **Step 1: Install project dependencies into the existing virtual environment.**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/pip install -r requirements.txt
```

Expected: `firecrawl-anydoc` and the declared runtime dependencies install successfully.

- [ ] **Step 2: Run the clean baseline suite with the virtual-environment interpreter.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q
```

Expected: collection succeeds and the existing suite passes. If dependency installation is unavailable, record the exact environment blocker and continue only with tests that do not import `anydoc`.

## Task 2: Add the backward-compatible chunk metadata contract

**Files:**
- Modify: `src/nexusmind/knowledge_chunking.py`
- Test: `tests/test_knowledge_chunking.py`

- [ ] **Step 1: Write failing contract tests.**

Add tests with these behaviors:

```python
def test_chunk_defaults_keep_legacy_constructor_and_expose_empty_structure() -> None:
    chunk = Chunk("doc", "chunk", "body", 0, 4)

    assert chunk.heading_path == ()
    assert chunk.section_title == ""
    assert chunk.source_location == ""
    assert chunk.retrieval_text == "body"


def test_structural_chunk_retrieval_text_contains_heading_context() -> None:
    chunk = Chunk(
        "doc", "chunk", "body", 0, 4,
        heading_path=("Android Security", "Binder"),
        section_title="Binder",
        source_location="notes.md:L3",
    )

    assert chunk.retrieval_text == "Android Security > Binder\nbody"


@pytest.mark.parametrize("heading_path", [("",), ["Binder"], (1,)])
def test_chunk_rejects_malformed_heading_path(heading_path: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Chunk("doc", "chunk", "body", 0, 4, heading_path=heading_path)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run only the new tests to verify the expected RED failure.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_chunking.py -k "defaults_keep_legacy or retrieval_text or malformed_heading_path"
```

Expected: failures report missing `heading_path`, `section_title`, or `retrieval_text`, not import or syntax errors.

- [ ] **Step 3: Implement the minimal validated fields and property.**

Extend the frozen, keyword-only tail of `Chunk` with:

```python
heading_path: tuple[str, ...] = ()
section_title: str = ""
source_location: str = ""
```

In `__post_init__`, require an exact tuple, non-empty string members, string title/location, and either an empty title or a title equal to the final path item. Add:

```python
@property
def retrieval_text(self) -> str:
    prefix = " > ".join(self.heading_path)
    return f"{prefix}\n{self.content}" if prefix else self.content
```

Keep the first five constructor parameters unchanged and do not alter canonical content validation.

- [ ] **Step 4: Run the focused tests and the existing chunking tests.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_chunking.py
```

Expected: all existing and new chunk contract tests pass.

- [ ] **Step 5: Commit the contract.**

```bash
git add src/nexusmind/knowledge_chunking.py tests/test_knowledge_chunking.py
git commit -m "feat: add structure metadata to chunks"
```

## Task 3: Propagate heading hierarchy through `StructureAwareChunker`

**Files:**
- Modify: `src/nexusmind/knowledge_chunking.py`
- Test: `tests/test_knowledge_chunking.py`

- [ ] **Step 1: Write failing hierarchy tests.**

Add a document containing `# Android Security`, `## Binder`, `### oneway`, a fenced code block containing `# not a heading`, and preamble text. Assert that emitted chunks covering each section have paths `("Android Security",)`, `("Android Security", "Binder")`, and `("Android Security", "Binder", "oneway")`; assert `section_title` is the final item and `source_location` is the logical path plus the 1-based heading line. Also assert all chunks remain exact slices and `len(content) <= chunk_size`.

- [ ] **Step 2: Run the hierarchy tests to verify RED.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_chunking.py -k "hierarchy or heading_path or source_location"
```

Expected: structure chunks have the current empty metadata, so the assertions fail.

- [ ] **Step 3: Implement metadata derivation over the existing structural spans.**

Add a heading descriptor containing start offset, heading level, title, and line number. Parse the same ATX heading lines recognized by `_markdown_blocks`; strip the marker and optional closing sequence, trim whitespace, and never parse inside fenced blocks. Walk the already packed spans in source order with a six-entry heading stack. Before creating each output `Chunk`, update the stack for headings whose start is at or before the span start, then attach the active path, final title, and `f"{document.logical_path}:L{line_number}"` (or `""` before the first heading). Use `algorithm="structure-v2"` in the stable ID input so the metadata contract has a distinct deterministic identity.

- [ ] **Step 4: Run all chunking and structure-boundary tests.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_chunking.py tests/test_structure_chunking_benchmark.py
```

Expected: hierarchy metadata is correct, existing block/limit tests remain green, and repeated chunking remains byte-for-byte deterministic.

- [ ] **Step 5: Commit the chunker propagation.**

```bash
git add src/nexusmind/knowledge_chunking.py tests/test_knowledge_chunking.py
git commit -m "feat: preserve markdown heading paths in chunks"
```

## Task 4: Index structural retrieval text in every backend

**Files:**
- Modify: `src/nexusmind/knowledge_retrieval.py`
- Modify: `src/nexusmind/semantic_retrieval.py`
- Modify: `src/nexusmind/reranking.py`
- Test: `tests/test_knowledge_retrieval.py`
- Test: `tests/test_semantic_retrieval.py`
- Test: `tests/test_reranking.py`

- [ ] **Step 1: Write failing retrieval tests.**

Add a lexical test with two chunks whose exact bodies omit `Binder` but whose heading paths differ; query `Binder` and assert only the structural chunk is returned. Add a semantic provider test that records document texts and asserts it receives `chunk.retrieval_text`, while the resulting `SearchHit.chunk.content` and offsets remain unchanged. Add a reranker test that checks its candidate receives the same canonical structural chunk and that returned hits retain its exact content.

- [ ] **Step 2: Run the new tests to verify RED.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py tests/test_semantic_retrieval.py tests/test_reranking.py -k "heading or structural or retrieval_text"
```

Expected: lexical search misses the heading-only term and the recording provider sees body-only text.

- [ ] **Step 3: Route backend indexing through `retrieval_text`.**

Change only the indexing/scoring inputs:

```python
tokens = self._analyze(chunk.retrieval_text)
```

and:

```python
tuple(chunk.retrieval_text for chunk in additions.values())
```

for semantic document embeddings. Keep `len(chunk.content)` for persisted/index resource limits that describe canonical payload size, and keep all hit/diagnostic chunk objects unchanged. Reranker adapters that score `hit.chunk.content` should score `hit.chunk.retrieval_text`; the reranker contract still returns the original `SearchHit` chunk.

- [ ] **Step 4: Run backend regression suites.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py tests/test_reranking.py
```

Expected: all backend tests pass, including structural term retrieval and diagnostic lineage checks.

- [ ] **Step 5: Commit backend propagation.**

```bash
git add src/nexusmind/knowledge_retrieval.py src/nexusmind/semantic_retrieval.py src/nexusmind/reranking.py tests/test_knowledge_retrieval.py tests/test_semantic_retrieval.py tests/test_reranking.py
git commit -m "feat: retrieve chunks with heading context"
```

## Task 5: Add precision-at-K to the evaluation contract

**Files:**
- Modify: `src/nexusmind/retrieval_evaluation.py`
- Test: `tests/test_retrieval_evaluation.py`
- Test: `tests/test_retrieval_evaluation_categories.py`

- [ ] **Step 1: Write failing precision tests.**

For one case with two returned documents and one relevant target, assert case `precision_at_k == 0.5`; for a two-case report assert overall precision is the mean of case precision values. Assert category reports carry the same aggregation. Keep the manually constructed `_result` helper in `tests/test_retrieval_evaluation_categories.py` valid by giving the new field a default.

- [ ] **Step 2: Run the tests to verify RED.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_categories.py -k "precision"
```

Expected: the new attributes are absent or the assertions fail.

- [ ] **Step 3: Implement bounded precision calculation.**

Add `precision_at_k: float = 0.0` after existing metric fields in the case, category, and overall report dataclasses. In `_report_from_rankings`, calculate `len(set(returned_targets) & relevant) / len(returned_targets)` with zero when no result is returned. Aggregate category and overall precision with the same arithmetic mean used for Hit@K/Recall@K/MRR. Do not change relevance target semantics: precision is document-target precision, while `returned_chunk_ids` remains chunk-level diagnostics.

- [ ] **Step 4: Run the full evaluation suite.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation*.py
```

Expected: all existing metrics and new precision assertions pass.

- [ ] **Step 5: Commit evaluation metrics.**

```bash
git add src/nexusmind/retrieval_evaluation.py tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_categories.py
git commit -m "feat: report retrieval precision at k"
```

## Task 6: Extend and document the structure benchmark

**Files:**
- Modify: `src/nexusmind/retrieval_benchmark.py`
- Modify: `tests/test_structure_chunking_benchmark.py`
- Modify: `evals/knowledge/chunking/cases.json`
- Create/Modify: `evals/knowledge/chunking/corpus/technical.md`
- Modify: `evals/knowledge/chunking.md`
- Modify: `docs/architecture.md`
- Modify: `README.md`

- [ ] **Step 1: Add a failing benchmark assertion.**

Add a nested technical case whose fixed-window result ranks below top 1 while the structure-aware result gets the heading term from `retrieval_text`. Assert candidate top-1 precision and MRR exceed baseline for that case, candidate Recall@3 is not lower, and reranked evaluation is deterministic.

- [ ] **Step 2: Run the benchmark test to verify RED.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_structure_chunking_benchmark.py
```

Expected: the new heading-context case fails before backend propagation or fixture updates.

- [ ] **Step 3: Add the deterministic fixture and render precision.**

Write the technical Markdown corpus with nested headings and a distractor document. Add exact relevance targets to `cases.json`. Update benchmark rendering tables to include `Precision@K`, and include the structure-aware backend/candidate comparison in the checked-in report. Use the existing deterministic embedding and reranker fixtures; do not add network calls or a real model.

- [ ] **Step 4: Update architecture and user-facing documentation.**

Document that the default `StructureAwareChunker` emits heading metadata, that `retrieval_text` is indexing-only context, and that displayed/cited content remains exact canonical text. Document the reproduction command and clarify that benchmark metrics are descriptive/non-gate.

- [ ] **Step 5: Run the benchmark and verify stable output.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_structure_chunking_benchmark.py tests/test_retrieval_benchmark*.py
```

Expected: the targeted structure case improves, Recall@3 does not regress, precision is rendered, and repeated runs produce identical reports/chunk IDs.

- [ ] **Step 6: Commit benchmark and documentation.**

```bash
git add src/nexusmind/retrieval_benchmark.py tests/test_structure_chunking_benchmark.py evals/knowledge/chunking docs/architecture.md README.md
git commit -m "docs: document structure-aware retrieval evaluation"
```

## Task 7: Full verification and PR handoff

**Files:** none beyond the committed implementation.

- [ ] **Step 1: Run the complete test suite with fresh dependencies.**

Run:

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q
```

Expected: exit code 0 with zero failures.

- [ ] **Step 2: Run repository quality checks.**

Run:

```bash
git diff --check origin/main...HEAD
```

Expected: no whitespace errors. Re-read the issue acceptance checklist and verify each requirement against tests or documentation.

- [ ] **Step 3: Inspect the final diff and branch state.**

Run:

```bash
git status --short --branch
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

Expected: only issue #126 implementation, tests, fixtures, docs, and the committed design/plan are present; no unrelated user changes are included.

- [ ] **Step 4: Push and create the PR.**

```bash
git push -u origin codex/issue-126-structure-aware-chunking
gh pr create --base main --head codex/issue-126-structure-aware-chunking --title "feat: add structure-aware chunk metadata" --body "$(cat <<'EOF'
## Summary
- preserve Markdown and AnyDoc heading hierarchy in chunk metadata
- index heading context without changing canonical chunk content or offsets
- add precision-aware before/after retrieval evaluation for issue #126

Closes #126

## Test plan
- [x] `PYTHONPATH=src .venv/bin/python -m pytest -q`
- [x] structure-aware benchmark and retrieval backend regression tests
- [x] `git diff --check`
EOF
)"
```

Expected: GitHub returns a new PR URL targeting `main`, and the worktree is preserved for any review iteration.
