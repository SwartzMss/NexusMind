# Retrieval Evaluation Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic document-relevance retrieval metrics and a checked-in offline baseline through the real Knowledge Runtime path.

**Architecture:** A focused `retrieval_evaluation` module owns immutable contracts, strict JSON loading, label-coherence validation, and Hit@K/Recall@K/MRR computation over `KnowledgeCollection.search()`. A small authored corpus and labels drive an end-to-end pytest baseline with fixed chunker and K configuration.

**Tech Stack:** Python 3.11+ dataclasses, standard-library JSON/path handling, pytest, LocalDirectoryAdapter, KnowledgeCollection, TextChunker, BM25 InMemoryChunkIndex.

---

### Task 1: Add evaluation contracts and metrics

**Files:**
- Create: `src/nexusmind/retrieval_evaluation.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_retrieval_evaluation.py`

- [ ] **Step 1: Write failing contract and metric tests**

Test non-empty target/case fields, exact non-empty unique target tuples, duplicate case IDs, empty case sets, and positive plain-int K. Build a real `KnowledgeCollection` with a small fake adapter and custom chunker that emits ranked chunks, then assert:

```python
assert case_result.returned_targets == (target_a, target_a, target_b)
assert case_result.relevant_targets_found == (target_a, target_b)
assert case_result.relevant_targets_missed == ()
assert case_result.first_relevant_rank == 1
assert case_result.hit_at_k == 1.0
assert case_result.recall_at_k == 1.0
assert case_result.reciprocal_rank == 1.0
```

Add hit-at-K miss, later reciprocal rank, multi-target partial recall, duplicate relevant-document chunks, aggregate `pytest.approx()` means, repeated-run determinism, unknown-label rejection before search, and snapshot equality before/after evaluation.

- [ ] **Step 2: Run tests to verify RED**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py`

Expected: collection error because the new module/contracts do not exist.

- [ ] **Step 3: Implement immutable contracts and evaluator**

Define controlled `RetrievalEvaluationError`, validated frozen/slotted target and case dataclasses, frozen/slotted case-result/report dataclasses, and:

```python
def evaluate_retrieval(
    collection: KnowledgeCollection,
    cases: tuple[RetrievalEvaluationCase, ...],
    *,
    k: int = 5,
) -> RetrievalEvaluationReport:
```

Validate all labels against `{(document.source_id, document.logical_path) for document in collection.snapshot().documents}` before searching. For each result, derive `RetrievalTarget(result.source.source_id, result.document.logical_path)`, preserve result/chunk order, deduplicate found relevant targets in first-hit order, derive missed targets in label order, and compute exact case metrics. Aggregate with arithmetic means. Export all public contracts from the module and package root.

- [ ] **Step 4: Run evaluator tests to verify GREEN**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py`

Expected: all evaluator tests pass.

- [ ] **Step 5: Commit evaluator**

```bash
git add src/nexusmind/retrieval_evaluation.py src/nexusmind/__init__.py tests/test_retrieval_evaluation.py
git commit -m "feat: add retrieval evaluation metrics"
```

### Task 2: Add strict JSON dataset loading

**Files:**
- Modify: `src/nexusmind/retrieval_evaluation.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_retrieval_evaluation_dataset.py`

- [ ] **Step 1: Write failing loader tests**

Test one valid UTF-8 dataset and parametrized failures for invalid JSON, non-object root, missing/extra root or case/target fields, non-list cases/targets, empty cases, wrong scalar types, empty IDs/queries/paths, duplicate case IDs, empty relevance, and duplicate targets.

```python
cases = load_retrieval_evaluation_cases(path)
assert cases == (
    RetrievalEvaluationCase(
        case_id="case-1",
        query="secure world",
        relevant_documents=(RetrievalTarget("eval-corpus", "trustzone.md"),),
    ),
)
```

- [ ] **Step 2: Run loader tests to verify RED**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation_dataset.py`

Expected: import failure because the loader is absent.

- [ ] **Step 3: Implement strict loader**

Add `RetrievalEvaluationDatasetError(RetrievalEvaluationError)` and `load_retrieval_evaluation_cases(path)`. Read UTF-8, wrap filesystem/JSON failures in the controlled error, require exact key sets at every object layer, require JSON arrays, instantiate validated target/case contracts, and reject duplicate case IDs before returning an exact tuple.

- [ ] **Step 4: Run loader and evaluator tests**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_dataset.py`

Expected: all tests pass.

- [ ] **Step 5: Commit loader**

```bash
git add src/nexusmind/retrieval_evaluation.py src/nexusmind/__init__.py tests/test_retrieval_evaluation_dataset.py
git commit -m "feat: load retrieval evaluation datasets"
```

### Task 3: Check in and measure the real baseline

**Files:**
- Create: `evals/knowledge/corpus/trustzone.md`
- Create: `evals/knowledge/corpus/qnx.md`
- Create: `evals/knowledge/corpus/android.md`
- Create: `evals/knowledge/corpus/cryptography.md`
- Create: `evals/knowledge/corpus/retrieval.md`
- Create: `evals/knowledge/cases.json`
- Create: `tests/test_retrieval_evaluation_baseline.py`

- [ ] **Step 1: Add authored corpus and 12–15 explicit labels**

Write original technical summaries with exact terms, multi-term concepts, rare terms, and deliberate distractors. Use `source_id="eval-corpus"` and root-relative markdown logical paths in every label. Ensure some documents exceed 240 characters.

- [ ] **Step 2: Write the failing end-to-end baseline test**

```python
def test_checked_in_retrieval_baseline_is_deterministic() -> None:
    collection = KnowledgeCollection(
        chunker=TextChunker(chunk_size=240, overlap=40)
    )
    collection.sync(LocalDirectoryAdapter(CORPUS, source_id="eval-corpus"))
    cases = load_retrieval_evaluation_cases(CASES)
    first = evaluate_retrieval(collection, cases, k=5)
    second = evaluate_retrieval(collection, cases, k=5)
    assert second == first
    assert first.k == 5
    assert len(first.case_results) == len(cases)
```

Also assert every label resolves and returned metrics equal constants recorded after the first measured run.

- [ ] **Step 3: Run baseline to obtain deterministic metrics**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation_baseline.py -vv`

Expected: initial failure only at deliberately impossible sentinel metric expectations (`-1.0`); capture the report values without changing retrieval/chunker behavior, then replace the sentinels with those exact measured values using `pytest.approx()`.

- [ ] **Step 4: Re-run baseline and focused evaluation tests**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_dataset.py tests/test_retrieval_evaluation_baseline.py`

Expected: all evaluation tests pass offline.

- [ ] **Step 5: Commit baseline corpus and test**

```bash
git add evals/knowledge/corpus evals/knowledge/cases.json tests/test_retrieval_evaluation_baseline.py
git commit -m "test: add deterministic retrieval baseline"
```

### Task 4: Record baseline and verify the repository

**Files:**
- Create: `evals/knowledge/baseline.md`
- Modify: `README.md`

- [ ] **Step 1: Document metrics and reproduction**

Record corpus/case locations, Document-level relevance versus chunk ranking, exact metric definitions, duplicate chunk semantics, fixed chunker/BM25/K configuration, measured Hit@5/Recall@5/MRR@5, reproduction pytest command, dataset limitations, and the no-tuning/non-gating status. Add a concise README pointer.

- [ ] **Step 2: Run full verification**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q`

Expected: complete suite passes with zero failures.

Run: `git diff --check`

Expected: exit 0 with no output.

- [ ] **Step 3: Commit documentation**

```bash
git add evals/knowledge/baseline.md README.md
git commit -m "docs: record retrieval evaluation baseline"
```

- [ ] **Step 4: Review branch scope**

Run: `git diff --stat origin/main...HEAD && git diff --check origin/main...HEAD`

Expected: only issue #69 spec/plan, evaluator code/tests, authored corpus/labels, baseline report, and README changes.
