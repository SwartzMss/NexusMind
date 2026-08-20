# Categorized Retrieval Evaluation and Backend Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic categorized, multi-K comparison of BM25, Semantic, and Hybrid retrieval over one checked-in offline benchmark, with reproducible Markdown diagnostics.

**Architecture:** Extend the current evaluator around one validated canonical snapshot and one `max(K)` ranking per case, then derive immutable per-K and per-category reports from ranking prefixes. A source-neutral comparison layer enforces exact snapshot equality across named backend factories, while a separate benchmark module owns offline fixture construction and a pure structured-report-to-Markdown renderer.

**Tech Stack:** Python 3.11–3.13, frozen/slotted dataclasses, `Enum`, existing `KnowledgeCollection`/BM25/Semantic/Hybrid APIs, authored JSON/Markdown fixtures, pytest.

---

## File structure

- Modify `src/nexusmind/retrieval_evaluation.py`: category contracts, strict loading, multi-K derivation, category aggregation, backend comparison, controlled errors.
- Modify `src/nexusmind/__init__.py`: public exports for new evaluation contracts and functions.
- Create `src/nexusmind/retrieval_benchmark.py`: deterministic provider, benchmark factories/runner, renderer configuration, pure Markdown renderer.
- Modify `evals/knowledge/{cases.json,cjk/cases.json,semantic/cases.json,hybrid/cases.json}`: explicit category migration.
- Create `evals/knowledge/benchmark/corpus/*.md` and `evals/knowledge/benchmark/cases.json`: common authored benchmark.
- Create `evals/knowledge/benchmark.md`: generated descriptive baseline.
- Modify `README.md`: evaluation workflow and policy documentation.
- Modify existing evaluation tests and create focused multi-K, comparison, and benchmark tests.

### Task 1: Categorized case contract and strict dataset migration

**Files:**
- Modify: `src/nexusmind/retrieval_evaluation.py`
- Modify: `src/nexusmind/__init__.py`
- Modify: `tests/test_retrieval_evaluation.py`
- Modify: `tests/test_retrieval_evaluation_dataset.py`
- Modify: `evals/knowledge/cases.json`
- Modify: `evals/knowledge/cjk/cases.json`
- Modify: `evals/knowledge/semantic/cases.json`
- Modify: `evals/knowledge/hybrid/cases.json`

- [ ] **Step 1: Write failing category contract tests**

Add tests that construct cases with `RetrievalCategory.EXACT_TERM`, reject strings/unknown values at the dataclass boundary, require `category` in exact JSON fields, reject unknown category strings, and prove IDs do not infer categories:

```python
case = RetrievalEvaluationCase(
    "exact-api", RetrievalCategory.EXACT_TERM, "KnowledgeCollection.search",
    (RetrievalTarget("source", "api.md"),),
)
assert case.category is RetrievalCategory.EXACT_TERM
with pytest.raises(RetrievalEvaluationDatasetError, match="unknown category"):
    load_retrieval_evaluation_cases(path_with_category("not-real"))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_dataset.py`

Expected: collection fails because `RetrievalCategory` and the category field do not exist.

- [ ] **Step 3: Implement the minimal categorized schema**

Define the ordered string enum and make category required:

```python
class RetrievalCategory(str, Enum):
    EXACT_TERM = "exact_term"
    IDENTIFIER = "identifier"
    CJK = "cjk"
    PARAPHRASE = "paraphrase"
    CROSS_LANGUAGE = "cross_language"
    MULTI_DOCUMENT = "multi_document"
    DISTRACTOR_HEAVY = "distractor_heavy"
    MIXED_SIGNAL = "mixed_signal"

@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    case_id: str
    category: RetrievalCategory
    query: str
    relevant_documents: tuple[RetrievalTarget, ...]
```

Require JSON case fields to equal `{"case_id", "category", "query", "relevant_documents"}`, parse via `RetrievalCategory(raw_case["category"])`, and wrap invalid values without echoing arbitrary input. Export the enum.

- [ ] **Step 4: Explicitly migrate every checked-in dataset and affected constructors**

Assign categories based on authored intent: exact named mechanisms to `exact_term`, tokens such as `ZXQ-417` to `identifier`, Chinese lexical cases to `cjk`, low-overlap natural-language cases to `paraphrase`, multi-target cases to `multi_document`, and Hybrid mixed signal to `mixed_signal`. Update all test constructors explicitly; do not add a default.

- [ ] **Step 5: Run categorized dataset regression tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_dataset.py tests/test_retrieval_evaluation_baseline.py tests/test_retrieval_evaluation_cjk.py tests/test_retrieval_evaluation_semantic.py tests/test_retrieval_evaluation_hybrid.py`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -- src/nexusmind/retrieval_evaluation.py src/nexusmind/__init__.py tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_dataset.py tests/test_retrieval_evaluation_baseline.py tests/test_retrieval_evaluation_cjk.py tests/test_retrieval_evaluation_semantic.py tests/test_retrieval_evaluation_hybrid.py evals/knowledge/cases.json evals/knowledge/cjk/cases.json evals/knowledge/semantic/cases.json evals/knowledge/hybrid/cases.json
git commit -m "feat: categorize retrieval evaluation cases"
```

### Task 2: Multi-K evaluation from one authoritative ranking

**Files:**
- Modify: `src/nexusmind/retrieval_evaluation.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_retrieval_evaluation_multi_k.py`

- [ ] **Step 1: Write failing validation and max-K prefix tests**

Cover empty/non-tuple K sets, booleans/non-integers/non-positive values, duplicates, more than `MAX_EVALUATION_K_VALUES`, values above `MAX_EVALUATION_K`, ascending output, one search call at maximum K, multi-document recall changes, and result-limit wrapping:

```python
reports = evaluate_retrieval_multi_k(collection, (case,), ks=(3, 1, 2))
assert tuple(report.k for report in reports) == (1, 2, 3)
assert collection.calls == [(case.query, 3)]
assert reports[0].case_results[0].returned_chunk_ids == ("rank-1",)
assert reports[1].case_results[0].returned_chunk_ids == ("rank-1", "rank-2")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation_multi_k.py`

Expected: import failure for the new function/constants.

- [ ] **Step 3: Extract snapshot validation and ranking derivation**

Add bounded constants, `_validate_cases`, `_canonical_targets`, `_search_rankings`, and `_report_from_rankings`. Search each case once at `max(ks)` and slice the immutable returned result tuple for each K. Include category and authored relevant targets in `RetrievalEvaluationCaseResult`.

- [ ] **Step 4: Implement public multi-K and preserve single-K**

```python
def evaluate_retrieval_multi_k(collection, cases, *, ks=(1, 3, 5, 10)):
    ordered_ks = _validate_ks(ks)
    snapshot = _require_snapshot(collection)
    _validate_relevance(cases, snapshot)
    rankings = _search_rankings(collection, cases, max(ordered_ks))
    return tuple(_report_from_rankings(cases, rankings, k) for k in ordered_ks)

def evaluate_retrieval(collection, cases, *, k=5):
    return evaluate_retrieval_multi_k(collection, cases, ks=(k,))[0]
```

- [ ] **Step 5: Run focused and existing evaluator tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_multi_k.py`

Expected: all pass and existing single-K metric values remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add -- src/nexusmind/retrieval_evaluation.py src/nexusmind/__init__.py tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_multi_k.py
git commit -m "feat: evaluate retrieval at multiple cutoffs"
```

### Task 3: Per-category aggregates and failure classifications

**Files:**
- Modify: `src/nexusmind/retrieval_evaluation.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_retrieval_evaluation_categories.py`

- [ ] **Step 1: Write failing category aggregate tests**

Use cases from two categories and a multi-document case to assert case counts, enum ordering, omission of empty categories, exact means, and reproducibility from underlying cases. Add deterministic helpers that classify a complete miss, a result found only below a smaller cutoff, and partial recall.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation_categories.py`

Expected: missing `RetrievalCategoryReport` and diagnostic classification contracts.

- [ ] **Step 3: Implement immutable category reports**

```python
@dataclass(frozen=True, slots=True)
class RetrievalCategoryReport:
    category: RetrievalCategory
    case_count: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
```

Add `category_reports` to each `RetrievalEvaluationReport`, aggregate in enum order, and compute only from the exact report's `case_results`.

- [ ] **Step 4: Implement deterministic failure classification**

Use a bounded enum (`missed`, `ranked_below_cutoff`, `partial_recall`) and a pure function comparing the same case across requested K reports. Do not generate prose or inspect backend scores.

- [ ] **Step 5: Run focused evaluation tests and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_multi_k.py tests/test_retrieval_evaluation_categories.py`

```bash
git add -- src/nexusmind/retrieval_evaluation.py src/nexusmind/__init__.py tests/test_retrieval_evaluation_categories.py
git commit -m "feat: aggregate retrieval metrics by category"
```

### Task 4: Source-neutral backend comparison and canonical snapshot equality

**Files:**
- Modify: `src/nexusmind/retrieval_evaluation.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_retrieval_evaluation_comparison.py`

- [ ] **Step 1: Write failing comparison contract tests**

Cover non-empty unique backend names, exact tuple inputs, non-callable factories, stable declaration order, identical cases/K values, factory/setup/snapshot/search errors with sanitized public messages and chained causes, result-limit incompatibility, and no partial report return.

- [ ] **Step 2: Write failing exact snapshot-equality test**

Build two fake backend collections whose snapshots differ only in document content or ordering, call comparison, and assert a controlled mismatch before either collection searches:

```python
with pytest.raises(RetrievalComparisonError, match="canonical snapshots differ"):
    compare_retrieval_backends(backends, cases, ks=(1, 3))
assert left.calls == right.calls == []
```

- [ ] **Step 3: Run tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation_comparison.py`

Expected: missing comparison types/functions.

- [ ] **Step 4: Implement comparison contracts and execution**

```python
@dataclass(frozen=True, slots=True)
class RetrievalBackend:
    name: str
    factory: Callable[[], KnowledgeCollection]

@dataclass(frozen=True, slots=True)
class RetrievalBackendReport:
    backend_name: str
    reports_by_k: tuple[RetrievalEvaluationReport, ...]

@dataclass(frozen=True, slots=True)
class RetrievalComparisonReport:
    ks: tuple[int, ...]
    backend_reports: tuple[RetrievalBackendReport, ...]
```

Construct all backends, capture snapshots, compare every snapshot by dataclass equality to the first, validate relevance once per snapshot, then evaluate. Preserve names and rankings without score normalization.

- [ ] **Step 5: Run comparison and evaluator tests and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_multi_k.py tests/test_retrieval_evaluation_categories.py tests/test_retrieval_evaluation_comparison.py`

```bash
git add -- src/nexusmind/retrieval_evaluation.py src/nexusmind/__init__.py tests/test_retrieval_evaluation_comparison.py
git commit -m "feat: compare retrieval backends deterministically"
```

### Task 5: Authored common benchmark and offline semantic fixture

**Files:**
- Create: `src/nexusmind/retrieval_benchmark.py`
- Create: `evals/knowledge/benchmark/corpus/*.md`
- Create: `evals/knowledge/benchmark/cases.json`
- Create: `tests/test_retrieval_benchmark.py`

- [ ] **Step 1: Write failing benchmark coverage tests**

Assert 10–20 documents, 30–50 unique cases, every enum category present, UTF-8 mixed-language content, at least one multi-document case, fixed K `(1, 3, 5, 10)`, exactly three backend names, repeat equality, identical snapshots, and no environment/API-key dependency.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_benchmark.py`

Expected: benchmark module/data are absent.

- [ ] **Step 3: Author the corpus and relevance cases**

Create focused documents for API/runtime terms, identifiers, CJK/mixed-language concepts, secure execution, IPC, cryptography, checkpoints, retrieval metrics, and strong near-duplicate distractors. Author independent category and document targets in JSON.

- [ ] **Step 4: Implement deterministic concept embeddings and backend factories**

Implement `BenchmarkEmbeddingProvider` using a fixed concept-vocabulary projection into a bounded vector dimension, deterministic normalization, and explicit document/query concept mappings. Build BM25, Semantic, and Hybrid collections with identical chunker/corpus/source configuration; do not perform I/O outside the checked-in corpus.

- [ ] **Step 5: Implement and verify the benchmark runner**

`run_retrieval_benchmark()` loads cases, constructs named backends, and calls `compare_retrieval_backends(..., ks=(1, 3, 5, 10))`. Repeated calls must compare equal.

- [ ] **Step 6: Run benchmark tests and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_benchmark.py`

```bash
git add -- src/nexusmind/retrieval_benchmark.py evals/knowledge/benchmark tests/test_retrieval_benchmark.py
git commit -m "test: add categorized retrieval benchmark"
```

### Task 6: Pure Markdown renderer and byte-for-byte baseline

**Files:**
- Modify: `src/nexusmind/retrieval_benchmark.py`
- Create: `evals/knowledge/benchmark.md`
- Create: `tests/test_retrieval_benchmark_report.py`

- [ ] **Step 1: Write failing pure-renderer tests**

Create a small structured comparison value directly, render twice, and assert equality, stable six-decimal metrics, backend/K/category ordering, selected diagnostics, no timestamp or absolute path, and no filesystem changes. Test the checked-in report with exact equality:

```python
expected = BENCHMARK_REPORT.read_text(encoding="utf-8")
assert render_retrieval_comparison(run_retrieval_benchmark(), REPORT_CONFIG) == expected
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_benchmark_report.py`

Expected: renderer/config/report are missing.

- [ ] **Step 3: Implement renderer-only immutable configuration and pure function**

```python
@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkRenderConfig:
    title: str
    corpus_summary: str
    reproduction_command: str

def render_retrieval_comparison(
    report: RetrievalComparisonReport,
    config: RetrievalBenchmarkRenderConfig,
) -> str:
    lines: list[str] = []
    # Read only report/config values; return "\n".join(lines) + "\n".
```

Render overall and per-category tables plus deterministic case diagnostics. Mention max-K prefix semantics, exact snapshot policy, fixture limitations, authored-label policy, and non-gate status.

- [ ] **Step 4: Generate the baseline through an explicit command**

Add a module CLI whose only write occurs in `main()` when passed the exact target path; the renderer itself remains pure. Generate `evals/knowledge/benchmark.md` once and verify a second generation is byte-identical.

- [ ] **Step 5: Run report and benchmark tests and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_benchmark.py tests/test_retrieval_benchmark_report.py`

```bash
git add -- src/nexusmind/retrieval_benchmark.py evals/knowledge/benchmark.md tests/test_retrieval_benchmark_report.py
git commit -m "docs: generate retrieval comparison baseline"
```

### Task 7: Documentation, full regression verification, and PR readiness

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-20-categorized-retrieval-comparison-design.md` only if implementation revealed an approved clarification.

- [ ] **Step 1: Write documentation assertions if the README has testable anchors**

Add or extend a documentation test to require category definitions, metric semantics, max-K prefixes, recall-vs-ranking interpretation, snapshot comparison, fixture limitations, label policy, regeneration command, and descriptive/non-gate wording.

- [ ] **Step 2: Update README**

Document the exact benchmark paths and reproduction command, all eight categories, Hit/Recall/MRR definitions, one-search max-K semantics, backend/snapshot rules, diagnostic interpretation, and explicit non-goals.

- [ ] **Step 3: Run focused retrieval/evaluation regression suite**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_knowledge_retrieval.py tests/test_lexical_analysis.py \
  tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py \
  tests/test_knowledge_collection.py tests/test_hybrid_knowledge_collection.py \
  tests/test_knowledge_snapshot.py tests/test_knowledge_store.py \
  tests/test_persistence.py tests/test_retrieval_evaluation.py \
  tests/test_retrieval_evaluation_dataset.py \
  tests/test_retrieval_evaluation_baseline.py tests/test_retrieval_evaluation_cjk.py \
  tests/test_retrieval_evaluation_semantic.py tests/test_retrieval_evaluation_hybrid.py \
  tests/test_retrieval_evaluation_multi_k.py \
  tests/test_retrieval_evaluation_categories.py \
  tests/test_retrieval_evaluation_comparison.py \
  tests/test_retrieval_benchmark.py tests/test_retrieval_benchmark_report.py
```

Expected: all pass offline.

- [ ] **Step 4: Run the entire suite and static repository checks**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q`

Run: `git diff --check && git status --short`

Expected: full suite passes; no whitespace errors; only intended files differ.

- [ ] **Step 5: Regenerate and compare the report explicitly**

Run the documented regeneration command to a temporary file and compare it byte-for-byte with `evals/knowledge/benchmark.md` using `cmp`.

- [ ] **Step 6: Commit documentation**

```bash
git add -- README.md
git commit -m "docs: explain categorized retrieval evaluation"
```

- [ ] **Step 7: Inspect final history and diff before publishing**

Run: `git log --oneline origin/main..HEAD`, `git diff --stat origin/main...HEAD`, and `git diff --check origin/main...HEAD`.

Expected: only issue #77 design, evaluation, benchmark, tests, report, and documentation changes are present.
