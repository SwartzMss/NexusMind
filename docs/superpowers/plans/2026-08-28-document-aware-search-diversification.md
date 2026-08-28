# Document-Aware Search Diversification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diversify final user-facing search results across canonical documents for every retrieval backend while preserving backend scores, selected-candidate raw order, result limits, and unmodified diagnostics.

**Architecture:** Add a backend-independent selector that operates on document IDs and same-query scores and returns raw candidate indices. `KnowledgeCollection.search()` oversamples the final backend output, resolves canonical provenance, applies the selector, and returns the selected raw-order subsequence; `diagnose_search()` remains untouched. A checked-in offline benchmark compares raw diagnostic Top-K with diversified search Top-K on broad and precise queries.

**Tech Stack:** Python 3.11+, dataclasses, lexical/semantic/hybrid/reranked retrieval indexes, pytest, JSON/Markdown offline evaluation.

---

### Task 1: Build the deterministic document-aware selector

**Files:**
- Create: `src/nexusmind/search_diversification.py`
- Create: `tests/test_search_diversification.py`

- [ ] **Step 1: Write failing selection and validation tests**

Create helpers and tests with exact candidate identity and score:

```python
from nexusmind.search_diversification import (
    RankedDocumentCandidate,
    search_candidate_depth,
    select_document_aware_indices,
)


def candidates(*values: tuple[str, float]) -> tuple[RankedDocumentCandidate, ...]:
    return tuple(RankedDocumentCandidate(document_id, score) for document_id, score in values)


def test_diversifies_inside_query_relative_score_window_and_preserves_raw_indices() -> None:
    ranked = candidates(
        ("a", 10.0), ("a", 9.0), ("a", 8.0), ("a", 7.0),
        ("a", 6.0), ("b", 5.5), ("c", 5.0),
    )
    assert select_document_aware_indices(ranked, limit=5) == (0, 1, 2, 5, 6)


def test_weak_cross_document_candidate_does_not_displace_strong_chunk() -> None:
    ranked = candidates(
        ("a", 10.0), ("a", 9.0), ("a", 8.0), ("a", 7.0),
        ("a", 6.0), ("b", 1.0),
    )
    assert select_document_aware_indices(ranked, limit=5) == (0, 1, 2, 3, 4)


def test_same_document_backfills_all_slots() -> None:
    ranked = candidates(("a", 4.0), ("a", 3.0), ("a", 2.0), ("a", 1.0))
    assert select_document_aware_indices(ranked, limit=4) == (0, 1, 2, 3)
```

Add parametrized tests proving equal scores, negative scores, and a positive affine transform select the same indices; repeated calls are identical; empty input returns `()`; output never exceeds `limit`; invalid candidate tuple, document ID, non-finite/non-float score, limit values, and limits above 100 are rejected. Assert depth bounds:

```python
@pytest.mark.parametrize(("limit", "depth"), [(1, 4), (10, 40), (25, 100), (100, 100)])
def test_candidate_depth_is_bounded(limit: int, depth: int) -> None:
    assert search_candidate_depth(limit) == depth
```

- [ ] **Step 2: Run selector tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_diversification.py -q`

Expected: collection fails because `nexusmind.search_diversification` does not exist.

- [ ] **Step 3: Implement the minimal selector**

Create the internal policy module:

```python
from dataclasses import dataclass
from math import isfinite

SEARCH_CANDIDATE_MULTIPLIER = 4
MAX_SEARCH_CANDIDATES = 100
PREFERRED_RESULTS_PER_DOCUMENT = 2
RELEVANCE_WINDOW_FACTOR = 0.25


@dataclass(frozen=True, slots=True)
class RankedDocumentCandidate:
    document_id: str
    score: float

    def __post_init__(self) -> None:
        if type(self.document_id) is not str or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if type(self.score) is not float or not isfinite(self.score):
            raise ValueError("score must be a finite float")


def search_candidate_depth(limit: int) -> int:
    _validate_limit(limit)
    return min(limit * SEARCH_CANDIDATE_MULTIPLIER, MAX_SEARCH_CANDIDATES)


def select_document_aware_indices(
    candidates: tuple[RankedDocumentCandidate, ...], *, limit: int
) -> tuple[int, ...]:
    _validate_limit(limit)
    if type(candidates) is not tuple:
        raise TypeError("candidates must be a tuple")
    if any(type(item) is not RankedDocumentCandidate for item in candidates):
        raise TypeError("candidates must contain RankedDocumentCandidate values")
    if not candidates:
        return ()
    raw_top_k = candidates[:limit]
    scores = tuple(item.score for item in raw_top_k)
    worst = min(scores)
    floor = worst - RELEVANCE_WINDOW_FACTOR * (max(scores) - worst)
    selected: list[int] = []
    counts: dict[str, int] = {}
    for index, item in enumerate(candidates):
        if len(selected) == limit:
            break
        if counts.get(item.document_id, 0) >= PREFERRED_RESULTS_PER_DOCUMENT:
            continue
        if index >= limit and item.score < floor:
            continue
        selected.append(index)
        counts[item.document_id] = counts.get(item.document_id, 0) + 1
    selected_set = set(selected)
    for index in range(len(candidates)):
        if len(selected) == limit:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    return tuple(sorted(selected))
```

Implement `_validate_limit()` with exact-int, positive, and maximum-100 checks.

- [ ] **Step 4: Run selector tests and verify GREEN**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_diversification.py -q`

Expected: all selector tests pass.

- [ ] **Step 5: Commit the selector**

```bash
git add src/nexusmind/search_diversification.py tests/test_search_diversification.py
git commit -m "feat: add document-aware search selector"
```

### Task 2: Specify collection, diagnostics, and backend behavior before integration

**Files:**
- Modify: `tests/test_knowledge_collection.py`
- Modify: `tests/test_knowledge_diagnostics.py`
- Modify: `tests/test_knowledge_base_diagnostics.py`
- Modify: `tests/test_semantic_retrieval.py`
- Modify: `tests/test_hybrid_retrieval.py`
- Modify: `tests/test_reranking.py`

- [ ] **Step 1: Write failing collection integration tests**

Add a scripted cloneable index that records requested limits and returns a ranked
tuple of exact `SearchHit` values. Synchronize three canonical documents and assert:

```python
results = collection.search("broad", limit=5)

assert index.search_calls == [("broad", 20)]
assert tuple(item.document.document_id for item in results) == (
    document_a.document_id,
    document_a.document_id,
    document_a.document_id,
    document_b.document_id,
    document_c.document_id,
)
assert tuple(item.hit.score for item in results) == (10.0, 9.0, 8.0, 5.5, 5.0)
assert tuple(item.hit.matched_terms for item in results) == tuple(
    raw_hits[index].matched_terms for index in (0, 1, 2, 5, 6)
)
```

Add tests that a one-document pool backfills to K, fewer-than-K returns every
candidate, selected metadata is detached, malformed oversampled candidates are
still rejected even when they would not be selected, and `limit=101` is rejected
before calling the backend.

- [ ] **Step 2: Add a raw diagnostics isolation regression**

Use one scripted index whose `search()` records the oversampled depth and whose
`diagnose()` records its requested raw limit:

```python
search_results = collection.search("broad", limit=3)
diagnostics = collection.diagnose_search("broad", limit=3)

assert state.search_limits == [12]
assert state.diagnose_limits == [3]
assert tuple(item.hit.chunk.chunk_id for item in search_results) == selected_ids
assert tuple(item.hit.chunk.chunk_id for item in diagnostics.results) == raw_ids[:3]
assert tuple(item.diagnostic.rank for item in diagnostics.candidates) == (1, 2, 3)
assert tuple(item.diagnostic.score for item in diagnostics.candidates) == raw_scores[:3]
```

- [ ] **Step 3: Add parametrized real-backend collection coverage**

For lexical, semantic, hybrid, and reranked factories, synchronize the same
multi-chunk documents, call `collection.search(query, limit=3)`, and assert:

```python
assert len(results) <= 3
assert len({item.document.document_id for item in results}) >= 2
assert results == collection.search(query, limit=3)
assert [item.hit.score for item in results] == [
    raw_scores[raw_chunk_ids.index(item.hit.chunk.chunk_id)] for item in results
]
```

Use deterministic in-memory embedding and reranker fixtures already defined in
the semantic/hybrid/reranking test modules. Do not assert identical scores across
backend types; compare each result only with its own backend's raw diagnostic row.

- [ ] **Step 4: Run the complete new integration surface and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_knowledge_collection.py \
  tests/test_knowledge_diagnostics.py \
  tests/test_knowledge_base_diagnostics.py \
  tests/test_semantic_retrieval.py \
  tests/test_hybrid_retrieval.py \
  tests/test_reranking.py -q
```

Expected: failures show search still requests exactly K backend results, returns
raw Top-K without document selection, and accepts limits above the diversification
maximum. Diagnostic raw-ranking assertions that already describe existing
behavior may pass, but the same test also asserts the not-yet-implemented
oversampled search path and therefore fails for the intended reason.

- [ ] **Step 5: Confirm no production files changed during RED**

```bash
git status --short
```

Expected: only the six specified test files are modified.

### Task 3: Integrate selection and make the full backend contract GREEN

**Files:**
- Modify: `src/nexusmind/knowledge_collection.py:331-336`
- Modify: `tests/test_knowledge_collection.py`
- Modify: `tests/test_knowledge_diagnostics.py`
- Modify: `tests/test_knowledge_base_diagnostics.py`
- Modify: `tests/test_semantic_retrieval.py`
- Modify: `tests/test_hybrid_retrieval.py`
- Modify: `tests/test_reranking.py`
- Modify if call-depth assertions require it: `tests/test_context_assembly.py`
- Modify if call-depth assertions require it: `tests/test_knowledge_answer.py`
- Modify if call-depth assertions require it: `tests/test_knowledge_query.py`

- [ ] **Step 1: Integrate selector without changing backend contracts**

Import the selector and replace `KnowledgeCollection.search()` with:

```python
def search(self, query: str, *, limit: int = 10) -> tuple[KnowledgeSearchResult, ...]:
    candidate_depth = search_candidate_depth(limit)
    hits = self._index.search(query, limit=candidate_depth)
    if type(hits) is not tuple:
        raise KnowledgeSearchResolutionError("index search result must be a tuple")
    resolved = tuple(self._resolve_hit(hit) for hit in hits)
    ranked = tuple(
        RankedDocumentCandidate(item.document.document_id, item.hit.score)
        for item in resolved
    )
    selected = select_document_aware_indices(ranked, limit=limit)
    return tuple(resolved[index] for index in selected)
```

Do not add backend-type checks, call diagnostics from search, or reconstruct
`SearchHit` values.

- [ ] **Step 2: Run collection, backend, and diagnostic tests and verify GREEN**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_knowledge_collection.py \
  tests/test_knowledge_diagnostics.py \
  tests/test_knowledge_base_diagnostics.py \
  tests/test_semantic_retrieval.py \
  tests/test_hybrid_retrieval.py \
  tests/test_reranking.py -q
```

Expected: all selected tests pass; diagnostic ranks/scores remain raw.

- [ ] **Step 3: Run context and answer regression tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_context_assembly.py tests/test_knowledge_answer.py tests/test_knowledge_query.py -q`

Expected: all selected tests pass after updating only scripted backend call-depth
assertions that intentionally change from K to bounded oversampling.

- [ ] **Step 4: Verify preservation properties explicitly**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_knowledge_collection.py -q \
  -k 'score or terms or order or backfill or deterministic'
```

Expected: selected raw order, exact score, exact matched terms, backfill, and
determinism tests pass.

- [ ] **Step 5: Commit integration and all prewritten tests**

```bash
git add src/nexusmind/knowledge_collection.py tests/test_knowledge_collection.py tests/test_knowledge_diagnostics.py tests/test_knowledge_base_diagnostics.py tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_context_assembly.py tests/test_knowledge_answer.py tests/test_knowledge_query.py
git commit -m "feat: diversify knowledge search results"
```

### Task 4: Add the raw-vs-diversified offline evaluation

**Files:**
- Create: `src/nexusmind/search_diversification_benchmark.py`
- Create: `tests/test_search_diversification_benchmark.py`
- Create: `evals/knowledge/diversification/cases.json`
- Create: `evals/knowledge/diversification/corpus/crypto-overview.md`
- Create: `evals/knowledge/diversification/corpus/crypto-permissions.md`
- Create: `evals/knowledge/diversification/corpus/binder.md`
- Create: `evals/knowledge/diversification/corpus/qnx.md`
- Create: `evals/knowledge/diversification/corpus/precise-import.md`
- Create: `evals/knowledge/diversification.md`

- [ ] **Step 1: Write failing benchmark contract tests**

Create tests that load the authored cases, build one default lexical collection,
and compare two views of the same synchronized state:

```python
report = run_search_diversification_benchmark()

assert {item.query for item in report.cases} >= {
    "Crypto", "Binder", "QNX", "权限校验", "lpRpcCrypto ImportFile exact permission flow"
}
assert report.diversified_broad_unique_relevant_documents > report.raw_broad_unique_relevant_documents
assert report.diversified_precise_mrr >= report.raw_precise_mrr
assert report.diversified_precise_recall >= report.raw_precise_recall
assert render_search_diversification_benchmark(report) == Path(
    "evals/knowledge/diversification.md"
).read_text(encoding="utf-8")
```

Assert every per-query row contains raw and diversified document sequences,
unique-document counts, relevant-document counts, hit/recall/MRR, and a stable
reproduction command.

- [ ] **Step 2: Run benchmark tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_diversification_benchmark.py -q`

Expected: benchmark module and dataset do not exist.

- [ ] **Step 3: Implement the benchmark using existing evaluation contracts**

Define a private raw view that delegates snapshot and uses diagnostics for search:

```python
class _RawDiagnosticSearchView:
    def __init__(self, collection: KnowledgeCollection) -> None:
        self._collection = collection

    def snapshot(self):
        return self._collection.snapshot()

    def search(self, query: str, *, limit: int = 10):
        return self._collection.diagnose_search(query, limit=limit).results
```

Load `RetrievalEvaluationCase` values with
`load_retrieval_evaluation_cases()`. Synchronize `LocalDirectoryAdapter` against
the authored corpus. Call `evaluate_retrieval(raw_view, cases, k=5)` and
`evaluate_retrieval(collection, cases, k=5)`. Derive total unique documents from
`returned_targets`, relevant unique documents from `relevant_targets_found`, and
reuse hit/recall/MRR from each `RetrievalEvaluationCaseResult`.

The CLI module entry point must support:

```bash
PYTHONPATH=src python -m nexusmind.search_diversification_benchmark \
  --write evals/knowledge/diversification.md
```

- [ ] **Step 4: Author deterministic broad and precise evaluation data**

Use source ID `diversification-docs`. Make `crypto-overview.md` long enough to
produce at least five Crypto-matching chunks; add relevant lower-ranked Crypto
material to `crypto-permissions.md`. Give Binder, QNX, and 权限校验 at least two
authored relevant documents across the corpus. Make `precise-import.md` contain
the only exact `lpRpcCrypto ImportFile exact permission flow` phrase so the
precise query should legitimately remain dominated by it.

Define cases with exact fields required by the existing dataset loader:

```json
{
  "cases": [
    {"case_id":"broad-crypto","category":"multi_document","query":"Crypto","relevant_documents":[{"source_id":"diversification-docs","logical_path":"crypto-overview.md"},{"source_id":"diversification-docs","logical_path":"crypto-permissions.md"}]},
    {"case_id":"broad-binder","category":"multi_document","query":"Binder","relevant_documents":[{"source_id":"diversification-docs","logical_path":"binder.md"},{"source_id":"diversification-docs","logical_path":"crypto-permissions.md"}]},
    {"case_id":"broad-qnx","category":"multi_document","query":"QNX","relevant_documents":[{"source_id":"diversification-docs","logical_path":"qnx.md"},{"source_id":"diversification-docs","logical_path":"binder.md"}]},
    {"case_id":"broad-permission","category":"multi_document","query":"权限校验","relevant_documents":[{"source_id":"diversification-docs","logical_path":"crypto-permissions.md"},{"source_id":"diversification-docs","logical_path":"precise-import.md"}]},
    {"case_id":"precise-import","category":"exact_term","query":"lpRpcCrypto ImportFile exact permission flow","relevant_documents":[{"source_id":"diversification-docs","logical_path":"precise-import.md"}]}
  ]
}
```

- [ ] **Step 5: Generate, inspect, and commit the report**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m nexusmind.search_diversification_benchmark \
  --write evals/knowledge/diversification.md
PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_diversification_benchmark.py -q
```

Expected: broad aggregate unique relevant-document coverage increases; precise
MRR and recall do not decrease; generated Markdown equals the checked-in report.

```bash
git add src/nexusmind/search_diversification_benchmark.py tests/test_search_diversification_benchmark.py evals/knowledge/diversification evals/knowledge/diversification.md
git commit -m "eval: compare raw and diversified search"
```

### Task 5: Document, verify, review, and create the PR

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `tests/test_project_metadata.py`

- [ ] **Step 1: Add failing documentation assertions**

Assert README and architecture state that search performs document-aware final
selection while diagnose preserves raw backend rank and score:

```python
assert "document-aware" in readme
assert "raw backend ranking" in readme
assert "document-aware" in architecture
assert "diagnose" in architecture and "raw backend ranking" in architecture
```

- [ ] **Step 2: Run metadata tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_metadata.py -q`

Expected: the new phrases are absent.

- [ ] **Step 3: Update public and architecture documentation**

Document this separation without exposing internal tuning constants as options:

```text
search: bounded backend candidate retrieval -> document-aware final selection
diagnose: unmodified raw backend ranking, ranks, stages, and scores
```

State that `--limit` remains the maximum final result count and that selected
scores are not rewritten.

- [ ] **Step 4: Run the full suite and diff checks**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'import sys,tomli; sys.modules["tomllib"]=tomli; import pytest; raise SystemExit(pytest.main(["-q"]))'
git diff origin/main..HEAD --check
git status --short
```

Expected: full suite passes with only declared skips; diff check is clean; only
intended files are modified.

- [ ] **Step 5: Request independent code review**

Invoke `superpowers:requesting-code-review` for `origin/main..HEAD`. Fix all
verified Critical and Important findings with a failing regression test first,
then repeat Step 4.

- [ ] **Step 6: Push and create PR**

```bash
git push -u origin codex/issue-110-diversify-search-results
gh pr create --repo SwartzMss/NexusMind --base main \
  --head codex/issue-110-diversify-search-results \
  --title "Diversify search results across documents" \
  --body-file /tmp/issue-110-pr-body.md
```

The PR body must summarize backend-agnostic placement, same-query relative score
safeguard, raw diagnostics isolation, evaluation results, verification evidence,
and include `Closes #110`.
