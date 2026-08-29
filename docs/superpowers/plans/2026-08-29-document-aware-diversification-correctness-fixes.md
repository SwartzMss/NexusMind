# Document-Aware Diversification Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve valid backend result-limit configurations and replace the outlier-sensitive relevance window with a robust same-query MAD cutoff while making the motivating Crypto benchmark diversify successfully.

**Architecture:** Built-in indexes expose an optional read-only search capacity; `KnowledgeCollection.search()` bounds oversampling by that capacity and conservatively uses the caller limit for third-party indexes without it. The selector derives its relevance floor from the lower median and lower median absolute deviation of raw Top-K scores, preserving raw order, scores, diagnostics, and positive-affine invariance.

**Tech Stack:** Python 3.11+, protocols/dataclasses, lexical/semantic/hybrid/reranked indexes, pytest, GitHub Actions, checked-in JSON/Markdown evaluation.

---

### Task 1: Respect optional backend result capacity

**Files:**
- Modify: `src/nexusmind/search_diversification.py`
- Modify: `src/nexusmind/knowledge_collection.py`
- Modify: `src/nexusmind/knowledge_retrieval.py`
- Modify: `src/nexusmind/semantic_retrieval.py`
- Modify: `src/nexusmind/hybrid_retrieval.py`
- Modify: `src/nexusmind/reranking.py`
- Modify: `tests/test_search_diversification.py`
- Modify: `tests/test_knowledge_collection.py`
- Modify: `tests/test_knowledge_diagnostics.py`
- Modify: `tests/test_knowledge_retrieval.py`
- Modify: `tests/test_semantic_retrieval.py`
- Modify: `tests/test_hybrid_retrieval.py`
- Modify: `tests/test_reranking.py`

- [ ] **Step 1: Write failing candidate-depth unit tests**

Replace the existing depth test with explicit capacity behavior:

```python
@pytest.mark.parametrize(
    ("limit", "capacity", "depth"),
    [
        (1, 100, 4),
        (5, 10, 10),
        (10, 100, 40),
        (25, 100, 100),
        (100, 100, 100),
        (5, None, 5),
        (5, True, 5),
        (5, 5.0, 5),
        (5, 0, 5),
        (5, 4, 5),
    ],
)
def test_candidate_depth_respects_optional_backend_capacity(
    limit: int, capacity: object, depth: int
) -> None:
    assert search_candidate_depth(limit, backend_capacity=capacity) == depth
```

Keep the existing invalid user-limit tests. Capacity is optional runtime
metadata, so malformed values fall back to `limit` instead of raising.

- [ ] **Step 2: Write failing collection compatibility regressions**

Extend the scripted test index with an opt-in capacity subclass:

```python
class _CapacityScriptedSearchIndex(_ScriptedSearchIndex):
    def __init__(self, state: _ScriptedSearchState, capacity: object) -> None:
        super().__init__(state)
        self._capacity = capacity

    @property
    def max_search_results(self) -> object:
        return self._capacity

    def clone(self) -> "_CapacityScriptedSearchIndex":
        return _CapacityScriptedSearchIndex(self.state, self._capacity)


class _RaisingCapacityScriptedSearchIndex(_ScriptedSearchIndex):
    @property
    def max_search_results(self) -> int:
        raise RuntimeError("private capacity failure")

    def clone(self) -> "_RaisingCapacityScriptedSearchIndex":
        return _RaisingCapacityScriptedSearchIndex(self.state)


def _scripted_collection(
    *, capacity: object | None, hit_count: int, raise_capacity: bool = False
) -> tuple[_ScriptedSearchState, KnowledgeCollection]:
    document = _document(
        "docs", "capacity.txt", " ".join(f"a{index}" for index in range(hit_count))
    )
    hits = tuple(
        _scripted_hit(document, f"a{index}", float(hit_count - index), ("broad",))
        for index in range(hit_count)
    )
    state = _ScriptedSearchState(hits, [])
    if raise_capacity:
        factory = lambda: _RaisingCapacityScriptedSearchIndex(state)
    elif capacity is None:
        factory = lambda: _ScriptedSearchIndex(state)
    else:
        factory = lambda: _CapacityScriptedSearchIndex(state, capacity)
    collection = KnowledgeCollection(index_factory=factory)  # type: ignore[arg-type]
    collection.sync(FakeAdapter("docs", (document,)))
    return state, collection
```

Use this subclass in tests that need oversampling. Add these exact contracts:

```python
def test_search_bounds_oversampling_by_advertised_backend_capacity() -> None:
    state, collection = _scripted_collection(capacity=10, hit_count=10)

    results = collection.search("broad", limit=5)

    assert len(results) == 5
    assert state.search_calls == [("broad", 10)]


def test_search_without_backend_capacity_preserves_requested_limit() -> None:
    state, collection = _scripted_collection(capacity=None, hit_count=10)

    collection.search("broad", limit=5)

    assert state.search_calls == [("broad", 5)]
```

Add subclasses whose `max_search_results` returns `True`, `5.0`, `0`, or raises
`RuntimeError`; assert each receives exactly `limit=5` and does not expose the
metadata failure:

```python
@pytest.mark.parametrize("capacity", [True, 5.0, 0])
def test_search_ignores_malformed_backend_capacity(capacity: object) -> None:
    state, collection = _scripted_collection(capacity=capacity, hit_count=10)

    collection.search("broad", limit=5)

    assert state.search_calls == [("broad", 5)]


def test_search_ignores_raising_backend_capacity() -> None:
    state, collection = _scripted_collection(
        capacity=None, hit_count=10, raise_capacity=True
    )

    collection.search("broad", limit=5)

    assert state.search_calls == [("broad", 5)]
```

Keep `limit=101` validation unchanged.

Give the existing `_RawIsolationIndex` diagnostic fixture an advertised capacity
without changing its `diagnose()` method:

```python
@property
def max_search_results(self) -> int:
    return 100
```

Its regression must continue to assert `search_limits == [12]`,
`diagnose_limits == [3]`, and the unchanged raw diagnostic ranks and scores.

- [ ] **Step 3: Write failing built-in capacity and clone tests**

In each backend test module construct limits with `max_results=10` and assert:

```python
assert index.max_search_results == 10
assert index.clone().max_search_results == 10
```

Use `ChunkIndexLimits`, `SemanticChunkIndexLimits`, `HybridChunkIndexLimits`,
and `RerankerLimits` respectively. For the hybrid fixture configure
`max_candidates_per_backend=10`, `max_fusion_entries=20`, and
`candidate_depth=10`; for reranking configure `candidate_depth=10` and
`max_candidates=10`.

- [ ] **Step 4: Run capacity tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_search_diversification.py \
  tests/test_knowledge_collection.py \
  tests/test_knowledge_diagnostics.py \
  tests/test_knowledge_retrieval.py \
  tests/test_semantic_retrieval.py \
  tests/test_hybrid_retrieval.py \
  tests/test_reranking.py -q
```

Expected: failures show `search_candidate_depth()` lacks the capacity argument,
built-ins lack `max_search_results`, and collection still requests `4K` from
capacity-less backends.

- [ ] **Step 5: Implement capacity-aware depth**

Change the selector helper to:

```python
def search_candidate_depth(limit: int, *, backend_capacity: object = None) -> int:
    _validate_limit(limit)
    if (
        type(backend_capacity) is not int
        or backend_capacity <= 0
        or limit > backend_capacity
    ):
        return limit
    return min(
        limit * SEARCH_CANDIDATE_MULTIPLIER,
        MAX_SEARCH_CANDIDATES,
        backend_capacity,
    )
```

In `KnowledgeCollection.search()`, isolate optional property access:

```python
try:
    backend_capacity = getattr(self._index, "max_search_results", None)
except Exception:
    backend_capacity = None
candidate_depth = search_candidate_depth(
    limit, backend_capacity=backend_capacity
)
```

Do not catch or retry exceptions from `self._index.search()`.

- [ ] **Step 6: Add the built-in properties**

Add the same read-only shape to each built-in class, returning its own immutable
limits object:

```python
@property
def max_search_results(self) -> int:
    """Return the configured maximum accepted final search limit."""

    return self._limits.max_results
```

Do not add this property to the mandatory `ChunkIndex` protocol.

- [ ] **Step 7: Run capacity tests and verify GREEN**

Run the Step 4 command again.

Expected: all selected tests pass; valid `max_results=10`, `limit=5` searches
request 10, custom indexes without valid capacity request 5, and clone capacity
tests pass.

- [ ] **Step 8: Commit the capacity fix**

```bash
git add src/nexusmind/search_diversification.py src/nexusmind/knowledge_collection.py src/nexusmind/knowledge_retrieval.py src/nexusmind/semantic_retrieval.py src/nexusmind/hybrid_retrieval.py src/nexusmind/reranking.py tests/test_search_diversification.py tests/test_knowledge_collection.py tests/test_knowledge_diagnostics.py tests/test_knowledge_retrieval.py tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py tests/test_reranking.py
git commit -m "fix: respect retrieval backend result capacity"
```

### Task 2: Replace the score range with a robust MAD cutoff

**Files:**
- Modify: `src/nexusmind/search_diversification.py`
- Modify: `tests/test_search_diversification.py`
- Modify: `tests/test_search_diversification_benchmark.py`
- Regenerate: `evals/knowledge/diversification.md`

- [ ] **Step 1: Write the failing outlier regression**

Add:

```python
def test_high_score_outlier_does_not_admit_near_zero_cross_document_results() -> None:
    ranked = candidates(
        ("a", 1000.0),
        ("a", 1.0),
        ("a", 1.0),
        ("a", 1.0),
        ("a", 1.0),
        ("b", 0.001),
        ("c", 0.0005),
        ("d", 0.0001),
    )

    assert select_document_aware_indices(ranked, limit=5) == (0, 1, 2, 3, 4)
```

Keep the existing smooth-distribution test expecting `(0, 1, 2, 5, 6)` for
scores `10, 9, 8, 7, 6, 5.5, 5`. Retain equal-score, negative-score,
determinism, and positive-affine tests.

- [ ] **Step 2: Run the selector tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_diversification.py -q
```

Expected: the new outlier test fails because the current max-to-worst range
admits indices 5, 6, and 7.

- [ ] **Step 3: Implement lower-median MAD**

Remove `RELEVANCE_WINDOW_FACTOR`. Add:

```python
def _lower_median(values: tuple[float, ...]) -> float:
    ordered = tuple(sorted(values))
    return ordered[(len(ordered) - 1) // 2]
```

Replace the floor calculation with:

```python
scores = tuple(item.score for item in raw_top_k)
center = _lower_median(scores)
robust_span = _lower_median(tuple(abs(score - center) for score in scores))
relevance_floor = min(scores) - robust_span
```

All inputs are already validated finite exact floats; do not add absolute score
constants or backend-type branches.

- [ ] **Step 4: Run selector tests and verify GREEN**

Run the Step 2 command again.

Expected: all selector tests pass, including the outlier and smooth examples.

- [ ] **Step 5: Tighten the Crypto benchmark contract before regeneration**

Add to `test_broad_coverage_improves_without_precise_relevance_regression()`:

```python
crypto = next(item for item in report.cases if item.case_id == "broad-crypto")
assert set(crypto.raw_documents) == {"crypto-overview.md"}
assert set(crypto.diversified_documents) >= {
    "crypto-overview.md",
    "crypto-permissions.md",
}
assert crypto.raw_relevant_document_count == 1
assert crypto.diversified_relevant_document_count == 2
```

- [ ] **Step 6: Run benchmark tests and verify RED against the old report**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_diversification_benchmark.py -q
```

Expected: the case-level behavior passes with the MAD selector, while the
byte-for-byte checked report assertion fails because the report still contains
the previous Crypto ranking.

- [ ] **Step 7: Regenerate and inspect the benchmark**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m nexusmind.search_diversification_benchmark --write evals/knowledge/diversification.md
PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_diversification_benchmark.py -q
```

Expected: all benchmark tests pass; Crypto diversified results include
`crypto-permissions.md`; aggregate broad relevant-document coverage exceeds the
raw value; precise MRR and Recall@5 do not decrease.

- [ ] **Step 8: Commit the robust selector and report**

```bash
git add src/nexusmind/search_diversification.py tests/test_search_diversification.py tests/test_search_diversification_benchmark.py evals/knowledge/diversification.md
git commit -m "fix: use robust relevance window for diversification"
```

### Task 3: Document, verify, push, and respond to review

**Files:**
- Modify: `docs/architecture.md`
- Modify: `tests/test_project_metadata.py`
- Create temporarily: `/tmp/pr-113-review-response.md`

- [ ] **Step 1: Add failing architecture assertions**

Extend `test_search_and_diagnostics_ranking_contract_is_documented()`:

```python
assert "backend capacity" in architecture
assert "median absolute deviation" in architecture
```

- [ ] **Step 2: Run metadata tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_metadata.py -q
```

Expected: architecture does not yet contain the two phrases.

- [ ] **Step 3: Update architecture documentation**

State that oversampling is capped by optional backend capacity, missing custom
capacity falls back to K, and the relevance floor uses the lower median absolute
deviation of only the same query/backend raw Top-K scores. Preserve the existing
statement that diagnostics are not oversampled. Add this paragraph:

```text
Search oversampling is bounded by optional backend capacity; a custom backend
without valid capacity metadata receives the caller's original K. The selector's
same-query relevance floor uses the lower median absolute deviation of only the
raw Top-K backend scores, so isolated score outliers cannot widen the window.
Diagnostics bypass both capacity-based oversampling and final selection.
```

- [ ] **Step 4: Run metadata tests and verify GREEN**

Run the Step 2 command again.

Expected: all metadata tests pass.

- [ ] **Step 5: Run fresh full verification**

Run:

```bash
PYTHONPATH=src .venv/bin/python -c 'import sys,tomli; sys.modules["tomllib"]=tomli; import pytest; raise SystemExit(pytest.main(["-q"]))'
git diff origin/main..HEAD --check
git diff --check
git status --short
```

Expected: the full suite reaches 100% with only declared skips; both diff checks
are clean; only the architecture and metadata test remain uncommitted.

- [ ] **Step 6: Commit documentation**

```bash
git add docs/architecture.md tests/test_project_metadata.py
git commit -m "docs: explain diversification safety policy"
```

- [ ] **Step 7: Push the reviewed fixes**

```bash
git push origin codex/issue-110-diversify-search-results
```

- [ ] **Step 8: Post a precise PR review response**

Create `/tmp/pr-113-review-response.md` with:

```markdown
Addressed review `#pullrequestreview-5056049029` in the latest commits:

- backend oversampling now uses `min(4K, 100, max_search_results)` for built-ins; unknown or malformed third-party capacity conservatively falls back to K, and valid caller limits no longer fail because of internal oversampling
- relevance now uses a lower-median MAD window over the same query/backend raw Top-K; the `1000, 1, 1, 1, 1` outlier regression keeps the near-zero cross-document candidates out
- the unchanged authored Crypto corpus now diversifies from only `crypto-overview.md` to include `crypto-permissions.md`; the checked report and aggregate safeguards were regenerated

Verification: full pytest suite passed locally; waiting for the refreshed Windows 3.11/3.12/3.13 and portable-package checks.
```

Post it:

```bash
gh pr comment 113 --repo SwartzMss/NexusMind --body-file /tmp/pr-113-review-response.md
```

- [ ] **Step 9: Monitor the refreshed CI**

Poll:

```bash
gh pr checks 113 --repo SwartzMss/NexusMind
```

Expected: Windows Python 3.11, 3.12, 3.13, and Windows portable package all
finish with `pass`. If any fails, read that job's log before changing code.
