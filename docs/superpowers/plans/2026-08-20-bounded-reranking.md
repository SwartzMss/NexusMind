# Bounded Second-Stage Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, provider-neutral reranking decorator and a deterministic fourth benchmark backend without changing canonical state or existing retrieval algorithms.

**Architecture:** A new `reranking.py` module owns the protocol, limits, controlled errors, lifecycle decorator, and coherence checks. The decorator clone-then-commits mutations to one base index and performs one fixed-depth base search before invoking a search-only reranker. The offline benchmark composes this decorator over Hybrid-RRF and retains the existing evaluator and pure renderer boundaries.

**Tech Stack:** Python 3.11–3.13, frozen dataclasses, typing protocols, pytest, existing KnowledgeCollection/retrieval evaluation stack.

---

### Task 1: Public reranking contracts and construction bounds

**Files:**
- Create: `src/nexusmind/reranking.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_reranking.py`

- [ ] **Step 1: Write failing contract tests**

Add tests that import `RerankerLimits`, `RerankedChunkIndex`, and the three controlled errors; reject `0`, negatives, floats, strings, and booleans for each limit and `candidate_depth`; reject non-callable factories and rerankers; reject depths above `max_candidates`; wrap factory failures; and reject base indexes without the complete cloneable lifecycle.

```python
@pytest.mark.parametrize("value", [0, -1, 1.0, True, "1"])
def test_reranker_limits_require_positive_plain_integers(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RerankerLimits(max_candidates=value)  # type: ignore[arg-type]
```

- [ ] **Step 2: Verify RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py`

Expected: collection fails because `nexusmind.reranking` does not exist.

- [ ] **Step 3: Implement minimal public contracts**

Create the error hierarchy, `Reranker` protocol, frozen keyword-only limits dataclass, and constructor validation. Construct exactly one base index through the factory, validate `add`, `replace_document`, `remove_document`, `search`, and `clone`, and store the immutable configuration.

```python
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: tuple[SearchHit, ...], *, limit: int
    ) -> tuple[SearchHit, ...]: ...

@dataclass(frozen=True, slots=True, kw_only=True)
class RerankerLimits:
    max_query_chars: int = 1_024
    max_candidates: int = 100
    max_total_candidate_chars: int = 1_000_000
    max_results: int = 100
```

- [ ] **Step 4: Verify GREEN and exports**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py tests/test_knowledge_retrieval.py`

Expected: all selected tests pass and package-root imports resolve.

- [ ] **Step 5: Commit contracts**

```bash
git add src/nexusmind/reranking.py src/nexusmind/__init__.py tests/test_reranking.py
git commit -m "feat: add bounded reranking contracts"
```

### Task 2: Fixed-candidate search and fail-closed coherence

**Files:**
- Modify: `src/nexusmind/reranking.py`
- Modify: `tests/test_reranking.py`

- [ ] **Step 1: Write failing search-bound tests**

Test strict query and result validation, query-length rejection before base/provider work, exactly one base search at `candidate_depth`, exact base tuple, base candidate count, total candidate characters, valid `SearchHit` fields, and duplicate base IDs. Use recording fakes so each assertion proves whether base and reranker work occurred.

```python
hits = index.search("bounded query", limit=2)
assert base.search_calls == [("bounded query", 5)]
assert reranker.calls == [("bounded query", base.hits, 2)]
assert len(hits) == 2
```

- [ ] **Step 2: Verify RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py -k 'search or candidate or query'`

Expected: failures identify the missing search/coherence implementation.

- [ ] **Step 3: Implement bounded search**

Validate caller-controlled bounds, call base search once with `candidate_depth`, validate and snapshot the exact candidate tuple, then call the reranker once with `min(limit, len(candidates))`. Catch private base/provider exceptions separately and raise stable `RerankerError` messages with exception chaining only.

- [ ] **Step 4: Write failing reranker-output tests**

Cover non-tuple output, too many results, non-`SearchHit` members, ghost IDs, duplicate IDs, same ID with conflicting chunk, changed `matched_terms`, non-float/NaN/infinite scores, provider exceptions, and private-text redaction. Add explicit empty/short underrun cases and prove no candidate fill or fallback occurs.

- [ ] **Step 5: Implement coherence and deterministic ordering**

Map results to the immutable candidate snapshot by `chunk_id`, require exact canonical chunk and matched terms, reconstruct hits from canonical data with reranker scores, and sort by `(-score, first_stage_rank, chunk_id)`. Return the shorter validated tuple unchanged in cardinality.

- [ ] **Step 6: Verify GREEN**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py`

Expected: all search, coherence, tie, underrun, and redaction tests pass.

- [ ] **Step 7: Commit search behavior**

```bash
git add src/nexusmind/reranking.py tests/test_reranking.py
git commit -m "feat: validate fixed reranking candidates"
```

### Task 3: Clone isolation and atomic lifecycle delegation

**Files:**
- Modify: `src/nexusmind/reranking.py`
- Modify: `tests/test_reranking.py`
- Create: `tests/test_reranked_knowledge_collection.py`

- [ ] **Step 1: Write failing lifecycle tests**

Prove `clone()` returns an independently owned base while sharing the deterministic reranker, rejects aliased/broken clones, and wraps clone failures without provider details. For add/replace/remove, prove the wrapper mutates a clone and swaps only on success; a failed base mutation must preserve prior search results.

- [ ] **Step 2: Verify RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py -k 'clone or add or replace or remove or atomic'`

Expected: lifecycle tests fail because delegation is incomplete.

- [ ] **Step 3: Implement lifecycle delegation**

Add a `_clone_base()` validation helper and `_mutate(method, *args)` clone-then-commit helper. Build wrapper clones with `object.__new__`, copy immutable configuration, share the reranker, and assign only the independent cloned base.

- [ ] **Step 4: Add collection integration tests**

Build a real `KnowledgeCollection` around `RerankedChunkIndex`; sync and search authored documents; verify ordered results retain exact canonical source/document/chunk provenance. Snapshot and restore through an independently constructed reranked collection and round-trip the snapshot through `SQLiteKnowledgeSnapshotStore`, proving no reranker/index state enters persistence.

- [ ] **Step 5: Verify GREEN**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py tests/test_reranked_knowledge_collection.py tests/test_knowledge_store.py`

Expected: lifecycle, provenance, restore, and SQLite tests pass.

- [ ] **Step 6: Commit lifecycle integration**

```bash
git add src/nexusmind/reranking.py tests/test_reranking.py tests/test_reranked_knowledge_collection.py
git commit -m "test: verify reranked index lifecycle"
```

### Task 4: Offline deterministic reranker and fourth benchmark backend

**Files:**
- Modify: `src/nexusmind/retrieval_benchmark.py`
- Modify: `tests/test_retrieval_benchmark.py`
- Modify: `tests/test_retrieval_benchmark_report.py`
- Modify: `evals/knowledge/benchmark.md`

- [ ] **Step 1: Write failing fixture and composition tests**

Require a deterministic content-driven benchmark reranker, no query/candidate mutation, stable tie behavior, and backend declaration order `BM25-only`, `Semantic-only`, `Hybrid-RRF`, `Hybrid-RRF + Rerank`. Assert all four collections have exactly equal canonical snapshots and reranking cannot return a chunk outside Hybrid's fixed candidate prefix.

- [ ] **Step 2: Verify RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_benchmark.py`

Expected: failures show the missing fourth backend.

- [ ] **Step 3: Implement the offline fixture**

Add a stateless `BenchmarkReranker` that derives finite scores only from case-folded query and candidate content using fixed authored concept groups. Compose it as `RerankedChunkIndex(base_index_factory=<hybrid factory>, candidate_depth=100, reranker=BenchmarkReranker())`; do not reference case IDs, labels, paths, or expected answers.

- [ ] **Step 4: Regenerate and byte-check the report**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m nexusmind.retrieval_benchmark --write evals/knowledge/benchmark.md`

Then test `render_retrieval_comparison(...).encode("utf-8") == REPORT.read_bytes()` and assert the fourth backend appears in overall, category, and diagnostics sections.

- [ ] **Step 5: Verify GREEN**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_retrieval_benchmark.py tests/test_retrieval_benchmark_report.py`

Expected: benchmark integration and byte-level reproduction tests pass.

- [ ] **Step 6: Commit benchmark**

```bash
git add src/nexusmind/retrieval_benchmark.py tests/test_retrieval_benchmark.py tests/test_retrieval_benchmark_report.py evals/knowledge/benchmark.md
git commit -m "feat: benchmark bounded reranking"
```

### Task 5: Documentation, roadmap, and complete verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ROADMAP.md` if present; otherwise update the existing roadmap section in `README.md`
- Add: `docs/superpowers/specs/2026-08-20-bounded-reranking-design.md`
- Add: `docs/superpowers/plans/2026-08-20-bounded-reranking.md`

- [ ] **Step 1: Document the stable boundary**

Explain fixed first-stage candidates, all four bounds, candidate depth ownership, canonical candidate preservation, replacement score semantics, stable ties, explicit shorter underrun, fail-closed errors, clone/atomicity behavior, and non-persisted reranker state. Update the benchmark description to four backends and state that metrics are descriptive. Mark Retrieval Runtime v1 complete and name user-facing Knowledge Base/Workspace APIs as next.

- [ ] **Step 2: Run focused compatibility suites**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py tests/test_reranked_knowledge_collection.py tests/test_hybrid_retrieval.py tests/test_semantic_retrieval.py tests/test_knowledge_collection.py tests/test_knowledge_store.py tests/test_retrieval_benchmark.py tests/test_retrieval_benchmark_report.py`

Expected: all focused tests pass.

- [ ] **Step 3: Run the full offline suite**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q`

Expected: all tests pass on the current supported interpreter.

- [ ] **Step 4: Check repository hygiene**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intentional issue #79 files are changed.

- [ ] **Step 5: Commit documentation and planning artifacts**

```bash
git add README.md docs evals/knowledge/benchmark.md
git commit -m "docs: describe bounded reranking"
```

- [ ] **Step 6: Push and open a draft PR**

Push `agent/issue-79-bounded-reranking`, confirm no existing PR for the branch, then open a draft PR targeting `main` with `Closes #79`, implementation summary, exact verification evidence, benchmark interpretation, and an explicit Windows CI checkbox.
