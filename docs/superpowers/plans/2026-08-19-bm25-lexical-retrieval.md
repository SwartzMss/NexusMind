# BM25 Lexical Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace distinct substring-match scoring in `InMemoryChunkIndex` with deterministic token-based BM25 ranking and coherent derived corpus statistics.

**Architecture:** Keep the public `ChunkIndex` protocol unchanged and evolve `SearchHit.score` to `float`. Cache normalized term frequencies and token lengths in the in-memory index; each successful bounded mutation atomically swaps candidate chunk maps and statistics rebuilt from the candidate corpus.

**Tech Stack:** Python 3.11+, standard-library `collections.Counter` and `math.log`, pytest, existing Knowledge snapshot/SQLite boundaries.

---

### Task 1: Implement analyzer and BM25 scoring

**Files:**
- Modify: `tests/test_knowledge_retrieval.py`
- Modify: `src/nexusmind/knowledge_retrieval.py`

- [ ] **Step 1: Write failing analyzer, score, and ranking tests**

Update old integer-score assertions to float semantics and add focused tests:

```python
def test_bm25_single_term_formula_and_score_type() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("only", "term"),))
    hit = index.search("term")[0]
    assert type(hit.score) is float
    assert math.isfinite(hit.score)
    assert hit.score == pytest.approx(math.log(4 / 3))


def test_matching_is_whitespace_token_based() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("substring", "concatenate"), _chunk("token", "cat")))
    assert [hit.chunk.chunk_id for hit in index.search("cat")] == ["token"]


def test_repeated_term_frequency_increases_score_at_equal_length() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("repeated", "term term filler"), _chunk("single", "term filler filler")))
    hits = {hit.chunk.chunk_id: hit for hit in index.search("term")}
    assert hits["repeated"].score > hits["single"].score


def test_rarer_term_has_higher_idf_contribution() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("one", "rare common"), _chunk("two", "other common")))
    assert index.search("rare")[0].score > index.search("common")[0].score


def test_shorter_chunk_scores_higher_at_equal_term_frequency() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("short", "term filler"), _chunk("long", "term filler filler filler")))
    hits = {hit.chunk.chunk_id: hit for hit in index.search("term")}
    assert hits["short"].score > hits["long"].score
```

Also assert duplicate normalized query terms do not change score or `matched_terms`, casefold remains deterministic, and exact BM25 ties use ascending `chunk_id`.

- [ ] **Step 2: Run retrieval tests to verify RED**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py`

Expected: FAIL because scores are integers, matching uses substrings, and TF/DF/length do not affect ranking.

- [ ] **Step 3: Implement normalized statistics and BM25 search**

In `knowledge_retrieval.py`, import `Counter` and `log`, define `_BM25_K1 = 1.2` and `_BM25_B = 0.75`, and change the score annotation:

```python
@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...]
```

Initialize `_term_frequencies`, `_token_counts`, `_document_frequencies`, and `_total_tokens`. Add a rebuild helper:

```python
def _statistics_for(self, chunks: dict[str, Chunk]):
    term_frequencies = {}
    token_counts = {}
    document_frequencies = Counter()
    total_tokens = 0
    for chunk_id, chunk in chunks.items():
        tokens = tuple(token.casefold() for token in chunk.content.split())
        frequencies = Counter(tokens)
        term_frequencies[chunk_id] = frequencies
        token_counts[chunk_id] = len(tokens)
        document_frequencies.update(frequencies.keys())
        total_tokens += len(tokens)
    return term_frequencies, token_counts, document_frequencies, total_tokens
```

Search deduplicates normalized query terms, uses cached TF/DF/length values, and computes:

```python
idf = log(1 + (chunk_count - df + 0.5) / (df + 0.5))
normalizer = tf + _BM25_K1 * (
    1 - _BM25_B + _BM25_B * chunk_length / average_length
)
term_score = idf * tf * (_BM25_K1 + 1) / normalizer
```

Only matched token terms contribute. Sort with `(-hit.score, hit.chunk.chunk_id)`.

- [ ] **Step 4: Run retrieval tests to verify GREEN**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py`

Expected: all retrieval tests pass.

- [ ] **Step 5: Commit analyzer and scoring**

```bash
git add src/nexusmind/knowledge_retrieval.py tests/test_knowledge_retrieval.py
git commit -m "feat: add BM25 lexical scoring"
```

### Task 2: Make mutation statistics atomic and clone-independent

**Files:**
- Modify: `tests/test_knowledge_retrieval.py`
- Modify: `src/nexusmind/knowledge_retrieval.py`

- [ ] **Step 1: Write failing mutation-statistics tests**

Add ranking assertions that prove:

```python
# add changes N/DF and therefore a prior hit's score
assert score_after_add != score_before_add

# replacement removes the old term and its length/DF contribution
assert index.search("old") == ()
assert index.search("new")

# removal changes rarity statistics for remaining chunks
assert score_after_remove < score_before_remove

# clone initially ranks identically, then mutation changes only the clone
assert clone.search("term") == original.search("term")
```

Extend failed add/replacement tests to snapshot complete search results before the exception and assert equality afterward.

- [ ] **Step 2: Run mutation tests to verify RED**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py -k 'statistics or clone or failed'`

Expected: mutation ranking assertions fail until every path commits rebuilt statistics and clone copies them.

- [ ] **Step 3: Atomically commit candidate maps and statistics**

Add `_commit_candidate(chunks, document_chunks, total_chars)` that calls `_statistics_for(chunks)` before assigning any field, then assigns all chunk, document, character, TF, length, DF, and token-total fields together.

For `add()`, copy current chunk/document maps, apply validated additions to candidates, then commit. For `replace_document()`, build candidates without the old IDs and with replacements, then commit. For `remove_document()`, build candidates without the removed IDs, then commit. Preserve all existing preflight checks before `_commit_candidate()`.

Clone copies all mutable statistics:

```python
clone._term_frequencies = {
    chunk_id: frequencies.copy()
    for chunk_id, frequencies in self._term_frequencies.items()
}
clone._token_counts = self._token_counts.copy()
clone._document_frequencies = self._document_frequencies.copy()
clone._total_tokens = self._total_tokens
```

- [ ] **Step 4: Run full retrieval tests**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py`

Expected: all retrieval tests pass, including existing resource bounds and identity conflicts.

- [ ] **Step 5: Commit atomic mutation state**

```bash
git add src/nexusmind/knowledge_retrieval.py tests/test_knowledge_retrieval.py
git commit -m "fix: keep BM25 statistics mutation-safe"
```

### Task 3: Preserve Knowledge provenance and persistence behavior

**Files:**
- Modify: `tests/test_knowledge_collection.py`
- Modify: `tests/test_knowledge_snapshot.py`
- Modify: `tests/test_knowledge_store.py`

- [ ] **Step 1: Write failing float-score compatibility assertions**

Update collection score assertions to type/relationship checks and verify the resolved result preserves the exact backend score object value:

```python
result = collection.search("checkpoint resume")[0]
assert type(result.hit.score) is float
assert result.hit.score > collection.search("checkpoint")[0].hit.score
```

For snapshot/restore, capture `(chunk_id, score, matched_terms)` before snapshot and assert the restored collection produces `pytest.approx()`-equivalent scores in the same order. Extend the SQLite restart test with finite float score and canonical provenance assertions.

- [ ] **Step 2: Run Knowledge and persistence tests to verify RED**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_collection.py tests/test_knowledge_snapshot.py tests/test_knowledge_store.py`

Expected: old exact integer score assertions fail under BM25 until updated to the new contract.

- [ ] **Step 3: Make only compatibility updates required by the float contract**

Keep production `KnowledgeCollection`, snapshot, and SQLite schema code unchanged. Replace brittle integer score expectations with `type(score) is float`, ranking relations, or `pytest.approx()` while retaining provenance, chunk coherence, ordering, and matched-term assertions.

- [ ] **Step 4: Run Knowledge and persistence tests**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_collection.py tests/test_knowledge_snapshot.py tests/test_knowledge_store.py`

Expected: all selected tests pass without production persistence changes.

- [ ] **Step 5: Commit compatibility coverage**

```bash
git add tests/test_knowledge_collection.py tests/test_knowledge_snapshot.py tests/test_knowledge_store.py
git commit -m "test: cover BM25 restore compatibility"
```

### Task 4: Document BM25 and verify the repository

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the lexical baseline**

Document the analyzer, formula, `k1=1.2`, `b=0.75`, positive IDF, score/tie ordering, mutation-derived statistics, restore rebuild behavior, and whitespace-tokenization limitation for languages without whitespace word boundaries. State that persistent postings, language analyzers, semantic retrieval, and RAG remain out of scope.

- [ ] **Step 2: Run full verification**

Run: `PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q`

Expected: complete suite passes with zero failures.

Run: `git diff --check`

Expected: exit 0 with no output.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs: describe BM25 lexical retrieval"
```

- [ ] **Step 4: Review branch scope**

Run: `git diff --stat origin/main...HEAD && git diff --check origin/main...HEAD`

Expected: only issue #67 spec, plan, retrieval implementation, focused tests, and README changes; no whitespace errors.
