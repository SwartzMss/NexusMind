# Unicode/CJK Lexical Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit deterministic lexical analyzer boundary, make Unicode/CJK analysis the default for BM25, and measure it against the legacy analyzer through the real Knowledge Runtime.

**Architecture:** Put the analyzer protocol and two immutable implementations in a focused module. Inject an analyzer into `InMemoryChunkIndex`, validate every analyzer result, and rebuild candidate BM25 statistics before atomically committing mutations. Keep analyzer/token state derived and add a separate authored CJK evaluation fixture that compares both analyzers without creating a quality gate.

**Tech Stack:** Python 3.11+, `unicodedata`, frozen dataclasses, `Protocol`, existing BM25/index/KnowledgeCollection/evaluation APIs, pytest.

---

### Task 1: Public analyzer contract and deterministic implementations

**Files:**
- Create: `src/nexusmind/lexical_analysis.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_lexical_analysis.py`

- [ ] **Step 1: Write failing analyzer contract and behavior tests**

Add tests that import all three public symbols and assert the wished-for API:

```python
from nexusmind import (
    LexicalAnalyzer,
    UnicodeCJKLexicalAnalyzer,
    WhitespaceLexicalAnalyzer,
)


def test_whitespace_analyzer_preserves_legacy_split_and_casefold() -> None:
    analyzer: LexicalAnalyzer = WhitespaceLexicalAnalyzer()
    assert analyzer.analyze(" Android Binder,  STRAßE\n") == (
        "android", "binder,", "strasse"
    )


def test_unicode_analyzer_normalizes_punctuation_width_and_case() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()
    assert analyzer.analyze("Ａｎｄｒｏｉｄ Binder， IPC １２") == (
        "android", "binder", "ipc", "12"
    )


def test_unicode_analyzer_emits_han_bigrams_and_singletons() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()
    assert analyzer.analyze("知识库安全检索") == (
        "知识", "识库", "库安", "安全", "全检", "检索"
    )
    assert analyzer.analyze("安") == ("安",)


def test_unicode_analyzer_separates_mixed_scripts_deterministically() -> None:
    analyzer = UnicodeCJKLexicalAnalyzer()
    expected = ("android", "14", "支持", "binder", "通信")
    assert analyzer.analyze("Android 14支持Binder通信") == expected
    assert analyzer.analyze("Android 14支持Binder通信") == expected
    assert all(type(token) is str and token for token in expected)
```

Also parameterize representative Han extension boundary code points and combining/punctuation boundaries so the documented classification is executable.

- [ ] **Step 2: Run the analyzer tests and verify RED**

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_lexical_analysis.py
```

Expected: collection fails because `LexicalAnalyzer`, `WhitespaceLexicalAnalyzer`, and `UnicodeCJKLexicalAnalyzer` are not exported.

- [ ] **Step 3: Implement the minimal analyzer module**

Create frozen stateless analyzers and explicit helpers:

```python
from dataclasses import dataclass
from typing import Protocol
import unicodedata


class LexicalAnalyzer(Protocol):
    def analyze(self, text: str) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class WhitespaceLexicalAnalyzer:
    def analyze(self, text: str) -> tuple[str, ...]:
        if type(text) is not str:
            raise TypeError("text must be a string")
        return tuple(term.casefold() for term in text.split())


@dataclass(frozen=True, slots=True)
class UnicodeCJKLexicalAnalyzer:
    def analyze(self, text: str) -> tuple[str, ...]:
        if type(text) is not str:
            raise TypeError("text must be a string")
        normalized = unicodedata.normalize("NFKC", text)
        # Flush non-Han L/N runs as casefolded tokens and Han runs as
        # overlapping bigrams (or the single code point for length one).
```

Implement `_is_han` with explicit audited ranges rather than locale or an external dependency. Export the new symbols from both the module and package root.

- [ ] **Step 4: Run analyzer tests and verify GREEN**

Run the focused test file and confirm all cases pass with no warnings.

- [ ] **Step 5: Commit the analyzer boundary**

```bash
git add src/nexusmind/lexical_analysis.py src/nexusmind/__init__.py tests/test_lexical_analysis.py
git commit -m "feat: add Unicode CJK lexical analyzers"
```

### Task 2: Analyzer-driven BM25 indexing and query analysis

**Files:**
- Modify: `src/nexusmind/knowledge_retrieval.py`
- Modify: `tests/test_knowledge_retrieval.py`

- [ ] **Step 1: Write failing index configuration and matching tests**

Add tests proving default Unicode behavior, explicit legacy behavior, identical index/query semantics, and unchanged BM25 token matching:

```python
def test_default_analyzer_matches_across_punctuation_and_han_bigrams() -> None:
    index = InMemoryChunkIndex()
    index.add((_chunk("chunk", "Android Binder，提供安全检索"),))
    hit = index.search("Binder 安全检索")[0]
    assert hit.matched_terms == ("binder", "安全", "全检", "检索")


def test_explicit_whitespace_analyzer_preserves_legacy_matching() -> None:
    index = InMemoryChunkIndex(analyzer=WhitespaceLexicalAnalyzer())
    index.add((_chunk("chunk", "Android Binder,"),))
    assert index.search("Binder") == ()
    assert index.search("Binder,")


def test_query_term_limit_counts_raw_analyzed_tokens_before_deduplication() -> None:
    index = InMemoryChunkIndex(limits=ChunkIndexLimits(max_query_terms=2))
    with pytest.raises(ChunkIndexLimitError, match="max_query_terms"):
        index.search("安全检索")  # 安全, 全检, 检索
    with pytest.raises(ChunkIndexLimitError, match="max_query_terms"):
        index.search("term term term")
```

Add a small spy analyzer test showing chunk content and query both pass through the configured analyzer.

- [ ] **Step 2: Run focused tests and verify RED**

Run the new tests by exact node IDs. Confirm failure is caused by the missing `analyzer=` constructor parameter/default behavior and pre-analysis query limit.

- [ ] **Step 3: Inject and validate analyzer output**

Change the constructor to:

```python
def __init__(
    self,
    *,
    limits: ChunkIndexLimits | None = None,
    analyzer: LexicalAnalyzer | None = None,
) -> None:
    ...
    self._analyzer = analyzer or UnicodeCJKLexicalAnalyzer()
```

Add `_analyze(text)` that calls the configured analyzer, requires an exact tuple of non-empty plain strings, and returns an isolated tuple. Route `_statistics_for` and `search` through it. Check `max_query_chars` before calling it and `max_query_terms` against its raw result before stable deduplication. Keep the BM25 formula and sort key byte-for-byte equivalent.

- [ ] **Step 4: Run retrieval and analyzer suites and verify GREEN**

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_lexical_analysis.py tests/test_knowledge_retrieval.py
```

Update only expectations whose lexical terms intentionally change under the new default; use `WhitespaceLexicalAnalyzer()` in tests that specifically assert legacy whitespace behavior.

- [ ] **Step 5: Commit analyzer-backed BM25 behavior**

```bash
git add src/nexusmind/knowledge_retrieval.py tests/test_knowledge_retrieval.py
git commit -m "feat: analyze BM25 corpus and queries consistently"
```

### Task 3: Fail-closed atomic mutations and clone semantics

**Files:**
- Modify: `tests/test_knowledge_retrieval.py`
- Modify: `src/nexusmind/knowledge_retrieval.py`

- [ ] **Step 1: Write failing hostile-analyzer tests**

Create analyzers that raise on selected content or return invalid values. Assert add and replace leave searchable old state unchanged:

```python
class SelectiveAnalyzer:
    def analyze(self, text: str) -> tuple[str, ...]:
        if "broken" in text:
            raise RuntimeError("analysis failed")
        return tuple(text.casefold().split())


def test_failed_analysis_does_not_commit_add_or_replacement() -> None:
    index = InMemoryChunkIndex(analyzer=SelectiveAnalyzer())
    old = _chunk("old", "stable", "doc-1")
    index.add((old,))
    with pytest.raises(ChunkIndexError, match="analyzer"):
        index.replace_document("doc-1", (_chunk("new", "broken", "doc-1"),))
    assert index.search("stable")[0].chunk == old
```

Parameterize invalid returns: list instead of tuple, empty token, non-string token, and a `str` subclass if exact plain-string validation is chosen. Add a stateful spy test showing `clone()` keeps equivalent analyzer behavior but mutations of clone index state do not affect the original.

- [ ] **Step 2: Run hostile tests and verify RED**

Confirm failures expose raw exceptions/invalid state or missing analyzer preservation, not test setup errors.

- [ ] **Step 3: Complete the atomic candidate-state boundary**

Introduce a controlled `LexicalAnalysisError(ChunkIndexError)` for analyzer exceptions and invalid output. Make `_statistics_for` an instance method using `_analyze`; ensure `_commit_candidate` computes every derived structure before assigning any instance field. Construct clones with `analyzer=self._analyzer` and copy derived mutable maps/counters independently.

- [ ] **Step 4: Verify mutations, statistics, and clone**

Run all retrieval tests, including existing add/replace/remove BM25-statistic assertions. Confirm failed candidate analysis preserves old scores and chunks.

- [ ] **Step 5: Commit fail-closed ownership behavior**

```bash
git add src/nexusmind/knowledge_retrieval.py src/nexusmind/__init__.py tests/test_knowledge_retrieval.py
git commit -m "feat: keep analyzed index mutations atomic"
```

### Task 4: Restore and persistence-boundary integration

**Files:**
- Modify: `tests/test_knowledge_snapshot.py`
- Modify: `tests/test_knowledge_store.py`

- [ ] **Step 1: Write restore and schema tests first**

Add a snapshot test that restores Chinese canonical content into a collection configured with a Unicode/CJK index and successfully searches a Han substring. Add a contrast using the whitespace analyzer. Extend the existing SQLite schema assertion to require exactly `knowledge_store_metadata`, `sources`, and `documents`, and assert no analyzer/tokens/postings columns or tables exist.

- [ ] **Step 2: Run the new integration tests and verify their initial result**

The Unicode restore test should already pass once Task 2 is complete; to preserve TDD value, first write it before any collection/persistence modification. If it passes immediately, that proves no production change is needed at this layer. The schema test should also pass and serves as an explicit invariant rather than authorization to change persistence.

- [ ] **Step 3: Make only required integration corrections**

If either test fails, change only the collection/index wiring needed to retain the caller-configured index during clone/restore. Do not add snapshot fields, database tables, or schema migrations.

- [ ] **Step 4: Run snapshot and store suites**

```bash
PYTHONPATH=src python -m pytest -q tests/test_knowledge_snapshot.py tests/test_knowledge_store.py
```

- [ ] **Step 5: Commit persistence invariant coverage**

```bash
git add tests/test_knowledge_snapshot.py tests/test_knowledge_store.py src/nexusmind/knowledge_collection.py
git commit -m "test: preserve lexical persistence boundaries"
```

Only stage `knowledge_collection.py` if a production correction was actually required.

### Task 5: Authored CJK comparative evaluation

**Files:**
- Create: `evals/knowledge/cjk/corpus/android.md`
- Create: `evals/knowledge/cjk/corpus/trustzone.md`
- Create: `evals/knowledge/cjk/corpus/qnx.md`
- Create: `evals/knowledge/cjk/corpus/cryptography.md`
- Create: `evals/knowledge/cjk/cases.json`
- Create: `evals/knowledge/cjk/baseline.md`
- Create: `tests/test_retrieval_evaluation_cjk.py`

- [ ] **Step 1: Author corpus and relevance labels independently of results**

Write short original Chinese technical prose for the four documents and 8–12 natural queries, including paraphrased Chinese and mixed Latin terms such as Android/Binder, TrustZone, QNX, AES-GCM. Label each case at `(source_id="cjk-docs", logical_path)` level before running either analyzer.

- [ ] **Step 2: Write the failing real-runtime comparison test**

Build two collections with identical `LocalDirectoryAdapter`, `TextChunker`, limits, and BM25 parameters, varying only analyzer. Load cases with `load_retrieval_evaluation_cases` and call `evaluate_retrieval`. Assert:

```python
assert unicode_first == unicode_second
assert len(unicode_first.case_results) == len(cases)
assert all(0.0 <= metric <= 1.0 for metric in (...))
assert unicode_first.hit_at_k > whitespace.hit_at_k
assert any(
    new.hit_at_k > old.hit_at_k
    for new, old in zip(unicode_first.case_results, whitespace.case_results)
)
```

Do not assert exact aggregate metric values.

- [ ] **Step 3: Run the comparison and verify RED for the intended reason**

Confirm the fixture or integration helper is missing, then add the minimal helper. If labels expose no difference, improve the authored lexical challenge rather than tuning BM25 or changing relevance after observing rankings.

- [ ] **Step 4: Record descriptive metrics**

Run the deterministic comparison, capture both reports, and write their six-decimal metrics to `baseline.md` with cutoff, chunker, BM25, analyzer configurations, reproduction command, and bigram limitations.

- [ ] **Step 5: Verify fixture determinism and commit**

```bash
PYTHONPATH=src python -m pytest -q tests/test_retrieval_evaluation_cjk.py
git add evals/knowledge/cjk tests/test_retrieval_evaluation_cjk.py
git commit -m "test: add CJK retrieval evaluation baseline"
```

### Task 6: Focused documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-19-unicode-cjk-lexical-analyzer-design.md` only if implementation reveals a material discrepancy

- [ ] **Step 1: Document the public configuration**

Add a focused README section showing default Unicode/CJK behavior and explicit compatibility configuration:

```python
index = InMemoryChunkIndex()  # NFKC + Unicode words + Han bigrams
legacy = InMemoryChunkIndex(analyzer=WhitespaceLexicalAnalyzer())
```

Explain that query term limits apply after analysis and before deduplication, analyzer/tokens are derived runtime state, restore rebuilds with current configuration, and CJK baseline numbers are descriptive.

- [ ] **Step 2: Run compile and focused retrieval verification on supported Python**

```bash
python -m compileall -q src tests
PYTHONPATH=src python -m pytest -q \
  tests/test_lexical_analysis.py \
  tests/test_knowledge_retrieval.py \
  tests/test_knowledge_collection.py \
  tests/test_knowledge_snapshot.py \
  tests/test_knowledge_store.py \
  tests/test_retrieval_evaluation.py \
  tests/test_retrieval_evaluation_dataset.py \
  tests/test_retrieval_evaluation_baseline.py \
  tests/test_retrieval_evaluation_cjk.py
```

Use Python 3.11, 3.12, or 3.13. Do not treat the local Python 3.10 venv as a valid project verification environment.

- [ ] **Step 3: Run the full suite**

```bash
PYTHONPATH=src python -m pytest -q
```

Expected: all tests pass on a supported interpreter. If only Python 3.10 is locally available, report that limitation and use GitHub Actions for the supported-version matrix before claiming completion.

- [ ] **Step 4: Check patch hygiene and scope**

```bash
git diff --check
git status --short
git diff --stat origin/main...HEAD
```

Confirm there are no BM25 parameter changes, persistence migrations, external NLP dependencies, or unrelated edits.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-19-unicode-cjk-lexical-analyzer-design.md
git commit -m "docs: explain Unicode CJK lexical analysis"
```

Stage the spec only if it was updated.

- [ ] **Step 6: Publish for review**

Use the repository's publish workflow to push `agent/issue-71-unicode-cjk-analyzer` and open a Draft PR against `main` with `Closes #71`, a summary of analyzer semantics and persistence invariants, the comparative CJK metrics, and exact verification evidence.
