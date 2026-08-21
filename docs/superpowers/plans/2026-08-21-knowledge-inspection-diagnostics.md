# Knowledge Inspection and Retrieval Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable, structured Python APIs that inspect KnowledgeBase state and chunks and explain BM25, semantic, Hybrid-RRF, and reranked retrieval decisions.

**Architecture:** Keep `KnowledgeBase` as the product boundary, add inspection-domain values in a focused module, and add backend-neutral retrieval trace values beside `SearchHit`. `KnowledgeCollection` owns canonical validation and provenance resolution; built-in indexes compute search hits and trace rows in the same single pass so diagnostics never change algorithms or execute providers twice.

**Tech Stack:** Python 3.10, frozen/slotted dataclasses, enums and protocols, pytest, existing in-memory retrieval backends and SQLite snapshot store.

---

### Task 1: Retrieval diagnostic contracts and BM25 trace

**Files:**
- Modify: `src/nexusmind/knowledge_retrieval.py`
- Modify: `src/nexusmind/__init__.py`
- Modify: `tests/test_knowledge_retrieval.py`

- [ ] **Step 1: Write failing public-contract tests**

Add imports for `RetrievalStage`, `RetrievalCandidateDiagnostic`,
`RetrievalDiagnostics`, and `DiagnosticChunkIndex`. Add tests that construct the
frozen values, reject mutation, and prove `InMemoryChunkIndex.diagnose()` returns
the exact final hits produced by `search()` plus one-based lexical rows:

```python
diagnostics = index.diagnose("binder uid", limit=2)
assert diagnostics.hits == index.search("binder uid", limit=2)
assert [item.stage for item in diagnostics.candidates] == [
    RetrievalStage.LEXICAL,
    RetrievalStage.LEXICAL,
]
assert [item.rank for item in diagnostics.candidates] == [1, 2]
assert [item.score for item in diagnostics.candidates] == [
    hit.score for hit in diagnostics.hits
]
assert all(item.rrf_contribution is None for item in diagnostics.candidates)
assert all(item.selected for item in diagnostics.candidates)
```

Also cover empty queries, `limit=1` with more than one corpus match (only
returned candidates are traced), finite-float score validation, exact tuples,
positive one-based ranks, `selected` as an exact bool, and optional finite
positive `rrf_contribution`.

- [ ] **Step 2: Run the tests and verify the missing contracts fail**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py
```

Expected: FAIL during import because retrieval diagnostic contracts and
`InMemoryChunkIndex.diagnose()` do not exist.

- [ ] **Step 3: Add strict backend-neutral diagnostic values**

In `knowledge_retrieval.py`, define and export:

```python
class RetrievalStage(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    FUSION = "fusion"
    RERANKER = "reranker"


@dataclass(frozen=True, slots=True)
class RetrievalCandidateDiagnostic:
    stage: RetrievalStage
    rank: int
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()
    rrf_contribution: float | None = None
    selected: bool = False

    def __post_init__(self) -> None:
        if type(self.stage) is not RetrievalStage:
            raise TypeError("stage must be RetrievalStage")
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not isinstance(self.chunk, Chunk):
            raise TypeError("chunk must be a Chunk")
        if type(self.score) is not float or not isfinite(self.score):
            raise ValueError("score must be a finite float")
        if type(self.matched_terms) is not tuple or any(
            type(term) is not str for term in self.matched_terms
        ):
            raise TypeError("matched_terms must be a tuple of strings")
        if self.rrf_contribution is not None and (
            type(self.rrf_contribution) is not float
            or not isfinite(self.rrf_contribution)
            or self.rrf_contribution <= 0.0
        ):
            raise ValueError("rrf_contribution must be a finite positive float")
        if type(self.selected) is not bool:
            raise TypeError("selected must be a boolean")


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    hits: tuple[SearchHit, ...]
    candidates: tuple[RetrievalCandidateDiagnostic, ...]


class DiagnosticChunkIndex(ChunkIndex, Protocol):
    def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics: ...
```

Validate exact tuple element types in `RetrievalDiagnostics.__post_init__`.
Import `Enum` and `isfinite`, and export all four names from package
`__init__.py`.

- [ ] **Step 4: Refactor BM25 into one computation with two projections**

Create a private `_search(query, *, limit, include_diagnostics)` helper that
retains all current validation, analysis, scoring, and sorting exactly. Keep
ordinary search as:

```python
def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
    return self._search(query, limit=limit, include_diagnostics=False).hits

def diagnose(self, query: str, *, limit: int = 10) -> RetrievalDiagnostics:
    return self._search(query, limit=limit, include_diagnostics=True)
```

After slicing final hits, build lexical candidates with rank from the final hit
tuple, the exact `SearchHit` score/terms/chunk, no RRF contribution, and
`selected=True`. Do not perform a second analysis or scoring pass.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_retrieval.py tests/test_lexical_analysis.py
```

Expected: PASS.

Commit:

```bash
git add -- src/nexusmind/knowledge_retrieval.py src/nexusmind/__init__.py tests/test_knowledge_retrieval.py
git commit -m "feat: add lexical retrieval diagnostics"
```

### Task 2: Semantic and Hybrid-RRF traces

**Files:**
- Modify: `src/nexusmind/semantic_retrieval.py`
- Modify: `src/nexusmind/hybrid_retrieval.py`
- Modify: `tests/test_semantic_retrieval.py`
- Modify: `tests/test_hybrid_retrieval.py`

- [ ] **Step 1: Write failing semantic single-pass tests**

Use a recording embedding provider and assert one query embedding call for
`diagnose()`, exact equality between diagnostic hits and ordinary search from an
equivalent cloned index, semantic ranks/scores, no matched terms or RRF
contributions, and empty diagnostics for blank/no-corpus queries:

```python
trace = index.diagnose("binder", limit=2)
assert provider.query_calls == ["binder"]
assert trace.hits == expected_index.search("binder", limit=2)
assert [(row.stage, row.rank, row.score) for row in trace.candidates] == [
    (RetrievalStage.SEMANTIC, rank, hit.score)
    for rank, hit in enumerate(trace.hits, start=1)
]
```

- [ ] **Step 2: Write failing exact Hybrid-RRF contribution tests**

Extend scripted child indexes with call recording. Call `diagnose()` and assert
each child is searched exactly once at the existing backend depth, followed by
fusion rows. For every child rank assert:

```python
assert row.rrf_contribution == pytest.approx(1.0 / (rrf_k + row.rank))
```

Assert fusion score equals the sum of matching lexical and semantic
contributions, child rows are marked selected only when the chunk survives the
requested final limit, fusion rows are all selected, and row ordering is
lexical ranks, semantic ranks, then fusion ranks. Retain tests for duplicate or
incoherent child hits and candidate bounds.

- [ ] **Step 3: Run tests and verify missing methods fail**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py
```

Expected: FAIL because semantic and hybrid indexes lack `diagnose()`.

- [ ] **Step 4: Implement semantic diagnostics without a second provider call**

Refactor semantic search into a private helper returning
`RetrievalDiagnostics`; preserve current validation, cosine clamping, sorting,
and limits. Build one semantic row for each returned hit. `search()` returns
`.hits`; `diagnose()` returns the whole value. Do not call `embed_query()` more
than once per public operation.

- [ ] **Step 5: Implement Hybrid-RRF diagnostics from the existing child hits**

Factor the current dual search and fusion block into a private helper. During
the existing two child loops, retain `rank`, exact backend score, terms, and:

```python
contribution = float(1.0 / (self._rrf_k + rank))
scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
```

For diagnostic mode, emit lexical/semantic child rows with that contribution,
then fusion rows for the sliced final hits. Mark child rows selected by
membership in final chunk IDs. Do not require child `diagnose()` support and do
not change custom child compatibility.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py tests/test_hybrid_knowledge_collection.py
```

Expected: PASS.

Commit:

```bash
git add -- src/nexusmind/semantic_retrieval.py src/nexusmind/hybrid_retrieval.py tests/test_semantic_retrieval.py tests/test_hybrid_retrieval.py
git commit -m "feat: explain semantic and hybrid retrieval"
```

### Task 3: Reranker trace over a fixed diagnostic base

**Files:**
- Modify: `src/nexusmind/reranking.py`
- Modify: `tests/test_reranking.py`
- Modify: `tests/test_reranked_knowledge_collection.py`

- [ ] **Step 1: Write failing fixed-candidate diagnostic tests**

Create a recording base implementing both `search()` and `diagnose()`. Assert
ordinary search still calls `base.search()` once, while diagnostic search calls
`base.diagnose()` once at `candidate_depth` and never calls `base.search()`.
Verify preserved base rows followed by reranker rows:

```python
trace = index.diagnose("binder uid", limit=2)
assert base.diagnose_calls == [("binder uid", candidate_depth)]
assert base.search_calls == []
assert trace.candidates[:len(base_trace.candidates)] == base_trace.candidates
assert [row.stage for row in trace.candidates[-2:]] == [
    RetrievalStage.RERANKER,
    RetrievalStage.RERANKER,
]
assert trace.hits == tuple(
    SearchHit(row.chunk, row.score, row.matched_terms)
    for row in trace.candidates[-2:]
)
```

Also test that base candidate rows have `selected` recomputed for final reranker
survival, underrun remains shorter, ties use first-stage rank then chunk ID,
unsupported base diagnostics fail before the reranker runs, and malformed base
trace hits must exactly equal its declared final candidate tuple.

- [ ] **Step 2: Run tests and verify the missing method fails**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py
```

Expected: FAIL because `RerankedChunkIndex.diagnose()` does not exist.

- [ ] **Step 3: Share reranker validation and projection**

Extract the existing reranker invocation, result coherence checks, canonical hit
reconstruction, and stable sorting into a private method accepting the already
validated base candidate tuple. Keep ordinary `search()` behavior byte-for-byte
equivalent.

Implement `diagnose()` by checking for a callable base `diagnose`, calling it
once at `candidate_depth`, validating `base_trace.hits` with existing bounds,
then invoking the shared reranker path. Rebuild base diagnostic rows with
`selected` based on final result IDs and append reranker rows in final rank
order. Wrap unsupported/malformed diagnostic bases in stable `RerankerError`
messages without provider text.

- [ ] **Step 4: Run reranking and collection regression tests and commit**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_reranking.py tests/test_reranked_knowledge_collection.py tests/test_retrieval_benchmark.py tests/test_retrieval_benchmark_report.py
```

Expected: PASS and benchmark report remains byte-identical.

Commit:

```bash
git add -- src/nexusmind/reranking.py tests/test_reranking.py tests/test_reranked_knowledge_collection.py
git commit -m "feat: trace bounded reranking decisions"
```

### Task 4: Canonical collection inspection and diagnostic provenance

**Files:**
- Create: `src/nexusmind/knowledge_inspection.py`
- Modify: `src/nexusmind/knowledge_collection.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_knowledge_inspection.py`
- Create: `tests/test_knowledge_diagnostics.py`

- [ ] **Step 1: Write failing chunk-inspection model and behavior tests**

Define expected imports for `KnowledgeChunkInspection` and
`KnowledgeDocumentInspection` from `knowledge_inspection`, plus
`KnowledgeRetrievalCandidateDiagnostic` and `KnowledgeRetrievalDiagnostics`
from `knowledge_collection`. Build a collection with a fixed small chunker,
sync metadata-bearing documents, and assert:

```python
inspection = collection.inspect_document(document.document_id, preview_chars=8)
assert inspection.source.source_id == document.source_id
assert inspection.document == document
assert [(item.ordinal, item.start_offset, item.end_offset) for item in inspection.chunks] == [
    (1, 0, 10),
    (2, 10, 20),
]
assert [item.preview for item in inspection.chunks] == ["abcdefgh", "klmnopqr"]
```

Mutate returned nested source/document metadata and prove a second inspection is
unchanged. Cover stable document iteration, empty documents, exact positive
integer `preview_chars`, unknown IDs, non-tuple chunker output, wrong document
IDs, duplicate chunk IDs, bad ordering/offsets, and content-slice mismatches.

- [ ] **Step 2: Write failing provenance-resolved diagnostic tests**

Create diagnostic and non-diagnostic fake indexes. Assert
`collection.diagnose_search()` invokes diagnose once, returns final
`KnowledgeSearchResult` equal to ordinary resolution semantics, and wraps every
candidate as:

```python
KnowledgeRetrievalCandidateDiagnostic(
    source=detached_source,
    document=detached_document,
    diagnostic=backend_row,
)
```

Test ghost, stale, duplicate, malformed-offset, content-conflicting, invalid
stage ordering, selected/final mismatch, and unsupported indexes. Confirm
failure returns no partial results and ordinary `search()` remains supported.

- [ ] **Step 3: Run tests and verify imports fail**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_inspection.py tests/test_knowledge_diagnostics.py
```

Expected: FAIL because the inspection module and collection methods do not
exist.

- [ ] **Step 4: Implement focused inspection-domain values**

Create `knowledge_inspection.py` with frozen/slotted chunk values:

```python
@dataclass(frozen=True, slots=True)
class KnowledgeChunkInspection:
    ordinal: int
    chunk_id: str
    start_offset: int
    end_offset: int
    character_count: int
    preview: str


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentInspection:
    source: KnowledgeSource
    document: Document
    chunks: tuple[KnowledgeChunkInspection, ...]
```

Define the two provenance-resolved values beside `KnowledgeSearchResult` in
`knowledge_collection.py`, avoiding a collection/inspection runtime cycle:

```python
@dataclass(frozen=True, slots=True)
class KnowledgeRetrievalCandidateDiagnostic:
    source: KnowledgeSource
    document: Document
    diagnostic: RetrievalCandidateDiagnostic


@dataclass(frozen=True, slots=True)
class KnowledgeRetrievalDiagnostics:
    query: str
    results: tuple[KnowledgeSearchResult, ...]
    candidates: tuple[KnowledgeRetrievalCandidateDiagnostic, ...]
```

Validate exact tuples and public field types in each value.

- [ ] **Step 5: Implement one canonical provenance resolver**

In `KnowledgeCollection`, extract current per-hit source/document/chunk checks
into `_resolve_hit(hit) -> KnowledgeSearchResult`. Use it from existing
`search()`, from final diagnostic hits, and from each candidate diagnostic.
Require exact one-to-one selected candidate/final hit IDs and coherent candidate
stage/rank ordering while permitting the same chunk in different stages.

- [ ] **Step 6: Implement deterministic document inspection**

Add `inspect_document(document_id, *, preview_chars=160)` and an internal stable
`inspect_documents()` helper. Locate canonical state without exposing internal
references, run the collection-owned chunker once per inspected document,
validate exact chunks against the canonical document, and return bounded
previews via `chunk.content[:preview_chars]`. Do not mutate `_index`, `_sources`,
or `_documents`.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_inspection.py tests/test_knowledge_diagnostics.py tests/test_knowledge_collection.py tests/test_knowledge_chunking.py
```

Expected: PASS.

Commit:

```bash
git add -- src/nexusmind/knowledge_inspection.py src/nexusmind/knowledge_collection.py src/nexusmind/__init__.py tests/test_knowledge_inspection.py tests/test_knowledge_diagnostics.py
git commit -m "feat: inspect canonical knowledge lifecycle"
```

### Task 5: Public KnowledgeBase inspection and diagnostics API

**Files:**
- Modify: `src/nexusmind/knowledge_inspection.py`
- Modify: `src/nexusmind/knowledge_base.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_knowledge_base_diagnostics.py`
- Modify: `tests/test_knowledge_base_sync.py`

- [ ] **Step 1: Write failing whole-base inspection tests**

Create a base with one unsynchronized registration, one synchronized non-empty
source, and one synchronized empty source. Assert `inspect()` returns stable
source/document order and:

```python
assert [(item.config.source_id, item.sync_status) for item in view.sources] == [
    ("empty", KnowledgeSourceSyncStatus.SYNCED),
    ("pending", KnowledgeSourceSyncStatus.REGISTERED),
    ("synced", KnowledgeSourceSyncStatus.SYNCED),
]
assert view.status == kb.status()
assert sum(item.document_count for item in view.sources) == len(view.documents)
assert sum(item.chunk_count for item in view.sources) == sum(
    item.chunk_count for item in view.documents
)
```

Assert document summaries omit full content, include detached metadata and
hash/count/type fields, and remain equivalent after `close()`/`open()` when
captured before close.

- [ ] **Step 2: Write failing product-wrapper tests**

Test `KnowledgeBase.inspect_document()` and `diagnose_search()` happy paths,
closed-handle guards, argument validation, custom non-diagnostic index behavior,
and error redaction. Patch lower layers to throw an exception containing a
private path/query and assert only stable public messages are visible.

- [ ] **Step 3: Run tests and verify public methods are absent**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base_diagnostics.py tests/test_knowledge_base_sync.py
```

Expected: FAIL because the new public values and methods do not exist.

- [ ] **Step 4: Add whole-base domain values**

Move the existing `KnowledgeBaseStatus` dataclass from `knowledge_base.py` into
`knowledge_inspection.py`, import and re-export it from `knowledge_base.py` so
existing imports (including `knowledge_base_ui.py`) remain compatible. Then add
and export:

```python
class KnowledgeSourceSyncStatus(str, Enum):
    REGISTERED = "registered"
    SYNCED = "synced"


@dataclass(frozen=True, slots=True)
class KnowledgeSourceInspection:
    config: RegisteredSourceConfig
    sync_status: KnowledgeSourceSyncStatus
    document_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentSummary:
    source_id: str
    document_id: str
    logical_path: str
    content_type: str
    content_hash: str
    metadata: dict[str, object]
    character_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class KnowledgeBaseInspection:
    status: KnowledgeBaseStatus
    sources: tuple[KnowledgeSourceInspection, ...]
    documents: tuple[KnowledgeDocumentSummary, ...]
```

Import `RegisteredSourceConfig` from `knowledge_base_manifest`, which does not
depend on inspection values. Validate counts as non-negative exact integers and
copy summary metadata at construction.

- [ ] **Step 5: Implement product-facing methods without persistence changes**

Add `KnowledgeBase.inspect()`, `inspect_document()`, and `diagnose_search()`.
Build inspection from the current manifest, collection snapshot, and validated
collection document inspections. Determine sync status only by membership in
canonical snapshot source IDs. Wrap lower-layer inspection/diagnostic failures
as stable `KnowledgeBaseSourceError("unable to inspect knowledge base")` and
`KnowledgeBaseSourceError("unable to diagnose knowledge search")`, preserving
existing public config/type errors where appropriate.

- [ ] **Step 6: Run KnowledgeBase suites and commit**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_knowledge_base_diagnostics.py tests/test_knowledge_base.py tests/test_knowledge_base_sync.py tests/test_knowledge_base_atomicity.py
```

Expected: PASS.

Commit:

```bash
git add -- src/nexusmind/knowledge_inspection.py src/nexusmind/knowledge_base.py src/nexusmind/__init__.py tests/test_knowledge_base_diagnostics.py tests/test_knowledge_base_sync.py
git commit -m "feat: expose knowledge diagnostics API"
```

### Task 6: Documentation, compatibility, and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-21-knowledge-inspection-diagnostics-design.md` only if implementation reveals a factual mismatch
- Test: all focused retrieval, collection, persistence, and KnowledgeBase suites

- [ ] **Step 1: Add public API examples and explicit boundaries**

Document `inspect()`, `inspect_document()`, and `diagnose_search()` with a small
Python example that renders structured fields without defining a CLI format.
Explain registered versus synced status, bounded previews, backend stages,
RRF contribution, reranker score, canonical provenance, custom-backend
unsupported behavior, no provider double execution, and non-persistence.

- [ ] **Step 2: Run formatting/static checks available in the project**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m compileall -q src tests
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 3: Run the complete relevant regression set outside the restricted sandbox**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q \
  tests/test_knowledge_retrieval.py \
  tests/test_semantic_retrieval.py \
  tests/test_hybrid_retrieval.py \
  tests/test_reranking.py \
  tests/test_knowledge_chunking.py \
  tests/test_knowledge_collection.py \
  tests/test_knowledge_store.py \
  tests/test_persistence.py \
  tests/test_knowledge_inspection.py \
  tests/test_knowledge_diagnostics.py \
  tests/test_knowledge_base.py \
  tests/test_knowledge_base_sync.py \
  tests/test_knowledge_base_atomicity.py \
  tests/test_knowledge_base_diagnostics.py \
  tests/test_retrieval_benchmark.py \
  tests/test_retrieval_benchmark_report.py
```

Expected: PASS. Run outside the restricted sandbox because an existing
checkpoint test proved thread/SQLite execution can hang inside it.

- [ ] **Step 4: Re-run the full suite and compare only against the recorded baseline**

Run:

```bash
/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q
```

Expected on this Linux environment: the same 38 pre-existing command-profile
failures caused by the Windows-only production guard, with no failures in files
changed by this issue and no new failures. Windows CI is the authoritative full
suite gate.

- [ ] **Step 5: Commit documentation**

```bash
git add -- README.md docs/superpowers/specs/2026-08-21-knowledge-inspection-diagnostics-design.md
git commit -m "docs: explain knowledge diagnostics"
```

- [ ] **Step 6: Review branch diff and publish a draft PR**

Inspect `git status --short`, `git diff --check`, and the complete
`origin/main...HEAD` diff. Push `agent/issue-85-knowledge-diagnostics`, confirm
there is no existing PR for that head, then create one draft PR targeting
`main` with `Closes #85`, a structured implementation summary, exact focused
test evidence, the recorded Linux baseline exception, and an explicit Windows
CI checkbox.
