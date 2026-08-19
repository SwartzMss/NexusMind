# Embedding-Backed Semantic Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-neutral synchronous embeddings and a bounded brute-force semantic `ChunkIndex` that works through the existing Knowledge Runtime and evaluation path.

**Architecture:** Keep embedding validation and the OpenAI-compatible HTTP adapter in `embeddings.py`, and keep semantic indexing/cosine lifecycle logic in `semantic_retrieval.py`. The semantic index incrementally embeds new/replacement chunks, prepares complete candidate state before commit, and plugs into `KnowledgeCollection` only through the existing `index_factory`/`ChunkIndex` contracts. Embeddings remain unpersisted derived state and are rebuilt on restore.

**Tech Stack:** Python 3.11–3.13, frozen dataclasses, `Protocol`, synchronous `httpx.Client`, standard-library `math`, existing KnowledgeCollection/evaluation APIs, pytest and `httpx.MockTransport`.

---

### Task 1: Embedding contracts and validated vector value object

**Files:**
- Create: `src/nexusmind/embeddings.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing public-contract and vector tests**

Create tests for imports, protocol assignment, numeric normalization, isolation,
and every invalid vector class:

```python
from nexusmind import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingValidationError,
    EmbeddingVector,
)


class FixtureProvider:
    def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((index + 1, 1)) for index, _ in enumerate(texts))

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector((1, 1))


def test_embedding_contracts_are_public_and_numeric_values_become_floats() -> None:
    provider: EmbeddingProvider = FixtureProvider()
    vector = provider.embed_query("query")
    assert vector == EmbeddingVector((1.0, 1.0))
    assert all(type(value) is float for value in vector.values)
```

Parameterize rejection of non-tuple values, empty tuples, bool/non-real values,
NaN, positive/negative infinity, and all-zero vectors. Assert the frozen vector
cannot be mutated and input mutable aliases cannot be retained because only an
exact tuple is accepted.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q tests/test_embeddings.py
```

Expected: collection fails because the embedding symbols are absent.

- [ ] **Step 3: Implement minimal contracts and validation**

Define the public error hierarchy and value object:

```python
class EmbeddingError(Exception): ...
class EmbeddingValidationError(EmbeddingError): ...
class EmbeddingProviderError(EmbeddingError): ...


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or not self.values:
            raise EmbeddingValidationError("embedding values must be a non-empty tuple")
        normalized: list[float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmbeddingValidationError("embedding values must be real numbers")
            converted = float(value)
            if not isfinite(converted):
                raise EmbeddingValidationError("embedding values must be finite")
            normalized.append(converted)
        if not any(value != 0.0 for value in normalized):
            raise EmbeddingValidationError("embedding vector must be non-zero")
        object.__setattr__(self, "values", tuple(normalized))
```

Add the synchronous `EmbeddingProvider` protocol with distinct batch-document
and single-query methods. Export all public symbols from the module and package
root.

- [ ] **Step 4: Run tests and verify GREEN**

Run `tests/test_embeddings.py`; confirm every vector case passes.

- [ ] **Step 5: Commit**

```bash
git add src/nexusmind/embeddings.py src/nexusmind/__init__.py tests/test_embeddings.py
git commit -m "feat: add embedding contracts"
```

### Task 2: Synchronous OpenAI-compatible embedding provider

**Files:**
- Modify: `src/nexusmind/embeddings.py`
- Modify: `src/nexusmind/__init__.py`
- Modify: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing mocked batching and request tests**

Use `httpx.MockTransport` to capture requests and return deliberately reversed
indexed data:

```python
def test_openai_provider_batches_documents_and_orders_by_response_index() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0, 1]},
            {"index": 0, "embedding": [1, 0]},
        ]})

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://provider.test/v1",
        api_key="secret-key",
        model="embed-model",
        transport=httpx.MockTransport(handler),
    )
    assert provider.embed_documents(("first", "second")) == (
        EmbeddingVector((1, 0)), EmbeddingVector((0, 1))
    )
    assert len(requests) == 1
    assert requests[0].url == httpx.URL("https://provider.test/v1/embeddings")
    assert json.loads(requests[0].content)["input"] == ["first", "second"]
```

Parse request JSON with `json.loads(request.content)`. Add a separate test that
`embed_query` sends one input and that `embed_documents(())` returns `()` with
no request.

- [ ] **Step 2: Write failing hostile-response and redaction tests**

Parameterize malformed top-level JSON, non-list `data`, wrong count, negative/
bool/out-of-range/missing/duplicate indexes, malformed vectors, inconsistent
dimensions, oversized response bytes, timeout, transport error, and HTTP 4xx/
5xx bodies containing the API key or query. Assert only controlled
`EmbeddingProviderError`/`EmbeddingValidationError` messages and verify secrets
and full input strings are absent from `str(error)`.

- [ ] **Step 3: Run focused tests and verify RED**

Expected: provider import/construction fails because it is not implemented.

- [ ] **Step 4: Implement the adapter and strict parser**

Implement plain-string configuration validation, positive finite timeout,
fixed conservative constants for maximum batch size, response bytes,
dimensions, and returned vector values, and a shared `_embed(texts)` helper.
Use:

```python
with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
    response = client.post(
        f"{self._base_url.rstrip('/')}/embeddings",
        headers={"Authorization": f"Bearer {self._api_key}"},
        json={"model": self._model, "input": list(texts)},
    )
```

Check `len(response.content)` before JSON parsing. Validate exact index coverage,
construct `EmbeddingVector` instances, require one dimension across the batch,
and order by numeric response index. Catch ordinary HTTP/JSON/provider-shape
exceptions, retain causes, and expose only stable redacted messages. Do not catch
`BaseException`.

- [ ] **Step 5: Run adapter tests and verify GREEN**

Run all `tests/test_embeddings.py`; no test may use live network or environment
API keys.

- [ ] **Step 6: Commit**

```bash
git add src/nexusmind/embeddings.py src/nexusmind/__init__.py tests/test_embeddings.py
git commit -m "feat: add OpenAI compatible embeddings"
```

### Task 3: Semantic index search, scores, and initial add lifecycle

**Files:**
- Create: `src/nexusmind/semantic_retrieval.py`
- Modify: `src/nexusmind/knowledge_retrieval.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `tests/test_semantic_retrieval.py`
- Modify: `tests/test_knowledge_retrieval.py`

- [ ] **Step 1: Write failing limit/export/empty-index tests**

Define a deterministic recording provider in the test file. Test public exports,
positive exact-int validation for every `SemanticChunkIndexLimits` field, empty
index and blank query returning `()` without provider calls, invalid query/
result arguments, and provider constructor validation.

- [ ] **Step 2: Write failing add and cosine-ranking tests**

Use authored mappings such as:

```python
documents = {"alpha": (1.0, 0.0), "diagonal": (1.0, 1.0), "opposite": (-1.0, 0.0)}
query = EmbeddingVector((1.0, 0.0))
```

Assert one batch document call, cosine scores `1.0`, approximately
`1 / sqrt(2)`, and `-1.0`, score domain, deterministic tie by `chunk_id`, limit
application, and `matched_terms == ()`.

- [ ] **Step 3: Verify RED**

Run `tests/test_semantic_retrieval.py`; expect missing semantic symbols.

- [ ] **Step 4: Make `SearchHit` diagnostics default-empty**

Change only:

```python
@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...] = ()
```

Run existing BM25 retrieval tests immediately to prove lexical behavior remains
unchanged.

- [ ] **Step 5: Implement limits, add, and search**

Define semantic errors (`SemanticChunkIndexError`,
`SemanticChunkIndexLimitError`, `SemanticDimensionError`,
`SemanticEmbeddingError`) and frozen limits. Validate the provider exposes both
callables. Implement strict provider-result trust boundaries: exact tuple/count
for documents and `EmbeddingVector` instances for all vectors.

For add, preserve exact-duplicate idempotency, batch only new chunk contents,
validate chunk/resource/dimension state, build candidate maps, and assign only
after success. For search, validate limits before provider calls, skip empty/
blank searches, enforce query dimension, compute with `math.fsum` and
`math.hypot`, clamp rounding to `[-1, 1]`, sort and return `SearchHit` objects.

- [ ] **Step 6: Run semantic and lexical tests**

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q \
  tests/test_semantic_retrieval.py tests/test_knowledge_retrieval.py
```

- [ ] **Step 7: Commit**

```bash
git add src/nexusmind/semantic_retrieval.py src/nexusmind/knowledge_retrieval.py src/nexusmind/__init__.py tests/test_semantic_retrieval.py tests/test_knowledge_retrieval.py
git commit -m "feat: add in-memory semantic search"
```

### Task 4: Semantic replacement, removal, clone, dimensions, and atomic failures

**Files:**
- Modify: `src/nexusmind/semantic_retrieval.py`
- Modify: `tests/test_semantic_retrieval.py`

- [ ] **Step 1: Write failing lifecycle and clone tests**

Cover replacement removing stale chunks, empty replacement removal, document
ownership errors, removal isolation, exact chunk identity conflicts, and clone
independence. Use a falsey provider object to prove constructor/clone use
explicit `is None` semantics. Assert provider instance sharing is documented and
only mutable index state diverges.

- [ ] **Step 2: Write failing dimension and resource tests**

Cover first commit dimension, mixed dimensions in one batch, dimension drift on
add, replacement and query, dimension persistence when replacing the only
document, reset after explicit empty state, maximum dimensions, chunk/content/
per-document/result/query limits, and total vector values. Every limit must
reject bool/zero configuration values.

- [ ] **Step 3: Write failing atomic-provider tests**

Use providers that raise, return non-tuples, wrong counts/types, zero/malformed
vectors, or change dimension. Start from searchable old state, capture exact
hits/scores, attempt add/replace, and assert the entire old result remains equal.
Verify outer error messages exclude sentinel secret/query/content text and chain
the underlying exception.

- [ ] **Step 4: Verify RED**

Run exact new lifecycle tests; confirm failures reflect missing methods and
validation rather than fixture errors.

- [ ] **Step 5: Implement candidate lifecycle**

Factor focused helpers for chunk validation, embedding batches, candidate
dimension/resource validation, and one `_commit_candidate`. Reuse existing
vectors for exact chunks; batch new replacement chunks once; enforce committed
dimension until the state becomes explicitly empty; reset dimension on empty
commit. Remove must not call the provider. Clone shares provider/limits and
copies every mutable dict/set/vector mapping and accounting field.

- [ ] **Step 6: Run tests and verify GREEN**

Run semantic tests plus existing collection-facing retrieval tests. Confirm
provider/vector/resource failures are atomic.

- [ ] **Step 7: Commit**

```bash
git add src/nexusmind/semantic_retrieval.py tests/test_semantic_retrieval.py
git commit -m "feat: complete semantic index lifecycle"
```

### Task 5: KnowledgeCollection, restore, SQLite, and semantic evaluation

**Files:**
- Modify: `tests/test_knowledge_collection.py`
- Modify: `tests/test_knowledge_snapshot.py`
- Modify: `tests/test_knowledge_store.py`
- Create: `tests/test_retrieval_evaluation_semantic.py`
- Create: `evals/knowledge/semantic/corpus/android.md`
- Create: `evals/knowledge/semantic/corpus/trustzone.md`
- Create: `evals/knowledge/semantic/corpus/qnx.md`
- Create: `evals/knowledge/semantic/corpus/cryptography.md`
- Create: `evals/knowledge/semantic/cases.json`
- Create: `evals/knowledge/semantic/baseline.md`

- [ ] **Step 1: Write semantic collection sync/provenance tests**

Inject `InMemorySemanticChunkIndex` through `index_factory`. Use a deterministic
provider mapping chunk content and natural-language queries to concept vectors.
Assert sync batches chunks, search returns canonical `KnowledgeSearchResult`,
chunk offset/content coherence remains valid, and failed provider/dimension
sync preserves the prior collection snapshot and search results.

- [ ] **Step 2: Write restore and persistence-boundary tests**

Restore canonical Chinese/English Documents into a semantic collection and
assert the current provider re-embeds them. Use two provider configurations to
prove restore uses current runtime configuration. Save/load with fresh
`SQLiteKnowledgeSnapshotStore` instances, restore, and search. Extend the exact
schema assertion only if needed; it must remain metadata/sources/documents with
no embedding/vector fields or tables.

- [ ] **Step 3: Author the offline semantic fixture before observing metrics**

Write four short original documents and 8–12 labels with intentionally low
lexical overlap but unambiguous canonical relevance. Freeze labels before
running the provider. The deterministic test provider maps authored concepts,
not exact result ranks, to small vectors and records distinct document/query
method calls.

- [ ] **Step 4: Write failing real-path evaluation tests**

Construct:

```python
collection = KnowledgeCollection(
    chunker=TextChunker(chunk_size=240, overlap=40),
    index_factory=lambda: InMemorySemanticChunkIndex(
        embedding_provider=provider
    ),
)
collection.sync(LocalDirectoryAdapter(CORPUS, source_id="semantic-docs"))
report = evaluate_retrieval(collection, cases, k=3)
```

Assert repeat equality, separate provider document/query calls, case count,
metric ranges, semantic `matched_terms == ()`, canonical targets/provenance, and
fixture coverage. Do not hard-code exact aggregate metrics as a release gate.

- [ ] **Step 5: Run integration/evaluation tests and record baseline**

Run the new tests, audit every missed/later-ranked relevance case for false
negatives, then record six-decimal Hit@3/Recall@3/MRR, fixed vectors/chunker/k,
reproduction command, descriptive non-gate policy, and explicit statement that
fixture vectors validate plumbing rather than real model quality.

- [ ] **Step 6: Verify existing fixtures are untouched and green**

```bash
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q \
  tests/test_knowledge_collection.py tests/test_knowledge_snapshot.py \
  tests/test_knowledge_store.py tests/test_retrieval_evaluation.py \
  tests/test_retrieval_evaluation_dataset.py \
  tests/test_retrieval_evaluation_baseline.py \
  tests/test_retrieval_evaluation_cjk.py \
  tests/test_retrieval_evaluation_semantic.py
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_knowledge_collection.py tests/test_knowledge_snapshot.py tests/test_knowledge_store.py tests/test_retrieval_evaluation_semantic.py evals/knowledge/semantic
git commit -m "test: add semantic retrieval integration baseline"
```

Do not stage collection/store production files unless a failing integration test
proves a minimal orchestration correction is required.

### Task 6: Documentation, full verification, and Draft PR

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-19-semantic-retrieval-design.md` only for material implementation discrepancies

- [ ] **Step 1: Document the public semantic path**

Explain provider query/document distinction, OpenAI-compatible configuration,
synchronous latency/cost, brute-force cosine and `[-1,1]` scores, backend-specific
score meaning, dimension/resource invariants, provider sharing, non-persisted
embeddings, restart re-embedding, fixture limitations, and deferred hybrid/RRF.
Include a minimal constructor example using `index_factory`.

- [ ] **Step 2: Run compile and complete focused verification**

```bash
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src python -m pytest -q \
  tests/test_embeddings.py tests/test_semantic_retrieval.py \
  tests/test_knowledge_retrieval.py tests/test_knowledge_collection.py \
  tests/test_knowledge_snapshot.py tests/test_knowledge_store.py \
  tests/test_retrieval_evaluation.py tests/test_retrieval_evaluation_dataset.py \
  tests/test_retrieval_evaluation_baseline.py \
  tests/test_retrieval_evaluation_cjk.py \
  tests/test_retrieval_evaluation_semantic.py \
  tests/test_openai_compatible_model.py
```

Use Python 3.11–3.13 for completion evidence. If only the unsupported shared
Python 3.10 venv exists locally, report it precisely and rely on PR Actions for
the supported full matrix rather than claiming unsupported full verification.

- [ ] **Step 3: Run full suite and scope checks**

```bash
PYTHONPATH=src python -m pytest -q
git diff --check
git status --short
git diff --stat origin/main...HEAD
```

Confirm there are no BM25/CJK fixture changes, schema migrations, new numerical
dependencies, hybrid fusion, persistent vectors/cache, ANN, or RAG scope.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-19-semantic-retrieval-design.md
git commit -m "docs: explain semantic retrieval"
```

- [ ] **Step 5: Publish**

Push `agent/issue-73-semantic-retrieval` and open a Draft PR against `main` with
`Closes #73`, semantic score/persistence notes, fixture metrics, and exact local
verification evidence. Preserve the worktree for review follow-up.
