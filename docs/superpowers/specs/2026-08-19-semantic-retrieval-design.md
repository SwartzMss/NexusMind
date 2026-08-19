# Embedding-Backed Semantic Retrieval Design

## Context and scope

NexusMind already has a synchronous `KnowledgeCollection` lifecycle and a
bounded BM25 `ChunkIndex`. Issue #73 adds semantic retrieval as a separate
provider-neutral backend while preserving that orchestration and the canonical
state boundary. It does not add hybrid fusion, persistent embeddings, ANN,
reranking, RAG, or a second semantic-specific collection type.

The semantic path is synchronous so it composes directly with the existing
`add`, `replace_document`, `remove_document`, `search`, `clone`, `sync`, and
`restore` contracts. Remote embedding latency and cost are explicit properties
of these synchronous calls.

## Embedding contracts

A focused `embeddings.py` module defines:

```python
class EmbeddingProvider(Protocol):
    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]: ...

    def embed_query(self, text: str) -> EmbeddingVector: ...
```

The two entry points remain distinct so asymmetric passage/query providers can
be represented. Providers are runtime configuration. Custom providers must be
safe to share between index clones; they need not remain deterministic across
real model revisions, while tests use deterministic fixture providers.

`EmbeddingVector` is a frozen, slotted value object. It accepts an exact tuple
of real numeric values, rejects booleans, converts accepted values to plain
floats, and requires a positive dimension, finite values, and a non-zero norm.
It retains no mutable input alias. Vector construction errors are controlled
embedding validation errors.

Embedding-layer exceptions distinguish invalid vectors/configuration from
provider transport/response failure. Public error text is stable and does not
include API keys, full input text, or raw response bodies.

## OpenAI-compatible provider

`OpenAICompatibleEmbeddingProvider` is a synchronous adapter for
`POST {base_url}/embeddings`. Construction requires non-empty plain-string
`base_url`, `api_key`, and `model`, plus a positive finite timeout. An optional
`httpx.BaseTransport` is injectable for offline tests.

`embed_documents` returns `()` without a request for an empty tuple and sends
one request for a non-empty tuple rather than one request per text.
`embed_query` sends a separate one-item request through the query-specific
method. Inputs must be exact strings; batching order is preserved by response
indexes, not response array order.

The adapter validates status, bounded response size, JSON shape, result count,
integer index coverage `0..n-1`, duplicate/missing indexes, vector shape, and
consistent dimensions within the response. It rejects empty, non-finite, zero,
or malformed vectors through `EmbeddingVector`. Transport, timeout, HTTP, JSON,
and provider-shape failures become `EmbeddingProviderError` with exception
chaining and redacted public messages. Tests use `httpx.MockTransport`; CI
requires no network or key.

The adapter has conservative fixed response/batch/vector bounds to prevent an
untrusted endpoint from amplifying a small request before the semantic index
can enforce its own retained-state limits. These are implementation safety
bounds, not model-quality parameters.

## Semantic index

A separate `semantic_retrieval.py` module defines
`SemanticChunkIndexLimits` and `InMemorySemanticChunkIndex`. The index
structurally implements the existing `ChunkIndex` lifecycle, so
`KnowledgeCollection` requires no semantic branch.

The limits are frozen positive plain integers covering:

- maximum chunks and chunks per document;
- total chunk-content characters;
- vector dimensions;
- total retained vector values;
- query characters; and
- returned results.

The implementation stores independent mutable maps for chunks,
document-to-chunk ownership, and chunk vectors, plus total content accounting
and the committed dimension. Python tuples are used for vectors; NumPy, FAISS,
and external vector stores are not introduced.

## Indexing and atomic mutation

Add validates chunk identity and resource preconditions, embeds only truly new
chunks in one document batch, and constructs a complete candidate state from
existing vectors plus new vectors. Replacement reuses an existing vector only
when the exact same chunk already exists; otherwise it batch-embeds replacement
chunks. Unchanged chunks belonging to other documents retain their vectors.
Removal does not call the provider.

Every provider call and vector/count/dimension/resource validation finishes
before any instance field is assigned. Provider exceptions are wrapped as a
controlled semantic-index embedding error without input leakage. Failed add,
replacement, removal preparation, sync staging, or restore leaves the previous
searchable/canonical state unchanged.

The first successful non-empty commit establishes the index dimension. All
new and replacement vectors must match it, even when replacing the only
document. Explicitly removing all chunks resets the dimension; a later add may
then establish a new one. Empty replacement has removal semantics and likewise
resets dimension only if no vectors remain. Query vectors must match the
committed dimension.

`clone()` shares the embedding provider instance and immutable limits while
copying chunks, ownership sets, vector mappings, dimension, and accounting.
Mutable semantic index state is therefore independent between original and
clone.

## Search and scores

Searching an empty index or a whitespace-only query returns `()` without
embedding the query or establishing a dimension. Otherwise the index validates
the plain-string query, query/result limits, obtains one query embedding through
`embed_query`, and enforces the committed dimension.

Cosine similarity is computed exactly by dot product divided by both norms.
The validated non-zero vectors make the denominator defined. Small floating
rounding excursions are clamped to `[-1.0, 1.0]`; scores are otherwise not
remapped. Every indexed chunk is a semantic candidate, including negative or
zero cosine results. Results use deterministic `score DESC, chunk_id ASC`
ordering and then apply the requested limit.

`SearchHit.matched_terms` gains a default empty tuple. BM25 continues supplying
its lexical diagnostics explicitly; semantic hits use `matched_terms == ()`.
Scores remain backend-specific: BM25 and cosine values have no defined absolute
cross-backend comparison.

## KnowledgeCollection and persistence

Callers inject semantic retrieval with an `index_factory` that creates a fresh
empty semantic index using the chosen provider and limits. Existing collection
sync staging, clone behavior, canonical Source/Document ownership, chunk
offset/content coherence checks, and `KnowledgeSearchResult` provenance remain
unchanged.

Embedding provider configuration, vectors, and semantic index state are
unpersisted runtime/derived state. `KnowledgeSnapshot` and SQLite schemas remain
unchanged. Restore and restart re-chunk canonical Documents and call the current
provider to rebuild vectors before atomically swapping collection state. The
associated latency, remote cost, and possible provider/model drift are explicit;
embedding caching/versioned persistence require a separate design.

## Deterministic evaluation

A small original fixture under `evals/knowledge/semantic/` contains authored
documents and canonical document-level labels with deliberately low lexical
overlap. Tests use a deterministic concept-vector provider that records the
document/query paths separately and produces low-dimensional validated vectors.

The fixture runs through:

```text
LocalDirectoryAdapter
  -> KnowledgeCollection
  -> TextChunker
  -> InMemorySemanticChunkIndex
  -> KnowledgeSearchResult provenance
  -> evaluate_retrieval
```

Repeated reports must be equal and metrics bounded. The checked-in baseline
records Hit@K, Recall@K, and MRR as descriptive plumbing data, not a release
threshold or evidence of external-model quality. Existing BM25 and CJK fixtures
are not changed to favor semantic retrieval.

## Testing strategy

Unit tests cover public contracts, numeric normalization, finite/non-empty/
non-zero vectors, provider batching and query/document separation, strict
mocked HTTP parsing/redaction, dimension consistency, cosine scores and ties,
empty-index behavior, semantic diagnostics, lifecycle operations, resource
bounds, provider sharing, clone independence, and atomic failure.

Integration tests cover semantic collection sync/search provenance, failed
sync/restore atomicity, restore re-embedding with current provider, SQLite
restart rebuild, unchanged canonical schema, and the deterministic real-path
evaluation fixture. Existing lexical retrieval and evaluation suites remain
regression coverage.

## Non-goals

This design does not implement hybrid/RRF or weighted fusion, score
normalization, reranking, metadata boosting, query rewriting, LLM judging,
FAISS/HNSW/pgvector, a vector database, persistent embeddings/cache/index,
ANN, NumPy optimization, semantic answer generation, RAG, Agent/Tool wiring,
or background indexing.
