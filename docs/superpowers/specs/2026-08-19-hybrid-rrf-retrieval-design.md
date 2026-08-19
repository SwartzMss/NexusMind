# Hybrid Retrieval with Reciprocal Rank Fusion

## Scope

`HybridChunkIndex` composes two collection-owned, cloneable chunk indexes and
structurally implements the existing `add`, `replace_document`,
`remove_document`, `search`, and `clone` lifecycle. The normal composition is
`InMemoryChunkIndex` plus `InMemorySemanticChunkIndex`, but the orchestration is
backend-neutral. Factories create fresh child indexes; pre-populated shared
children are outside the factory contract, and returning the same object for
both children is rejected.

Hybrid retrieval does not add filtering, reranking, query rewriting, fallback,
learned fusion, vector databases, persistent embeddings, RAG, or Agent wiring.

## RRF and score meaning

The backend uses standard 1-based Reciprocal Rank Fusion:

```text
RRF(d) = sum_i 1 / (rrf_k + rank_i(d))
```

`rrf_k` is a positive plain integer and defaults to 60. Only rank contributes;
raw BM25 and cosine magnitudes are never normalized, compared, or added. Final
`SearchHit.score` is the RRF score, ordered by `score DESC, chunk_id ASC`.

Fusion uses canonical `chunk_id`. Duplicate IDs inside a child result are
invalid. If both children return an ID, their `Chunk` values must be identical
or search fails closed. Lexical `matched_terms` are preserved exactly whenever
the lexical child returns the chunk; semantic-only results use `()`.

## Candidate and temporary bounds

`candidate_depth` is an explicit positive per-backend target. For a valid final
result limit, each child receives `max(limit, candidate_depth)`. Construction
requires candidate depth not to exceed `max_candidates_per_backend`; search
never silently truncates the configured depth. `HybridChunkIndexLimits` bounds
final results, candidates per backend, and the combined temporary child-hit
count (`max_fusion_entries`). Child results that exceed their requested limit,
or whose combined tuple lengths exceed the fusion bound, are rejected before
per-hit validation allocates duplicate-detection bookkeeping. Child indexes may
impose tighter result/query limits; an incompatible child configuration fails
predictably as a chained, redacted hybrid backend error.

## Ownership, mutation, and clone behavior

Every mutation clones both committed children, applies the operation to the two
candidates, and replaces both committed references only after both calls
succeed. Thus a later semantic/provider failure cannot retain an earlier
lexical mutation, and the reverse ordering is also atomic. Child error text is
not exposed through the public hybrid message.

`clone()` clones both children and rejects a child that returns itself or two
children that return the same candidate object. Mutation staging enforces the
same cross-child independence. Frozen limits and fusion configuration, child
factories, and child-specific immutable runtime configuration may be shared.
Mutable child corpus state is independent.

Search invokes lexical then semantic for the same query and is fail-closed: a
failure in either child returns no partial fusion result. Graceful degradation
is deferred because it needs an explicit correctness and observability policy.

## Collection and persistence boundary

`KnowledgeCollection` uses hybrid retrieval through its existing
`index_factory`; it has no hybrid-specific provenance path. Copy-on-write sync
staging composes with the hybrid's own two-child atomic mutation. Fused chunks
are resolved to canonical `KnowledgeSource` and `Document` values using the
existing offset/content coherence checks.

Sources and Documents remain the only canonical persisted state. Chunks,
lexical statistics, vectors, child indexes, ranks, and RRF state are derived and
are absent from snapshots and SQLite. Restore/restart creates fresh children and
rebuilds both indexes from canonical Documents, including current-runtime
embedding calls and their latency, cost, and possible model drift.

## Deterministic evaluation

`evals/knowledge/hybrid/` contains lexical-favored, semantic-favored, and
mixed-signal cases with one shared corpus and canonical document labels. Tests
run BM25-only, Semantic-only, and Hybrid-RRF through
`LocalDirectoryAdapter -> KnowledgeCollection -> evaluate_retrieval` using
offline authored vectors. The recorded Hit@K, Recall@K, and MRR values are a
descriptive non-gate plumbing baseline, not an external-model quality claim.
