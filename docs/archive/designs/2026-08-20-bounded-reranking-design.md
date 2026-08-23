# Bounded Second-Stage Reranking Design

> Historical design record. It describes the scope and assumptions at implementation time and is not the current product specification. See [the current architecture](../../architecture.md).

## Objective

Add a provider-neutral, deterministic, resource-bounded second retrieval stage that reranks only a fixed first-stage candidate set. The feature composes with every cloneable `ChunkIndex` without changing BM25, semantic retrieval, hybrid RRF, canonical knowledge state, or persistence formats.

## Public Contracts

`Reranker` is a search-only protocol:

```python
class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: tuple[SearchHit, ...],
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]: ...
```

`RerankedChunkIndex` is a lifecycle decorator constructed from a `base_index_factory`, a reranker, a composition-owned `candidate_depth`, and `RerankerLimits`. It implements `add`, `replace_document`, `remove_document`, `search`, and `clone`.

The controlled error hierarchy is:

- `RerankerError(ChunkIndexError)` for factory, clone, mutation, base-search, or provider failures.
- `RerankerLimitError(RerankerError)` for configured or per-request resource-bound violations.
- `RerankerCoherenceError(RerankerError)` for malformed or incoherent base/reranker output.

Public errors contain stable, source-neutral messages. They never include query text, candidate content, provider exception text, or other private provider details.

## Resource Bounds

`RerankerLimits` contains positive plain integers:

- `max_query_chars`
- `max_candidates`
- `max_total_candidate_chars`
- `max_results`

Boolean values are rejected even though `bool` subclasses `int`. `candidate_depth` is also a positive plain integer and cannot exceed `max_candidates`.

Before any reranker work, `search` validates the requested `limit`, query type and length, candidate tuple shape and count, candidate contents, unique identity, and total candidate characters. The base search is requested exactly once with `candidate_depth`; no later stage can fetch more candidates. The result limit passed to the reranker is `min(limit, len(candidates))`.

## Search and Coherence Semantics

The base index must return an exact tuple of valid, unique `SearchHit` values within `candidate_depth`. Each hit must contain a `Chunk`, a finite plain-float score, and an exact tuple of strings for `matched_terms`.

The reranker must return an exact tuple no longer than its requested result limit. Every returned hit must correspond to exactly one supplied candidate by `chunk_id`, and its `Chunk` and `matched_terms` must equal the supplied canonical values. Ghost IDs, duplicates, conflicting chunk data, invalid scores, or malformed fields fail closed.

The reranker owns only the second-stage score and ordering signal. The wrapper reconstructs final `SearchHit` objects from the canonical candidate snapshot, using the reranker score while preserving the candidate `Chunk` and `matched_terms`.

Final ordering is deterministic: descending reranker score, then ascending original first-stage rank, then ascending `chunk_id`. A reranker may return fewer results than requested. That underrun is returned exactly as a shorter tuple; the wrapper never fills from the original ranking and never falls back after a failure.

## Lifecycle, Cloning, and Atomicity

The wrapper exclusively owns one cloneable base index. The reranker is immutable, deterministic runtime configuration and is shared by clones; it never receives mutation calls and owns no canonical state.

`clone()` clones the base index and rejects aliasing or an incomplete lifecycle. Mutations use clone-then-commit: clone the base, apply the mutation to the candidate clone, and swap it into committed state only after success. Any factory, clone, base mutation, base search, reranker call, or coherence failure is wrapped and leaves committed state unchanged.

Because the canonical state remains exclusively in the base index, existing `KnowledgeCollection`, provenance, SQLite restore, and canonical snapshot semantics remain unchanged.

## Offline Deterministic Fixture

The benchmark reranker is authored offline and scores candidates from query/chunk content rather than case IDs or expected answers. It performs no network or mutable external I/O. Its deterministic lexical feature weights improve first-result ordering while respecting the fixed Hybrid-RRF candidate set.

The checked-in benchmark compares four backends in declaration order:

1. BM25-only
2. Semantic-only
3. Hybrid-RRF
4. Hybrid-RRF + Rerank

All use the same LF-canonical corpus, query set, relevance labels, and exact canonical snapshot. Results remain descriptive rather than release gates. The report emphasizes Hit@1 and MRR while retaining larger-K recall, category aggregates, diagnostics, and byte-for-byte reproducibility.

## Testing

Tests cover strict integer validation, query/result/candidate/character preflight bounds, exactly one bounded first-stage search, exact tuple requirements, invalid base hits, ghost/duplicate/conflicting reranker hits, invalid score and matched terms, stable tie-breaking, explicit underrun, provider error redaction, fail-closed behavior, clone isolation, atomic mutations, provenance preservation, collection restore/SQLite compatibility, four-backend benchmark ordering, canonical snapshot equality, and byte-identical report regeneration.

## Documentation and Non-Goals

README documentation explains the wrapper, bounds, provider contract, underrun behavior, deterministic ordering, failure model, and benchmark interpretation. The roadmap records this as the final Retrieval Runtime v1 capability and identifies user-facing Knowledge Base/Workspace APIs as the next layer.

This change does not add RAG generation, LLM judging, remote providers, query rewriting, metadata boosting, score blending, ANN, caches, schema changes, new persisted state, or product-facing APIs.
