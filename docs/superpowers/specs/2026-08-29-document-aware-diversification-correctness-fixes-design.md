# Document-Aware Diversification Correctness Fixes

## Status and scope

This design addresses two blocking findings from PR #113 review:

1. unconditional `4K` oversampling can exceed a retrieval backend's otherwise
   valid configured `max_results`;
2. a `max(raw_top_k) - min(raw_top_k)` relevance window is vulnerable to a
   single high-score outlier and can promote obviously weak cross-document
   candidates.

This document supersedes the candidate-depth and relevance-floor policy in
`2026-08-28-document-aware-search-diversification-design.md`. The surrounding
architecture remains unchanged: diversification is backend-independent, runs
after the backend's final ranking stage, returns a raw-order subsequence, and is
never applied to `diagnose_search()`.

## Backend result capacity

The four built-in final retrieval backends expose a read-only
`max_search_results` property whose value is safe for collection-level
oversampling:

- `InMemoryChunkIndex`;
- `InMemorySemanticChunkIndex`;
- `HybridChunkIndex`;
- `RerankedChunkIndex`.

For lexical, semantic, and reranked indexes this is their configured public
result or candidate capacity. Hybrid uses the minimum of `max_results`,
`max_candidates_per_backend`, and half of `max_fusion_entries`, accounting for
disjoint candidates from both children. If Hybrid's fixed `candidate_depth`
already exceeds that safe value, it advertises no capacity and opts out of
collection-level oversampling. Clones preserve the same property value because
they preserve the immutable limits and candidate depth. The base `ChunkIndex`
protocol does not require this property; existing third-party implementations
remain valid.

For user limit `K`, `KnowledgeCollection.search()` determines backend depth as
follows:

```text
desired = max(K, min(4K, 100))

if max_search_results is a positive plain integer and K <= max_search_results:
    backend_depth = min(desired, max_search_results)
else:
    backend_depth = K
```

Here, `100` is the maximum oversampling candidate depth, not a public search
result limit. Consequently, `K > 100` remains valid and `desired` never becomes
smaller than `K`.

The fallback is deliberately conservative. A third-party backend without the
optional property receives exactly the same limit it received before
diversification. A missing, raising, boolean, non-integer, or non-positive
property is treated as unavailable rather than becoming a new failure mode.
When `K` exceeds an advertised capacity, the collection still sends `K`; the
backend retains responsibility for raising its existing public limit error.

The selector runs for every backend output. With only `K` candidates it returns
the original Top-K sequence after raw-order backfill, so an unadvertised custom
backend remains behaviorally compatible while opting out of oversampling.

`diagnose_search(query, limit=K)` does not inspect this property and continues
to request exactly `K` raw results.

## Robust same-query relevance safeguard

The selector derives its scale only from scores in the same query and final
backend output. Given the first `min(K, candidate_count)` scores:

```text
center = lower_median(raw_top_k_scores)
deviations = abs(score - center) for each raw Top-K score
robust_span = lower_median(deviations)
relevance_floor = min(raw_top_k_scores) - robust_span
```

`lower_median` sorts finite floats ascending and selects index
`(count - 1) // 2`. This avoids averaging across two score populations for an
even candidate count. The construction is deterministic, supports negative and
equal scores, and is invariant under every positive affine score transform.

The two-pass selection remains:

1. scan candidates in raw rank order and prefer at most two results per
   canonical document;
2. raw Top-K candidates always satisfy relevance, while oversampled candidates
   must meet `relevance_floor`;
3. backfill remaining slots in raw rank order without the per-document cap;
4. sort selected indices so the returned results remain a raw-ranking
   subsequence.

For scores `1000, 1, 1, 1, 1`, both robust center and floor are `1`; candidates
scoring `0.001` or less cannot displace the remaining score-`1` chunks. For
scores `10, 9, 8, 7, 6`, center is `8`, robust span is `1`, and floor is `5`,
so nearby candidates at `5.5` and `5` remain eligible.

## Evaluation and tests

Tests must demonstrate:

- a backend with `max_search_results=10` and user `limit=5` receives depth 10
  and succeeds;
- a backend without the optional capacity receives exactly `limit`;
- a third-party backend without capacity receives `K > 100` unchanged;
- malformed or raising third-party capacity properties do not introduce a new
  failure;
- all four built-in backends expose a safe capacity and clones retain it;
- Hybrid custom limits bound oversampling by per-backend and worst-case fusion
  capacity at the collection boundary;
- the high-score outlier example rejects weak cross-document candidates;
- equal scores, negative scores, deterministic ordering, and positive affine
  invariance remain intact;
- diagnostics continue requesting the raw user limit and retain raw ranks and
  scores.

The checked-in offline benchmark keeps its aggregate broad/precise safeguards
and adds a case-level requirement for `broad-crypto`: raw Top-5 contains only
`crypto-overview.md`, while diversified Top-5 includes both authored relevant
Crypto documents. The corpus is not rewritten to force this result; the robust
policy is evaluated against the existing checked-in data.

## Non-goals

- No backend-specific branching in the selector.
- No exception-driven retry that could mask query, provider, or coherence
  failures.
- No mandatory protocol change for third-party indexes.
- No backend-independent absolute score threshold.
- No changes to diagnostic ranking, persistence, or public search result data.
