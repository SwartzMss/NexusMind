# Document-Aware Search Diversification Design

## Context

Chunk-level retrieval can legitimately rank many chunks from one long document
ahead of every other document. Returning raw Top-K chunks directly from the
user-facing search path can therefore reduce useful document coverage for broad
queries even when relevant candidates from other documents occur just below K.

Diversification applies to the final ranked candidates produced by every search
backend, including lexical, semantic, hybrid, and reranked indexes. It runs after
all backend-specific scoring, fusion, and reranking and before the user-facing
result limit is applied. Raw backend diagnostics remain unchanged.

## Public Contract

- `KnowledgeBase.search(query, limit=K)` returns at most K results.
- `limit` continues to describe final results, not candidate depth.
- Returned `SearchHit.score` and `matched_terms` are copied unchanged.
- The relative order of selected candidates matches their raw backend order.
- `KnowledgeBase.diagnose_search(query, limit=K)` continues to expose the
  backend's unmodified diagnostic result and candidate ranking.
- No diversification controls are added to the CLI or public Python API.

## Pipeline

```text
backend-specific retrieval / fusion / reranking
    -> bounded final candidate pool
    -> canonical provenance resolution
    -> relevance-aware document selection
    -> raw-rank-preserving final results
```

`KnowledgeCollection.search()` requests an internal candidate depth of:

```text
min(final_limit * 4, 100)
```

The constants are internal safeguards, not public tuning contracts. The backend
still owns candidate ranking and scores. The collection resolves candidates to
canonical documents before selection so grouping uses canonical `document_id`.

## Selection Policy

The selector receives an already ranked tuple of provenance-resolved results and
the final limit. It never calls a backend and has no backend-specific branches.

Let `raw_top_k` be the first `min(K, candidate_count)` candidates, then define:

- `best_top_k_score = max(score for raw_top_k)`;
- `worst_top_k_score = min(score for raw_top_k)`;
- `spread = best_top_k_score - worst_top_k_score`;
- `relevance_floor = worst_top_k_score - 0.25 * spread`.

The formula uses only scores from the same query and backend output. It is stable
under positive affine score transformations, so it supports positive, negative,
shifted, fused, and reranked score scales without an absolute cross-backend
threshold. Raw tuple order remains authoritative even for a custom backend whose
scores contain ties; the formula does not sort or otherwise reinterpret that order.

Selection has two passes:

1. Traverse candidates in raw order. Select candidates whose document currently
   has fewer than two preferred results and which are either already within raw
   Top-K or meet `relevance_floor`. Stop when K results are selected or candidates
   are exhausted.
2. Traverse candidates again in raw order and backfill any remaining slots with
   deferred candidates, without a per-document cap.

After selection, order the selected set by original raw rank. This makes the final
output a subsequence of the backend ranking: diversification can omit candidates,
but cannot reorder selected candidates or rewrite scores.

The two-per-document allowance and 0.25 score-window factor are internal first
implementation values. They become acceptable only if the evaluation described
below improves broad-query coverage without material precise-query regression.

## Relevance Behavior

Candidates already present in raw Top-K remain eligible for the preferred pass,
regardless of score. Oversampled candidates can displace deferred same-document
chunks only when their score lies inside the query-relative relevance window.

When no suitable additional documents exist, the second pass restores deferred
chunks in raw order, so search still returns up to K results. When the Top-K score
spread is zero, the relevance floor equals the worst Top-K score and only tied
oversampled candidates can participate in diversification.

## Diagnostics Isolation

`KnowledgeCollection.diagnose_search()` retains its current direct diagnostic
backend call with the requested diagnostic limit. It does not oversample for the
selector, invoke the selector, or mutate diagnostic `rank`, `score`, `stage`, or
`selected` values. CLI `diagnose` therefore remains a view of raw backend behavior.

## Error Handling and Bounds

- Existing public query and limit validation remains unchanged.
- Candidate depth multiplication is bounded before the backend call.
- Backend tuple/type/provenance validation remains authoritative.
- The selector rejects invalid final limits, non-exact result tuples, malformed
  results, and non-finite scores through a controlled collection error.
- Empty candidate input returns an empty tuple.
- If the candidate pool contains fewer than K values, every valid candidate is
  returned in raw order.

## Evaluation

Add a small descriptive raw-vs-diversified evaluation using realistic multi-chunk
documents. It must contain:

- broad keyword queries such as `Crypto`, `Binder`, `QNX`, and `权限校验`;
- exploratory queries with multiple authored relevant documents;
- precise queries where one document is the intended dominant result.

For raw Top-K and diversified Top-K, report at least:

- existing hit/recall/MRR relevance metrics;
- unique relevant documents returned;
- total unique documents returned;
- per-query returned document sequence.

Keep the policy only if broad and multi-document cases improve useful document
coverage and precise-query first-relevant rank and relevant-document recall do not
materially regress. The report is descriptive and checked into `evals/knowledge`;
it does not become a network-dependent or nondeterministic CI gate.

## Testing

Focused tests cover:

- a raw ranking dominated by one document surfaces a relevant additional document;
- a very weak oversampled document does not displace a strong same-document chunk;
- deferred chunks backfill when diversity cannot fill K;
- selected results preserve raw score, matched terms, and relative rank;
- equal, negative, and positively shifted scores behave deterministically;
- repeated calls produce identical results;
- final count never exceeds K and candidate requests never exceed 100;
- lexical, semantic, hybrid, and reranked collection paths use the same selector;
- diagnostics request and return the original raw limit/ranking without selection;
- existing query/context behavior remains covered by regression tests.

## Non-goals

This change does not modify backend scoring, lexical analysis, semantic similarity,
RRF, reranking, chunking, raw diagnostics, or public tuning options. It does not
introduce MMR, embedding diversity, a document-level reranker, or a permanent hard
per-document result cap.
