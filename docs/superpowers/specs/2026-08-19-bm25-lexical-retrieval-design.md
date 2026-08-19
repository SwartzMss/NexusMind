# BM25 Lexical Retrieval Design

## Goal

Upgrade `InMemoryChunkIndex` from distinct substring-match counting to deterministic, dependency-free BM25 lexical ranking while preserving the `ChunkIndex`, Knowledge provenance, and canonical persistence boundaries.

## Analyzer and score contract

Both indexed chunk content and queries use the same small analyzer:

```text
Unicode string -> str.split() whitespace tokens -> token.casefold()
```

Matching is therefore token-based: `cat` does not match `concatenate`. Query tokens are deduplicated in first-occurrence order, so repeated query terms do not multiply query weight. `matched_terms` contains those normalized, deduplicated query terms that have positive term frequency in the chunk, in query order.

`SearchHit.score` becomes `float`. The built-in index returns finite positive scores for matches and preserves them unchanged through `KnowledgeSearchResult.hit.score`.

## BM25 formula

The built-in index uses `k1 = 1.2` and `b = 0.75`. For each distinct matched query term:

```text
idf(term) = log(1 + (N - df(term) + 0.5) / (df(term) + 0.5))

term_score = idf(term) *
             tf(term, chunk) * (k1 + 1)
             ------------------------------------------------------
             tf(term, chunk) + k1 * (1 - b + b * dl / avgdl)

score(query, chunk) = sum(term_score for each matched query term)
```

`N` is indexed chunk count, `df` is the number of indexed chunks containing the term, `dl` is the chunk's analyzed token count, and `avgdl` is the average analyzed token count. The positive IDF form prevents negative scores for common terms in small corpora. Empty-token chunks cannot match a non-empty query; when the corpus has no analyzed tokens, search returns no hits and avoids division by zero.

Results sort by descending score and then ascending `chunk_id`.

## Derived index state

`InMemoryChunkIndex` owns these process-local derived structures alongside its existing chunk and document maps:

- per-chunk normalized term-frequency counters;
- per-chunk token counts;
- corpus document/chunk-frequency counter;
- total analyzed token count, from which average length is derived.

No analyzer or BM25 state enters `KnowledgeSnapshot` or SQLite.

## Atomic mutations and cloning

All mutation inputs retain the current validation, identity-conflict, and resource-limit checks. After validation, each mutation builds candidate chunk/document maps, rebuilds all candidate analyzer/statistics state from those maps, and swaps every field only after rebuilding succeeds. This bounded `O(N)` mutation strategy favors a single auditable invariant over complex incremental rollback logic; `N` remains capped by `max_chunks`.

`add()` remains idempotent for exact duplicate chunks. `replace_document()` removes all old chunks before candidate statistics are rebuilt. `remove_document()` rebuilds without the removed document. `clone()` copies every mutable chunk/document/statistics mapping and counter so ranking is initially equivalent and later mutations are independent.

Existing character-count and per-document resource bounds remain unchanged. Query character, raw query-term, result, and positive plain-integer limit checks also remain unchanged.

## Knowledge and persistence compatibility

`ChunkIndex.search()` continues returning `SearchHit`; `KnowledgeCollection.search()` continues validating chunk/document coherence and resolving detached canonical provenance. Only the score type and built-in ranking semantics change.

Snapshot and SQLite schemas remain canonical Source/Document-only. `restore()` rebuilds chunks and a fresh BM25 index using the configured chunker, so equivalent canonical state and chunker configuration produce equivalent ranking after process restart.

## Testing

Offline deterministic tests will cover:

- float, finite, positive scores and formula-specific `pytest.approx()` checks;
- whitespace/casefold token matching, including no substring matches;
- duplicate query-term deduplication and deterministic `matched_terms`;
- TF, positive IDF rarity, and length-normalization ranking relations;
- score-descending and `chunk_id` tie ordering plus limit behavior;
- add, replace, remove, clone, exact-duplicate, and failed-mutation statistics;
- unchanged query/index resource bounds;
- Knowledge resolved provenance preserving the backend float score;
- snapshot/restore and SQLite restart ranking equivalence;
- complete regression suite.

## Documentation and non-goals

README will document the exact formula, constants, analyzer behavior, deterministic ordering, derived-state rebuild after restore, and the whitespace analyzer's limitation for languages without whitespace word boundaries.

This does not add persistent postings, FTS5, stemming, stop words, Chinese segmentation, language detection, metadata/path/source boosting, semantic or hybrid retrieval, embeddings, reranking, query rewriting, citations, RAG, LLM generation, adapters, or UI/API work.
