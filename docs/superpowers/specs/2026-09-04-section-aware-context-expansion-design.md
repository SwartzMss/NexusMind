# Section-Aware Context Expansion Design

**Goal:** Make `KnowledgeBase.query()` include deterministic, provenance-preserving neighboring chunks around retrieved anchors by default, while allowing callers to reproduce the previous context behavior with `expand_context=False`.

**Scope:** Issue #128. Search APIs, retrieval ranking, backend scores, matched terms, and retrieval diagnostics remain unchanged. The existing `assemble_context()` default behavior remains unchanged.

## Design

`KnowledgeBase.query()` gains a keyword-only `expand_context: bool = True` parameter. The value is validated as a real boolean. When false, the existing query path is used byte-for-byte in terms of retrieval and context candidate selection. When true, the query still obtains the same final ranked retrieval results first; an independent expansion step then appends bounded supporting candidates before invoking `assemble_context()`.

The expansion step lives in a new `context_expansion` module. It receives the ranked anchor results and a read-only catalog of the canonical chunks produced during synchronization. For each anchor it examines at most the immediately previous and next chunks in canonical offset order. A neighbor is eligible only when it belongs to the same document and has the exact same `heading_path`; this strict rule includes unheaded chunks with `()` while preventing automatic movement between sibling Markdown sections. Parent-section expansion is intentionally deferred.

All anchor candidates are emitted first in original retrieval order. Expansion-only candidates follow in deterministic anchor order and distance order. Expansion candidates reuse the exact canonical `Chunk`, `Document`, and `KnowledgeSource` provenance and are represented as `SearchHit` values with score `0.0` and no matched terms. They do not participate in retrieval or reranking. The expansion helper has fixed hard bounds on neighbors and total candidates so it cannot produce unbounded input.

`KnowledgeCollection` maintains the chunk catalog as derived process-local state. Sync and restore update the catalog in the same commit point as the cloned index; snapshots and the SQLite schema remain unchanged. This preserves custom chunker output and stable chunk IDs without re-running a chunker during query execution.

The expanded candidate tuple is passed to the existing `assemble_context()` function. Its existing passage, character, token, duplicate, overlap, and provenance rules remain authoritative. The query path uses a bounded candidate cap large enough for the fixed expansion bound; the no-expansion path retains the existing retrieval-limit cap. Because anchors precede expansion candidates, supporting context cannot evict an earlier anchor when a budget is tight.

## Diagnostics

`KnowledgeQueryTrace` and its JSON debug representation expose whether expansion was enabled and bounded counts for selected anchor passages, selected expansion passages, expansion documents, and skipped section boundaries. Existing retrieval diagnostics and score fields are not modified. Expansion metadata is derived after context assembly, so counts reflect what survived the authoritative context budgets.

## Evaluation

Add a deterministic offline context-expansion benchmark with authored chunk fixtures and stable output. It covers evidence in the previous chunk, caveats in the next chunk, exclusion of an adjacent sibling section, multiple documents competing under passage/character/token budgets, and repeated-run equality. The report measures anchor retention, relevant-context coverage, expansion precision/irrelevant rate, boundary skips, and deterministic serialization.

## Error handling and compatibility

- Invalid `expand_context` values fail with `TypeError` before retrieval.
- Missing or incoherent canonical chunk catalog entries fail closed as a knowledge-base source error in the query orchestration layer.
- `search()` and `diagnose_search()` do not call or depend on context expansion.
- `expand_context=False` remains the explicit compatibility/A-B path.
- No new persistent fields, global configuration, embedding model, retrieval backend, or LLM call is introduced.

## Acceptance mapping

- Anchors and retrieval diagnostics are untouched; expansion is appended after retrieval.
- Same-document neighbors are selected from canonical offsets with strict section compatibility.
- Passages continue to slice `Document.content` exactly and retain canonical offsets/hashes.
- Existing context assembly enforces final budgets and overlap handling.
- Expansion metadata distinguishes anchors from supporting context.
- Offline fixtures cover cross-chunk evidence, sibling exclusion, budget competition, and determinism.
