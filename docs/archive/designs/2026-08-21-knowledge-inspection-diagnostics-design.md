# Knowledge Inspection and Retrieval Diagnostics Design

> Historical design record. It describes the scope and assumptions at implementation time and is not the current product specification. See [the current architecture](../../architecture.md).

## Goal and Boundary

Add structured, read-only inspection and retrieval diagnostics to the public
Python `KnowledgeBase` API. These capabilities make registered and canonical
knowledge, document-to-chunk transformation, and retrieval decisions
observable without coupling the domain API to CLI, UI, or Agent presentation.

`KnowledgeBase` remains the product-facing boundary. `KnowledgeCollection`
resolves canonical state and provenance, while retrieval indexes expose an
optional structured diagnostic protocol. Existing `search()` behavior and all
BM25, semantic, Hybrid-RRF, and reranking formulas remain unchanged.

## Public Inspection Model

New frozen, slotted dataclasses and a string enum describe inspection results:

- `KnowledgeSourceSyncStatus` has `REGISTERED` and `SYNCED`. A registered source
  is `SYNCED` when committed canonical state contains that source, including a
  successfully synchronized source with zero documents. Otherwise it is
  `REGISTERED`. Inspection does not scan the filesystem or claim that source
  contents are currently clean.
- `KnowledgeSourceInspection` contains the detached registered source config,
  sync status, canonical document count, and derived chunk count.
- `KnowledgeDocumentSummary` contains detached basic canonical document fields:
  source/document identity, logical path, content type, content hash, metadata,
  content character count, and derived chunk count. It does not duplicate full
  document content in a whole-base response.
- `KnowledgeBaseInspection` contains the current `KnowledgeBaseStatus`, source
  inspections ordered by `source_id`, and document summaries ordered by
  `(source_id, logical_path)`.
- `KnowledgeChunkInspection` contains stable ordinal, chunk ID, start/end
  offsets, character count, and a bounded preview.
- `KnowledgeDocumentInspection` contains detached canonical source and document
  values plus all derived chunk inspections in chunk order.

`KnowledgeBase.inspect()` returns the whole-base view. `inspect_document()`
accepts a canonical `document_id` and an optional positive `preview_chars`
bound, defaulting to a conservative fixed value. Unknown IDs and invalid bounds
raise controlled public errors. Returned mutable nested metadata is deep-copied
from internal state.

Source status is derived from manifest registration plus committed canonical
state, so it remains meaningful after `KnowledgeBase.open()` without adding a
new persistence schema or ephemeral last-sync history. This design deliberately
does not expose dirty-state detection, timestamps, failed-sync history, or
background sync state.

## Chunk Inspection

`KnowledgeCollection` owns the configured chunker and adds a read-only document
inspection operation. It locates the committed canonical document, deep-copies
it, runs the same collection-owned chunker, and validates the exact tuple,
document identity, stable ordering, offsets, IDs, and canonical content slices
before constructing inspection values.

Recomputing chunks preserves the existing rule that chunks and indexes are
derived runtime state, not canonical snapshot or SQLite state. Inspection never
adds a chunk cache or changes restore behavior. A nondeterministic, malformed,
or failing custom chunker fails closed with a controlled collection error.

Whole-base inspection uses the same validated derivation to calculate chunk
counts. Work is explicitly requested by the caller and bounded by the existing
collection document/chunk limits; preview allocation is additionally bounded by
`preview_chars`.

## Retrieval Diagnostic Model

The retrieval layer adds backend-neutral frozen values:

- `RetrievalStage` identifies `lexical`, `semantic`, `fusion`, or `reranker`.
- `RetrievalCandidateDiagnostic` records stage, one-based stage rank, canonical
  chunk, backend score, matched terms, optional RRF contribution, and whether
  the candidate survives into the final result.
- `RetrievalDiagnostics` contains the final `SearchHit` tuple and the ordered
  candidate diagnostic tuple for one query.
- `KnowledgeRetrievalDiagnostics` contains the query, provenance-resolved final
  `KnowledgeSearchResult` tuple, and candidate diagnostics whose chunks have
  each been checked against committed source/document state.

`DiagnosticChunkIndex` is an optional protocol with
`diagnose(query, *, limit) -> RetrievalDiagnostics`. The four built-in retrieval
compositions implement it. `KnowledgeCollection.diagnose_search()` calls the
diagnostic method exactly once, validates its complete result, resolves final
hits and every diagnostic candidate through the same canonical provenance rules
as `search()`, and returns one coherent snapshot. `KnowledgeBase.diagnose_search()`
is the public product wrapper and preserves controlled public error redaction.

A custom `CloneableChunkIndex` that implements `add()`, `replace_document()`,
`remove_document()`, `search()`, and `clone()`, but not optional `diagnose()`,
remains valid for ordinary search, sync, restore, and persistence. Explicitly
requesting diagnostics fails with a stable unsupported-diagnostics error. The
protocol is optional so this feature does not break existing third-party index
factories.

## Backend Traces

Each backend refactors its existing search computation into one private
single-pass implementation used by both `search()` and `diagnose()`. `search()`
projects only final hits; `diagnose()` additionally projects already-computed
intermediate values. This prevents a diagnostic request from running providers
twice or drifting from the returned final result.

- BM25 emits one `lexical` row per ranked match with its existing score and
  matched terms.
- Semantic retrieval emits one `semantic` row per ranked candidate with its
  existing cosine score.
- Hybrid retrieval preserves lexical and semantic child rows, records each
  child's one-based rank and exact `1 / (rrf_k + rank)` contribution, then emits
  a `fusion` row with the existing summed score and final fusion rank.
- Reranking preserves all base diagnostics while updating `selected` for the
  final output, then emits `reranker` rows with the provider score and final
  rank. It never asks the base for candidates beyond the existing configured
  depth.

The ordinary `SearchHit` contract is not enlarged with backend-specific fields.
Diagnostic rows are ordered by pipeline stage and then rank, with the existing
chunk-ID tie rules retained. Every score remains a finite float, all tuples are
exact tuples, and all existing candidate/result/resource limits apply before
diagnostic values are returned.

## Consistency and Failure Handling

Inspection and diagnostics require an open `KnowledgeBase`. They are read-only
and do not acquire the cross-process mutation lock, touch source files, save
SQLite state, update the manifest, or retain query history.

Collection validation rejects unknown documents, ghost or duplicate chunks,
invalid offsets, content that differs from its canonical document slice,
incoherent final hits, invalid stages/ranks/scores/contributions, and diagnostics
that claim a selected candidate without a matching final hit. Built-in indexes
continue to use their existing controlled error families. `KnowledgeBase`
converts lower-layer failures to stable `KnowledgeBaseSourceError` messages
without exposing provider exceptions, private paths, queries, or document text.

## Compatibility and Persistence

Existing `KnowledgeBase.status()`, `list_sources()`, `list_documents()`,
`search()`, synchronization, close semantics, manifest encoding, snapshot
encoding, and SQLite schema remain compatible. No diagnostics, chunk values,
query values, scores, vectors, sync timestamps, or failure history are persisted.

The existing local UI is unchanged in this issue. A later CLI, UI, or Agent
adapter can render the public diagnostic dataclasses without reading private
collection/index state or reimplementing retrieval explanations.

## Testing

Tests cover:

- stable source/document ordering, registered-versus-synced status, empty
  synchronized sources, chunk counts, detached metadata, bounded previews, and
  exact chunk offsets;
- unknown document IDs, invalid preview bounds, malformed custom chunkers,
  closed handles, and read-only inspection behavior across reopen;
- BM25 scores and terms, semantic cosine candidates, exact Hybrid-RRF ranks and
  contributions, reranker first-stage/final ranks, deterministic ties, empty
  results, and existing resource limits;
- one provider execution per diagnostic query, equality between diagnostic
  final hits and ordinary search semantics, and full canonical provenance for
  every final and intermediate chunk;
- malformed custom diagnostic implementations and unsupported custom indexes
  failing closed without affecting ordinary search;
- unchanged snapshot/SQLite formats and the focused KnowledgeBase, collection,
  lexical, semantic, hybrid, and reranking regression suites.

The repository's Linux baseline currently has 38 unrelated command-profile
failures because production rejects non-Windows command execution while those
tests still run on Linux. They are recorded as pre-existing and are outside this
issue; focused retrieval and KnowledgeBase suites must pass, and no new failures
may be introduced.

## Non-Goals

This change does not add or modify CLI/UI screens, retrieval algorithms, source
formats, PDF/DOCX/OCR ingestion, RAG or LLM answer generation, query rewriting,
knowledge graphs, vector database migration, persistent embeddings, background
sync, filesystem dirty-state scanning, diagnostic history, or telemetry.
