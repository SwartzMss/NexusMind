# Resolved Knowledge Search Design

## Goal

Make `KnowledgeCollection.search()` return Knowledge-layer results whose retrieval hit is resolved to the collection's committed canonical `KnowledgeSource` and `Document`. This adds trustworthy provenance without changing ranking, chunking, persistence schemas, or retrieval backends.

## Public contract

Add a frozen, slotted `KnowledgeSearchResult` dataclass in `knowledge_collection.py` with three fields:

- `source: KnowledgeSource`
- `document: Document`
- `hit: SearchHit`

`KnowledgeCollection.search(query, limit=10)` returns `tuple[KnowledgeSearchResult, ...]`. `ChunkIndex.search()` remains unchanged and continues to return `tuple[SearchHit, ...]`, keeping backend retrieval details below the collection boundary.

The existing `source`, `document`, `hit.chunk`, `hit.score`, and `hit.matched_terms` contracts expose every required identity, path, offset, content, and scoring field without duplicating data in a flattened result.

## Resolution and isolation

For each backend hit, the collection resolves `hit.chunk.document_id` against its committed `_documents` state and then resolves the document's `source_id` against committed `_sources`. Results preserve the backend tuple order exactly.

Each result receives deep copies of the canonical source and document. This prevents mutation of nested metadata mappings from changing collection-owned state. The hit remains the backend value because chunks and search-hit fields are immutable and do not contain mutable metadata.

Resolution never trusts a backend-provided source/document association. An unknown document, a missing owning source, a malformed non-`SearchHit` value, or a chunk whose document identity is incoherent with canonical state raises a controlled `KnowledgeSearchResolutionError`, derived from `KnowledgeCollectionError`. No partial result tuple is returned.

Sync, source removal, restore, and SQLite restart need no new state: they already atomically replace canonical documents and rebuild or update the derived index. Search always resolves against the currently committed dictionaries, so stale or changed provenance cannot be fabricated.

## Compatibility and exports

This intentionally evolves the collection-level `search()` return type. Direct retrieval callers using `ChunkIndex.search()` retain the existing `SearchHit` API. Public module exports will include the new result and error contracts wherever the collection's existing public contracts are exported.

## Testing

Tests will follow the existing offline pytest patterns and cover:

- correct source/document resolution across one and multiple sources;
- logical path, chunk content and offsets, score, and matched terms;
- backend ordering and empty results;
- deep-copy isolation for source and document metadata;
- changed and removed documents after sync;
- snapshot/restore and SQLite save/load/restore provenance;
- controlled failure for ghost, malformed, and incoherent hits;
- unchanged direct `ChunkIndex.search()` behavior.

Existing collection tests that inspect returned chunks will be updated to traverse `result.hit.chunk`, reflecting the deliberate API boundary change.

## Documentation and non-goals

README will document the `SearchHit -> KnowledgeSearchResult` layering and state that this feature only resolves canonical provenance. It does not add citation formatting, semantic retrieval, ranking changes, persistent indexes, RAG, UI, or schema changes.
