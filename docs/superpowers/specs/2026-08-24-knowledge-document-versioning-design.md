# Knowledge Document Provenance and Version Tracking Design

## Objective

Add internal, durable document-version history to the KnowledgeBase lifecycle.
Synchronization will detect content changes, append immutable versions, and
preserve older versions without changing the existing adapter, query, search,
or current-document contracts.

This change implements issue #99's versioning foundation only. Historical
versions are persisted as part of collection snapshots but are not exposed
through a public listing or retrieval API. Search and query behavior continues
to use only the current active document for each logical document identity.

## Architecture and Compatibility Boundary

`KnowledgeCollection` remains the owner of canonical current documents and the
derived chunk index. It gains a separate version-history store keyed by stable
`document_id`. The two stores have distinct responsibilities:

- `_documents` contains only the current active documents and continues to
  resolve search hits.
- `_document_versions` contains the immutable history of every document seen by
  a successful sync, including versions of documents no longer active.

`Document` and `KnowledgeSourceAdapter` remain unchanged. Adapters continue to
return source snapshots made only of current `Document` values and need no
knowledge of hashes, clocks, version relationships, or synchronization
contexts. Existing callers that inspect `KnowledgeSnapshot.sources` and
`KnowledgeSnapshot.documents` retain their current semantics.

## Version Contract

Introduce a frozen, slotted `DocumentVersion` value with these fields:

```python
@dataclass(frozen=True, slots=True)
class DocumentVersion:
    version_id: str
    document_id: str
    source_id: str
    logical_path: str
    content: str
    content_hash: str
    created_at: str
    previous_version_id: str | None
    sync_context: str
```

The full content is stored so that each historical record is a self-contained
source snapshot rather than metadata pointing at mutable current state.
`source_id` and `logical_path` retain provenance, while `document_id` and
`content_hash` retain logical and content identity.

`created_at` is a canonical UTC RFC 3339 timestamp ending in `Z`. The collection
uses an injectable internal clock so tests can be deterministic. The default
clock uses the current UTC time. `sync_context` is an opaque, collection-created
identifier for the successful source sync that produced the version; all new
versions created by one sync share it.

`version_id` is deterministic and collision-resistant: a `version-` prefixed
SHA-256 digest of an unambiguous canonical encoding of `document_id`,
`content_hash`, and `previous_version_id`. This makes the version relationship
part of the identity while allowing a previously removed document with the same
content to append a distinct occurrence after an intervening version. Exact
duplicates against the current version never create a new record.

## Synchronization Semantics

Sync keeps the existing stable-document identity and content-hash comparison:

1. A document not previously active is compared with the latest retained
   version for that `document_id`.
2. If it has never been seen, sync appends the first version.
3. If it is currently active with the same content hash, it is unchanged and no
   version is appended.
4. If it is currently active with a different content hash, sync appends a
   version whose `previous_version_id` points to the latest retained version.
5. If it was removed and later reappears, sync appends a new version linked to
   the retained latest version. This records the new synchronization occurrence
   even when its content matches an older retained version.
6. Removing a document from an adapter snapshot removes it from the active
   document store and index but never deletes its retained history.

Version preparation participates in the existing staged synchronization
boundary. Incoming validation, chunking, index mutations, source/current-state
updates, and history updates either commit together or leave all committed state
unchanged. A failed sync creates neither a current-document change nor a ghost
version.

The current document-count limit continues to bound active documents. A new
positive `max_document_versions` collection limit bounds retained history.
Preflight rejects a sync that would exceed it before chunking or state mutation.

## Snapshot and Restore

`KnowledgeSnapshot` gains a trailing `document_versions` tuple with an empty
tuple default, so existing code that constructs a two-field snapshot remains
source-compatible. Snapshot output remains
deterministic: sources and current documents keep their existing ordering, and
versions are ordered by document identity followed by chain order. All values
are detached copies.

A snapshot containing current documents but an empty history is treated as a
legacy snapshot. Restore synthesizes one root version per current document in a
single restore context using the injected clock. Once restored, subsequent
snapshots always contain explicit version history. A partially populated
history is not treated as legacy and must satisfy all normal coherence rules.

Restore validates the complete snapshot before committing:

- every version has valid non-empty identity and provenance fields;
- the stored content hash matches its content;
- its `document_id` matches the stable identity of `source_id` and
  `logical_path`;
- version IDs match their deterministic inputs and are globally unique;
- each document's chain has exactly one root, no cycles, no forks, and no
  missing predecessor;
- every current document has a version chain whose latest version matches its
  identity, provenance, content, and content hash;
- historical versions may belong to documents or sources that are no longer
  active; their embedded source identifier and logical path remain the retained
  provenance;
- configured source, current-document, and version limits are satisfied.

Restore rebuilds the search index exclusively from current documents. It stages
the validated version history alongside sources, current documents, and the
index, then swaps all state atomically. Existing restore summaries remain about
active sources, active documents, and indexed chunks; no history count is added
to public operational results.

## Query and Search Behavior

No search or query API reads historical versions. A chunk continues to resolve
through `_documents` to the current canonical `Document`. Historical content is
never chunked during restore and cannot appear in lexical, semantic, hybrid, or
reranked results.

The new version type is a snapshot serialization detail rather than a new
history-browsing service. It may be exported as needed for snapshot construction
and persistence typing, but no collection method will list, fetch, diff, or
search versions in this issue.

## Error Handling and Security

Malformed or incoherent history raises the existing controlled snapshot or
restore error families. Messages identify the violated invariant without
including document content, absolute source paths, or adapter exception text.

All incoming source, document, and version values are detached before commit.
Callers cannot mutate retained history through adapter objects, snapshot values,
or restored inputs. Version-limit failure and clock failure occur before commit
and preserve both canonical and derived state.

## Testing

Focused tests will cover:

- first sync creating one root version with complete provenance;
- identical sync remaining unchanged without another version;
- changed content appending a linked version while indexing only the new text;
- deletion preserving history while removing current search results;
- reappearance extending the retained chain, including same-content return;
- multiple changed documents sharing one synchronization context;
- deterministic version identity and UTC timestamp validation;
- version-history limit preflight;
- chunk/index/clock failures preserving current state and history atomically;
- deterministic, detached snapshot round trips;
- restore rebuilding search only from current versions;
- rejection of bad hashes, IDs, provenance, roots, predecessors, cycles, forks,
  duplicate IDs, stale current versions, malformed provenance, and limit
  overflow;
- unchanged behavior for existing source adapters and all retrieval backends.

The complete test suite will run after focused tests to confirm compatibility.

## Non-Goals

This issue does not add a public version-history API, historical search, diffs,
Git integration or history analysis, background synchronization, UI timelines,
external version-control dependencies, adapter-specific version metadata,
cross-document content deduplication, retention pruning, or migration from an
external persistence schema.
