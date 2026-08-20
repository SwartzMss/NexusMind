# User-Facing KnowledgeBase and Source Registry Design

## Objective

Add a narrow product API that lets callers create or open one knowledge base, register local sources, explicitly synchronize them, inspect canonical state, and search without assembling chunkers or retrieval indexes. The product layer composes the existing Knowledge Runtime and does not alter retrieval, provenance, snapshot, or SQLite canonical schemas.

The name is `KnowledgeBase`; Agent Runtime already owns the unrelated `Workspace` term.

## Persistence Layout and State Boundaries

One knowledge base is one directory:

```text
knowledge-base/
├── manifest.json
└── knowledge.db
```

`manifest.json` stores product configuration: stable KnowledgeBase identity, optional display name, registered source configurations, and schema version. `knowledge.db` remains an unmodified `SQLiteKnowledgeSnapshotStore` containing only canonical `KnowledgeSource` and `Document` values. Chunks, lexical state, embeddings, hybrid fusion state, reranker state, factories, and provider configuration remain derived runtime state and are rebuilt on open.

The KnowledgeBase root must be a text filesystem path. `create()` rejects a non-directory target, symlink/reparse-point root, or non-empty existing directory. It may use an existing empty directory. `open()` requires a real, non-symlink directory containing a valid manifest; a missing canonical database is a controlled persistence error rather than silently creating an empty replacement.

## Manifest and Source Configuration

The v1 manifest has this exact logical schema:

```json
{
  "format_version": "1",
  "knowledge_base_id": "security-docs",
  "display_name": "Security Docs",
  "sources": [
    {
      "config_version": "1",
      "source_id": "docs",
      "type": "local_directory",
      "path": "/absolute/path/to/docs"
    }
  ]
}
```

`display_name` is either a non-empty string or `null`. Unknown, missing, or mistyped fields fail closed at every object layer. Both format and source config versions are exact strings. Source entries are serialized in ascending `source_id` order. JSON uses UTF-8, LF, `ensure_ascii=False`, sorted keys, compact separators, and one trailing LF so identical state produces identical bytes across supported platforms.

Public frozen source types are `LocalFileSourceConfig` and `LocalDirectorySourceConfig`. Each has the fixed config version `"1"`, an explicit class-owned type discriminator, a non-empty unique `source_id`, and a text path. Registration resolves the path with `strict=False` and persists an absolute path, making later opens independent of process working directory. Registration validates syntax and configured character bounds but does not require the target to exist and never constructs an adapter. Existing `LocalFileAdapter` or `LocalDirectoryAdapter` performs existence, type, encoding, traversal, extension, symlink, and ingestion validation only during explicit synchronization.

The source configuration is distinct from canonical `KnowledgeSource`: registration says where future synchronization should load data, while the canonical model proves what has already been ingested.

`KnowledgeBaseLimits` uses positive plain integers and bounds at least manifest bytes, registered source count, knowledge-base ID characters, display-name characters, source ID characters, and path characters. Booleans are rejected. Bounds are checked before retaining or writing configuration.

## Public API

```python
class KnowledgeBase:
    @classmethod
    def create(
        cls,
        path,
        *,
        knowledge_base_id: str,
        display_name: str | None = None,
        index_factory=None,
        limits: KnowledgeBaseLimits | None = None,
    ) -> KnowledgeBase: ...

    @classmethod
    def open(
        cls,
        path,
        *,
        index_factory=None,
        limits: KnowledgeBaseLimits | None = None,
    ) -> KnowledgeBase: ...

    def add_source(self, config: RegisteredSourceConfig) -> None: ...
    def unregister_source(self, source_id: str) -> None: ...
    def remove_source(self, source_id: str) -> None: ...
    def sync(self) -> tuple[KnowledgeSyncResult, ...]: ...
    def sync_source(self, source_id: str) -> KnowledgeSyncResult: ...
    def list_sources(self) -> tuple[RegisteredSourceConfig, ...]: ...
    def list_documents(self) -> tuple[Document, ...]: ...
    def search(self, query: str, *, limit: int = 10) -> tuple[KnowledgeSearchResult, ...]: ...
    def status(self) -> KnowledgeBaseStatus: ...
    def close(self) -> None: ...
```

`create()` writes an empty manifest and initializes the canonical store without source, chunk, embedding, or provider work. `open()` strictly loads both stores, creates a fresh collection using current runtime configuration, and restores the canonical snapshot to rebuild derived state. It also validates cross-store coherence: every canonical `KnowledgeSource.source_id` must have a registration, and its canonical source type must match that registration's local-file/local-directory discriminator. Unsynchronized registrations without canonical state are valid. Orphan or type-conflicting canonical sources fail closed instead of opening a misleading KnowledgeBase.

The default runtime is deterministic and offline: `TextChunker` plus `InMemoryChunkIndex` configured with `UnicodeCJKLexicalAnalyzer`. `index_factory` is an explicit non-persisted injection point for Semantic, Hybrid, or Reranked indexes. The benchmark fixture reranker is never a production default. Low-level runtime APIs remain public and independently usable.

Every method except idempotent `close()` rejects use after close. The object also becomes unusable after an unrecoverable persistence compensation failure.

## Registration and Inspection Semantics

`add_source()` persists registration only. Duplicate source IDs fail closed even when the duplicate object is equal. It performs no adapter construction, filesystem ingestion, embedding, or retrieval provider work.

`unregister_source()` removes only a registration and is allowed only when the current canonical snapshot contains no `KnowledgeSource` with that ID. This makes it useful for unused registrations without leaving ingested orphan knowledge. Unknown registrations and attempts to unregister synchronized knowledge fail closed.

`remove_source()` removes both registration and canonical source/documents using the atomic protocol below. Unknown registrations fail closed.

`list_sources()` returns frozen source config values sorted by `source_id`. `list_documents()` returns deep-detached canonical `Document` values in snapshot order. `KnowledgeBaseStatus` is a frozen value containing knowledge-base ID, display name, registered source count, canonical source count, and document count. It performs no filesystem dirty detection and stores no invented last-sync timestamp or status.

`search()` delegates directly to `KnowledgeCollection.search()` and retains existing `KnowledgeSearchResult` ordering and canonical provenance semantics.

## Synchronization and Atomicity

Synchronization is explicit, synchronous, deterministic, and caller-triggered. There are no watchers, background tasks, or automatic provider calls.

For `sync()`:

1. Snapshot the currently committed collection.
2. Create a fresh staging collection with the same runtime configuration and restore that snapshot.
3. Instantiate adapters and call the existing `KnowledgeCollection.sync()` for every registered source in ascending `source_id` order.
4. Stop at the first failure and discard the staging collection; the live collection and SQLite snapshot remain unchanged.
5. After all sources succeed, save the complete staging snapshot through `SQLiteKnowledgeSnapshotStore.save()`.
6. Only after the SQLite transaction commits, swap the live collection reference.

The result is an ordered tuple of each source's `KnowledgeSyncResult`. This is an all-or-nothing batch contract, not a partial-success report. `sync_source()` uses the same protocol for exactly one known registration and returns one result. Sources registered but excluded from `sync_source()` remain untouched in the restored staging state.

Calling `sync()` with no registrations is a successful no-op returning an empty tuple; it does not rewrite the canonical store.

Registration-only manifest changes use atomic file replacement: encode and bound the complete next manifest, write a same-directory temporary file with explicit UTF-8 bytes, flush and fsync it, `os.replace` it over `manifest.json`, and fsync the containing directory where supported. Temporary files are cleaned up after controlled failures.

`remove_source()` stages both next manifest and next collection. It saves the new canonical snapshot first, then atomically replaces the manifest, then swaps memory. If manifest replacement fails after the SQLite commit, it immediately saves the old canonical snapshot as compensation. When compensation succeeds, the in-memory object and both persistent stores remain at the old state and the operation raises a controlled persistence error. If compensation also fails, the object is poisoned/closed and raises an explicit recovery failure; callers must reopen after repairing persistent state. This is a narrow two-store recovery contract, not a general distributed transaction system.

## Controlled Errors

The product-layer hierarchy is:

- `KnowledgeBaseError`
- `KnowledgeBaseConfigError`
- `KnowledgeBaseSourceError`
- `KnowledgeBasePersistenceError`
- `KnowledgeBaseClosedError`

Manifest schema/version/bound failures use `KnowledgeBaseConfigError`. Duplicate, unknown, invalid lifecycle, adapter, and synchronization failures use `KnowledgeBaseSourceError`. Filesystem, manifest atomic-write, SQLite load/save, and compensation failures use `KnowledgeBasePersistenceError`. Closed or poisoned object use raises `KnowledgeBaseClosedError`.

Owned lower-level failures are chained as causes but public messages never include document contents, provider exception text, API secrets, or query text. No method returns a partially committed value after failure.

## Testing

Tests cover:

- create/open/reopen, stable identity/name, idempotent close, and closed-use rejection;
- byte-deterministic manifest serialization and exact schema/version/type validation;
- LocalFile and LocalDirectory source config round trips, absolute path resolution, duplicates, unknown types, malformed paths, and registry/config/byte bounds;
- proof that registration performs no adapter, ingestion, index, embedding, or reranker work;
- one-source sync and deterministic multi-source order;
- batch sync failure preserving live memory and SQLite bytes/state;
- restart preserving registrations and canonical documents, with search rebuilding derived default state;
- injected alternative cloneable index factory behavior without manifest changes;
- unregister-before-sync, rejection of unregister-after-sync, and full source removal;
- canonical-save, manifest-replace, compensation-success, and compensation-failure removal paths;
- corrupt, missing, oversized, and incompatible manifests and missing/corrupt canonical store;
- orphan and source-type-conflicting canonical state rejection during open;
- detached, stable `list_sources`, `list_documents`, and `status` values;
- persistence proof that manifest contains no document content and SQLite/snapshot contains no registered configs, chunks, embeddings, hybrid state, or reranker state;
- unchanged KnowledgeCollection, store, BM25, Semantic, Hybrid, Rerank, and evaluation suites.

## Documentation and Non-Goals

README documentation includes a create → register → sync → search → reopen example and explains product configuration versus canonical knowledge, registration versus synchronization, both removal operations, the two-file persistence invariant, deterministic offline default, runtime factory injection, and synchronous/manual-sync limitations.

This issue does not add CLI commands, UI, Office/PDF ingestion, Git/Web/MCP sources, RAG or citations, LLM judging, query rewriting, new retrieval algorithms, benchmark tuning, persistent embeddings/indexes, watchers, background synchronization, permissions, cloud sync, or Agent Runtime integration. A thin `nexusmind kb` CLI is the intended follow-up after this API stabilizes.
