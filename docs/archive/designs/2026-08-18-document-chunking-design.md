# Document Chunking Design

> Historical design record. It describes the scope and assumptions at implementation time and is not the current product specification. See [the current architecture](../../architecture.md).

## Goal

Extend the provider-neutral Knowledge Runtime from `KnowledgeSource -> Document`
to `KnowledgeSource -> Document -> Chunk` with one deterministic, bounded,
character-based text chunker. Indexing, retrieval, embeddings, persistence, and
model-specific tokenization remain outside this milestone.

## Architecture

Create `src/nexusmind/knowledge_chunking.py` as a focused Knowledge Chunking
module. `src/nexusmind/knowledge.py` remains responsible for the source-neutral
`KnowledgeSource` and `Document` contracts, while the new module owns the
derived `Chunk` contract, chunk identity, configuration validation, resource
bounds, and the first-party `TextChunker` implementation.

The public types will be re-exported from `nexusmind.__init__` so callers can use
the same package-level API style as the existing Knowledge contracts. The
chunking module depends only on `Document` and the Python standard library; it
does not depend on ingestion adapters, Agent Runtime, tools, MCP, providers,
tokenizers, indexes, or retrieval components.

## Public Contracts

`Chunk` is a frozen, slotted dataclass with these derived-data fields:

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    document_id: str
    chunk_id: str
    content: str
    start_offset: int
    end_offset: int
```

Offsets are Python string indices and use the half-open interval
`[start_offset, end_offset)`. For every emitted chunk:

```python
chunk.content == document.content[chunk.start_offset:chunk.end_offset]
```

`TextChunker` exposes keyword configuration with defaults:

```python
TextChunker(chunk_size=1000, overlap=100, max_chunks=10000)
```

and converts a `Document` through:

```python
chunks = chunker.chunk(document)
```

The result is a tuple ordered by increasing `start_offset`.

## Chunking Algorithm

The chunker validates configuration during construction:

- `chunk_size` must be an integer greater than zero;
- `overlap` must be an integer satisfying `0 <= overlap < chunk_size`;
- `max_chunks` must be an integer greater than zero.

For non-empty content, the step is `chunk_size - overlap`. Starting at offset
zero, each chunk ends at `min(start + chunk_size, len(content))`. Iteration
stops as soon as the end reaches the document length, which prevents an empty
trailing chunk. Empty documents return `()`.

Before constructing chunks, the implementation computes the required chunk
count arithmetically. If it exceeds `max_chunks`, chunking raises
`ChunkLimitError` and returns no partial result. Invalid configuration raises
`ValueError`; wrong configuration or document types raise `TypeError`. These
errors are controlled and do not include document content.

## Deterministic Identity

Chunk IDs use SHA-256 over an unambiguous canonical JSON array containing:

- the document's stable `document_id`;
- the document's `content_hash`;
- `start_offset` and `end_offset`;
- `chunk_size` and `overlap`.

The digest is prefixed with `chunk-`. Including `content_hash` prevents the same
chunk ID from referring to different derived text after a logical document's
content changes. Including the offsets and effective configuration makes
identity deterministic for a given document version and chunking operation.
`max_chunks` is excluded because it is a safety bound and does not affect the
content or boundaries of successful output.

This milestone does not attempt cross-version deduplication or incremental
chunk reuse.

## Documentation

README documentation will show the explicit layering:

```text
Knowledge Ingestion -> Document
Knowledge Chunking  -> Chunk
future              -> Index -> Retrieval
```

It will describe character offsets, deterministic version-safe chunk identity,
configuration validation, and the maximum chunk-count bound without presenting
Index or Retrieval as implemented capabilities.

## Testing

`tests/test_knowledge_chunking.py` will cover:

- short, exact-boundary, multi-chunk, and empty documents;
- overlap boundaries and absence of empty trailing chunks;
- exact preservation of source slices for every chunk;
- Unicode content using Python character offsets;
- repeated-run determinism and stable IDs for identical inputs/configuration;
- different content versions not ambiguously reusing IDs;
- invalid `chunk_size`, `overlap`, and `max_chunks` values and types;
- enforcement of `max_chunks` without partial output;
- rejection of non-`Document` inputs.

The full existing test suite will run after the focused tests to verify that
Knowledge Core and local ingestion behavior remain unchanged.

## Out of Scope

No token-aware or semantic splitting, source-specific fields, embeddings,
indexes, vector databases, retrieval APIs, RAG assembly, persistence schema,
incremental synchronization, new ingestion adapters, UI, or Web API will be
added.
