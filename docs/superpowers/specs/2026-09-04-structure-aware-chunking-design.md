# Structure-Aware Chunk Metadata and Retrieval Design

## Goal

Complete issue #126 by preserving Markdown/AnyDoc heading hierarchy in derived
chunks, exposing that hierarchy as backward-compatible metadata, using it in
retrieval, and extending the deterministic offline benchmark to measure the
before/after effect without changing canonical document storage.

## Current context

The current `origin/main` already contains `StructureAwareChunker`, Markdown
block boundary protection, bounded fallback splitting, a default switch from
`TextChunker`, and a small structure-chunking benchmark. `Chunk` still only
contains document identity, content, and offsets. Retrieval backends index only
`chunk.content`, so a heading that falls outside a body slice cannot help a
query match. The existing collection validates every chunk as an exact slice of
the canonical `Document`.

## Design

### Chunk contract

Extend `Chunk` with optional fields that default to the legacy empty values:

```python
heading_path: tuple[str, ...] = ()
section_title: str = ""
source_location: str = ""
```

`TextChunker` continues to create empty metadata. `StructureAwareChunker`
derives metadata from Markdown ATX headings and creates the same metadata for
every span belonging to the section. The heading path is ordered from the
outermost heading to the current heading; `section_title` is its last item or
the empty string for preamble content. `source_location` is a stable
document-relative line marker for the current section heading, or the empty
string for content before the first heading. It does not expose an absolute
filesystem path.

All new fields are validated at construction: the path is a tuple of non-empty
strings, the title and location are strings, and the title is consistent with
the final path item. Metadata is immutable at the chunk boundary. Existing
five-positional-argument `Chunk(...)` calls remain valid.

### Structure-aware metadata derivation

Keep the existing protected-block parser and bounded packing algorithm. Add a
heading-state pass over the resulting structural spans:

1. Recognize only ATX headings already recognized by the Markdown block parser;
   headings inside fenced code remain ordinary content.
2. Maintain a six-level stack. A heading at level `n` replaces the current
   level and discards deeper levels.
3. Associate each emitted span with the active path at its effective section
   start. A heading-only or packed heading prelude uses the deepest heading it
   contains before body content; later content inherits that section path.
4. Preserve exact source offsets and the existing size, overlap, line-boundary,
   and `max_chunks` guarantees.

The plain `TextChunker` remains the stable fixed-window baseline for benchmark
comparisons and for callers that explicitly inject it.

### Retrieval representation

Add a read-only `retrieval_text` property to `Chunk`:

```text
<heading_path joined with " > ">
<exact chunk content>
```

For chunks without structural metadata it equals `content`, preserving all
existing behavior. Lexical analyzers, semantic embedding input, and reranker
scoring use `retrieval_text`; context assembly, displayed content, provenance
checks, offsets, and citation passages continue to use exact `content`.

Chunk IDs include the algorithm/metadata identity so identical source slices
under different heading paths cannot collide. Legacy chunks constructed without
metadata retain their existing identity formula unless they are produced by
the structure-aware chunker, whose explicit algorithm version remains part of
the ID.

### Evaluation

Extend the existing `evals/knowledge/chunking` corpus and cases with a nested
technical document where the query contains a heading term and a body term,
but a fixed window separates them. The benchmark will evaluate:

- `TextChunker` as the before baseline;
- `StructureAwareChunker` as the after candidate;
- existing `evaluate_retrieval_multi_k` metrics for Hit@K, Recall@K, and MRR;
- a precision-at-K field in the benchmark report, derived from authored
  relevant-document targets;
- the existing reranked backend comparison, with structural retrieval text
  available to the candidate and reranker.

The benchmark remains deterministic, offline, descriptive, and non-gating
outside its targeted regression assertions. It must show the structure-aware
candidate improves the boundary case without reducing Recall@3 on the corpus.

### Compatibility and error handling

- Canonical `Document`, manifest, and SQLite schemas do not change because
  chunk metadata is derived runtime state.
- Existing custom chunkers and custom indexes continue to work with legacy
  `Chunk` objects.
- All four index families (lexical, semantic, hybrid, reranked) must agree on
  the same chunk identity and retrieval text; incoherent backend behavior is
  rejected through existing collection/index validation paths.
- Malformed metadata raises the same controlled type/value errors used by the
  existing chunk contract; no user-provided heading text is interpreted as a
  path or executed.
- Existing exact-slice and offset validation remains authoritative.

## Files and responsibilities

- `src/nexusmind/knowledge_chunking.py`: extend `Chunk`, derive section
  metadata, and preserve bounded structural splitting.
- `src/nexusmind/knowledge_retrieval.py`: index `retrieval_text` in BM25.
- `src/nexusmind/semantic_retrieval.py`: embed `retrieval_text`.
- `src/nexusmind/reranking.py`: score and budget candidate retrieval text while
  keeping returned content exact.
- `src/nexusmind/__init__.py`: preserve public exports and expose any new
  public metadata type only if needed by the final contract.
- `tests/test_knowledge_chunking.py`: contract, hierarchy, validation,
  determinism, and limit coverage.
- `tests/test_knowledge_retrieval.py`, `tests/test_semantic_retrieval.py`,
  `tests/test_reranking.py`: verify structural terms participate in retrieval
  without changing content/offset invariants.
- `tests/test_structure_chunking_benchmark.py` and
  `evals/knowledge/chunking/`: deterministic before/after quality fixture.
- `docs/architecture.md`, `evals/knowledge/chunking.md`, and `README.md`:
  document the public behavior and reproduction command.

## Acceptance criteria

- Nested Markdown and AnyDoc-generated Markdown headings produce correct
  `heading_path`, `section_title`, and `source_location` values.
- `Chunk.content` remains an exact canonical document slice and all configured
  size/overlap/limit constraints still hold.
- Explicit legacy chunkers and custom callers remain compatible.
- Structural heading terms are searchable through lexical, semantic, hybrid,
  and reranked paths.
- The offline benchmark is repeatable and reports before/after retrieval
  metrics, including precision, while preserving existing recall.
- The full existing test suite passes with the project dependencies installed.

## Out of scope

Knowledge Graph, web search, new embedding models, vector database changes,
parser rewrites, and persistence of derived chunk metadata are excluded.
