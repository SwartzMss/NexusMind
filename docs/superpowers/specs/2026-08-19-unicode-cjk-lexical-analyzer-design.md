# Unicode/CJK Lexical Analyzer Design

## Context

NexusMind's in-memory BM25 backend currently embeds whitespace splitting and
`str.casefold()` directly in search and statistics construction. That makes
punctuation part of a token and leaves an unspaced Han sentence as one token.
Issue #71 introduces an explicit analyzer boundary so lexical improvements are
deterministic, measurable, and independent of canonical knowledge persistence.

## Scope and invariants

This change adds dependency-free lexical analysis only. It does not change the
BM25 formula, `k1` or `b`, positive IDF, result ordering, chunking, provenance,
snapshot contents, or SQLite schema. Sources and Documents remain canonical;
chunks, analyzer tokens, and BM25 statistics remain derived runtime state.

`InMemoryChunkIndex()` will default to the Unicode/CJK analyzer. Callers that
need exact historical behavior can explicitly provide the whitespace analyzer.

## Analyzer contract and implementations

`LexicalAnalyzer` is a source-neutral protocol with one method:

```python
class LexicalAnalyzer(Protocol):
    def analyze(self, text: str) -> tuple[str, ...]: ...
```

The index validates analyzer output at its trust boundary. The result must be
an exact tuple containing only non-empty plain strings. Analyzer exceptions and
invalid output fail closed. The index retains no mutable alias to analyzer
output.

Two frozen, stateless implementations are public package exports:

- `WhitespaceLexicalAnalyzer` uses `text.split()` followed by `casefold()` and
  exactly preserves the legacy behavior, including punctuation attachment.
- `UnicodeCJKLexicalAnalyzer` first applies Unicode NFKC normalization, then
  extracts tokens deterministically without locale-sensitive behavior.

The Unicode/CJK analyzer classifies characters as follows:

- Han characters are the code points in the explicitly documented CJK Unified
  Ideographs ranges supported by the implementation. A contiguous Han run
  emits overlapping character bigrams. A one-character run emits that single
  character.
- All other Unicode letters and numbers form contiguous word-like runs. Each
  run is normalized with `casefold()` and emitted as one token.
- Whitespace, punctuation, symbols, marks outside a word run, controls, and
  unassigned characters are boundaries and are not emitted.
- A transition between Han and non-Han letters or numbers is a boundary. Thus
  mixed text such as `Android 14支持Binder通信` produces stable Latin/digit
  tokens and Han bigrams without cross-script tokens.

NFKC is chosen so compatibility-equivalent spellings, including full-width
Latin letters and digits, share lexical terms. This is a retrieval
normalization policy, not a mutation of Document content.

## Index state and atomic mutation

`InMemoryChunkIndex` receives an analyzer as runtime configuration. Analyzer
instances for this issue are immutable and stateless; `clone()` preserves the
same analyzer semantics while copying all mutable index state independently.

The index stores derived per-chunk analyzed tokens plus the BM25 TF/DF and
length statistics needed by the existing scorer. Add and replacement operations
build and validate the complete candidate derived state before committing it.
If analysis, output validation, a resource check, or statistics construction
fails, the previous chunks and statistics remain untouched. Removal follows the
same candidate-state path so all mutation methods share one atomic rebuild
invariant.

This full candidate rebuild is preferred over incremental postings in this
issue: it is easier to audit, preserves current bounded in-memory scale, and
avoids introducing a second persistence/index subsystem.

## Search and resource limits

`max_query_chars` remains a bound on the original query string and is checked
before analysis. The analyzer then runs once, its output is validated, and
`max_query_terms` is applied to the raw analyzed tuple before duplicate removal.
Counting before deduplication ensures repeated input and CJK token amplification
cannot bypass the resource limit. BM25 query behavior continues to use the
stable first occurrence of each distinct analyzed term.

Corpus analysis remains bounded by the existing document, chunk, and character
limits. Because the built-in Unicode/CJK analyzer emits at most one token per
input code point, it has a clear finite amplification bound and needs no new
general token-budget framework. Custom analyzers must return a finite tuple;
candidate state is not committed until all output has been validated.

## KnowledgeCollection and persistence

No analyzer configuration or token state is added to `KnowledgeSnapshot` or
SQLite. A collection restore reads canonical Sources and Documents, re-chunks
the Documents, and rebuilds the index using the analyzer configured on the
current collection/index. Provenance resolution and canonical chunk coherence
checks are unchanged.

## Evaluation fixture

A checked-in `evals/knowledge/cjk/` fixture will contain a small original
Chinese and mixed Chinese/Latin corpus, document-level relevance labels, and a
baseline report. Tests run the fixture through
`LocalDirectoryAdapter -> KnowledgeCollection -> TextChunker ->
InMemoryChunkIndex -> KnowledgeSearchResult -> retrieval evaluator` with both
analyzers.

The report records the fixed chunker, BM25 settings, cutoff, analyzer
configuration, whitespace metrics, Unicode/CJK metrics, and known bigram
limitations. Exact metric values are descriptive output, not CI quality gates;
tests assert determinism, valid ranges, fixture coverage, and that authored
cases expose the intended lexical difference.

Known limitations are explicit: character bigrams do not understand word
boundaries, synonyms, or semantics; isolated-character queries have weak
specificity; and compatibility normalization may intentionally conflate some
visual forms.

## Testing strategy

Unit tests cover public exports, legacy compatibility, NFKC, Unicode casefold,
punctuation boundaries, Han runs and single characters, mixed scripts, stable
output, and absence of empty tokens. Index tests prove the same analyzer is used
for corpus and query text, analyzed-token query limits, BM25 statistics,
replace/remove rebuilds, analyzer failure atomicity, and clone independence.

Integration tests cover punctuation and CJK retrieval through
`KnowledgeCollection`, restore with the current analyzer, unchanged persistence
schema, and deterministic comparative evaluation. The existing retrieval,
provenance, snapshot, store, and evaluation suites remain regression coverage.

## Non-goals

This design does not add dictionary segmentation, stemming, stop words,
synonyms, language detection, query rewriting, BM25 tuning, FTS, persistent
postings, embeddings, semantic or hybrid retrieval, reranking, external data,
or RAG behavior.
