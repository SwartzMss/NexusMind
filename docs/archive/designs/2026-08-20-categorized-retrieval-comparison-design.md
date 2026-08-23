# Categorized Retrieval Evaluation and Backend Comparison

> Historical design record. It describes the scope and assumptions at implementation time and is not the current product specification. See [the current architecture](../../architecture.md).

## Goal and boundaries

Extend NexusMind's deterministic offline retrieval evaluation so one authored
corpus and one categorized relevance dataset can compare named retrieval
backends at several bounded cutoffs. The result exposes global metrics,
per-category metrics, and deterministic failure diagnostics without changing
retrieval, ranking, provenance, persistence, or runtime routing behavior.

The change must not alter BM25 analysis or scoring, semantic cosine ranking,
Hybrid-RRF parameters or candidate depth, `KnowledgeCollection` provenance,
snapshot or SQLite schemas, or persist evaluation state. The checked-in report
is descriptive and non-gating.

## Categorized cases and datasets

`RetrievalEvaluationCase` gains one required `category` field. Categories use a
strict enum with the bounded values `exact_term`, `identifier`, `cjk`,
`paraphrase`, `cross_language`, `multi_document`, `distractor_heavy`, and
`mixed_signal`. Every case has exactly one category; no category is inferred
from its ID and unknown values fail closed.

The loader continues to require exact JSON object fields and UTF-8 input. All
checked-in datasets are explicitly migrated to include `category`. Case IDs
remain globally unique within a dataset, relevance targets remain canonical
document identities, and duplicate relevance targets remain invalid.

## Ranking and multi-K semantics

The existing single-K `evaluate_retrieval()` API remains available and keeps
its current metric definitions. The multi-K evaluator accepts a non-empty exact
tuple of positive plain integers. Values must be unique, the tuple length and
largest K are bounded by documented constants, and reports are emitted in
ascending K order regardless of caller order.

For each case and backend, evaluation performs exactly one search with
`limit=max(K)`. That returned sequence is the authoritative max-K ranking.
Every smaller cutoff is a strict prefix of that same sequence; evaluation does
not re-run retrieval, deduplicate hits, reinterpret relevance, or allow a
backend-specific ranking at a smaller K. Hit@K, Recall@K, MRR, returned targets,
returned chunk IDs, found targets, missed targets, and first relevant rank are
all derived from the prefix. A backend that cannot serve `max(K)` fails with a
controlled evaluation error rather than emitting partial reports.

Canonical relevance validation is performed once against the backend's
snapshot before any search and reused for every K.

## Metrics and diagnostics

Each per-K report retains case results and global case-weighted means for
Hit@K, Recall@K, and MRR. A category report uses the identical formulas over
only the cases in that category and includes its case count. Categories are
ordered by the enum definition and empty categories are omitted. Every case
contributes to exactly one category and all metrics must be finite values in
`[0, 1]`.

Case results carry category, query, authored relevant targets, returned targets
and chunk IDs, first relevant rank, distinct relevant targets found and missed,
and the three metrics. These fields deterministically expose complete misses,
ranking losses between cutoffs, partial multi-document recall, and backend
disagreement without generated explanations.

## Backend comparison and snapshot equality

A named backend specification contains a non-empty unique name and a factory
that builds a collection over the common benchmark corpus. Comparison preserves
the declared backend order and does not normalize or combine backend scores.
BM25-only, Semantic-only, and Hybrid-RRF are benchmark configurations, not
hard-coded branches in evaluation logic.

Before retrieval, comparison captures each backend's canonical snapshot. The
first backend establishes the reference snapshot, and every later snapshot
must equal it exactly using `KnowledgeSnapshot` value equality, including
sources, document identities, logical paths, content, metadata, and ordering.
Any difference fails before case searches with a controlled comparison error.
This guarantees all backend reports use the same canonical corpus rather than
merely similar relevance labels.

Factory, setup, snapshot, and search failures are chained as causes but exposed
through stable public messages that do not include private corpus, query, or
provider exception text. Duplicate names and malformed configuration fail
before backend construction where possible.

## Authored benchmark and semantic fixture

Add a checked-in UTF-8 benchmark of roughly 10–20 focused documents and 30–50
authored queries covering every category. It includes English and CJK terms,
mixed-language technical text, identifiers, paraphrases with low lexical
overlap, multiple relevant documents, strong distractors, near duplicates, and
mixed lexical/semantic signals. Labels are written independently of observed
backend output and are not adjusted to make Hybrid win.

The semantic fixture is an offline deterministic `EmbeddingProvider` with
bounded, dimensionally valid authored concept vectors. It creates complementary
lexical and semantic behavior but is explicitly not a real-model quality
benchmark and does not encode every relevance label as a perfect answer. No
network access, API key, random state, or private data is used.

## Renderer boundary and checked-in report

Markdown rendering is a pure function over an immutable structured comparison
report plus an immutable display configuration containing stable corpus facts
and the literal reproduction command. The renderer does not access a
`KnowledgeCollection`, invoke retrieval, read or write files, inspect the
environment, resolve machine paths, or consult clocks or randomness. It returns
one Markdown string.

A separate benchmark runner owns corpus loading, backend construction,
evaluation, calling the renderer, and optional file output. The checked-in
`evals/knowledge/benchmark.md` contains stable configuration and corpus summary,
backend and K lists, overall and per-category tables, selected deterministic
failure diagnostics, reproduction instructions, semantic-fixture limitations,
authored-label policy, and the descriptive/non-gate policy. A test regenerates
the string and compares it byte-for-byte with the checked-in UTF-8 file.

## Components

- `retrieval_evaluation.py` owns categories, validation, ranking-to-metrics
  derivation, single- and multi-K reports, category aggregation, backend
  comparison contracts, and controlled errors.
- A focused benchmark module owns the deterministic fixture provider, benchmark
  collection factories, runner configuration, and pure Markdown renderer.
- `evals/knowledge/benchmark/` owns the authored corpus and categorized cases;
  `evals/knowledge/benchmark.md` is generated output.
- Existing evaluation datasets and tests are migrated without silently adding a
  default category.

## Testing strategy

Implementation follows red-green-refactor. Unit tests cover category and JSON
validation; duplicate cases, targets, K values, and backend names; K bounds and
deterministic ordering; exactly one max-K search; prefix-derived metrics;
multi-document recall; global and category aggregates; snapshot equality;
controlled factory, setup, snapshot, search, and result-limit failures; and
backend-specific rankings and diagnostics.

Integration tests cover all three backends over the identical authored corpus,
offline repeatability, category coverage, deterministic vectors, and
byte-for-byte Markdown reproduction. The existing lexical, CJK, semantic,
hybrid, provenance, snapshot, SQLite, and evaluation suites remain regression
coverage. No quality threshold is introduced.

## Documentation

Documentation defines every category and metric, explains max-K prefix
semantics, distinguishes poor large-K recall from poor small-K ranking, records
backend and snapshot rules, describes fixture and authored-label limitations,
documents report regeneration, and explains how diagnostics may inform future
work without prescribing reranking, persistence, ANN indexing, or another
out-of-scope feature.
