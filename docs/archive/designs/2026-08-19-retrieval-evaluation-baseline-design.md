# Retrieval Evaluation Baseline Design

> Historical design record. It describes the scope and assumptions at implementation time and is not the current product specification. See [the current architecture](../../architecture.md).

## Goal

Add a deterministic offline evaluation boundary that measures the real provenance-resolved Knowledge search path over a small checked-in corpus, without changing retrieval or chunking behavior.

## Contracts

Create `src/nexusmind/retrieval_evaluation.py` and export its public contracts from the package root.

`RetrievalTarget(source_id, logical_path)` is a frozen, slotted canonical Document identity. Both fields must be non-empty strings.

`RetrievalEvaluationCase(case_id, query, relevant_documents)` is frozen and slotted. IDs and queries must be non-empty strings, `relevant_documents` must be an exact non-empty tuple of unique `RetrievalTarget` values, and duplicate targets are rejected.

`RetrievalEvaluationCaseResult` contains:

- case ID and query;
- returned targets in actual chunk-result rank order, including repeated documents;
- returned chunk IDs in the same order;
- unique relevant targets found in first-hit order;
- relevant targets missed in declared-label order;
- first relevant chunk rank or `None`;
- Hit@K, Recall@K, and reciprocal-rank floats.

`RetrievalEvaluationReport` contains `k`, ordered case results, and aggregate mean Hit@K, Recall@K, and MRR.

All containers are immutable bounded tuples. The contracts retain no mutable aliases.

## Evaluation semantics

`evaluate_retrieval(collection, cases, k=5)` requires an exact non-empty tuple of unique case IDs and a positive plain-integer `k`. Before any query runs, it snapshots the committed collection and validates every relevance target against the snapshot's `(source_id, logical_path)` identities. Broken labels raise a controlled `RetrievalEvaluationError`, rather than becoming false misses.

Each case calls `KnowledgeCollection.search(case.query, limit=k)`. Relevance comparisons use `result.source.source_id` and `result.document.logical_path`; chunk IDs never define relevance.

Ranking remains chunk-level and diagnostics preserve every returned result. For a case:

```text
Hit@K = 1.0 when any relevant target appears, else 0.0

Recall@K = distinct relevant targets observed / declared relevant targets

Reciprocal Rank = 1 / rank of first relevant chunk, else 0.0
```

Repeated chunks from one relevant document remain visible in returned diagnostics, count once for Recall, and retain their actual ranks for reciprocal-rank semantics. Aggregate values are arithmetic means of case values and are finite floats in `[0.0, 1.0]`. Input case order is preserved.

The evaluator only reads snapshot/search results and does not mutate the collection.

## Dataset loader

`load_retrieval_evaluation_cases(path)` reads UTF-8 JSON with this strict schema:

```json
{
  "cases": [
    {
      "case_id": "trustzone-secure-world",
      "query": "TrustZone secure world",
      "relevant_documents": [
        {"source_id": "eval-corpus", "logical_path": "trustzone.md"}
      ]
    }
  ]
}
```

The loader rejects invalid JSON, non-object roots, missing or extra fields, non-list collections, empty cases, invalid field values, duplicate targets, and duplicate case IDs with `RetrievalEvaluationDatasetError`. JSON arrays are converted to validated tuple contracts.

## Checked-in baseline

Add authored, copyright-safe files under:

```text
evals/knowledge/
├── corpus/
│   ├── trustzone.md
│   ├── qnx.md
│   ├── android.md
│   ├── cryptography.md
│   └── retrieval.md
├── cases.json
└── baseline.md
```

The 12–15 labels cover exact terms, multi-term queries, rare terms, and distractors. The integration test uses:

- `LocalDirectoryAdapter(source_id="eval-corpus")`;
- `TextChunker(chunk_size=240, overlap=40)`;
- the default `InMemoryChunkIndex` BM25 constants;
- `k=5`.

At least some documents exceed one chunk so the baseline measures chunk-ranked/document-relevant behavior. The end-to-end test loads the checked-in JSON, ingests the directory through the real adapter, evaluates twice, and asserts equal reports, bounded metrics, and complete case execution. Measured values are copied verbatim into `baseline.md` with the reproduction command, but are not hard-coded as CI quality thresholds.

The initial numbers are descriptive, not quality thresholds or a statistically representative benchmark.

## Testing

Unit tests cover contract validation, strict loader failures, duplicate IDs and targets, empty cases, invalid `k` including bool, hit/miss, multi-target recall, duplicate returned-document chunks, reciprocal rank at first/later/no hit, aggregate means, rank-preserving diagnostics, label coherence, collection-state non-mutation, Document identity rather than chunk identity, and deterministic repetition.

The checked-in integration test covers JSON loading and the full local adapter -> collection -> chunker -> BM25 -> provenance -> evaluator flow entirely offline. The full existing suite must remain green.

## Non-goals

This issue does not tune BM25 parameters, IDF, tokenization, chunking defaults, or tie-breaking. It does not add a CLI, release gate, persistent experiment store, external benchmark, generated labels, semantic/vector/hybrid retrieval, reranking, query rewriting, LLM judge, dashboard, API, or RAG evaluation.
