# Retrieval Evaluation Baseline

This directory contains NexusMind's first deterministic offline retrieval baseline. It is an authored regression fixture, not a statistically representative public benchmark or a release gate.

## Configuration

- Corpus: `evals/knowledge/corpus/*.md` (5 original documents)
- Labels: `evals/knowledge/cases.json` (15 cases)
- Canonical source ID: `eval-corpus`
- Ingestion: `LocalDirectoryAdapter`
- Chunker: `TextChunker(chunk_size=240, overlap=40)`
- Retrieval: default `InMemoryChunkIndex` BM25 (`k1=1.2`, `b=0.75`)
- Evaluation cutoff: `k=5`

## Semantics

Relevance identity is the canonical Document pair `(source_id, logical_path)`. Retrieval ranking remains chunk-level. Every returned chunk remains in diagnostics, including repeated chunks from the same document.

- Hit@5 is `1.0` when any relevant document occurs in the first five chunks, otherwise `0.0`.
- Recall@5 is the number of distinct relevant documents represented in the first five chunks divided by the number of declared relevant documents. Repeated chunks from one document count once.
- Reciprocal rank is `1 / rank` for the first relevant chunk within the first five, otherwise `0.0`.
- Aggregate values are arithmetic means of the 15 per-case values.

## Recorded result

| Metric | Value |
|---|---:|
| Hit@5 | 1.000000 |
| Recall@5 | 1.000000 |
| MRR@5 | 1.000000 |

## Reproduce

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation_baseline.py -vv
```

The test ingests the checked-in corpus through the real local adapter, synchronizes a `KnowledgeCollection`, rebuilds the BM25 index, resolves canonical provenance, loads the strict JSON labels, evaluates twice, and asserts identical reports and the metrics above. It requires no network, model, generated labels, or external data.

## Limitations

The corpus is intentionally small and technical, the relevance judgments are authored for regression coverage, and `k=5` is the only recorded cutoff. Perfect initial scores do not imply general retrieval quality. Future retrieval changes should first report movement against this baseline; BM25 parameters, analyzer behavior, chunking, and labels must not be silently tuned merely to preserve or improve these numbers.
