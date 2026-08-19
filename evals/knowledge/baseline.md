# Retrieval Evaluation Baseline

This directory contains NexusMind's first deterministic offline retrieval baseline. It is an authored regression fixture, not a statistically representative public benchmark or a release gate.

## Configuration

- Corpus: `evals/knowledge/corpus/*.md` (5 original documents)
- Labels: `evals/knowledge/cases.json` (15 cases)
- Canonical source ID: `eval-corpus`
- Ingestion: `LocalDirectoryAdapter`
- Chunker: `TextChunker(chunk_size=240, overlap=40)`
- Retrieval: `InMemoryChunkIndex` BM25 (`k1=1.2`, `b=0.75`)
- Current analyzer: explicit `UnicodeCJKLexicalAnalyzer`
- Evaluation cutoff: `k=5`

## Semantics

Relevance identity is the canonical Document pair `(source_id, logical_path)`. Retrieval ranking remains chunk-level. Every returned chunk remains in diagnostics, including repeated chunks from the same document.

- Hit@5 is `1.0` when any relevant document occurs in the first five chunks, otherwise `0.0`.
- Recall@5 is the number of distinct relevant documents represented in the first five chunks divided by the number of declared relevant documents. Repeated chunks from one document count once.
- Reciprocal rank is `1 / rank` for the first relevant chunk within the first five, otherwise `0.0`.
- Aggregate values are arithmetic means of the 15 per-case values.

## Recorded results

The original #69 baseline used the then-default whitespace analyzer. Issue #71
changed the default index analyzer, so both configurations are recorded rather
than silently carrying the old values forward.

### Previous baseline: `WhitespaceLexicalAnalyzer`

| Metric | Value |
|---|---:|
| Hit@5 | 0.933333 |
| Recall@5 | 0.933333 |
| MRR@5 | 0.822222 |

### Current baseline: `UnicodeCJKLexicalAnalyzer`

| Metric | Value | Change from whitespace |
|---|---:|---:|
| Hit@5 | 0.933333 | 0.000000 |
| Recall@5 | 0.900000 | -0.033333 |
| MRR@5 | 0.866667 | +0.044445 |

The higher MRR and lower Recall@5 show that this analyzer change is not a
uniform quality improvement. These movements are diagnostic data for future
semantic or hybrid retrieval work, not acceptance thresholds.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_retrieval_evaluation_baseline.py -vv
```

The test explicitly configures `UnicodeCJKLexicalAnalyzer`, ingests the
checked-in corpus through the real local adapter, synchronizes a
`KnowledgeCollection`, rebuilds the BM25 index, resolves canonical provenance,
loads the strict JSON labels, and asserts identical repeated reports with
bounded metrics. It deliberately does not hard-code the values above as CI
quality thresholds. The run requires no network, model, generated labels, or
external data. To reproduce the previous row, use the same configuration with
`WhitespaceLexicalAnalyzer` as the index analyzer.

## Limitations

The corpus is intentionally small and technical, the relevance judgments are authored for regression coverage, and `k=5` is the only recorded cutoff. Several paraphrase and distractor-overlap cases intentionally expose lexical misses and later-ranked relevant chunks, leaving room to measure future retrieval improvements. Future changes should report movement against this baseline; a separate policy must define any eventual quality gate. BM25 parameters, analyzer behavior, chunking, and labels must not be silently tuned merely to preserve or improve these numbers.
