# Semantic Retrieval Fixture Baseline

This is a descriptive, non-gate baseline for the deterministic semantic
fixture. It validates semantic retrieval plumbing without claiming the vectors
represent the quality, latency, or cost of a production embedding model.

Configuration:

- Corpus: four original UTF-8 Markdown documents in `corpus/`
- Cases: eight labels in `cases.json`
- Provider: fixed four-dimensional test fixture provider
- Chunker: `TextChunker(chunk_size=240, overlap=40)`
- Index: `InMemorySemanticChunkIndex` brute-force cosine search
- Evaluation: `k=3`

Recorded results:

| Metric | Value |
| --- | ---: |
| Hit@3 | 1.000000 |
| Recall@3 | 1.000000 |
| MRR@3 | 0.937500 |

The `android-app-boundary` case ranks the closely related protected-service
concept first and Android second, deliberately retaining honest ranking
headroom. Aggregate values are recorded for observation and are not asserted as
release thresholds.

Reproduce from the repository root:

```bash
PYTHONPATH=src python -m pytest -q tests/test_retrieval_evaluation_semantic.py
```

Limitations: the fixture uses authored concept vectors and a tiny corpus. It
tests batching, query/document separation, cosine ordering, canonical
provenance, and evaluator integration; it does not compare real embedding
models or measure general semantic retrieval quality.
