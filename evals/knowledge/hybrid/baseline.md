# Hybrid Retrieval Fixture Baseline

This baseline is a descriptive, non-gate comparison over an authored offline
fixture. It exercises complementary lexical, semantic, and mixed-signal cases;
it is not evidence of production embedding-model quality and is not tuned to
make Hybrid-RRF win every metric.

Configuration: `chunk_size=240`, `overlap=40`, `k=2`, `rrf_k=60`, and hybrid
candidate depth 100. All modes use the same corpus and canonical relevance
labels. Semantic vectors are deterministic test fixtures.

| Mode | Hit@2 | Recall@2 | MRR |
|---|---:|---:|---:|
| BM25-only | 0.666667 | 0.666667 | 0.666667 |
| Semantic-only | 0.666667 | 0.666667 | 0.666667 |
| Hybrid-RRF | 1.000000 | 1.000000 | 1.000000 |

The exact identifier case is missed by Semantic-only at `k=2`, the credential
paraphrase is missed by BM25-only, and the checkpoint case supplies signal to
both. Hybrid-RRF covers all three in this deliberately small fixture; that is a
fixture observation, not a general claim that hybrid always wins.
RRF combines ranks only; it never adds BM25 and cosine score magnitudes.

Reproduce with:

```text
pytest -q tests/test_retrieval_evaluation_hybrid.py
```
