# Structure-aware Chunking Benchmark

This deterministic offline gate compares `TextChunker(chunk_size=80, overlap=0)` with
`StructureAwareChunker(chunk_size=80, overlap=0)` on the same Markdown corpus and BM25 queries.

- Baseline Top-1 Hit / MRR: 0.75 / 0.75
- Structure-aware Top-1 Hit / MRR: 1.00 / 1.00
- Both strategies retain full Recall@3.
- The targeted gain is the `fixed-window-boundary` case; code, table, and list cases do not regress.
- Repeated candidate runs must return identical metrics and chunk IDs.

Reproduce with:

```powershell
python -m pytest tests/test_structure_chunking_benchmark.py -q
```
