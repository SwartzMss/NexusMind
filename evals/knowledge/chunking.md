# Structure-aware Chunking Benchmark

This deterministic offline gate compares `TextChunker(chunk_size=80, overlap=0)` with
`StructureAwareChunker(chunk_size=80, overlap=0)` on the same Markdown corpus and BM25 queries.
The structure-aware candidate also indexes the derived heading path while keeping
the returned chunk content as an exact canonical slice.

- Baseline Top-1 Hit / Precision / MRR: 0.80 / 0.80 / 0.80
- Structure-aware Top-1 Hit / Precision / MRR: 1.00 / 1.00 / 1.00
- Both strategies retain full Recall@3.
- The targeted gains are the `fixed-window-boundary` and `nested-heading-context` cases;
  code, table, and list cases do not regress.
- At K=3, Recall remains 1.00 for both strategies; precision is document-target precision
  and can be lower when multiple chunks from the same document are returned.
- Repeated candidate runs must return identical metrics and chunk IDs.

Reproduce with:

```powershell
python -m pytest tests/test_structure_chunking_benchmark.py -q
```
