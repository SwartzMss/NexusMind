# CJK lexical retrieval baseline

This descriptive, non-gate baseline compares two lexical policies on a small,
copyright-safe Chinese fixture. The corpus contains seven UTF-8 Markdown files
under source `cjk-docs`: four canonical documents covering Android/Binder IPC,
Arm TrustZone isolation, QNX microkernel fault isolation, and AES-GCM/HKDF,
plus plausible near neighbors about iOS/XPC, virtualization, and the TLS 1.3
key schedule. The ten queries and their
canonical `(source_id, logical_path)` relevance labels were authored before
the revised corpus was ranked. The near neighbors and three harder paraphrases
were added without changing any existing relevance target.

## Fixed configuration

- ingestion: `LocalDirectoryAdapter` over `evals/knowledge/cjk/corpus`
- composition: `KnowledgeCollection` with provenance-preserving search results
- chunking: `TextChunker(chunk_size=240, overlap=40)`
- index: `InMemoryChunkIndex`, BM25 `k1=1.2`, `b=0.75`
- cutoff: `k=3`
- Unicode policy: `UnicodeCJKLexicalAnalyzer` with NFKC word runs and
  overlapping Han bigram tokens
- comparison policy: `WhitespaceLexicalAnalyzer` with case-folded whitespace
  splitting

## Recorded metrics

| Analyzer | Hit@3 | Recall@3 | MRR |
| --- | ---: | ---: | ---: |
| `UnicodeCJKLexicalAnalyzer` | 1.000000 | 1.000000 | 0.850000 |
| `WhitespaceLexicalAnalyzer` | 0.600000 | 0.600000 | 0.500000 |

Reproduce offline from the repository root with:

```console
pytest -q tests/test_retrieval_evaluation_cjk.py
```

These values document the checked-in fixture; they are not exact aggregate
release gates. Tests require determinism, bounded metrics, a qualitative Hit@k
improvement, and non-saturated Unicode MRR as a fixture-quality guard instead
of pinning the exact metric triplet.

Han bigram analysis improves matches when Chinese prose has no spaces, but it
has limitations: it does not perform word segmentation, can over-match common
adjacent characters, and cannot by itself resolve synonyms or meaning. Mixed
Latin terms remain ordinary word tokens. The fixture deliberately mixes
Chinese paraphrases with Latin technical terms, so the comparison exposes the
whitespace policy's token-boundary limitation without claiming general search
quality.
