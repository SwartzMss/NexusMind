# Document-Aware Search Diversification

This offline lexical evaluation compares raw backend Top-K diagnostics with
the final document-aware search selection. The relevance safeguard uses only
a same-query relative score window; backend scores and raw diagnostics are unchanged.

## Aggregate safeguards

- K: 5
- Broad relevant-document coverage: 6 raw -> 8 diversified
- Precise MRR: 1.000000 raw -> 1.000000 diversified
- Precise Recall@K: 1.000000 raw -> 1.000000 diversified

## Per-query metrics

| Case | Category | Raw unique/relevant | Diversified unique/relevant | Raw Hit/Recall/MRR | Diversified Hit/Recall/MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| broad-crypto | multi_document | 1/1 | 2/2 | 1.000000/0.500000/1.000000 | 1.000000/1.000000/1.000000 |
| broad-binder | multi_document | 2/2 | 2/2 | 1.000000/1.000000/1.000000 | 1.000000/1.000000/1.000000 |
| broad-qnx | multi_document | 1/1 | 2/2 | 1.000000/0.500000/1.000000 | 1.000000/1.000000/1.000000 |
| broad-permission | multi_document | 2/2 | 2/2 | 1.000000/1.000000/1.000000 | 1.000000/1.000000/1.000000 |
| precise-import | exact_term | 2/1 | 2/1 | 1.000000/1.000000/1.000000 | 1.000000/1.000000/1.000000 |

## Rankings

### broad-crypto: Crypto
- Relevant documents: crypto-overview.md, crypto-permissions.md
- Raw documents: crypto-overview.md, crypto-overview.md, crypto-overview.md, crypto-overview.md, crypto-overview.md
- Diversified documents: crypto-overview.md, crypto-overview.md, crypto-overview.md, crypto-permissions.md, crypto-permissions.md

### broad-binder: Binder
- Relevant documents: binder.md, crypto-permissions.md
- Raw documents: binder.md, binder.md, crypto-permissions.md, binder.md, binder.md
- Diversified documents: binder.md, binder.md, crypto-permissions.md, binder.md, binder.md

### broad-qnx: QNX
- Relevant documents: qnx.md, binder.md
- Raw documents: qnx.md, qnx.md, qnx.md, qnx.md, qnx.md
- Diversified documents: qnx.md, qnx.md, qnx.md, qnx.md, binder.md

### broad-permission: 权限校验
- Relevant documents: crypto-permissions.md, precise-import.md
- Raw documents: crypto-permissions.md, crypto-permissions.md, precise-import.md
- Diversified documents: crypto-permissions.md, crypto-permissions.md, precise-import.md

### precise-import: lpRpcCrypto ImportFile exact permission flow
- Relevant documents: precise-import.md
- Raw documents: precise-import.md, precise-import.md, precise-import.md, crypto-permissions.md, crypto-permissions.md
- Diversified documents: precise-import.md, precise-import.md, precise-import.md, crypto-permissions.md, crypto-permissions.md

## Reproduction

```bash
PYTHONPATH=src python -m nexusmind.search_diversification_benchmark --write evals/knowledge/diversification.md
```
