# Categorized Retrieval Backend Comparison

10 UTF-8 documents and 32 authored relevance cases. The baseline is descriptive/non-gate.
All cutoffs are strict prefixes of one search at `max(K)` per case.
Backends are compared only after their canonical snapshots are exactly equal.

## Configuration

- Backends: BM25-only, Semantic-only, Hybrid-RRF
- K values: 1, 3, 5, 10
- Semantic vectors: deterministic authored concept fixture; not real-model quality
- Relevance labels: authored independently of backend output

## Overall metrics

| Backend | K | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 0.843750 | 0.765625 | 0.843750 |
| BM25-only | 3 | 0.968750 | 0.921875 | 0.895833 |
| BM25-only | 5 | 0.968750 | 0.921875 | 0.895833 |
| BM25-only | 10 | 0.968750 | 0.921875 | 0.895833 |
| Semantic-only | 1 | 0.750000 | 0.671875 | 0.750000 |
| Semantic-only | 3 | 0.968750 | 0.937500 | 0.854167 |
| Semantic-only | 5 | 1.000000 | 0.984375 | 0.861979 |
| Semantic-only | 10 | 1.000000 | 1.000000 | 0.861979 |
| Hybrid-RRF | 1 | 0.812500 | 0.734375 | 0.812500 |
| Hybrid-RRF | 3 | 0.968750 | 0.921875 | 0.885417 |
| Hybrid-RRF | 5 | 0.968750 | 0.968750 | 0.885417 |
| Hybrid-RRF | 10 | 1.000000 | 1.000000 | 0.888889 |

## Per-category metrics

### exact_term

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |

### identifier

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |

### cjk

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 1 | 4 | 0.500000 | 0.500000 | 0.500000 |
| Semantic-only | 3 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Semantic-only | 5 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Hybrid-RRF | 1 | 4 | 0.750000 | 0.750000 | 0.750000 |
| Hybrid-RRF | 3 | 4 | 1.000000 | 1.000000 | 0.875000 |
| Hybrid-RRF | 5 | 4 | 1.000000 | 1.000000 | 0.875000 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 0.875000 |

### paraphrase

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 0.250000 | 0.250000 | 0.250000 |
| BM25-only | 3 | 4 | 0.750000 | 0.750000 | 0.458333 |
| BM25-only | 5 | 4 | 0.750000 | 0.750000 | 0.458333 |
| BM25-only | 10 | 4 | 0.750000 | 0.750000 | 0.458333 |
| Semantic-only | 1 | 4 | 0.500000 | 0.500000 | 0.500000 |
| Semantic-only | 3 | 4 | 0.750000 | 0.750000 | 0.583333 |
| Semantic-only | 5 | 4 | 1.000000 | 1.000000 | 0.645833 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 0.645833 |
| Hybrid-RRF | 1 | 4 | 0.500000 | 0.500000 | 0.500000 |
| Hybrid-RRF | 3 | 4 | 0.750000 | 0.750000 | 0.583333 |
| Hybrid-RRF | 5 | 4 | 0.750000 | 0.750000 | 0.583333 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 0.611111 |

### cross_language

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 1.000000 | 0.750000 | 1.000000 |
| BM25-only | 3 | 4 | 1.000000 | 0.875000 | 1.000000 |
| BM25-only | 5 | 4 | 1.000000 | 0.875000 | 1.000000 |
| BM25-only | 10 | 4 | 1.000000 | 0.875000 | 1.000000 |
| Semantic-only | 1 | 4 | 0.750000 | 0.500000 | 0.750000 |
| Semantic-only | 3 | 4 | 1.000000 | 1.000000 | 0.875000 |
| Semantic-only | 5 | 4 | 1.000000 | 1.000000 | 0.875000 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 0.875000 |
| Hybrid-RRF | 1 | 4 | 1.000000 | 0.750000 | 1.000000 |
| Hybrid-RRF | 3 | 4 | 1.000000 | 0.875000 | 1.000000 |
| Hybrid-RRF | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |

### multi_document

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 0.750000 | 0.375000 | 0.750000 |
| BM25-only | 3 | 4 | 1.000000 | 0.750000 | 0.875000 |
| BM25-only | 5 | 4 | 1.000000 | 0.750000 | 0.875000 |
| BM25-only | 10 | 4 | 1.000000 | 0.750000 | 0.875000 |
| Semantic-only | 1 | 4 | 0.750000 | 0.375000 | 0.750000 |
| Semantic-only | 3 | 4 | 1.000000 | 0.750000 | 0.875000 |
| Semantic-only | 5 | 4 | 1.000000 | 0.875000 | 0.875000 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 0.875000 |
| Hybrid-RRF | 1 | 4 | 0.750000 | 0.375000 | 0.750000 |
| Hybrid-RRF | 3 | 4 | 1.000000 | 0.750000 | 0.875000 |
| Hybrid-RRF | 5 | 4 | 1.000000 | 1.000000 | 0.875000 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 0.875000 |

### distractor_heavy

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 0.750000 | 0.750000 | 0.750000 |
| BM25-only | 3 | 4 | 1.000000 | 1.000000 | 0.833333 |
| BM25-only | 5 | 4 | 1.000000 | 1.000000 | 0.833333 |
| BM25-only | 10 | 4 | 1.000000 | 1.000000 | 0.833333 |
| Semantic-only | 1 | 4 | 0.500000 | 0.500000 | 0.500000 |
| Semantic-only | 3 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Semantic-only | 5 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Hybrid-RRF | 1 | 4 | 0.500000 | 0.500000 | 0.500000 |
| Hybrid-RRF | 3 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Hybrid-RRF | 5 | 4 | 1.000000 | 1.000000 | 0.750000 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 0.750000 |

### mixed_signal

| Backend | K | Cases | Hit@K | Recall@K | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25-only | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| BM25-only | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Semantic-only | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 1 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 3 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 5 | 4 | 1.000000 | 1.000000 | 1.000000 |
| Hybrid-RRF | 10 | 4 | 1.000000 | 1.000000 | 1.000000 |

## Selected diagnostics

### case: para-remote-service
- category: paraphrase
- query: How can a mobile app call a service in another process?
- relevant targets: benchmark-docs/ipc.md
- BM25-only: first relevant rank=2; found=ipc.md; missed=none; returned chunks=chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe
- Semantic-only: first relevant rank=1; found=ipc.md; missed=none; returned chunks=chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=1; found=ipc.md; missed=none; returned chunks=chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59

### case: para-resume
- category: paraphrase
- query: continue a workflow after the process starts again
- relevant targets: benchmark-docs/mixed-language.md
- BM25-only: first relevant rank=missed; found=none; missed=mixed-language.md; returned chunks=chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236
- Semantic-only: first relevant rank=4; found=mixed-language.md; missed=none; returned chunks=chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=9; found=mixed-language.md; missed=none; returned chunks=chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59

### case: para-ranking
- category: paraphrase
- query: tell whether useful evidence is present but ordered too late
- relevant targets: benchmark-docs/retrieval.md
- BM25-only: first relevant rank=3; found=retrieval.md; missed=none; returned chunks=chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45
- Semantic-only: first relevant rank=3; found=retrieval.md; missed=none; returned chunks=chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=3; found=retrieval.md; missed=none; returned chunks=chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900

### case: cross-binder-cn
- category: cross_language
- query: mobile remote procedure 跨进程 调用方身份
- relevant targets: benchmark-docs/cjk.md, benchmark-docs/ipc.md
- BM25-only: first relevant rank=1; found=cjk.md; missed=ipc.md; returned chunks=chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e
- Semantic-only: first relevant rank=1; found=cjk.md, ipc.md; missed=none; returned chunks=chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=1; found=cjk.md, ipc.md; missed=none; returned chunks=chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59

### case: cross-trustzone
- category: cross_language
- query: Arm 安全世界 isolation ordinary OS
- relevant targets: benchmark-docs/security.md, benchmark-docs/cjk.md
- BM25-only: first relevant rank=1; found=cjk.md, security.md; missed=none; returned chunks=chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe
- Semantic-only: first relevant rank=1; found=cjk.md, security.md; missed=none; returned chunks=chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=1; found=cjk.md, security.md; missed=none; returned chunks=chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59

### case: multi-ipc
- category: multi_document
- query: isolated services communicate across process boundaries
- relevant targets: benchmark-docs/ipc.md, benchmark-docs/cjk.md
- BM25-only: first relevant rank=1; found=ipc.md; missed=cjk.md; returned chunks=chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe
- Semantic-only: first relevant rank=1; found=ipc.md, cjk.md; missed=none; returned chunks=chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=1; found=ipc.md, cjk.md; missed=none; returned chunks=chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59

### case: multi-security
- category: multi_document
- query: separate protected execution from ordinary applications
- relevant targets: benchmark-docs/security.md, benchmark-docs/cjk.md
- BM25-only: first relevant rank=1; found=security.md; missed=cjk.md; returned chunks=chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e
- Semantic-only: first relevant rank=2; found=cjk.md, security.md; missed=none; returned chunks=chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=2; found=security.md, cjk.md; missed=none; returned chunks=chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900

### case: multi-ranking
- category: multi_document
- query: ranking metrics and reciprocal fusion
- relevant targets: benchmark-docs/retrieval.md, benchmark-docs/mixed-language.md
- BM25-only: first relevant rank=2; found=mixed-language.md, retrieval.md; missed=none; returned chunks=chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45
- Semantic-only: first relevant rank=1; found=mixed-language.md, retrieval.md; missed=none; returned chunks=chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=1; found=mixed-language.md, retrieval.md; missed=none; returned chunks=chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900

### case: multi-runtime
- category: multi_document
- query: canonical runtime state and durable workflow state
- relevant targets: benchmark-docs/api.md, benchmark-docs/mixed-language.md
- BM25-only: first relevant rank=1; found=api.md, mixed-language.md; missed=none; returned chunks=chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d
- Semantic-only: first relevant rank=1; found=api.md, mixed-language.md; missed=none; returned chunks=chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=1; found=api.md, mixed-language.md; missed=none; returned chunks=chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900

### case: distractor-recall
- category: distractor_heavy
- query: recall useful evidence sources
- relevant targets: benchmark-docs/retrieval.md
- BM25-only: first relevant rank=3; found=retrieval.md; missed=none; returned chunks=chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236
- Semantic-only: first relevant rank=1; found=retrieval.md; missed=none; returned chunks=chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59
- Hybrid-RRF: first relevant rank=2; found=retrieval.md; missed=none; returned chunks=chunk-c4cfff3162f53055e3fae7427b3d72db4af32d593f5f5b03d010a00aaa55d22d, chunk-c04353c3b46b9b36e0f5756427ebed09e0ae165d621d2d9bcb1bafa829717236, chunk-1b534f81fc92740e29d67fcbc7617bc18b6afdec6c2bf4619e3c6b8a328456d0, chunk-b49e3dfd5266a84a31c58db70a65699182fc234c29dcd39458134c3873c08b45, chunk-e416f89f014b1be6fdb86f81f0a5409dc4bc23b5ee18064947638a2fd5e6d78e, chunk-db368da093f76f01b73888b2bfd69e10cf0c73d8b907b9d1f7afec1a644cb66b, chunk-367f366a67327bc038e18a20076ab11ca6fd9b45b57f21568e1358d25676a9fe, chunk-addda90c64af6311fe4abe0f2a107d9601296cee844039305c16a904d4209a13, chunk-a129f466775d0ebe20a551f68328f9d126e6278254a84a6687adce0d23f5a900, chunk-318715c75a138dded497b89777f7b751697391ab89d50fcc9846c11370997d59

## Reproduction

```bash
PYTHONPATH=src python -m nexusmind.retrieval_benchmark --write evals/knowledge/benchmark.md
```

This report exposes regressions for review but defines no quality threshold.
