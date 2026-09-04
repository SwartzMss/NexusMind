# Section-Aware Context Expansion Benchmark

The benchmark holds ranked retrieval anchors constant and evaluates deterministic context expansion.

| Case | Anchor retention | Relevant coverage | Expansion precision | Irrelevant expansion | Boundary skips |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi-document-budget | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1 |
| next-caveat | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0 |
| previous-definition | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0 |
| sibling-boundary | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1 |

## Reproduction

    PYTHONPATH=src python -m nexusmind.context_expansion_evaluation --write evals/knowledge/context_expansion/baseline.md

All fixtures, anchors, labels, budgets, ordering, and metric formatting are deterministic.
