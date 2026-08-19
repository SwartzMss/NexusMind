# Lexical Retrieval Evaluation

BM25 combines term frequency, inverse document frequency, and document length normalization to rank lexical matches. Rare query terms generally contribute more than ubiquitous terms, while repeated occurrences have diminishing returns. Deterministic tie breaking makes repeated experiments comparable.

Retrieval evaluation separates ranked chunk results from document-level relevance labels. Hit at K checks whether any relevant document appears. Recall at K measures distinct relevant documents covered, and reciprocal rank uses the actual rank of the first relevant chunk. An offline baseline can reveal regressions without semantic retrieval, model calls, or generated labels.
