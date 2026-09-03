from pathlib import Path

from nexusmind import (
    KnowledgeCollection,
    LocalDirectoryAdapter,
    StructureAwareChunker,
    TextChunker,
)
from nexusmind.retrieval_evaluation import (
    evaluate_retrieval_multi_k,
    load_retrieval_evaluation_cases,
)


ROOT = Path(__file__).parents[1] / "evals" / "knowledge" / "chunking"


def _run(chunker):
    collection = KnowledgeCollection(chunker=chunker)
    collection.sync(LocalDirectoryAdapter(ROOT / "corpus", source_id="chunking-docs"))
    cases = load_retrieval_evaluation_cases(ROOT / "cases.json")
    return evaluate_retrieval_multi_k(collection, cases, ks=(1, 3))


def test_structure_chunking_improves_boundary_case_without_retrieval_regression() -> None:
    baseline = _run(TextChunker(chunk_size=80, overlap=0))
    candidate = _run(StructureAwareChunker(chunk_size=80, overlap=0))

    assert candidate == _run(StructureAwareChunker(chunk_size=80, overlap=0))
    assert candidate[0].hit_at_k > baseline[0].hit_at_k
    assert candidate[0].mrr > baseline[0].mrr
    assert candidate[1].recall_at_k >= baseline[1].recall_at_k == 1.0
    boundary = next(item for item in candidate[0].case_results if item.case_id == "fixed-window-boundary")
    assert boundary.first_relevant_rank == 1


def test_structure_benchmark_exposes_nested_heading_context_metadata() -> None:
    collection = KnowledgeCollection(chunker=StructureAwareChunker(chunk_size=80, overlap=0))
    collection.sync(LocalDirectoryAdapter(ROOT / "corpus", source_id="chunking-docs"))

    results = collection.search("Binder oneway pid zero", limit=10)

    technical = next(result for result in results if result.document.logical_path == "technical.md")
    assert technical.hit.chunk.heading_path == (
        "Android Security",
        "Binder",
        "Transaction",
    )
    assert technical.hit.chunk.section_title == "Transaction"
    assert technical.hit.chunk.source_location == "technical.md:L9"
    assert "Android Security > Binder > Transaction" in technical.hit.chunk.retrieval_text
    assert technical.hit.chunk.content in technical.hit.chunk.retrieval_text
