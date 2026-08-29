from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import httpx

from nexusmind import (
    Chunk,
    Document,
    GeneratedAnswer,
    KnowledgeBase,
    KnowledgeSearchResult,
    KnowledgeSource,
    KnowledgeQueryOptions,
    LocalFileSourceConfig,
    QueryExpansion,
    SearchHit,
)
from nexusmind.config import ModelConfig
from nexusmind.query_expansion import OpenAICompatibleQueryExpander


@dataclass
class Generator:
    @property
    def config_identity(self) -> str:
        return "fake"

    def generate(self, question, context, *, model_context, limits):
        return GeneratedAnswer("Recovered [K1]", ("K1",))


@dataclass
class Expander:
    queries: tuple[str, ...]
    failure: bool = False

    def expand(self, question: str) -> QueryExpansion:
        if self.failure:
            raise RuntimeError("offline")
        return QueryExpansion(question, self.queries)


class GapIndex:
    """Backend with a deliberately large relevance gap across documents."""

    max_search_results = 100

    def __init__(self, chunks: dict[str, tuple[Chunk, ...]] | None = None) -> None:
        self._chunks = {} if chunks is None else dict(chunks)

    def clone(self) -> "GapIndex":
        return GapIndex(self._chunks)

    def add(self, chunk: Chunk) -> None:
        self._chunks[chunk.document_id] = self._chunks.get(chunk.document_id, ()) + (
            chunk,
        )

    def replace_document(self, document_id: str, chunks: tuple[Chunk, ...]) -> None:
        self._chunks[document_id] = chunks

    def remove_document(self, document_id: str) -> None:
        self._chunks.pop(document_id, None)

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        del query
        scored: list[SearchHit] = []
        for chunks in self._chunks.values():
            if len(chunks) == 1 and chunks[0].content == "B":
                scored.append(SearchHit(chunks[0], 1.0))
            else:
                scored.extend(
                    SearchHit(chunk, float(100 - ordinal * 10))
                    for ordinal, chunk in enumerate(chunks)
                )
        return tuple(sorted(scored, key=lambda hit: -hit.score)[:limit])


def _kb(tmp_path: Path, expander: Expander) -> KnowledgeBase:
    source = tmp_path / "runbook.md"
    source.write_text(
        "IAM_Master crash watchdog restart. Binder kernel credentials.", encoding="utf-8"
    )
    kb = KnowledgeBase.create(
        str(tmp_path / "kb"), answer_generator=Generator(), query_expander=expander
    )
    kb.add_source(LocalFileSourceConfig(path=str(source)))
    kb.sync()
    return kb


def test_expanded_query_recovers_terminology_mismatch_and_traces_ranks(tmp_path: Path) -> None:
    result = _kb(tmp_path, Expander(("IAM_Master crash watchdog restart",))).query(
        "IAM_Master 挂了怎么办"
    )

    assert result.trace.retrieval_queries == (
        "IAM_Master 挂了怎么办",
        "IAM_Master crash watchdog restart",
    )
    assert result.trace.fused_result_provenance[0][1] == ((0, 1), (1, 1))
    assert result.trace.query_expansion_error is None


def test_expansion_failure_falls_back_to_original_retrieval(tmp_path: Path) -> None:
    kb = _kb(tmp_path, Expander((), failure=True))
    expected = kb._collection.search("Binder credentials", limit=8)
    result = kb.query("Binder credentials")

    assert result.trace.retrieval_queries == ("Binder credentials",)
    assert result.trace.query_expansion_error == "RuntimeError"
    assert tuple(passage.chunk_id for passage in result.answer.model_context.passages) == tuple(
        item.hit.chunk.chunk_id for item in expected
    )


def test_single_query_path_uses_backend_scores_for_diversification(tmp_path: Path) -> None:
    kb = _kb(tmp_path, Expander((), failure=True))
    calls: list[tuple[str, int]] = []
    original_search = kb._collection.search

    def search(query: str, *, limit: int = 10):
        calls.append((query, limit))
        return original_search(query, limit=limit)

    kb._collection.search = search  # type: ignore[method-assign]

    kb.query("Binder credentials")

    assert calls == [("Binder credentials", 8)]


def test_expansion_fallback_preserves_large_backend_relevance_gap(tmp_path: Path) -> None:
    strong = tmp_path / "a.txt"
    weak = tmp_path / "b.txt"
    strong.write_text("A" * 4600, encoding="utf-8")
    weak.write_text("B", encoding="utf-8")
    kb = KnowledgeBase.create(
        str(tmp_path / "gap-kb"),
        index_factory=GapIndex,
        answer_generator=Generator(),
        query_expander=Expander((), failure=True),
    )
    kb.add_source(LocalFileSourceConfig(path=str(strong)))
    kb.add_source(LocalFileSourceConfig(path=str(weak)))
    kb.sync()

    result = kb.query("failure", options=KnowledgeQueryOptions(retrieval_limit=5))

    assert result.trace.query_expansion_error == "RuntimeError"
    assert len(result.trace.passages) == 5
    assert {passage.logical_path for passage in result.trace.passages} == {"a.txt"}


def test_multi_query_rrf_does_not_rewrite_backend_scores() -> None:
    source = KnowledgeSource(source_id="docs", source_type="fake", display_name="Docs")
    first_document = Document(source_id="docs", logical_path="a.txt", content="alpha")
    second_document = Document(source_id="docs", logical_path="b.txt", content="beta")
    first = KnowledgeSearchResult(
        source,
        first_document,
        SearchHit(Chunk(first_document.document_id, "chunk-a", "alpha", 0, 5), 100.0),
    )
    second = KnowledgeSearchResult(
        source,
        second_document,
        SearchHit(Chunk(second_document.document_id, "chunk-b", "beta", 0, 4), 1.0),
    )

    fused, provenance = KnowledgeBase._fuse_query_results(
        ((first, second), (second, first)), limit=2
    )

    assert tuple(item.hit.score for item in fused) == (100.0, 1.0)
    assert provenance == (
        ("chunk-a", ((0, 1), (1, 2))),
        ("chunk-b", ((0, 2), (1, 1))),
    )


def test_openai_expander_requires_strict_bounded_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "Preserve every exact identifier" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"queries":["IAM_Master crash restart"]}'},
                    }
                ]
            },
        )

    expander = OpenAICompatibleQueryExpander(
        ModelConfig("https://example.test", "secret", "model"),
        transport=httpx.MockTransport(handler),
    )
    assert expander.expand("IAM_Master 挂了").expanded_queries == (
        "IAM_Master crash restart",
    )
