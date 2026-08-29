from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import httpx

from nexusmind import (
    GeneratedAnswer,
    KnowledgeBase,
    KnowledgeQueryOptions,
    LocalFileSourceConfig,
    QueryExpansion,
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
    result = _kb(tmp_path, Expander((), failure=True)).query("Binder credentials")

    assert result.trace.retrieval_queries == ("Binder credentials",)
    assert result.trace.query_expansion_error == "RuntimeError"


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
