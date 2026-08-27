from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json
from pathlib import Path

import pytest

from nexusmind import (
    AnswerGenerationLimits,
    ContextPackage,
    GeneratedAnswer,
    KnowledgeBase,
    KnowledgeQueryOptions,
    KnowledgeQueryResult,
    LocalFileSourceConfig,
    knowledge_query_result_dict,
)


@dataclass
class FakeGenerator:
    @property
    def config_identity(self) -> str:
        return "fake/query-v1"

    def generate(self, question, context, *, model_context, limits):
        return GeneratedAnswer("Binder uses kernel credentials [K1]", ("K1",))


def _knowledge_base(tmp_path: Path) -> KnowledgeBase:
    document = tmp_path / "binder.md"
    document.write_text(
        "Binder authenticates callers using kernel-provided credentials.",
        encoding="utf-8",
    )
    knowledge = KnowledgeBase.create(
        str(tmp_path / "kb"),
        knowledge_base_id="query-test",
        answer_generator=FakeGenerator(),
    )
    knowledge.add_source(LocalFileSourceConfig(path=str(document)))
    knowledge.sync()
    return knowledge


def test_query_runs_existing_pipeline_and_returns_validated_debug_trace(
    tmp_path: Path,
) -> None:
    knowledge = _knowledge_base(tmp_path)

    result = knowledge.query("How does Binder authenticate callers?")

    assert isinstance(result, KnowledgeQueryResult)
    assert result.answer.text == "Binder uses kernel credentials [K1]"
    assert result.citations == result.answer.citations
    assert result.citations[0].logical_path == "binder.md"
    assert result.trace_id
    assert result.trace.retrieval_backend == "InMemoryChunkIndex"
    assert result.trace.passages == result.answer.model_context.passages
    assert result.trace.candidate_count == 1
    assert result.trace.context_character_count > 0


def test_knowledge_base_exposes_only_the_query_answer_api(tmp_path: Path) -> None:
    knowledge = _knowledge_base(tmp_path)

    assert not hasattr(knowledge, "answer")
    assert knowledge.query("Binder credentials?").answer.text


def test_query_result_has_stable_json_schema_and_is_frozen(tmp_path: Path) -> None:
    result = _knowledge_base(tmp_path).query("Binder credentials?")

    payload = knowledge_query_result_dict(result, include_debug=True)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)

    assert list(payload) == ["answer", "citations", "trace_id", "debug"]
    assert json.loads(encoded)["citations"][0]["citation_id"] == "K1"
    assert json.loads(encoded)["debug"]["passage_count"] == 1
    with pytest.raises(FrozenInstanceError):
        result.trace_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("retrieval_limit", [True, 0])
def test_query_options_reject_invalid_retrieval_limits(retrieval_limit: object) -> None:
    error = TypeError if retrieval_limit is True else ValueError
    with pytest.raises(error):
        KnowledgeQueryOptions(retrieval_limit=retrieval_limit)  # type: ignore[arg-type]
