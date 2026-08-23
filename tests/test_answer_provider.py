from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from nexusmind.answer_provider import OpenAICompatibleAnswerProvider
from nexusmind.config import ModelConfig
from nexusmind.context_assembly import ContextPackage
from nexusmind.knowledge_answer import AnswerGenerationLimits, AnswerGeneratorError


def _record():
    return SimpleNamespace(rendered_context="[K1]\nEvidence")


def _provider(handler) -> OpenAICompatibleAnswerProvider:
    return OpenAICompatibleAnswerProvider(
        ModelConfig("https://example.test/v1", "secret", "test-model", 3.0),
        transport=httpx.MockTransport(handler),
    )


def test_provider_uses_knowledge_native_chat_completion_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["messages"][1]["content"].endswith("[K1]\nEvidence")
        content = json.dumps({"answer": "Supported [K1]", "citations": ["K1"]})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    result = _provider(handler).generate(
        "What?", ContextPackage("What?", (), {}), model_context=_record(), limits=AnswerGenerationLimits()
    )
    assert result.text == "Supported [K1]"
    assert result.citation_ids == ("K1",)


@pytest.mark.parametrize("status", [400, 401, 500])
def test_provider_maps_http_failures_without_leaking_response(status: int) -> None:
    provider = _provider(lambda request: httpx.Response(status, text="secret provider body"))
    with pytest.raises(AnswerGeneratorError, match="request failed") as error:
        provider.generate("What?", ContextPackage("What?", (), {}), model_context=_record(), limits=AnswerGenerationLimits())
    assert "secret" not in str(error.value)


def test_provider_rejects_unstructured_output() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "not-json"}}]}))
    with pytest.raises(AnswerGeneratorError, match="invalid output"):
        provider.generate("What?", ContextPackage("What?", (), {}), model_context=_record(), limits=AnswerGenerationLimits())
