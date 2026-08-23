"""Knowledge-native answer providers without Agent Runtime dependencies."""

from __future__ import annotations

import json
import httpx

from nexusmind.config import ModelConfig
from nexusmind.context_assembly import ContextPackage
from nexusmind.knowledge_answer import AnswerGenerationLimits, AnswerGeneratorError, GeneratedAnswer, ModelContextRecord


class OpenAICompatibleAnswerProvider:
    """Generate one structured knowledge answer through Chat Completions."""

    def __init__(self, config: ModelConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        if type(config) is not ModelConfig:
            raise TypeError("config must be ModelConfig")
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            raise TypeError("transport must be an httpx BaseTransport")
        self._config = config
        self._transport = transport

    @property
    def config_identity(self) -> str:
        return f"openai-compatible/{self._config.model}"

    def generate(self, question: str, context: ContextPackage, *, model_context: ModelContextRecord, limits: AnswerGenerationLimits) -> GeneratedAnswer:
        del context
        system = (
            "Answer only from the supplied evidence. Return exactly one JSON object "
            'with keys "answer" (string) and "citations" (array of K handles). '
            "Every factual answer must cite at least one supplied handle. Do not use markdown fences."
        )
        user = f"Question:\n{question}\n\nEvidence:\n{model_context.rendered_context}"
        maximum = limits.max_answer_chars + (limits.max_citations * 16) + 1_024
        request = {"model": self._config.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "stream": False}
        try:
            with httpx.Client(base_url=self._config.base_url.rstrip("/") + "/", headers={"Authorization": f"Bearer {self._config.api_key}"}, timeout=self._config.timeout, transport=self._transport) as client:
                with client.stream("POST", "chat/completions", json=request) as response:
                    if not 200 <= response.status_code < 300:
                        raise AnswerGeneratorError("answer provider request failed")
                    raw = bytearray()
                    response_limit = max(4_096, maximum * 4)
                    for chunk in response.iter_bytes():
                        if len(raw) + len(chunk) > response_limit:
                            raise AnswerGeneratorError("answer provider returned invalid output")
                        raw.extend(chunk)
        except AnswerGeneratorError:
            raise
        except (httpx.HTTPError, OSError):
            raise AnswerGeneratorError("answer provider request failed") from None

        try:
            envelope = json.loads(bytes(raw), parse_constant=_reject_constant)
            choices = envelope["choices"]
            if type(choices) is not list or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            if type(choice) is not dict or choice.get("finish_reason") != "stop":
                raise ValueError
            message = choice["message"]
            if type(message) is not dict or type(message.get("content")) is not str:
                raise ValueError
            content = message["content"]
            if len(content) > maximum:
                raise ValueError
            payload = json.loads(content, parse_constant=_reject_constant)
        except (KeyError, TypeError, ValueError, RecursionError, UnicodeDecodeError):
            raise AnswerGeneratorError("answer provider returned invalid output") from None
        if type(payload) is not dict or set(payload) != {"answer", "citations"}:
            raise AnswerGeneratorError("answer provider returned invalid output")
        answer, citations = payload["answer"], payload["citations"]
        if type(answer) is not str or type(citations) is not list or any(type(item) is not str for item in citations):
            raise AnswerGeneratorError("answer provider returned invalid output")
        return GeneratedAnswer(answer, tuple(citations))


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = ["OpenAICompatibleAnswerProvider"]
