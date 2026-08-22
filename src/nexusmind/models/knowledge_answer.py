"""AnswerGenerator adapter for the existing provider-neutral ChatModel."""

from __future__ import annotations

import asyncio
import json

from nexusmind.context_assembly import ContextPackage
from nexusmind.knowledge_answer import (
    AnswerGenerationLimits,
    AnswerGeneratorError,
    GeneratedAnswer,
    ModelContextRecord,
)
from nexusmind.models.base import ChatModel
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole


class ChatModelAnswerGenerator:
    """Generate one strictly structured answer without entering an agent loop."""

    def __init__(self, model: ChatModel, *, config_identity: str) -> None:
        if not isinstance(model, ChatModel):
            raise TypeError("model must be a ChatModel")
        if type(config_identity) is not str or not config_identity.strip():
            raise ValueError("config_identity must be a non-empty string")
        self._model = model
        self._config_identity = config_identity

    @property
    def config_identity(self) -> str:
        return self._config_identity

    def generate(
        self,
        question: str,
        context: ContextPackage,
        *,
        model_context: ModelContextRecord,
        limits: AnswerGenerationLimits,
    ) -> GeneratedAnswer:
        del context
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AnswerGeneratorError(
                "synchronous answer generation cannot run inside an event loop"
            )
        return asyncio.run(self._generate(question, model_context, limits))

    async def _generate(
        self,
        question: str,
        model_context: ModelContextRecord,
        limits: AnswerGenerationLimits,
    ) -> GeneratedAnswer:
        system = (
            "Answer only from the supplied evidence. Return exactly one JSON object "
            'with keys "answer" (string) and "citations" (array of K handles). '
            "Every factual answer must cite at least one supplied handle. Do not use markdown fences."
        )
        user = f"Question:\n{question}\n\nEvidence:\n{model_context.rendered_context}"
        output: list[str] = []
        character_count = 0
        maximum = limits.max_answer_chars + (limits.max_citations * 16) + 1_024
        completed = False
        async for event in self._model.stream(
            [Message(MessageRole.SYSTEM, system), Message(MessageRole.USER, user)]
        ):
            if event.type == RuntimeEventType.TEXT_DELTA and event.text:
                character_count += len(event.text)
                if character_count > maximum:
                    raise AnswerGeneratorError("answer generator returned invalid output")
                output.append(event.text)
            elif event.type == RuntimeEventType.MODEL_TURN_COMPLETED:
                if event.finish_reason != "stop":
                    raise AnswerGeneratorError("answer generator returned invalid output")
                completed = True
            elif event.type in {
                RuntimeEventType.MODEL_FAILED,
                RuntimeEventType.RUN_FAILED,
                RuntimeEventType.TOOL_CALL,
                RuntimeEventType.TOOL_CALL_DELTA,
                RuntimeEventType.TOOL_CALL_COMPLETED,
            }:
                raise AnswerGeneratorError("answer generator returned invalid output")
        if not completed:
            raise AnswerGeneratorError("answer generator returned invalid output")
        try:
            payload = json.loads("".join(output), parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            raise AnswerGeneratorError("answer generator returned invalid output") from None
        if type(payload) is not dict or set(payload) != {"answer", "citations"}:
            raise AnswerGeneratorError("answer generator returned invalid output")
        answer = payload["answer"]
        citations = payload["citations"]
        if type(answer) is not str or type(citations) is not list or any(
            type(item) is not str for item in citations
        ):
            raise AnswerGeneratorError("answer generator returned invalid output")
        return GeneratedAnswer(answer, tuple(citations))


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = ["ChatModelAnswerGenerator"]
