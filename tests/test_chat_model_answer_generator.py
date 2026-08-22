from __future__ import annotations

import pytest
from types import SimpleNamespace

from nexusmind import AnswerGenerationLimits, AnswerGeneratorError
from nexusmind.models.base import ChatModel
from nexusmind.models.knowledge_answer import ChatModelAnswerGenerator
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType


class StubChatModel(ChatModel):
    def __init__(self, output: str, *, finish_reason: str = "stop") -> None:
        self.output = output
        self.finish_reason = finish_reason
        self.messages = None

    async def stream(self, messages, tools=None):
        self.messages = messages
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text=self.output)
        yield RuntimeEvent(
            RuntimeEventType.MODEL_TURN_COMPLETED,
            finish_reason=self.finish_reason,
        )


def test_chat_model_answer_generator_returns_only_untrusted_structured_output() -> None:
    model = StubChatModel('{"answer":"Supported [K1]","citations":["K1"]}')
    generator = ChatModelAnswerGenerator(model, config_identity="fake/v1")
    record = SimpleNamespace(rendered_context="[K1]\ncontent: evidence")

    result = generator.generate(
        "question",
        object(),
        model_context=record,
        limits=AnswerGenerationLimits(),
    )

    assert result.text == "Supported [K1]"
    assert result.citation_ids == ("K1",)
    assert model.messages[0].content.startswith("Answer only from the supplied evidence")
    assert "Evidence:" in model.messages[1].content


@pytest.mark.parametrize(
    "output,finish_reason",
    [
        ('{"answer":"x","citations":["K1"],"extra":true}', "stop"),
        ("```json\n{}\n```", "stop"),
        ('{"answer":"x","citations":["K1"]}', "length"),
    ],
)
def test_chat_model_answer_generator_rejects_non_strict_output(
    output: str, finish_reason: str
) -> None:
    generator = ChatModelAnswerGenerator(
        StubChatModel(output, finish_reason=finish_reason),
        config_identity="fake/v1",
    )

    with pytest.raises(AnswerGeneratorError, match="invalid output"):
        generator.generate(
            "question",
            object(),
            model_context=SimpleNamespace(rendered_context="[K1]\ncontent: evidence"),
            limits=AnswerGenerationLimits(),
        )
