import asyncio

from nexusmind.models.base import ChatModel
from nexusmind.models.fake import FakeChatModel
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole


def test_fake_model_implements_chat_model() -> None:
    assert isinstance(FakeChatModel(), ChatModel)


def test_fake_model_streams_text_deltas() -> None:
    async def collect():
        model = FakeChatModel(["one", "two"])
        messages = [Message(role=MessageRole.USER, content="hello")]
        return [event async for event in model.stream(messages)]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.MODEL_TURN_COMPLETED,
    ]
    assert [event.text for event in events if event.type == RuntimeEventType.TEXT_DELTA] == ["one", "two"]

