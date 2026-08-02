from __future__ import annotations

import asyncio

from nexusmind.models.fake import FakeChatModel
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.runtime.harness import HarnessRequest, HarnessRunner
from nexusmind.runtime.messages import Message, MessageRole


def test_harness_request_snapshots_inputs() -> None:
    metadata = {"origin": {"name": "chat"}}
    messages = [Message(role=MessageRole.USER, content="hello")]
    request = HarnessRequest(messages=tuple(messages), metadata=metadata)

    metadata["origin"]["name"] = "mutated"
    assert request.metadata == {"origin": {"name": "chat"}}


def test_harness_runner_streams_provider_neutral_events() -> None:
    async def collect():
        runner = HarnessRunner(FakeChatModel(["hello"]))
        request = HarnessRequest(
            messages=(Message(role=MessageRole.USER, content="hi"),)
        )
        return [event async for event in runner.stream(request)]

    events = asyncio.run(collect())
    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.RUN_COMPLETED,
    ]
