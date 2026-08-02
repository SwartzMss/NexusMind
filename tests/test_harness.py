from __future__ import annotations

import asyncio

from nexusmind.models.fake import FakeChatModel
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.runtime.harness import HarnessRequest, HarnessRunner
from nexusmind.runtime.harness import HarnessLimits
from nexusmind.runtime.harness import HarnessStatus, StopReason
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


def test_harness_runner_preserves_message_history() -> None:
    class RecordingModel(FakeChatModel):
        def __init__(self):
            super().__init__(["ok"])
            self.seen = None

        async def stream(self, messages, tools=None):
            self.seen = messages
            async for event in super().stream(messages, tools=tools):
                yield event

    async def collect():
        model = RecordingModel()
        runner = HarnessRunner(model)
        request = HarnessRequest(
            messages=(
                Message(role=MessageRole.SYSTEM, content="system"),
                Message(role=MessageRole.USER, content="first"),
                Message(role=MessageRole.ASSISTANT, content="reply"),
                Message(role=MessageRole.USER, content="second"),
            )
        )
        [event async for event in runner.stream(request)]
        return model.seen

    seen = asyncio.run(collect())
    assert [message.content for message in seen] == ["system", "first", "reply", "second"]


def test_harness_runner_uses_request_limits() -> None:
    async def collect():
        runner = HarnessRunner(FakeChatModel(["hello"]))
        request = HarnessRequest(
            messages=(Message(role=MessageRole.USER, content="hi"),),
            limits=HarnessLimits(max_model_turns=1),
        )
        return [event async for event in runner.stream(request)]

    events = asyncio.run(collect())
    assert events[-1].type is RuntimeEventType.RUN_COMPLETED


def test_harness_runner_constructor_limits_are_used_when_request_omits_limits() -> None:
    async def collect():
        runner = HarnessRunner(FakeChatModel(["hello", "again"]), limits=HarnessLimits(max_model_turns=1))
        request = HarnessRequest(messages=(Message(role=MessageRole.USER, content="hi"),))
        return [event async for event in runner.stream(request)]

    events = asyncio.run(collect())
    assert events[-1].type is RuntimeEventType.RUN_COMPLETED


def test_harness_runner_records_terminal_state() -> None:
    async def collect(runner, request):
        [event async for event in runner.stream(request)]

    runner = HarnessRunner(FakeChatModel(["hello"]))
    request = HarnessRequest(messages=(Message(role=MessageRole.USER, content="hi"),))
    asyncio.run(collect(runner, request))
    assert runner.state is not None
    assert runner.state.status is HarnessStatus.COMPLETED
    assert runner.stop_reason is StopReason.MODEL_COMPLETED
