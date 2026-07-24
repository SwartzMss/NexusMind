import asyncio

from nexusmind.models.fake import FakeChatModel
from nexusmind.models.tool_calls import ToolCallDelta
from nexusmind.runtime.chat import ChatRuntime
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.tools.contracts import ToolCall


def test_runtime_streams_model_events_in_order() -> None:
    async def collect():
        runtime = ChatRuntime(FakeChatModel(["a", "b", "c"]))
        return [event async for event in runtime.stream_user_message("hello")]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.RUN_COMPLETED,
    ]
    assert "".join(event.text or "" for event in events) == "abc"


def test_runtime_converts_model_exception_to_run_failed() -> None:
    async def collect():
        runtime = ChatRuntime(FakeChatModel(error=RuntimeError("provider failed")))
        return [event async for event in runtime.stream_user_message("hello")]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert events[-1].error == "Model execution failed"


def test_runtime_passes_through_tool_call_events() -> None:
    class ToolCallModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_DELTA,
                tool_call_delta=ToolCallDelta(index=0, call_id_fragment="call_1"),
            )
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="echo", arguments={"text": "hello"}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolCallModel())
        return [event async for event in runtime.stream_user_message("hello")]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TOOL_CALL_DELTA,
        RuntimeEventType.TOOL_CALL_COMPLETED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
    ]


def test_runtime_completes_run_for_stop_model_turn() -> None:
    class StopModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect():
        runtime = ChatRuntime(StopModel())
        return [event async for event in runtime.stream_user_message("hello")]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.RUN_COMPLETED,
    ]


def test_runtime_does_not_complete_run_when_stop_turn_contains_tool_call() -> None:
    class ToolCallStopModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="echo", arguments={}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect():
        return [
            event
            async for event in ChatRuntime(ToolCallStopModel()).stream_user_message("hello")
        ]

    events = asyncio.run(collect())

    assert [event.type for event in events[-2:]] == [
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]


def test_runtime_treats_model_failed_as_terminal() -> None:
    class FailedModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.MODEL_FAILED, error="provider failed")
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect():
        return [event async for event in ChatRuntime(FailedModel()).stream_user_message("hello")]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]
    assert events[-1].error == "Model execution failed"


def test_runtime_fails_when_tool_finish_has_no_completed_tool_call() -> None:
    class MissingToolCallModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.MODEL_TURN_COMPLETED,
                finish_reason="tool_calls",
            )

    async def collect():
        return [
            event
            async for event in ChatRuntime(MissingToolCallModel()).stream_user_message("hello")
        ]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]


def test_runtime_rejects_model_events_outside_the_model_whitelist() -> None:
    invalid_types = [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.RUN_COMPLETED,
        RuntimeEventType.RUN_FAILED,
        RuntimeEventType.TOOL_CALL,
        RuntimeEventType.TOOL_RESULT,
    ]

    for invalid_type in invalid_types:
        class InvalidEventModel:
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                yield RuntimeEvent(invalid_type)

        async def collect():
            return [
                event
                async for event in ChatRuntime(InvalidEventModel()).stream_user_message(
                    "hello"
                )
            ]

        events = asyncio.run(collect())
        assert events[-2].type == RuntimeEventType.MODEL_FAILED
        assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_rejects_model_events_with_missing_payloads() -> None:
    invalid_events = [
        RuntimeEvent(RuntimeEventType.TEXT_DELTA),
        RuntimeEvent(RuntimeEventType.TOOL_CALL_DELTA),
        RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED),
        RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED),
        RuntimeEvent(RuntimeEventType.MODEL_FAILED),
    ]

    for invalid_event in invalid_events:
        class InvalidPayloadModel:
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                yield invalid_event

        async def collect():
            return [
                event
                async for event in ChatRuntime(InvalidPayloadModel()).stream_user_message(
                    "hello"
                )
            ]

        events = asyncio.run(collect())
        assert events[-2].type == RuntimeEventType.MODEL_FAILED
        assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_rejects_wrong_tool_dto_types_and_finish_reasons() -> None:
    invalid_events = [
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_DELTA,
            tool_call_delta=object(),  # type: ignore[arg-type]
        ),
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_COMPLETED,
            tool_call=object(),  # type: ignore[arg-type]
        ),
        RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="custom"),
    ]

    for invalid_event in invalid_events:
        class InvalidPayloadModel:
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                yield invalid_event

        async def collect():
            return [
                event
                async for event in ChatRuntime(InvalidPayloadModel()).stream_user_message(
                    "hello"
                )
            ]

        events = asyncio.run(collect())
        assert events[-2].type == RuntimeEventType.MODEL_FAILED
        assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_does_not_expose_arbitrary_exception_text() -> None:
    class UnsafeFailureModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            raise RuntimeError("sk-live-secret")

    async def collect():
        return [
            event
            async for event in ChatRuntime(UnsafeFailureModel()).stream_user_message(
                "hello"
            )
        ]

    events = asyncio.run(collect())

    assert events[-1].error == "Model execution failed"
    assert "sk-live-secret" not in repr(events)


def test_runtime_fails_when_model_turn_completion_is_missing() -> None:
    class MissingCompletionModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="hello")

    async def collect():
        runtime = ChatRuntime(MissingCompletionModel())
        return [event async for event in runtime.stream_user_message("hello")]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]


def test_runtime_fails_on_duplicate_turn_completion_or_events_after_completion() -> None:
    class DuplicateCompletionModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="late")

    async def collect():
        runtime = ChatRuntime(DuplicateCompletionModel())
        return [event async for event in runtime.stream_user_message("hello")]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]


def test_runtime_fails_when_model_emits_before_model_started() -> None:
    class BadOrderModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="early")

    async def collect():
        runtime = ChatRuntime(BadOrderModel())
        return [event async for event in runtime.stream_user_message("hello")]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]

