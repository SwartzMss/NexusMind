import asyncio

from nexusmind.models.fake import FakeChatModel
from nexusmind.models.tool_calls import ToolCallDelta
from nexusmind.runtime.chat import AgentLoopLimits, ChatRuntime
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.runtime.policy import ApprovalDecision, ToolApproval, ToolPolicyDecision
from nexusmind.tools.builtin import EchoTool
from nexusmind.tools.contracts import ToolCall, ToolDefinition, ToolError, ToolErrorCode, ToolResult, ToolRiskLevel
from nexusmind.tools.executor import ToolExecutor
from nexusmind.tools.registry import ToolRegistry


def _executor_with_echo() -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return ToolExecutor(registry)


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
        RuntimeEventType.RUN_FAILED,
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
    assert RuntimeEventType.MODEL_TURN_COMPLETED not in [
        event.type for event in events
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


def test_runtime_rejects_raw_string_event_type_before_dispatch() -> None:
    class RawStringTypeModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent("tool_call_completed", tool_call=ToolCall(id="call_1", name="echo", arguments={}))  # type: ignore[arg-type]

    async def collect():
        runtime = ChatRuntime(RawStringTypeModel(), tool_executor=_executor_with_echo())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-2].type == RuntimeEventType.MODEL_FAILED
    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_rejects_model_event_with_non_dict_metadata() -> None:
    class BadMetadataModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(
                RuntimeEventType.MODEL_STARTED,
                metadata="not-a-dict",  # type: ignore[arg-type]
            )

    async def collect():
        return [event async for event in ChatRuntime(BadMetadataModel()).stream_user_message("hello")]

    events = asyncio.run(collect())

    assert events[-2].type == RuntimeEventType.MODEL_FAILED
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_rejects_non_runtime_event_dto_and_does_not_execute_tool() -> None:
    class FakeEvent:
        type = RuntimeEventType.TOOL_CALL_COMPLETED
        text = None
        error = None
        tool_call_delta = None
        tool_call = ToolCall(id="call_1", name="echo", arguments={})
        tool_result = None
        finish_reason = None
        metadata = {}

    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class FakeEventModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield FakeEvent()

    async def collect(executor):
        runtime = ChatRuntime(FakeEventModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
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


def test_runtime_rejects_model_events_with_conflicting_payload_fields() -> None:
    invalid_events = [
        RuntimeEvent(RuntimeEventType.MODEL_STARTED, tool_call=ToolCall(id="call_1", name="echo", arguments={})),
        RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="hello", tool_result=ToolResult(call_id="call_1", name="echo", output={})),
        RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop", error="unexpected"),
        RuntimeEvent(
            RuntimeEventType.MODEL_STARTED,
            tool_approval=ToolApproval("req_1", "call_1", "echo", ToolRiskLevel.READ_ONLY, "Approve echo"),
        ),
    ]

    for invalid_event in invalid_events:
        class ConflictingPayloadModel:
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                yield invalid_event

        async def collect():
            return [event async for event in ChatRuntime(ConflictingPayloadModel()).stream_user_message("hello")]

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


def test_runtime_rejects_invalid_fields_inside_tool_dtos() -> None:
    invalid_events = [
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_DELTA,
            tool_call_delta=ToolCallDelta(index=True),
        ),
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_DELTA,
            tool_call_delta=ToolCallDelta(index=-1),
        ),
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_DELTA,
            tool_call_delta=ToolCallDelta(
                index=0,
                arguments_fragment=None,  # type: ignore[arg-type]
            ),
        ),
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_COMPLETED,
            tool_call=ToolCall(id="", name="echo", arguments={}),
        ),
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_COMPLETED,
            tool_call=ToolCall(id="call_1", name="", arguments={}),
        ),
        RuntimeEvent(
            RuntimeEventType.TOOL_CALL_COMPLETED,
            tool_call=ToolCall(
                id="call_1",
                name="echo",
                arguments=[],  # type: ignore[arg-type]
            ),
        ),
    ]

    for invalid_event in invalid_events:
        class InvalidDtoModel:
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                yield invalid_event

        async def collect():
            return [
                event
                async for event in ChatRuntime(InvalidDtoModel()).stream_user_message(
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


def test_runtime_executes_tool_call_and_feeds_result_to_next_turn() -> None:
    class ToolLoopModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_1", name="echo", arguments={"text": "hello"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model):
        runtime = ChatRuntime(model, tool_executor=_executor_with_echo())
        tools = [ToolDefinition(name="echo")]
        return [event async for event in runtime.stream_user_message("hello", tools=tools)]

    model = ToolLoopModel()
    events = asyncio.run(collect(model))

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TOOL_CALL_COMPLETED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.TOOL_RESULT,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.MODEL_TURN_COMPLETED,
        RuntimeEventType.RUN_COMPLETED,
    ]
    second_turn_messages = model.messages_by_turn[1]
    assert second_turn_messages[-2].tool_calls[0].id == "call_1"
    assert second_turn_messages[-1].role.value == "tool"
    assert second_turn_messages[-1].tool_call_id == "call_1"
    assert second_turn_messages[-1].content == '{"ok":true,"output":{"text":"hello"}}'
    assert events[4].tool_result.output == {"text": "hello"}


def test_runtime_default_policy_allows_read_only_tool_without_approval() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ToolModel:
        def __init__(self) -> None:
            self.turns = 0

        async def stream(self, messages, tools=None):
            self.turns += 1
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if self.turns == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(executor):
        runtime = ChatRuntime(ToolModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo", risk_level=ToolRiskLevel.READ_ONLY)])]

    executor = RecordingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 1
    assert RuntimeEventType.TOOL_APPROVAL_REQUIRED not in [event.type for event in events]
    assert events[-1].type == RuntimeEventType.RUN_COMPLETED


def test_runtime_rejects_invalid_tool_definition_risk_level_before_execution() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(ToolModel(), tool_executor=executor)
        bad_tool = ToolDefinition(name="echo", risk_level="read_only")  # type: ignore[arg-type]
        return [event async for event in runtime.stream_user_message("hello", tools=[bad_tool])]

    executor = RecordingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_requires_approval_for_local_write_and_allows_once() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call):
            self.calls.append(call.id)
            return ToolResult(call_id=call.id, name=call.name, output={"ok": True})

    class AllowApproval:
        def __init__(self) -> None:
            self.requests = []

        async def request(self, request):
            self.requests.append(request)
            return ApprovalDecision.ALLOW_ONCE

    class ToolModel:
        def __init__(self) -> None:
            self.turns = 0

        async def stream(self, messages, tools=None):
            self.turns += 1
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if self.turns == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="write_file", arguments={"path": "secret.txt"}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(executor, approval):
        runtime = ChatRuntime(ToolModel(), tool_executor=executor, approval_provider=approval)
        tools = [ToolDefinition(name="write_file", risk_level=ToolRiskLevel.LOCAL_WRITE)]
        return [event async for event in runtime.stream_user_message("hello", tools=tools)]

    executor = RecordingExecutor()
    approval = AllowApproval()
    events = asyncio.run(collect(executor, approval))

    assert executor.calls == ["call_1"]
    assert len(approval.requests) == 1
    assert "secret.txt" not in repr(approval.requests[0])
    approval_events = [event for event in events if event.type in {RuntimeEventType.TOOL_APPROVAL_REQUIRED, RuntimeEventType.TOOL_APPROVAL_RESOLVED}]
    assert [event.type for event in approval_events] == [
        RuntimeEventType.TOOL_APPROVAL_REQUIRED,
        RuntimeEventType.TOOL_APPROVAL_RESOLVED,
    ]
    assert approval_events[0].tool_approval.decision is None
    assert approval_events[1].tool_approval.decision == ApprovalDecision.ALLOW_ONCE
    assert events[-1].type == RuntimeEventType.RUN_COMPLETED


def test_runtime_approval_deny_returns_permission_denied_and_continues() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class DenyApproval:
        async def request(self, request):
            return ApprovalDecision.DENY

    class ToolModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="write_file", arguments={}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model, executor):
        runtime = ChatRuntime(model, tool_executor=executor, approval_provider=DenyApproval())
        tools = [ToolDefinition(name="write_file", risk_level=ToolRiskLevel.LOCAL_WRITE)]
        return [event async for event in runtime.stream_user_message("hello", tools=tools)]

    model = ToolModel()
    executor = RecordingExecutor()
    events = asyncio.run(collect(model, executor))

    assert executor.calls == 0
    assert RuntimeEventType.RUN_FAILED not in [event.type for event in events]
    result_event = [event for event in events if event.type == RuntimeEventType.TOOL_RESULT][0]
    assert result_event.tool_result.error.code == ToolErrorCode.PERMISSION_DENIED
    assert model.messages_by_turn[1][-1].content == (
        '{"ok":false,"error":{"code":"PERMISSION_DENIED",'
        '"message":"Tool execution was denied","retryable":false}}'
    )


def test_runtime_policy_deny_returns_permission_denied_without_approval_or_execution() -> None:
    class DenyPolicy:
        async def evaluate(self, call, context):
            return ToolPolicyDecision.DENY

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ToolModel:
        def __init__(self) -> None:
            self.turns = 0

        async def stream(self, messages, tools=None):
            self.turns += 1
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if self.turns == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(executor):
        runtime = ChatRuntime(ToolModel(), tool_executor=executor, tool_policy=DenyPolicy())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = RecordingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert RuntimeEventType.TOOL_APPROVAL_REQUIRED not in [event.type for event in events]
    assert [event for event in events if event.type == RuntimeEventType.TOOL_RESULT][0].tool_result.error.code == ToolErrorCode.PERMISSION_DENIED


def test_runtime_policy_and_approval_failures_are_run_failed_without_secret_leak() -> None:
    class BadPolicy:
        async def evaluate(self, call, context):
            raise RuntimeError("sk-live-secret")

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=_executor_with_echo(), tool_policy=BadPolicy())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert "sk-live-secret" not in repr(events)


def test_runtime_invalid_policy_decision_fails_before_execution() -> None:
    class BadPolicy:
        async def evaluate(self, call, context):
            return "allow"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(ToolModel(), tool_executor=executor, tool_policy=BadPolicy())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = RecordingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_invalid_approval_decision_fails_without_execution() -> None:
    class BadApproval:
        async def request(self, request):
            return "allow_once"

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="write_file", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(ToolModel(), tool_executor=executor, approval_provider=BadApproval())
        tools = [ToolDefinition(name="write_file", risk_level=ToolRiskLevel.LOCAL_WRITE)]
        return [event async for event in runtime.stream_user_message("hello", tools=tools)]

    executor = RecordingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert [event.type for event in events if event.type == RuntimeEventType.TOOL_APPROVAL_REQUIRED]
    assert RuntimeEventType.TOOL_APPROVAL_RESOLVED not in [event.type for event in events]
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_policy_cancel_propagates_without_terminal_events() -> None:
    class CancelPolicy:
        async def evaluate(self, call, context):
            raise asyncio.CancelledError()

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(events):
        runtime = ChatRuntime(ToolModel(), tool_executor=_executor_with_echo(), tool_policy=CancelPolicy())
        async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")]):
            events.append(event)

    events = []
    try:
        asyncio.run(collect(events))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert RuntimeEventType.RUN_FAILED not in [event.type for event in events]
    assert RuntimeEventType.RUN_COMPLETED not in [event.type for event in events]


def test_runtime_approval_cancel_has_no_resolved_result_or_next_turn() -> None:
    class CancelApproval:
        async def request(self, request):
            raise asyncio.CancelledError()

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="write_file", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(events):
        runtime = ChatRuntime(ToolModel(), tool_executor=_executor_with_echo(), approval_provider=CancelApproval())
        tools = [ToolDefinition(name="write_file", risk_level=ToolRiskLevel.LOCAL_WRITE)]
        async for event in runtime.stream_user_message("hello", tools=tools):
            events.append(event)

    events = []
    try:
        asyncio.run(collect(events))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert [event.type for event in events][-1] == RuntimeEventType.TOOL_APPROVAL_REQUIRED
    assert RuntimeEventType.TOOL_APPROVAL_RESOLVED not in [event.type for event in events]
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]
    assert RuntimeEventType.RUN_FAILED not in [event.type for event in events]


def test_runtime_multiple_approvals_are_processed_in_order_after_denial() -> None:
    class MixedApproval:
        def __init__(self) -> None:
            self.requests = []

        async def request(self, request):
            self.requests.append(request.call_id)
            return ApprovalDecision.DENY if request.call_id == "one" else ApprovalDecision.ALLOW_ONCE

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call):
            self.calls.append(call.id)
            return ToolResult(call_id=call.id, name=call.name, output={})

    class MultiToolModel:
        def __init__(self) -> None:
            self.turns = 0

        async def stream(self, messages, tools=None):
            self.turns += 1
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if self.turns == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="one", name="write_file", arguments={}))
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="two", name="send_email", arguments={}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(approval, executor):
        runtime = ChatRuntime(MultiToolModel(), tool_executor=executor, approval_provider=approval)
        tools = [
            ToolDefinition(name="write_file", risk_level=ToolRiskLevel.LOCAL_WRITE),
            ToolDefinition(name="send_email", risk_level=ToolRiskLevel.EXTERNAL_WRITE),
        ]
        return [event async for event in runtime.stream_user_message("hello", tools=tools)]

    approval = MixedApproval()
    executor = RecordingExecutor()
    events = asyncio.run(collect(approval, executor))

    assert approval.requests == ["one", "two"]
    assert executor.calls == ["two"]
    assert [event.tool_approval.decision for event in events if event.type == RuntimeEventType.TOOL_APPROVAL_RESOLVED] == [
        ApprovalDecision.DENY,
        ApprovalDecision.ALLOW_ONCE,
    ]
    assert [event.tool_result.error.code if event.tool_result.error else None for event in events if event.type == RuntimeEventType.TOOL_RESULT] == [
        ToolErrorCode.PERMISSION_DENIED,
        None,
    ]


def test_runtime_completes_after_multiple_tool_turns_with_history_preserved() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call):
            self.calls.append((call.id, call.name, dict(call.arguments)))
            return ToolResult(call_id=call.id, name=call.name, output={"text": call.arguments["text"]})

    class MultiTurnToolModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="checking first")
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_1", name="echo", arguments={"text": "one"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            elif len(self.messages_by_turn) == 2:
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="checking second")
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_2", name="echo", arguments={"text": "two"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model, executor):
        runtime = ChatRuntime(model, tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    model = MultiTurnToolModel()
    executor = RecordingExecutor()
    events = asyncio.run(collect(model, executor))

    assert executor.calls == [
        ("call_1", "echo", {"text": "one"}),
        ("call_2", "echo", {"text": "two"}),
    ]
    assert [event.type for event in events].count(RuntimeEventType.RUN_STARTED) == 1
    assert [event.type for event in events].count(RuntimeEventType.RUN_COMPLETED) == 1
    assert events[-1].type == RuntimeEventType.RUN_COMPLETED
    assert [event.tool_result.call_id for event in events if event.type == RuntimeEventType.TOOL_RESULT] == [
        "call_1",
        "call_2",
    ]

    third_turn_messages = model.messages_by_turn[2]
    assert [message.role for message in third_turn_messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert third_turn_messages[1].content == "checking first"
    assert third_turn_messages[1].tool_calls[0].id == "call_1"
    assert third_turn_messages[2].tool_call_id == "call_1"
    assert third_turn_messages[2].content == '{"ok":true,"output":{"text":"one"}}'
    assert third_turn_messages[3].content == "checking second"
    assert third_turn_messages[3].tool_calls[0].id == "call_2"
    assert third_turn_messages[4].tool_call_id == "call_2"
    assert third_turn_messages[4].content == '{"ok":true,"output":{"text":"two"}}'


def test_runtime_model_cannot_mutate_internal_message_history() -> None:
    class RecordingExecutor:
        async def execute(self, call):
            return ToolResult(call_id=call.id, name=call.name, output={"path": call.arguments["path"]})

    class MutatingModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_1", name="echo", arguments={"path": "original"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
                messages.clear()
            elif len(self.messages_by_turn) == 2:
                messages.clear()
                self.messages_by_turn[-1][-2].metadata["mutated"] = True
                self.messages_by_turn[-1][-2].tool_calls[0].arguments["path"] = "changed"
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_2", name="echo", arguments={"path": "second"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model):
        runtime = ChatRuntime(model, tool_executor=RecordingExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    model = MutatingModel()
    events = asyncio.run(collect(model))

    assert events[-1].type == RuntimeEventType.RUN_COMPLETED
    third_turn_messages = model.messages_by_turn[2]
    assert [message.role for message in third_turn_messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert third_turn_messages[1].metadata == {}
    assert third_turn_messages[1].tool_calls[0].arguments == {"path": "original"}
    assert third_turn_messages[2].content == '{"ok":true,"output":{"path":"original"}}'
    assert third_turn_messages[3].tool_calls[0].arguments == {"path": "second"}


def test_runtime_executes_multiple_tool_calls_in_order_even_when_one_fails() -> None:
    class MultiToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(messages) == 1:
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_missing", name="missing", arguments={}),
                )
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_echo", name="echo", arguments={"text": "ok"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect():
        runtime = ChatRuntime(MultiToolModel(), tool_executor=_executor_with_echo())
        return [
            event
            async for event in runtime.stream_user_message(
                "hello",
                tools=[ToolDefinition(name="missing"), ToolDefinition(name="echo")],
            )
        ]

    events = asyncio.run(collect())
    results = [event.tool_result for event in events if event.type == RuntimeEventType.TOOL_RESULT]

    assert [result.call_id for result in results] == ["call_missing", "call_echo"]
    assert results[0].error is not None
    assert results[1].output == {"text": "ok"}
    assert events[-1].type == RuntimeEventType.RUN_COMPLETED


def test_runtime_feeds_timeout_and_execution_failed_results_and_continues_same_turn() -> None:
    class MixedFailureExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call):
            self.calls.append(call.id)
            if call.id == "timeout":
                return ToolResult(
                    call_id=call.id,
                    name=call.name,
                    error=ToolError(
                        code=ToolErrorCode.EXECUTION_TIMEOUT,
                        message="Tool timed out after 1 seconds",
                    ),
                )
            if call.id == "failed":
                return ToolResult(
                    call_id=call.id,
                    name=call.name,
                    error=ToolError(
                        code=ToolErrorCode.EXECUTION_FAILED,
                        message="Tool execution failed",
                    ),
                )
            return ToolResult(call_id=call.id, name=call.name, output={"text": "ok"})

    class MixedFailureModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="timeout", name="slow", arguments={}))
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="failed", name="fail", arguments={}))
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="ok", name="echo", arguments={}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model, executor):
        runtime = ChatRuntime(model, tool_executor=executor)
        tools = [ToolDefinition(name="slow"), ToolDefinition(name="fail"), ToolDefinition(name="echo")]
        return [event async for event in runtime.stream_user_message("hello", tools=tools)]

    model = MixedFailureModel()
    executor = MixedFailureExecutor()
    events = asyncio.run(collect(model, executor))

    assert executor.calls == ["timeout", "failed", "ok"]
    assert RuntimeEventType.RUN_FAILED not in [event.type for event in events]
    assert events[-1].type == RuntimeEventType.RUN_COMPLETED
    tool_results = [event.tool_result for event in events if event.type == RuntimeEventType.TOOL_RESULT]
    assert [result.error.code for result in tool_results[:2]] == [
        ToolErrorCode.EXECUTION_TIMEOUT,
        ToolErrorCode.EXECUTION_FAILED,
    ]

    second_turn_messages = model.messages_by_turn[1]
    assert second_turn_messages[-3].content == (
        '{"ok":false,"error":{"code":"EXECUTION_TIMEOUT",'
        '"message":"Tool timed out after 1 seconds","retryable":false}}'
    )
    assert second_turn_messages[-2].content == (
        '{"ok":false,"error":{"code":"EXECUTION_FAILED",'
        '"message":"Tool execution failed","retryable":false}}'
    )
    assert second_turn_messages[-1].content == '{"ok":true,"output":{"text":"ok"}}'


def test_runtime_fails_duplicate_tool_call_ids_before_executing() -> None:
    class DuplicateToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="dup", name="echo", arguments={"text": "a"}))
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="dup", name="echo", arguments={"text": "b"}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(DuplicateToolModel(), tool_executor=_executor_with_echo())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_fails_when_tool_result_identity_mismatches_call() -> None:
    class BadExecutor:
        async def execute(self, call):
            return ToolResult(call_id="other", name=call.name, output={})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=BadExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_fails_when_tool_call_total_limit_would_be_exceeded() -> None:
    class TooManyToolsModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="one", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="two", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(
            TooManyToolsModel(),
            tool_executor=_executor_with_echo(),
            limits=AgentLoopLimits(max_tool_calls_total=1),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_fails_when_tool_result_is_not_json_serializable_without_echoing_output() -> None:
    class BadExecutor:
        async def execute(self, call):
            return ToolResult(call_id=call.id, name=call.name, output={"secret": {1, 2, 3}})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=BadExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert "secret" not in repr(events[-1])
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_fails_repeated_tool_call_id_across_turns_before_reexecution() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={"ok": True})

    class RepeatingModel:
        def __init__(self) -> None:
            self.turns = 0

        async def stream(self, messages, tools=None):
            self.turns += 1
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(RepeatingModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 1
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_rejects_non_json_tool_arguments_before_executing_tool() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class BadArgumentModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={"value": {1, 2, 3}}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(BadArgumentModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_snapshots_nested_tool_arguments_before_tool_mutation() -> None:
    class MutatingExecutor:
        async def execute(self, call):
            call.arguments["nested"]["text"] = "mutated"
            return ToolResult(call_id=call.id, name=call.name, output={"ok": True})

    class SnapshotModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={"nested": {"text": "original"}}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model):
        runtime = ChatRuntime(model, tool_executor=MutatingExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    model = SnapshotModel()
    asyncio.run(collect(model))

    assert model.messages_by_turn[1][-2].tool_calls[0].arguments == {"nested": {"text": "original"}}


def test_runtime_rejects_conflicting_tool_result_output_and_error() -> None:
    class BadExecutor:
        async def execute(self, call):
            from nexusmind.tools.contracts import ToolError, ToolErrorCode

            return ToolResult(
                call_id=call.id,
                name=call.name,
                output={"result": "ok"},
                error=ToolError(code=ToolErrorCode.EXECUTION_FAILED, message="failed"),
            )

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=BadExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_reports_tool_executor_exception_as_run_failure_only() -> None:
    class ExplodingExecutor:
        async def execute(self, call):
            raise RuntimeError("sk-live-secret")

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=ExplodingExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.MODEL_FAILED not in [event.type for event in events]
    assert "sk-live-secret" not in repr(events)


def test_runtime_enforces_tool_result_size_limit_without_tool_result_event() -> None:
    class BigExecutor:
        async def execute(self, call):
            return ToolResult(call_id=call.id, name=call.name, output={"text": "x" * 100})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(
            ToolModel(),
            tool_executor=BigExecutor(),
            limits=AgentLoopLimits(max_tool_result_bytes_per_call=20),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_rejects_tool_call_name_not_advertised_for_this_run() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={"ok": True})

    class UnadvertisedToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="delete_files", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(UnadvertisedToolModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_allows_successful_tool_result_with_null_output() -> None:
    class NullExecutor:
        async def execute(self, call):
            return ToolResult(call_id=call.id, name=call.name, output=None)

    class NullToolModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="nullable", arguments={}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model):
        runtime = ChatRuntime(model, tool_executor=NullExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="nullable")])]

    model = NullToolModel()
    events = asyncio.run(collect(model))

    assert events[-1].type == RuntimeEventType.RUN_COMPLETED
    assert model.messages_by_turn[1][-1].content == '{"ok":true,"output":null}'


def test_runtime_rejects_invalid_tool_error_shape_before_event() -> None:
    class BadExecutor:
        async def execute(self, call):
            return ToolResult(
                call_id=call.id,
                name=call.name,
                error=ToolError(
                    code=ToolErrorCode.EXECUTION_FAILED,
                    message="failed",
                    retryable="yes",  # type: ignore[arg-type]
                ),
            )

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=BadExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_enforces_tool_argument_size_limit_before_executing_tool() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class BigArgumentModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={"text": "x" * 100}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(
            BigArgumentModel(),
            tool_executor=executor,
            limits=AgentLoopLimits(max_tool_arguments_bytes_per_call=20),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_tool_result_repr_redacts_error_message_and_metadata() -> None:
    result = ToolResult(
        call_id="call_1",
        name="echo",
        error=ToolError(code=ToolErrorCode.EXECUTION_FAILED, message="sk-live-secret"),
        metadata={"token": "secret"},
    )

    assert "sk-live-secret" not in repr(result)
    assert "secret" not in repr(result)


def test_runtime_executes_original_tool_arguments_when_consumer_mutates_event() -> None:
    class RecordingExecutor:
        def __init__(self) -> None:
            self.arguments = None

        async def execute(self, call):
            self.arguments = dict(call.arguments)
            return ToolResult(call_id=call.id, name=call.name, output={"ok": True})

    class ToolModel:
        def __init__(self) -> None:
            self.turns = 0

        async def stream(self, messages, tools=None):
            self.turns += 1
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if self.turns == 1:
                yield RuntimeEvent(
                    RuntimeEventType.TOOL_CALL_COMPLETED,
                    tool_call=ToolCall(id="call_1", name="echo", arguments={"path": "original"}),
                )
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def run(executor):
        stream = ChatRuntime(ToolModel(), tool_executor=executor).stream_user_message(
            "hello",
            tools=[ToolDefinition(name="echo")],
        )
        events = []
        async for event in stream:
            if event.tool_call is not None:
                event.tool_call.arguments["path"] = "mutated"
            events.append(event)
        return events

    executor = RecordingExecutor()
    events = asyncio.run(run(executor))

    assert events[-1].type == RuntimeEventType.RUN_COMPLETED
    assert executor.arguments == {"path": "original"}


def test_runtime_does_not_execute_tools_when_no_model_turn_budget_remains() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(
            ToolModel(),
            tool_executor=executor,
            limits=AgentLoopLimits(max_model_turns=1),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_enforces_tool_call_count_limit_as_calls_arrive() -> None:
    class TooManyToolCallsModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="one", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="two", name="echo", arguments={}))
            raise AssertionError("runtime should stop before consuming more tool calls")

    async def collect():
        runtime = ChatRuntime(
            TooManyToolCallsModel(),
            tool_executor=_executor_with_echo(),
            limits=AgentLoopLimits(max_tool_calls_total=1),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert [event.type for event in events].count(RuntimeEventType.TOOL_CALL_COMPLETED) == 1


def test_runtime_does_not_start_next_tool_when_result_budget_is_exhausted() -> None:
    class BudgetExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call):
            self.calls.append(call.id)
            return ToolResult(call_id=call.id, name=call.name, output="")

    class TwoToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="one", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="two", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(
            TwoToolModel(),
            tool_executor=executor,
            limits=AgentLoopLimits(max_tool_result_bytes_total=23),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = BudgetExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == ["one"]
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_json_budget_accepts_short_control_character_escape() -> None:
    class NewlineExecutor:
        async def execute(self, call):
            return ToolResult(call_id=call.id, name=call.name, output={"x": "\n"})

    class ToolModel:
        def __init__(self) -> None:
            self.messages_by_turn = []

        async def stream(self, messages, tools=None):
            self.messages_by_turn.append(list(messages))
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            if len(self.messages_by_turn) == 1:
                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")
            else:
                yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    async def collect(model):
        runtime = ChatRuntime(
            model,
            tool_executor=NewlineExecutor(),
            limits=AgentLoopLimits(max_tool_result_bytes_per_call=31),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    model = ToolModel()
    events = asyncio.run(collect(model))

    assert events[-1].type == RuntimeEventType.RUN_COMPLETED
    assert model.messages_by_turn[1][-1].content == '{"ok":true,"output":{"x":"\\n"}}'


def test_message_content_none_is_limited_to_assistant_tool_call_messages() -> None:
    invalid_messages = [
        lambda: Message(MessageRole.USER, None),
        lambda: Message(MessageRole.SYSTEM, None),
        lambda: Message(MessageRole.TOOL, None, tool_call_id="call_1"),
        lambda: Message(MessageRole.ASSISTANT, None),
    ]

    for build in invalid_messages:
        try:
            build()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    message = Message(
        MessageRole.ASSISTANT,
        None,
        tool_calls=(ToolCall(id="call_1", name="echo", arguments={}),),
    )
    assert message.content is None


def test_runtime_reports_non_json_tool_arguments_as_model_failure() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class BadArgumentModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="echo", arguments={"value": object()}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(BadArgumentModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert [event.type for event in events[-2:]] == [
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]


def test_runtime_does_not_execute_tool_when_minimum_result_envelope_cannot_fit() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output=None)

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(
            ToolModel(),
            tool_executor=executor,
            limits=AgentLoopLimits(max_tool_result_bytes_per_call=10),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_rejects_executor_result_that_is_not_tool_result() -> None:
    class FakeResult:
        call_id = "call_1"
        name = "echo"
        output = {"ok": True}
        error = None

    class BadExecutor:
        async def execute(self, call):
            return FakeResult()

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=BadExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_json_budget_counts_container_punctuation_before_encoding() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ContainerHeavyModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="echo", arguments={"items": [[], [], []]}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(
            ContainerHeavyModel(),
            tool_executor=executor,
            limits=AgentLoopLimits(max_tool_arguments_bytes_per_call=14),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_json_budget_rejects_excessive_nodes_before_encoding() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    class ManyNodesModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="echo", arguments={"items": [1, 2, 3]}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(
            ManyNodesModel(),
            tool_executor=executor,
            limits=AgentLoopLimits(max_json_nodes_per_payload=3),
        )
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert events[-1].type == RuntimeEventType.RUN_FAILED


def test_runtime_json_budget_rejects_circular_arguments() -> None:
    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id=call.id, name=call.name, output={})

    circular = {}
    circular["self"] = circular

    class CircularModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id="call_1", name="echo", arguments=circular),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(CircularModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert [event.type for event in events[-2:]] == [
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]


def test_runtime_rejects_tool_result_string_subclass_identity_fields() -> None:
    class SneakyStr(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    class BadExecutor:
        async def execute(self, call):
            return ToolResult(call_id=SneakyStr("other"), name=SneakyStr("other"), output={})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=BadExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_runtime_rejects_tool_call_string_subclass_identity_before_execution() -> None:
    class SneakyStr(str):
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

    class CountingExecutor:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, call):
            self.calls += 1
            return ToolResult(call_id="call_1", name="echo", output={})

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(
                RuntimeEventType.TOOL_CALL_COMPLETED,
                tool_call=ToolCall(id=SneakyStr("call_1"), name=SneakyStr("echo"), arguments={}),
            )
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(ToolModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CountingExecutor()
    events = asyncio.run(collect(executor))

    assert executor.calls == 0
    assert [event.type for event in events[-2:]] == [
        RuntimeEventType.MODEL_FAILED,
        RuntimeEventType.RUN_FAILED,
    ]


def test_runtime_rejects_tool_result_metadata_that_is_not_dict() -> None:
    class BadExecutor:
        async def execute(self, call):
            return ToolResult(
                call_id=call.id,
                name=call.name,
                output={},
                metadata="not-a-dict",  # type: ignore[arg-type]
            )

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect():
        runtime = ChatRuntime(ToolModel(), tool_executor=BadExecutor())
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.RUN_FAILED
    assert RuntimeEventType.TOOL_RESULT not in [event.type for event in events]


def test_agent_loop_limits_reject_result_node_budget_that_cannot_fit_minimum_envelope() -> None:
    try:
        AgentLoopLimits(max_json_nodes_per_payload=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_runtime_propagates_cancel_during_model_call() -> None:
    class CancelModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            raise asyncio.CancelledError()

    async def collect(events):
        async for event in ChatRuntime(CancelModel()).stream_user_message("hello"):
            events.append(event)

    events = []
    try:
        asyncio.run(collect(events))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
    ]
    assert RuntimeEventType.MODEL_FAILED not in [event.type for event in events]
    assert RuntimeEventType.RUN_FAILED not in [event.type for event in events]
    assert RuntimeEventType.RUN_COMPLETED not in [event.type for event in events]


def test_runtime_propagates_cancel_during_tool_call() -> None:
    class CancelExecutor:
        async def execute(self, call):
            raise asyncio.CancelledError()

    class ToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="call_1", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(events):
        runtime = ChatRuntime(ToolModel(), tool_executor=CancelExecutor())
        async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")]):
            events.append(event)

    events = []
    try:
        asyncio.run(collect(events))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert [event.type for event in events] == [
        RuntimeEventType.RUN_STARTED,
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TOOL_CALL_COMPLETED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
    ]
    assert RuntimeEventType.MODEL_FAILED not in [event.type for event in events]
    assert RuntimeEventType.RUN_FAILED not in [event.type for event in events]
    assert RuntimeEventType.RUN_COMPLETED not in [event.type for event in events]
    assert [event.type for event in events].count(RuntimeEventType.MODEL_STARTED) == 1


def test_runtime_does_not_start_later_tools_after_mid_batch_cancel() -> None:
    class CancelSecondExecutor:
        def __init__(self) -> None:
            self.calls = []

        async def execute(self, call):
            self.calls.append(call.id)
            if call.id == "two":
                raise asyncio.CancelledError()
            return ToolResult(call_id=call.id, name=call.name, output={})

    class MultiToolModel:
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="one", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="two", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=ToolCall(id="three", name="echo", arguments={}))
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

    async def collect(executor):
        runtime = ChatRuntime(MultiToolModel(), tool_executor=executor)
        return [event async for event in runtime.stream_user_message("hello", tools=[ToolDefinition(name="echo")])]

    executor = CancelSecondExecutor()
    try:
        asyncio.run(collect(executor))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("expected CancelledError")

    assert executor.calls == ["one", "two"]

