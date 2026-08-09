import asyncio
import json
import pytest

from nexusmind.models.fake import FakeChatModel
from nexusmind.runtime.harness import (
    CheckpointBoundary, HarnessCheckpoint, HarnessPhase, HarnessRequest,
    HarnessResumeCompatibilityError, HarnessResumeRequest, HarnessResumeStateError, HarnessRunner, HarnessState,
    HarnessStatus, SQLiteCheckpointStore, StopReason,
)
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolCall, ToolDefinition, ToolResult
from nexusmind.tools.builtin import EchoTool
from nexusmind.tools.executor import ToolExecutor
from nexusmind.tools.registry import ToolRegistry

def _argument_bytes(*calls):
    return sum(len(json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for call in calls)

def _echo_executor():
    registry = ToolRegistry()
    registry.register(EchoTool())
    return ToolExecutor(registry)

def _before_model_checkpoint():
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello")], phase=HarnessPhase.BEFORE_MODEL)
    return HarnessCheckpoint.create(state, "run-1", 3)

def test_resume_before_model_preserves_state_and_isolated_messages():
    async def run():
        execution = HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(_before_model_checkpoint()))
        [event async for event in execution.stream()]
        assert execution.state.status is HarnessStatus.COMPLETED
        assert execution.state.messages[0].content == "hello"
    asyncio.run(run())

def test_before_model_rejects_partial_tool_batch():
    calls = (
        ToolCall(id="call-a", name="echo", arguments={"text": "a"}),
        ToolCall(id="call-b", name="echo", arguments={"text": "b"}),
    )
    result = Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-a", content="{}")
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="go"),
        Message(role=MessageRole.ASSISTANT, content=None, tool_calls=calls), result],
        model_turns=1, tool_calls_total=1, tool_argument_bytes_total=_argument_bytes(*calls),
        tool_result_bytes_total=2, started_tool_call_ids={"call-a"}, executed_tool_call_ids={"call-a"},
        phase=HarnessPhase.BEFORE_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-partial-before-model", 0)
    with pytest.raises(HarnessResumeStateError, match="completed Tool batch"):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

def test_before_model_rejects_assistant_followed_by_system():
    state = HarnessState(messages=[
        Message(role=MessageRole.USER, content="go"),
        Message(role=MessageRole.ASSISTANT, content="done"),
        Message(role=MessageRole.SYSTEM, content="late"),
    ], model_turns=1, phase=HarnessPhase.BEFORE_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-late-system", 0)
    with pytest.raises(HarnessResumeStateError, match="terminal Assistant"):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

def test_resume_rejects_terminal_assistant_before_tool_batch():
    call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[
        Message(role=MessageRole.USER, content="go"),
        Message(role=MessageRole.ASSISTANT, content="final"),
        Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,)),
        Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-1", content="{}"),
    ], model_turns=2, tool_calls_total=1, tool_argument_bytes_total=_argument_bytes(call),
        tool_result_bytes_total=2, started_tool_call_ids={"call-1"}, executed_tool_call_ids={"call-1"},
        phase=HarnessPhase.BEFORE_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-terminal-trailing", 0)
    with pytest.raises(HarnessResumeStateError, match="terminal Assistant"):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

def test_resume_rejects_unhashable_tool_call_id():
    call = ToolCall(id=[], name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
        model_turns=1, phase=HarnessPhase.AFTER_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-unhashable", 0)
    with pytest.raises(HarnessResumeStateError, match="Tool Call ID"):
        HarnessRunner(FakeChatModel(["done"]), tool_executor=_echo_executor()).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )

@pytest.mark.parametrize("tool_call_id,name", [
    (["call-1"], "echo"),
    ({"id": "call-1"}, "echo"),
    ("call-1", ["echo"]),
    ("call-1", ""),
])
def test_resume_rejects_invalid_tool_result_identity(tool_call_id, name):
    call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[
        Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,)),
        Message(role=MessageRole.TOOL, name=name, tool_call_id=tool_call_id, content="{}"),
    ], model_turns=1, phase=HarnessPhase.AFTER_TOOL, tool_result_bytes_total=2)
    checkpoint = HarnessCheckpoint.create(state, "run-invalid-result", 0)
    with pytest.raises(HarnessResumeStateError, match="Tool result"):
        HarnessRunner(FakeChatModel(["done"]), tool_executor=_echo_executor()).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )

@pytest.mark.parametrize("call_id,name", [("", "echo"), ("call-1", "")])
def test_resume_rejects_empty_tool_identity_before_executor(call_id, name):
    call = ToolCall(id=call_id, name=name, arguments={"text": "hello"})
    state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
        model_turns=1, phase=HarnessPhase.AFTER_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-empty-identity", 0)
    executor = _echo_executor()
    with pytest.raises(HarnessResumeStateError, match="Tool Call"):
        HarnessRunner(FakeChatModel(["done"]), tool_executor=executor).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )

def test_resumed_execution_preserves_source_phase_before_stream():
    call = ToolCall(id="call-phase", name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
        model_turns=1, phase=HarnessPhase.BEFORE_TOOL,
        tool_argument_bytes_total=_argument_bytes(call))
    checkpoint = HarnessCheckpoint.create(state, "run-phase", 0)
    execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=_echo_executor()).resume_execution(
        HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
    )
    assert execution.state.phase is HarnessPhase.BEFORE_TOOL

def test_resume_rejects_terminal_checkpoint():
    state = HarnessState(messages=[], status=HarnessStatus.COMPLETED, stop_reason=StopReason.MODEL_COMPLETED, phase=HarnessPhase.TERMINAL)
    checkpoint = HarnessCheckpoint.create(state, "run-1", 0)
    with pytest.raises(HarnessResumeStateError):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

def test_resume_rejects_internal_metadata_controls():
    with pytest.raises(HarnessResumeStateError, match="reserved"):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(
            HarnessResumeRequest(_before_model_checkpoint(), metadata={"resume_tool_batch": True})
        )

def test_execution_stream_is_single_use():
    async def run():
        execution = HarnessRunner(FakeChatModel(["done"])).create_execution(
            HarnessRequest(messages=(Message(role=MessageRole.USER, content="hello"),))
        )
        [event async for event in execution.stream()]
        with pytest.raises(RuntimeError, match="only be streamed once"):
            [event async for event in execution.stream()]
    asyncio.run(run())

def test_normal_request_rejects_internal_resume_metadata():
    request = HarnessRequest(messages=(Message(role=MessageRole.USER, content="hello"),),
        metadata={"resume_existing_assistant": True})
    with pytest.raises(ValueError, match="reserved"):
        HarnessRunner(FakeChatModel(["done"])).create_execution(request)

def test_resume_after_model_text_completes_without_model_call():
    class CountingModel(FakeChatModel):
        def __init__(self):
            super().__init__(["unexpected"])
            self.calls = 0
        async def stream(self, messages, tools=None):
            self.calls += 1
            async for event in super().stream(messages, tools=tools):
                yield event
    async def run():
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content="answer")], model_turns=1, status=HarnessStatus.RUNNING, phase=HarnessPhase.AFTER_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-text", 0)
        model = CountingModel()
        execution = HarnessRunner(model).resume_execution(HarnessResumeRequest(checkpoint))
        [event async for event in execution.stream()]
        assert model.calls == 0
        assert execution.state.status is HarnessStatus.COMPLETED
    asyncio.run(run())

def test_resumed_checkpoint_defaults_to_source_run_id():
    state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello")], phase=HarnessPhase.BEFORE_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-lineage", 3)
    execution = HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))
    resumed = execution.create_checkpoint(sequence=4)
    assert resumed.run_id == "run-lineage"
    with pytest.raises(HarnessResumeStateError):
        execution.create_checkpoint(sequence=3)
    with pytest.raises(HarnessResumeStateError):
        execution.create_checkpoint(run_id="other-run", sequence=5)

def test_resumed_execution_requires_monotonic_checkpoint_sequences():
    checkpoint = _before_model_checkpoint()
    execution = HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))
    first = execution.create_checkpoint(sequence=4)
    assert first.sequence == 4
    with pytest.raises(HarnessResumeStateError, match="monotonically"):
        execution.create_checkpoint(sequence=4)
    second = execution.create_checkpoint(sequence=5)
    assert second.sequence == 5
    with pytest.raises(HarnessResumeStateError, match="monotonically"):
        execution.create_checkpoint(sequence=4)

def test_two_resumes_are_isolated_from_each_other_and_checkpoint():
    async def run():
        source_message = Message(role=MessageRole.USER, content="hello", metadata={"nested": {"value": 1}})
        state = HarnessState(messages=[source_message], model_turns=3, tool_calls_total=0,
            tool_argument_bytes_total=10, tool_result_bytes_total=20,
            phase=HarnessPhase.BEFORE_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-isolated", 4)
        runner = HarnessRunner(FakeChatModel(["done"]))
        first = runner.resume_execution(HarnessResumeRequest(checkpoint))
        second = runner.resume_execution(HarnessResumeRequest(checkpoint))
        first.state.messages[0].metadata["nested"]["value"] = 99
        assert second.state.messages[0].metadata["nested"]["value"] == 1
        assert checkpoint.state.messages[0].metadata["nested"]["value"] == 1
        assert first.state.model_turns == second.state.model_turns == 3
        assert first.state.tool_calls_total == second.state.tool_calls_total == 0
    asyncio.run(run())

def test_resume_after_model_rejects_trailing_transcript():
    state = HarnessState(messages=[
        Message(role=MessageRole.ASSISTANT, content="answer"),
        Message(role=MessageRole.USER, content="later"),
    ], model_turns=1, status=HarnessStatus.RUNNING, phase=HarnessPhase.AFTER_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-text", 0)
    with pytest.raises(HarnessResumeStateError):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

def test_resume_at_model_limit_emits_limit_failure():
    async def run():
        state = HarnessState(messages=[Message(role=MessageRole.USER, content="hello")], model_turns=1, status=HarnessStatus.RUNNING, phase=HarnessPhase.BEFORE_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-limit", 0)
        execution = HarnessRunner(FakeChatModel(["unexpected"]), limits=HarnessLimits(max_model_turns=1)).resume_execution(HarnessResumeRequest(checkpoint))
        events = [event async for event in execution.stream()]
        assert events[-1].type.value == "run_failed"
        assert execution.stop_reason is StopReason.LIMIT_EXCEEDED
        assert execution.state.phase is HarnessPhase.TERMINAL
    asyncio.run(run())

def test_resume_after_tool_with_completed_batch_continues_model():
    async def run():
        call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
        state = HarnessState(messages=[
            Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,)),
            Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-1", content='{"ok":true}'),
            ], model_turns=1, tool_calls_total=1, tool_argument_bytes_total=_argument_bytes(call),
                tool_result_bytes_total=len('{"ok":true}'.encode("utf-8")), started_tool_call_ids={"call-1"},
            executed_tool_call_ids={"call-1"}, status=HarnessStatus.RUNNING,
            phase=HarnessPhase.AFTER_TOOL)
        checkpoint = HarnessCheckpoint.create(state, "run-tool", 0)
        execution = HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))
        events = [event async for event in execution.stream()]
        assert events[-1].type.value == "run_completed"
        assert execution.state.executed_tool_call_ids == {"call-1"}
    asyncio.run(run())

def test_resume_after_model_executes_pending_tool_without_model_replay():
    async def run():
        call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
            model_turns=1, status=HarnessStatus.RUNNING, phase=HarnessPhase.AFTER_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-tool", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=ToolExecutor(registry)).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )
        events = [event async for event in execution.stream()]
        assert any(event.type.value == "tool_result" for event in events)
        assert not any(event.metadata.get("resume_internal") for event in events)
        assert execution.state.executed_tool_call_ids == {"call-1"}
    asyncio.run(run())

def test_resume_pending_tool_at_model_limit_executes_before_next_model():
    async def run():
        call = ToolCall(id="call-limit", name="echo", arguments={"text": "hi"})
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
            model_turns=1, status=HarnessStatus.RUNNING, phase=HarnessPhase.AFTER_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-limit-tool", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["unexpected"]), tool_executor=ToolExecutor(registry),
            limits=HarnessLimits(max_model_turns=1)).resume_execution(
                HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
            )
        events = [event async for event in execution.stream()]
        assert any(event.type.value == "tool_result" for event in events)
        assert execution.stop_reason is StopReason.LIMIT_EXCEEDED
        assert execution.state.executed_tool_call_ids == {"call-limit"}
    asyncio.run(run())

@pytest.mark.parametrize("phase", [HarnessPhase.BEFORE_TOOL, HarnessPhase.AFTER_TOOL])
def test_resume_tool_phase_at_model_limit_executes_pending_tool(phase):
    async def run():
        first = ToolCall(id="call-done", name="echo", arguments={"text": "done"})
        pending = ToolCall(id="call-pending", name="echo", arguments={"text": "pending"})
        messages = [Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(first, pending))]
        kwargs = {}
        if phase is HarnessPhase.AFTER_TOOL:
            messages.append(Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-done", content="{}"))
            kwargs = {"tool_calls_total": 1, "tool_result_bytes_total": len("{}".encode("utf-8")), "started_tool_call_ids": {"call-done"}, "executed_tool_call_ids": {"call-done"}}
        else:
            messages = [Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(pending,))]
        accounted = (pending,) if phase is HarnessPhase.BEFORE_TOOL else (first, pending)
        state = HarnessState(messages=messages, model_turns=1, tool_argument_bytes_total=_argument_bytes(*accounted), phase=phase, **kwargs)
        checkpoint = HarnessCheckpoint.create(state, "run-phase-limit", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["unexpected"]), tool_executor=ToolExecutor(registry),
            limits=HarnessLimits(max_model_turns=1)).resume_execution(
                HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
            )
        events = [event async for event in execution.stream()]
        assert any(event.type.value == "tool_result" and event.tool_result.call_id == "call-pending" for event in events)
        assert execution.stop_reason is StopReason.LIMIT_EXCEEDED
    asyncio.run(run())

def test_resume_rejects_checkpoint_before_pending_cursor_advances():
    call = ToolCall(id="call-pending", name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
        model_turns=1, phase=HarnessPhase.AFTER_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-cursor", 0)
    execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=_echo_executor()).resume_execution(
        HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
    )
    with pytest.raises(HarnessResumeStateError, match="resume cursor"):
        execution.create_checkpoint(sequence=1)

def test_resume_cursor_stays_blocked_after_run_started():
    async def run():
        call = ToolCall(id="call-pending", name="echo", arguments={"text": "hi"})
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
            model_turns=1, phase=HarnessPhase.AFTER_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-cursor-started", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=ToolExecutor(registry)).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )
        stream = execution.stream()
        first = await anext(stream)
        assert first.type.value == "run_started"
        with pytest.raises(HarnessResumeStateError, match="resume cursor"):
            execution.create_checkpoint(sequence=1)
        await stream.aclose()
    asyncio.run(run())

def test_resume_after_model_executes_multiple_pending_tools():
    async def run():
        calls = (
            ToolCall(id="call-1", name="echo", arguments={"text": "one"}),
            ToolCall(id="call-2", name="echo", arguments={"text": "two"}),
        )
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=calls)],
            model_turns=1, phase=HarnessPhase.AFTER_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-multiple", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=ToolExecutor(registry)).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )
        events = [event async for event in execution.stream()]
        results = [event.tool_result.call_id for event in events if event.type.value == "tool_result"]
        assert results == ["call-1", "call-2"]
        assert execution.state.executed_tool_call_ids == {"call-1", "call-2"}
    asyncio.run(run())

def test_checkpoint_allowed_after_first_resumed_tool_result():
    async def run():
        calls = (
            ToolCall(id="call-1", name="echo", arguments={"text": "one"}),
            ToolCall(id="call-2", name="echo", arguments={"text": "two"}),
        )
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=calls)],
            model_turns=1, phase=HarnessPhase.AFTER_MODEL)
        source = HarnessCheckpoint.create(state, "run-mid-tool", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=ToolExecutor(registry)).resume_execution(
            HarnessResumeRequest(source, tools=(EchoTool().definition,))
        )
        stream = execution.stream()
        async for event in stream:
            if event.type.value == "tool_result":
                checkpoint = execution.create_checkpoint(sequence=1)
                assert checkpoint.state.executed_tool_call_ids == ("call-1",)
                await stream.aclose()
                return
        pytest.fail("missing resumed tool result")
    asyncio.run(run())

def test_resume_after_model_allows_arguments_at_total_limit():
    async def run():
        call = ToolCall(id="call-budget", name="echo", arguments={"text": "hi"})
        argument_bytes = len('{"text":"hi"}'.encode("utf-8"))
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
            model_turns=1, phase=HarnessPhase.AFTER_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-budget", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        limits = HarnessLimits(max_tool_arguments_bytes_total=argument_bytes)
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=ToolExecutor(registry), limits=limits).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )
        events = [event async for event in execution.stream()]
        assert any(event.type.value == "tool_result" for event in events)
        assert execution.state.tool_argument_bytes_total == argument_bytes
    asyncio.run(run())

def test_resume_rejects_missing_pending_tool_before_execution():
    call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
        model_turns=1, phase=HarnessPhase.AFTER_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-tool", 0)
    with pytest.raises(HarnessResumeCompatibilityError, match="missing tools"):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

def test_resume_rejects_unknown_tool_result_in_batch():
    call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[
        Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,)),
        Message(role=MessageRole.TOOL, name="echo", tool_call_id="unknown", content="{}"),
    ], model_turns=1, tool_calls_total=1, executed_tool_call_ids={"unknown"},
        phase=HarnessPhase.AFTER_TOOL)
    with pytest.raises(ValueError):
        HarnessCheckpoint.create(state, "run-invalid", 0)

def test_resume_rejects_incomplete_previous_tool_batch():
    first = ToolCall(id="call-a", name="echo", arguments={"text": "a"})
    missing = ToolCall(id="call-b", name="echo", arguments={"text": "b"})
    next_call = ToolCall(id="call-c", name="echo", arguments={"text": "c"})
    state = HarnessState(messages=[
        Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(first, missing)),
        Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-a", content="{}"),
        Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(next_call,)),
    ], model_turns=2, tool_calls_total=1, started_tool_call_ids={"call-a"},
        executed_tool_call_ids={"call-a"}, phase=HarnessPhase.BEFORE_TOOL)
    checkpoint = HarnessCheckpoint.create(state, "run-invalid", 0)
    with pytest.raises(HarnessResumeStateError, match="previous Tool Call batch"):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

def test_resume_before_tool_executes_pending_tool():
    async def run():
        call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
            model_turns=1, tool_argument_bytes_total=_argument_bytes(call), status=HarnessStatus.RUNNING, phase=HarnessPhase.BEFORE_TOOL)
        checkpoint = HarnessCheckpoint.create(state, "run-before-tool", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["unexpected"]), tool_executor=ToolExecutor(registry)).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )
        events = [event async for event in execution.stream()]
        assert any(event.type.value == "tool_result" for event in events)
        assert execution.state.executed_tool_call_ids == {"call-1"}
    asyncio.run(run())

def test_resume_before_tool_mid_batch_executes_only_remaining_call():
    async def run():
        first = ToolCall(id="call-1", name="echo", arguments={"text": "one"})
        second = ToolCall(id="call-2", name="echo", arguments={"text": "two"})
        state = HarnessState(messages=[
            Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(first, second)),
            Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-1", content="{}"),
        ], model_turns=1, tool_calls_total=1, tool_argument_bytes_total=_argument_bytes(first, second),
            tool_result_bytes_total=2, started_tool_call_ids={"call-1"}, executed_tool_call_ids={"call-1"},
            phase=HarnessPhase.BEFORE_TOOL)
        checkpoint = HarnessCheckpoint.create(state, "run-before-second", 0)
        registry = ToolRegistry(); registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=ToolExecutor(registry)).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )
        events = [event async for event in execution.stream()]
        assert [event.tool_result.call_id for event in events if event.type.value == "tool_result"] == ["call-2"]
        assert [message.tool_call_id for message in execution.state.messages if message.role is MessageRole.TOOL] == ["call-1", "call-2"]
    asyncio.run(run())

def test_sqlite_load_latest_resume_partial_tool_batch(tmp_path):
    class RecordingExecutor(ToolExecutor):
        def __init__(self, registry):
            super().__init__(registry); self.calls = []
        async def execute_with_result_budget(self, call, *, result_budget):
            self.calls.append(call.id)
            return await super().execute_with_result_budget(call, result_budget=result_budget)

    async def run():
        path = tmp_path / "resume.db"
        calls = (
            ToolCall(id="call-1", name="echo", arguments={"text": "one"}),
            ToolCall(id="call-2", name="echo", arguments={"text": "two"}),
        )
        source = HarnessCheckpoint.create(HarnessState(
            messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=calls)],
            model_turns=1, phase=HarnessPhase.AFTER_MODEL,
        ), "sqlite-run", 0)
        store = SQLiteCheckpointStore(path); await store.initialize(); await store.save(source); await store.close()
        reopened = SQLiteCheckpointStore(path); await reopened.initialize(); loaded = await reopened.load_latest("sqlite-run")
        registry = ToolRegistry(); registry.register(EchoTool())
        first_executor = RecordingExecutor(registry)
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=first_executor).resume_execution(
            HarnessResumeRequest(loaded, tools=(EchoTool().definition,))
        )
        stream = execution.stream()
        async for event in stream:
            if event.type.value == "tool_result":
                partial = execution.create_checkpoint(sequence=1)
                await reopened.save(partial); await stream.aclose(); break
        await reopened.close()
        assert first_executor.calls == ["call-1"]
        final_store = SQLiteCheckpointStore(path); await final_store.initialize(); partial_loaded = await final_store.load_latest("sqlite-run")
        second_executor = RecordingExecutor(registry)
        final_execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=second_executor).resume_execution(
            HarnessResumeRequest(partial_loaded, tools=(EchoTool().definition,))
        )
        [event async for event in final_execution.stream()]
        assert second_executor.calls == ["call-2"]
        final = final_execution.create_checkpoint(sequence=2); await final_store.save(final); await final_store.close()
        verify = SQLiteCheckpointStore(path); await verify.initialize(); latest = await verify.load_latest("sqlite-run"); await verify.close()
        assert latest.sequence == 2 and latest.run_id == "sqlite-run"
        assert set(latest.state.executed_tool_call_ids) == {"call-1", "call-2"}
    asyncio.run(run())

def test_resume_after_tool_executes_only_remaining_calls():
    async def run():
        first = ToolCall(id="call-1", name="echo", arguments={"text": "one"})
        second = ToolCall(id="call-2", name="echo", arguments={"text": "two"})
        state = HarnessState(messages=[
            Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(first, second)),
            Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-1", content='{"ok":true}'),
        ], model_turns=1, tool_calls_total=1, tool_argument_bytes_total=_argument_bytes(first, second),
            tool_result_bytes_total=len('{"ok":true}'.encode("utf-8")), started_tool_call_ids={"call-1"},
            executed_tool_call_ids={"call-1"}, phase=HarnessPhase.AFTER_TOOL)
        checkpoint = HarnessCheckpoint.create(state, "run-partial-tool", 0)
        registry = ToolRegistry()
        registry.register(EchoTool())
        execution = HarnessRunner(FakeChatModel(["unexpected"]), tool_executor=ToolExecutor(registry)).resume_execution(
            HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
        )
        events = [event async for event in execution.stream()]
        results = [event.tool_result.call_id for event in events if event.type.value == "tool_result"]
        assert results == ["call-2"]
        assert execution.state.executed_tool_call_ids == {"call-1", "call-2"}
        assert [message.tool_call_id for message in execution.state.messages if message.role is MessageRole.TOOL] == ["call-1", "call-2"]
        assistants = [message for message in execution.state.messages if message.role is MessageRole.ASSISTANT]
        assert len(assistants) == 2
        assert assistants[-1].content == "unexpected"
    asyncio.run(run())
