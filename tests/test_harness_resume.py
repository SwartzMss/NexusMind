import asyncio
import pytest

from nexusmind.models.fake import FakeChatModel
from nexusmind.runtime.harness import (
    CheckpointBoundary, HarnessCheckpoint, HarnessPhase, HarnessRequest,
    HarnessResumeCompatibilityError, HarnessResumeRequest, HarnessResumeStateError, HarnessRunner, HarnessState,
    HarnessStatus, StopReason,
)
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolCall, ToolDefinition, ToolResult
from nexusmind.tools.builtin import EchoTool
from nexusmind.tools.executor import ToolExecutor
from nexusmind.tools.registry import ToolRegistry

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

def test_resume_rejects_terminal_checkpoint():
    state = HarnessState(messages=[], status=HarnessStatus.COMPLETED, stop_reason=StopReason.MODEL_COMPLETED, phase=HarnessPhase.TERMINAL)
    checkpoint = HarnessCheckpoint.create(state, "run-1", 0)
    with pytest.raises(HarnessResumeStateError):
        HarnessRunner(FakeChatModel(["done"])).resume_execution(HarnessResumeRequest(checkpoint))

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

def test_two_resumes_are_isolated_from_each_other_and_checkpoint():
    async def run():
        source_message = Message(role=MessageRole.USER, content="hello", metadata={"nested": {"value": 1}})
        state = HarnessState(messages=[source_message], model_turns=3, tool_calls_total=2,
            tool_argument_bytes_total=10, tool_result_bytes_total=20,
            started_tool_call_ids={"call-1", "call-2"},
            executed_tool_call_ids={"call-1", "call-2"},
            phase=HarnessPhase.BEFORE_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-isolated", 4)
        runner = HarnessRunner(FakeChatModel(["done"]))
        first = runner.resume_execution(HarnessResumeRequest(checkpoint))
        second = runner.resume_execution(HarnessResumeRequest(checkpoint))
        first.state.messages[0].metadata["nested"]["value"] = 99
        assert second.state.messages[0].metadata["nested"]["value"] == 1
        assert checkpoint.state.messages[0].metadata["nested"]["value"] == 1
        assert first.state.model_turns == second.state.model_turns == 3
        assert first.state.tool_calls_total == second.state.tool_calls_total == 2
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
        ], model_turns=1, tool_calls_total=1, started_tool_call_ids={"call-1"},
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

def test_resume_rejects_checkpoint_before_pending_cursor_advances():
    call = ToolCall(id="call-pending", name="echo", arguments={"text": "hi"})
    state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))],
        model_turns=1, phase=HarnessPhase.AFTER_MODEL)
    checkpoint = HarnessCheckpoint.create(state, "run-cursor", 0)
    execution = HarnessRunner(FakeChatModel(["done"])).resume_execution(
        HarnessResumeRequest(checkpoint, tools=(EchoTool().definition,))
    )
    with pytest.raises(HarnessResumeStateError, match="resume cursor"):
        execution.create_checkpoint(sequence=1)

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
            model_turns=1, status=HarnessStatus.RUNNING, phase=HarnessPhase.BEFORE_TOOL)
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

def test_resume_after_tool_executes_only_remaining_calls():
    async def run():
        first = ToolCall(id="call-1", name="echo", arguments={"text": "one"})
        second = ToolCall(id="call-2", name="echo", arguments={"text": "two"})
        state = HarnessState(messages=[
            Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(first, second)),
            Message(role=MessageRole.TOOL, name="echo", tool_call_id="call-1", content='{"ok":true}'),
        ], model_turns=1, tool_calls_total=1, started_tool_call_ids={"call-1"},
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
