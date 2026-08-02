import asyncio
import pytest

from nexusmind.models.fake import FakeChatModel
from nexusmind.runtime.harness import (
    CheckpointBoundary, HarnessCheckpoint, HarnessPhase, HarnessRequest,
    HarnessResumeRequest, HarnessResumeStateError, HarnessRunner, HarnessState,
    HarnessStatus, StopReason,
)
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolCall, ToolDefinition, ToolResult

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

def test_resume_after_model_tool_call_executes_without_model_replay():
    class Executor:
        def __init__(self): self.calls = []
        def definition(self, name): return ToolDefinition(name=name)
        def result_requirements(self, call):
            from nexusmind.tools.contracts import ToolResultRequirements
            return ToolResultRequirements(1, 1, 1)
        async def execute(self, call):
            return await self.execute_with_result_budget(call, result_budget=None)
        async def execute_with_result_budget(self, call, *, result_budget):
            self.calls.append(call.id)
            return ToolResult(call_id=call.id, name=call.name, output={"ok": True})
    async def run():
        call = ToolCall(id="call-1", name="echo", arguments={"text": "hi"})
        state = HarnessState(messages=[Message(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,))], model_turns=1, status=HarnessStatus.RUNNING, phase=HarnessPhase.AFTER_MODEL)
        checkpoint = HarnessCheckpoint.create(state, "run-tool", 0)
        executor = Executor()
        execution = HarnessRunner(FakeChatModel(["done"]), tool_executor=executor).resume_execution(HarnessResumeRequest(checkpoint, tools=(ToolDefinition(name="echo"),)))
        events = [event async for event in execution.stream()]
        assert executor.calls == ["call-1"], [(event.type, event.error) for event in events]
        assert execution.state.executed_tool_call_ids == {"call-1"}
    asyncio.run(run())
