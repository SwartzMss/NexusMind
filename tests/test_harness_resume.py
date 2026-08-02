import asyncio
import pytest

from nexusmind.models.fake import FakeChatModel
from nexusmind.runtime.harness import (
    CheckpointBoundary, HarnessCheckpoint, HarnessPhase, HarnessRequest,
    HarnessResumeRequest, HarnessResumeStateError, HarnessRunner, HarnessState,
    HarnessStatus, StopReason,
)
from nexusmind.runtime.messages import Message, MessageRole

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
