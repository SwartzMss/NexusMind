from __future__ import annotations
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4
from collections.abc import AsyncIterator
from nexusmind.models.base import ChatModel
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.harness.runner_impl import _LegacyHarnessRuntime
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.harness.state import HarnessPhase, HarnessState, HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.harness.resume import HarnessResumeCompatibilityError, HarnessResumeRequest, HarnessResumeStateError, state_from_checkpoint
from nexusmind.runtime.harness.checkpoint import CheckpointBoundary, HarnessCheckpoint, HarnessStateSnapshot
from nexusmind.runtime.policy import ApprovalProvider, ToolApprovalSummarizer, ToolPolicy
from nexusmind.tools.executor import ToolExecutorProtocol
from nexusmind.tools.contracts import ToolCall

class _ResumeToolBatchModel:
    def __init__(self, original: ChatModel, calls: tuple[ToolCall, ...]) -> None:
        self._original = original
        self._calls = calls
        self._first = True

    async def stream(self, messages, tools=None):
        if not self._first:
            async for event in self._original.stream(messages, tools=tools):
                yield event
            return
        self._first = False
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
        for call in self._calls:
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=deepcopy(call))
        yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls")

class HarnessExecution:
    def __init__(self, runner: "HarnessRunner", request: HarnessRequest) -> None:
        self._runner = runner
        self._request = request
        self.state = HarnessState(messages=list(request.messages))
        self.stop_reason: StopReason | None = None
        self._resume_complete = False
        self._resume_source = None

    def create_checkpoint(self, run_id: str, sequence: int, boundary: CheckpointBoundary | None = None) -> HarnessCheckpoint:
        if self._resume_source is not None:
            if run_id != self._resume_source.run_id:
                raise HarnessResumeStateError("Resumed execution must keep the source run ID")
            if sequence <= self._resume_source.sequence:
                raise HarnessResumeStateError("Resumed checkpoint sequence must increase")
        return HarnessCheckpoint.create(
            state=self.state,
            run_id=run_id,
            sequence=sequence,
            boundary=boundary,
            stop_reason=self.stop_reason,
        )

    async def stream(self) -> AsyncIterator[RuntimeEvent]:
        if self._resume_complete:
            yield RuntimeEvent(RuntimeEventType.RUN_STARTED, metadata={"resumed": True, "checkpoint_id": self._resume_source.checkpoint_id, "checkpoint_sequence": self._resume_source.sequence, "checkpoint_boundary": self._resume_source.boundary.value})
            self.state.status = HarnessStatus.COMPLETED
            self.state.stop_reason = StopReason.MODEL_COMPLETED
            self.state.phase = HarnessPhase.TERMINAL
            self.stop_reason = StopReason.MODEL_COMPLETED
            yield RuntimeEvent(RuntimeEventType.RUN_COMPLETED)
            return
        try:
          async for event in self._runner._stream(self._request, self.state):
            if event.type.value == "run_completed":
                self.stop_reason = StopReason.MODEL_COMPLETED
                self.state.stop_reason = self.stop_reason
            elif event.type.value == "model_failed" and self.stop_reason is None:
                self.stop_reason = StopReason.MODEL_FAILED
                self.state.stop_reason = self.stop_reason
            elif event.type.value == "run_failed":
                if self.stop_reason is None:
                    reason = event.metadata.get("stop_reason")
                    if reason in StopReason._value2member_map_:
                        self.stop_reason = StopReason(reason)
                    elif event.metadata.get("tool_execution_started"):
                        self.stop_reason = StopReason.TOOL_FAILED
                    elif event.error == "Agent loop limit exceeded":
                        self.stop_reason = StopReason.LIMIT_EXCEEDED
                    else:
                        self.stop_reason = StopReason.RUNTIME_ERROR
                self.state.stop_reason = self.stop_reason
                self.state.status = HarnessStatus.FAILED
                self.state.phase = HarnessPhase.TERMINAL
            yield event
        except asyncio.CancelledError:
            self.state.status = HarnessStatus.CANCELLED
            self.state.phase = HarnessPhase.TERMINAL
            self.stop_reason = StopReason.CANCELLED
            self.state.stop_reason = self.stop_reason
            raise


class HarnessRunner:
    """Bounded, provider-neutral execution boundary."""
    def __init__(self, model: ChatModel, tool_executor: ToolExecutorProtocol | None = None,
                 limits: HarnessLimits | None = None, tool_policy: ToolPolicy | None = None,
                 approval_provider: ApprovalProvider | None = None,
                 approval_summarizer: ToolApprovalSummarizer | None = None) -> None:
        self._model = model
        self._tool_executor = tool_executor
        self._default_limits = limits or HarnessLimits()
        self._tool_policy = tool_policy
        self._approval_provider = approval_provider
        self._approval_summarizer = approval_summarizer
        self.state: HarnessState | None = None
        self.stop_reason: StopReason | None = None

    @property
    def limits(self) -> HarnessLimits:
        return self._default_limits

    async def stream(self, request: HarnessRequest) -> AsyncIterator[RuntimeEvent]:
        execution = self.create_execution(request)
        async for event in execution.stream():
            yield event
        # Compatibility snapshot for callers that used the old runner fields.
        self.state = execution.state
        self.stop_reason = execution.stop_reason

    def create_execution(self, request: HarnessRequest) -> HarnessExecution:
        """Create isolated mutable state for one run; executions may run concurrently."""
        return HarnessExecution(self, request)

    def resume_execution(self, request: HarnessResumeRequest) -> HarnessExecution:
        checkpoint = request.checkpoint
        state = state_from_checkpoint(checkpoint) if checkpoint.state.phase is HarnessPhase.BEFORE_MODEL else None
        if checkpoint.state.phase in {HarnessPhase.AFTER_MODEL, HarnessPhase.BEFORE_TOOL, HarnessPhase.AFTER_TOOL}:
            assistants = [message for message in checkpoint.state.messages if message.role.value == "assistant"]
            if not assistants:
                raise HarnessResumeStateError("Checkpoint has no resumable Assistant Tool Call batch")
            if checkpoint.state.phase is HarnessPhase.AFTER_MODEL and assistants[-1] is not checkpoint.state.messages[-1]:
                raise HarnessResumeStateError("AFTER_MODEL checkpoint has trailing transcript entries")
            pending = tuple(call for call in assistants[-1].tool_calls if call.id not in checkpoint.state.executed_tool_call_ids)
            if checkpoint.state.phase is HarnessPhase.AFTER_MODEL and not assistants[-1].tool_calls:
                state = HarnessState(messages=deepcopy(list(checkpoint.state.messages)), model_turns=checkpoint.state.model_turns, tool_calls_total=checkpoint.state.tool_calls_total, tool_argument_bytes_total=checkpoint.state.tool_argument_bytes_total, tool_result_bytes_total=checkpoint.state.tool_result_bytes_total, started_tool_call_ids=set(checkpoint.state.started_tool_call_ids), executed_tool_call_ids=set(checkpoint.state.executed_tool_call_ids), status=checkpoint.state.status, phase=checkpoint.state.phase)
                resume_runner = self
                limits = request.limits or self._default_limits
            elif not pending:
                raise HarnessResumeStateError("Checkpoint has no pending Tool Call")
            elif pending:
                state = HarnessState(messages=deepcopy(list(checkpoint.state.messages)), model_turns=checkpoint.state.model_turns, tool_calls_total=checkpoint.state.tool_calls_total, tool_argument_bytes_total=checkpoint.state.tool_argument_bytes_total, tool_result_bytes_total=checkpoint.state.tool_result_bytes_total, started_tool_call_ids=set(checkpoint.state.started_tool_call_ids), executed_tool_call_ids=set(checkpoint.state.executed_tool_call_ids), status=checkpoint.state.status, phase=checkpoint.state.phase)
                state.messages.pop()
                state.phase = HarnessPhase.BEFORE_MODEL
                state.model_turns = max(0, state.model_turns - 1)
                if checkpoint.state.phase is not HarnessPhase.AFTER_MODEL:
                    state.messages.append(deepcopy(assistants[-1]))
                resume_runner = HarnessRunner(_ResumeToolBatchModel(self._model, pending), self._tool_executor, limits=self._default_limits, tool_policy=self._tool_policy, approval_provider=self._approval_provider, approval_summarizer=self._approval_summarizer)
            else:
                resume_runner = self
            limits = request.limits or self._default_limits
        else:
            resume_runner = self
        if state is None:
            raise HarnessResumeStateError("Checkpoint phase is not safely resumable")
        limits = request.limits or self._default_limits
        if state.model_turns > limits.max_model_turns or state.tool_calls_total > limits.max_tool_calls_total or state.tool_argument_bytes_total > limits.max_tool_arguments_bytes_total or state.tool_result_bytes_total > limits.max_tool_result_bytes_total:
            raise HarnessResumeCompatibilityError("Checkpoint consumption exceeds selected limits")
        harness_request = HarnessRequest(
            messages=tuple(state.messages),
            tools=tuple(deepcopy(request.tools)),
            limits=limits,
            metadata={"resumed": True, "checkpoint_id": request.checkpoint.checkpoint_id, "checkpoint_sequence": request.checkpoint.sequence},
        )
        if checkpoint.state.phase in {HarnessPhase.BEFORE_TOOL, HarnessPhase.AFTER_TOOL}:
            harness_request = HarnessRequest(messages=harness_request.messages, tools=harness_request.tools, limits=harness_request.limits, metadata={**harness_request.metadata, "resume_existing_assistant": True})
        execution = HarnessExecution(resume_runner, harness_request)
        execution.state = state
        execution._resume_source = checkpoint
        execution._resume_complete = checkpoint.state.phase is HarnessPhase.AFTER_MODEL and not assistants[-1].tool_calls if checkpoint.state.phase is HarnessPhase.AFTER_MODEL else False
        return execution

    async def _stream(self, request: HarnessRequest, state: HarnessState) -> AsyncIterator[RuntimeEvent]:
        limits = request.limits or self._default_limits
        runtime = _LegacyHarnessRuntime(
            self._model,
            tool_executor=self._tool_executor,
            limits=limits,
            tool_policy=self._tool_policy,
            approval_provider=self._approval_provider,
            approval_summarizer=self._approval_summarizer,
        )
        user = next((m.content for m in reversed(request.messages) if m.role.value == "user"), "")
        try:
            async for event in runtime.stream_user_message(
                user or "",
                tools=list(request.tools),
                _initial_messages=list(request.messages),
                _state=state,
                _skip_assistant_message=bool(request.metadata.get("resume_existing_assistant")),
            ):
                if event.type.value == "run_completed":
                    state.status = HarnessStatus.COMPLETED
                elif event.type.value == "run_failed":
                    state.status = HarnessStatus.FAILED
                yield event
        except asyncio.CancelledError:
            state.status = HarnessStatus.CANCELLED
            raise
