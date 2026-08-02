from __future__ import annotations
import asyncio
import json
from copy import deepcopy
from dataclasses import replace
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
        self._resume_limit_exceeded = False

    def create_checkpoint(self, run_id: str | None = None, sequence: int | None = None, boundary: CheckpointBoundary | None = None) -> HarnessCheckpoint:
        if self._resume_source is not None:
            if run_id is None:
                run_id = self._resume_source.run_id
            if sequence is None:
                raise HarnessResumeStateError("Resumed checkpoint requires a new sequence")
            if run_id != self._resume_source.run_id:
                raise HarnessResumeStateError("Resumed execution must keep the source run ID")
            if sequence <= self._resume_source.sequence:
                raise HarnessResumeStateError("Resumed checkpoint sequence must increase")
        elif run_id is None or sequence is None:
            raise ValueError("Checkpoint run_id and sequence are required")
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
        if self._resume_limit_exceeded:
            yield RuntimeEvent(RuntimeEventType.RUN_STARTED, metadata={"resumed": True, "checkpoint_id": self._resume_source.checkpoint_id, "checkpoint_sequence": self._resume_source.sequence, "checkpoint_boundary": self._resume_source.boundary.value})
            self.state.status = HarnessStatus.FAILED
            self.state.stop_reason = StopReason.LIMIT_EXCEEDED
            self.state.phase = HarnessPhase.TERMINAL
            self.stop_reason = StopReason.LIMIT_EXCEEDED
            yield RuntimeEvent(RuntimeEventType.RUN_FAILED, error="Agent loop limit exceeded", metadata={"stop_reason": StopReason.LIMIT_EXCEEDED.value})
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
        try:
            checkpoint.validate()
        except ValueError as exc:
            raise HarnessResumeStateError("Checkpoint is not valid for resume") from exc
        state = state_from_checkpoint(checkpoint) if checkpoint.state.phase is HarnessPhase.BEFORE_MODEL else None
        if checkpoint.state.phase in {HarnessPhase.AFTER_MODEL, HarnessPhase.BEFORE_TOOL, HarnessPhase.AFTER_TOOL}:
            messages = checkpoint.state.messages
            assistants = [message for message in messages if message.role.value == "assistant"]
            if not assistants:
                raise HarnessResumeStateError("Checkpoint has no resumable Assistant Tool Call batch")
            last = messages[-1]
            if checkpoint.state.phase is HarnessPhase.AFTER_MODEL and last is not assistants[-1]:
                raise HarnessResumeStateError("AFTER_MODEL checkpoint has trailing transcript entries")
            if checkpoint.state.phase is HarnessPhase.BEFORE_TOOL and last is not assistants[-1]:
                raise HarnessResumeStateError("BEFORE_TOOL checkpoint has trailing transcript entries")
            if checkpoint.state.phase is HarnessPhase.AFTER_TOOL:
                if last.role.value != "tool" or not last.tool_call_id:
                    raise HarnessResumeStateError("AFTER_TOOL checkpoint must end with a Tool result")
                if last.tool_call_id not in {call.id for call in assistants[-1].tool_calls}:
                    raise HarnessResumeStateError("AFTER_TOOL result does not match the Assistant Tool Call batch")
                if last.tool_call_id not in checkpoint.state.executed_tool_call_ids:
                    raise HarnessResumeStateError("AFTER_TOOL result is not recorded as executed")
            pending = tuple(call for call in assistants[-1].tool_calls if call.id not in checkpoint.state.executed_tool_call_ids)
            if pending:
                requested_tool_names = [tool.name for tool in request.tools]
                if len(set(requested_tool_names)) != len(requested_tool_names):
                    raise HarnessResumeCompatibilityError("Resume request contains duplicate tool definitions")
                missing_tools = {call.name for call in pending} - set(requested_tool_names)
                if missing_tools:
                    raise HarnessResumeCompatibilityError(
                        "Resume request is missing tools required by the checkpoint"
                    )
            if checkpoint.state.phase is HarnessPhase.AFTER_MODEL and not assistants[-1].tool_calls:
                state = HarnessState(messages=deepcopy(list(checkpoint.state.messages)), model_turns=checkpoint.state.model_turns, tool_calls_total=checkpoint.state.tool_calls_total, tool_argument_bytes_total=checkpoint.state.tool_argument_bytes_total, tool_result_bytes_total=checkpoint.state.tool_result_bytes_total, started_tool_call_ids=set(checkpoint.state.started_tool_call_ids), executed_tool_call_ids=set(checkpoint.state.executed_tool_call_ids), status=checkpoint.state.status, phase=checkpoint.state.phase)
                resume_runner = self
                limits = request.limits or self._default_limits
            elif not pending:
                if checkpoint.state.phase is not HarnessPhase.AFTER_TOOL:
                    raise HarnessResumeStateError("Checkpoint has no pending Tool Call")
                state = HarnessState(
                    messages=deepcopy(list(checkpoint.state.messages)),
                    model_turns=checkpoint.state.model_turns,
                    tool_calls_total=checkpoint.state.tool_calls_total,
                    tool_argument_bytes_total=checkpoint.state.tool_argument_bytes_total,
                    tool_result_bytes_total=checkpoint.state.tool_result_bytes_total,
                    started_tool_call_ids=set(checkpoint.state.started_tool_call_ids),
                    executed_tool_call_ids=set(checkpoint.state.executed_tool_call_ids),
                    status=checkpoint.state.status,
                    phase=HarnessPhase.BEFORE_MODEL,
                )
                resume_runner = self
            elif pending:
                state = HarnessState(messages=deepcopy(list(checkpoint.state.messages)), model_turns=checkpoint.state.model_turns, tool_calls_total=checkpoint.state.tool_calls_total, tool_argument_bytes_total=checkpoint.state.tool_argument_bytes_total, tool_result_bytes_total=checkpoint.state.tool_result_bytes_total, started_tool_call_ids=set(checkpoint.state.started_tool_call_ids), executed_tool_call_ids=set(checkpoint.state.executed_tool_call_ids), status=checkpoint.state.status, phase=checkpoint.state.phase)
                pending_argument_bytes = sum(len(json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for call in pending)
                state.tool_argument_bytes_total = max(0, state.tool_argument_bytes_total - pending_argument_bytes)
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
        needs_model = checkpoint.state.phase is HarnessPhase.BEFORE_MODEL or bool(
            checkpoint.state.phase in {HarnessPhase.BEFORE_TOOL, HarnessPhase.AFTER_TOOL}
            and state.phase is HarnessPhase.BEFORE_MODEL
        )
        if needs_model and state.model_turns >= limits.max_model_turns:
            execution_limit_exceeded = True
        else:
            execution_limit_exceeded = False
        harness_request = HarnessRequest(
            messages=tuple(state.messages),
            tools=tuple(deepcopy(request.tools)),
            limits=limits,
            metadata={
                **deepcopy(request.metadata),
                "resumed": True,
                "checkpoint_id": request.checkpoint.checkpoint_id,
                "checkpoint_sequence": request.checkpoint.sequence,
                "checkpoint_boundary": request.checkpoint.boundary.value,
            },
        )
        if checkpoint.state.phase is HarnessPhase.BEFORE_TOOL or (
            checkpoint.state.phase is HarnessPhase.AFTER_TOOL and bool(pending)
        ):
            harness_request = HarnessRequest(messages=harness_request.messages, tools=harness_request.tools, limits=harness_request.limits, metadata={**harness_request.metadata, "resume_existing_assistant": True})
        execution = HarnessExecution(resume_runner, harness_request)
        execution.state = state
        execution._resume_source = checkpoint
        execution._resume_limit_exceeded = execution_limit_exceeded
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
                if event.type is RuntimeEventType.RUN_STARTED and request.metadata.get("resumed"):
                    event = replace(
                        event,
                        metadata={
                            **event.metadata,
                            "resumed": True,
                            "checkpoint_id": request.metadata["checkpoint_id"],
                            "checkpoint_sequence": request.metadata["checkpoint_sequence"],
                            "checkpoint_boundary": request.metadata["checkpoint_boundary"],
                        },
                    )
                if event.type.value == "run_completed":
                    state.status = HarnessStatus.COMPLETED
                elif event.type.value == "run_failed":
                    state.status = HarnessStatus.FAILED
                yield event
        except asyncio.CancelledError:
            state.status = HarnessStatus.CANCELLED
            raise
