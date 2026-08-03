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
from nexusmind.runtime.harness.runner_impl import _LegacyHarnessRuntime, _snapshot_tool_call
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.harness.state import HarnessPhase, HarnessState, HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.harness.resume import HarnessResumeCompatibilityError, HarnessResumeRequest, HarnessResumeStateError, state_from_checkpoint
from nexusmind.runtime.harness.checkpoint import CheckpointBoundary, HarnessCheckpoint, HarnessStateSnapshot
from nexusmind.runtime.policy import ApprovalProvider, ToolApprovalSummarizer, ToolPolicy
from nexusmind.tools.executor import ToolExecutorProtocol
from nexusmind.tools.contracts import ToolCall

_RESERVED_RESUME_METADATA = {
    "resume_tool_batch", "resume_existing_assistant", "resume_internal",
    "resumed", "checkpoint_id", "checkpoint_sequence", "checkpoint_boundary",
}

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
        internal = {"resume_internal": True}
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED, metadata=internal)
        for call in self._calls:
            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=deepcopy(call), metadata=internal)
        yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls", metadata=internal)

def _validate_resume_batches(messages) -> None:
    batches = []
    for index, message in enumerate(messages):
        if message.role.value != "assistant" or not message.tool_calls:
            continue
        calls = {call.id: call for call in message.tool_calls}
        if len(calls) != len(message.tool_calls):
            raise HarnessResumeStateError("Assistant Tool Call batch contains duplicate IDs")
        results = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].role.value == "tool":
            result = messages[cursor]
            if not result.tool_call_id or result.tool_call_id not in calls:
                raise HarnessResumeStateError("Tool result does not match its Assistant Tool Call batch")
            if result.name != calls[result.tool_call_id].name:
                raise HarnessResumeStateError("Tool result name does not match its Tool Call")
            results.append(result.tool_call_id)
            cursor += 1
        if len(set(results)) != len(results):
            raise HarnessResumeStateError("Tool batch contains duplicate Tool results")
        ordered_call_ids = list(calls)
        if results != ordered_call_ids[:len(results)]:
            raise HarnessResumeStateError("Tool results are not an ordered prefix of the Tool Call batch")
        batches.append((calls, results, cursor == len(messages)))
    for calls, results, reaches_end in batches:
        if not reaches_end and results != list(calls):
            raise HarnessResumeStateError("A previous Tool Call batch is incomplete")

def _validate_run_consumption(checkpoint: HarnessCheckpoint) -> None:
    messages = checkpoint.state.messages
    user_indexes = [index for index, message in enumerate(messages) if message.role.value == "user"]
    run_messages = messages[user_indexes[-1] + 1:] if user_indexes else messages
    _validate_resume_batches(run_messages)
    if checkpoint.state.phase is HarnessPhase.BEFORE_MODEL and run_messages:
        last = run_messages[-1]
        if last.role.value == "assistant":
            raise HarnessResumeStateError("BEFORE_MODEL cannot follow an Assistant message")
        if last.role.value == "tool":
            assistant_index = next(
                (index for index in range(len(run_messages) - 1, -1, -1)
                 if run_messages[index].role.value == "assistant" and run_messages[index].tool_calls),
                None,
            )
            if assistant_index is None:
                raise HarnessResumeStateError("BEFORE_MODEL Tool result has no Tool Call batch")
            call_ids = [call.id for call in run_messages[assistant_index].tool_calls]
            result_ids = [
                message.tool_call_id for message in run_messages[assistant_index + 1:]
                if message.role.value == "tool"
            ]
            if result_ids != call_ids:
                raise HarnessResumeStateError("BEFORE_MODEL requires a completed Tool batch")
    calls = [call for message in run_messages if message.role.value == "assistant" for call in message.tool_calls]
    results = [message for message in run_messages if message.role.value == "tool"]
    call_ids = [call.id for call in calls]
    result_ids = [message.tool_call_id for message in results]
    if len(set(call_ids)) != len(call_ids):
        raise HarnessResumeStateError("Checkpoint transcript contains duplicate Tool Call IDs")
    if any(not result_id or result_id not in set(call_ids) for result_id in result_ids):
        raise HarnessResumeStateError("Checkpoint transcript contains an unknown Tool result")
    if set(checkpoint.state.executed_tool_call_ids) != set(result_ids) or checkpoint.state.tool_calls_total != len(results):
        raise HarnessResumeStateError("Tool counters do not match the current Run transcript")
    if not set(checkpoint.state.started_tool_call_ids).issubset(set(call_ids)):
        raise HarnessResumeStateError("Started Tool Call IDs do not match the current Run transcript")
    accounted_calls = calls
    if checkpoint.state.phase is HarnessPhase.AFTER_MODEL:
        assistants = [message for message in run_messages if message.role.value == "assistant"]
        if assistants and assistants[-1].tool_calls:
            accounted_calls = calls[:-len(assistants[-1].tool_calls)]
    required_argument_bytes = sum(len(json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) for call in accounted_calls)
    required_result_bytes = sum(len((message.content or "").encode("utf-8")) for message in results)
    required_model_turns = sum(1 for message in run_messages if message.role.value == "assistant")
    if checkpoint.state.model_turns < required_model_turns:
        raise HarnessResumeStateError("Model turn counter is below transcript consumption")
    if checkpoint.state.tool_argument_bytes_total < required_argument_bytes:
        raise HarnessResumeStateError("Tool argument byte counter is below transcript consumption")
    if checkpoint.state.tool_result_bytes_total < required_result_bytes:
        raise HarnessResumeStateError("Tool result byte counter is below transcript consumption")

class HarnessExecution:
    def __init__(self, runner: "HarnessRunner", request: HarnessRequest) -> None:
        self._runner = runner
        self._request = request
        self.state = HarnessState(messages=list(request.messages))
        self.stop_reason: StopReason | None = None
        self._resume_complete = False
        self._resume_source = None
        self._resume_limit_exceeded = False
        self._resume_cursor_pending = False
        self._stream_started = False
        self._resume_tool_batch = False
        self._skip_assistant_once = False
        self._last_checkpoint_sequence: int | None = None

    def create_checkpoint(self, run_id: str | None = None, sequence: int | None = None, boundary: CheckpointBoundary | None = None) -> HarnessCheckpoint:
        if self._resume_cursor_pending:
            raise HarnessResumeStateError("Cannot checkpoint before the resume cursor advances")
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
        if self._last_checkpoint_sequence is not None and sequence <= self._last_checkpoint_sequence:
            raise HarnessResumeStateError("Checkpoint sequence must increase monotonically")
        checkpoint = HarnessCheckpoint.create(
            state=self.state,
            run_id=run_id,
            sequence=sequence,
            boundary=boundary,
            stop_reason=self.stop_reason,
        )
        self._last_checkpoint_sequence = sequence
        return checkpoint

    async def stream(self) -> AsyncIterator[RuntimeEvent]:
        if self._stream_started:
            raise RuntimeError("HarnessExecution can only be streamed once")
        self._stream_started = True
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
          async for event in self._runner._stream(
              self._request,
              self.state,
              resume_tool_batch=self._resume_tool_batch,
              skip_assistant_once=self._skip_assistant_once,
          ):
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
            if self._resume_cursor_pending and event.type is RuntimeEventType.TOOL_RESULT:
                self._resume_cursor_pending = False
            yield event
          self._resume_cursor_pending = False
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
        if _RESERVED_RESUME_METADATA & set(request.metadata):
            raise ValueError("Harness request metadata contains reserved internal fields")
        return HarnessExecution(self, request)

    def resume_execution(self, request: HarnessResumeRequest) -> HarnessExecution:
        if _RESERVED_RESUME_METADATA & set(request.metadata):
            raise HarnessResumeStateError("Resume metadata contains reserved internal fields")
        checkpoint = request.checkpoint
        try:
            checkpoint.validate()
        except ValueError as exc:
            raise HarnessResumeStateError("Checkpoint is not valid for resume") from exc
        _validate_run_consumption(checkpoint)
        pending: tuple[ToolCall, ...] = ()
        state = state_from_checkpoint(checkpoint) if checkpoint.state.phase is HarnessPhase.BEFORE_MODEL else None
        if checkpoint.state.phase in {HarnessPhase.AFTER_MODEL, HarnessPhase.BEFORE_TOOL, HarnessPhase.AFTER_TOOL}:
            messages = checkpoint.state.messages
            user_indexes = [index for index, message in enumerate(messages) if message.role.value == "user"]
            run_messages = messages[user_indexes[-1] + 1:] if user_indexes else messages
            _validate_resume_batches(run_messages)
            assistants = [message for message in messages if message.role.value == "assistant"]
            if not assistants:
                raise HarnessResumeStateError("Checkpoint has no resumable Assistant Tool Call batch")
            last = messages[-1]
            if checkpoint.state.phase is HarnessPhase.AFTER_MODEL and last is not assistants[-1]:
                raise HarnessResumeStateError("AFTER_MODEL checkpoint has trailing transcript entries")
            if checkpoint.state.phase is HarnessPhase.AFTER_TOOL:
                if last.role.value != "tool" or not last.tool_call_id:
                    raise HarnessResumeStateError("AFTER_TOOL checkpoint must end with a Tool result")
                if last.tool_call_id not in {call.id for call in assistants[-1].tool_calls}:
                    raise HarnessResumeStateError("AFTER_TOOL result does not match the Assistant Tool Call batch")
                if last.tool_call_id not in checkpoint.state.executed_tool_call_ids:
                    raise HarnessResumeStateError("AFTER_TOOL result is not recorded as executed")
            pending = tuple(call for call in assistants[-1].tool_calls if call.id not in checkpoint.state.executed_tool_call_ids)
            batch_calls = {call.id: call for call in assistants[-1].tool_calls}
            if len(batch_calls) != len(assistants[-1].tool_calls):
                raise HarnessResumeStateError("Assistant Tool Call batch contains duplicate IDs")
            assistant_index = next(
                index for index in range(len(messages) - 1, -1, -1)
                if messages[index] is assistants[-1]
            )
            batch_results = messages[assistant_index + 1:]
            result_ids: list[str] = []
            for message in batch_results:
                if message.role.value != "tool":
                    raise HarnessResumeStateError("Tool batch has an invalid trailing transcript entry")
                if not message.tool_call_id or message.tool_call_id not in batch_calls:
                    raise HarnessResumeStateError("Tool result does not match the current Assistant Tool Call batch")
                if message.name != batch_calls[message.tool_call_id].name:
                    raise HarnessResumeStateError("Tool result name does not match its Tool Call")
                result_ids.append(message.tool_call_id)
            if len(set(result_ids)) != len(result_ids):
                raise HarnessResumeStateError("Tool batch contains duplicate Tool results")
            ordered_call_ids = [call.id for call in assistants[-1].tool_calls]
            if result_ids != ordered_call_ids[:len(result_ids)]:
                raise HarnessResumeStateError("Tool results are not an ordered prefix of the Tool Call batch")
            executed_batch_ids = set(checkpoint.state.executed_tool_call_ids) & set(batch_calls)
            if set(result_ids) != executed_batch_ids:
                raise HarnessResumeStateError("Tool results do not match executed Tool Call IDs")
            transcript_call_ids = [
                call.id
                for message in run_messages
                if message.role.value == "assistant"
                for call in message.tool_calls
            ]
            transcript_result_ids = [
                message.tool_call_id
                for message in run_messages
                if message.role.value == "tool" and message.tool_call_id
            ]
            if len(set(transcript_call_ids)) != len(transcript_call_ids):
                raise HarnessResumeStateError("Checkpoint transcript contains duplicate Tool Call IDs")
            if not set(checkpoint.state.started_tool_call_ids).issubset(set(transcript_call_ids)):
                raise HarnessResumeStateError("Started Tool Call IDs do not match the transcript")
            if len(transcript_result_ids) != checkpoint.state.tool_calls_total:
                raise HarnessResumeStateError("Tool call count does not match the transcript")
            if set(checkpoint.state.executed_tool_call_ids) != set(transcript_result_ids):
                raise HarnessResumeStateError("Executed Tool Call IDs do not match the transcript")
            accounted_calls = transcript_call_ids
            if checkpoint.state.phase is HarnessPhase.AFTER_MODEL:
                accounted_calls = transcript_call_ids[:-len(assistants[-1].tool_calls)] if assistants[-1].tool_calls else transcript_call_ids
            calls_by_id = {
                call.id: call
                for message in run_messages if message.role.value == "assistant"
                for call in message.tool_calls
            }
            required_argument_bytes = sum(
                len(json.dumps(calls_by_id[call_id].arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                for call_id in accounted_calls
            )
            required_result_bytes = sum(
                len((message.content or "").encode("utf-8"))
                for message in run_messages if message.role.value == "tool"
            )
            required_model_turns = sum(1 for message in run_messages if message.role.value == "assistant")
            if checkpoint.state.model_turns < required_model_turns:
                raise HarnessResumeStateError("Model turn counter is below transcript consumption")
            if checkpoint.state.tool_argument_bytes_total < required_argument_bytes:
                raise HarnessResumeStateError("Tool argument byte counter is below transcript consumption")
            if checkpoint.state.tool_result_bytes_total < required_result_bytes:
                raise HarnessResumeStateError("Tool result byte counter is below transcript consumption")
            if pending:
                requested_tool_names = [tool.name for tool in request.tools]
                if len(set(requested_tool_names)) != len(requested_tool_names):
                    raise HarnessResumeCompatibilityError("Resume request contains duplicate tool definitions")
                missing_tools = {call.name for call in pending} - set(requested_tool_names)
                if missing_tools:
                    raise HarnessResumeCompatibilityError(
                        "Resume request is missing tools required by the checkpoint"
                    )
                if self._tool_executor is None:
                    raise HarnessResumeCompatibilityError("Resume requires a Tool executor")
                requested_definitions = {tool.name: tool for tool in request.tools}
                for call in pending:
                    try:
                        _snapshot_tool_call(
                            call,
                            max_bytes_per_call=(request.limits or self._default_limits).max_tool_arguments_bytes_per_call,
                            remaining_total_bytes=(request.limits or self._default_limits).max_tool_arguments_bytes_total,
                            max_nodes=(request.limits or self._default_limits).max_json_nodes_per_payload,
                            max_depth=(request.limits or self._default_limits).max_json_depth,
                        )
                    except Exception as exc:
                        raise HarnessResumeStateError("Pending Tool Call is not valid for resume") from exc
                    if self._tool_executor.definition(call.name) != requested_definitions[call.name]:
                        raise HarnessResumeCompatibilityError("Executor Tool definition does not match the Resume request")
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
                    phase=checkpoint.state.phase,
                )
                resume_runner = self
            elif pending:
                state = HarnessState(messages=deepcopy(list(checkpoint.state.messages)), model_turns=checkpoint.state.model_turns, tool_calls_total=checkpoint.state.tool_calls_total, tool_argument_bytes_total=checkpoint.state.tool_argument_bytes_total, tool_result_bytes_total=checkpoint.state.tool_result_bytes_total, started_tool_call_ids=set(checkpoint.state.started_tool_call_ids), executed_tool_call_ids=set(checkpoint.state.executed_tool_call_ids), status=checkpoint.state.status, phase=checkpoint.state.phase)
                if checkpoint.state.phase is HarnessPhase.AFTER_MODEL:
                    pending_argument_bytes = sum(
                        len(json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                        for call in pending
                    )
                    state.tool_argument_bytes_total += pending_argument_bytes
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
        needs_model = checkpoint.state.phase is HarnessPhase.BEFORE_MODEL or (
            checkpoint.state.phase is HarnessPhase.AFTER_TOOL and not pending
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
        execution = HarnessExecution(resume_runner, harness_request)
        execution.state = state
        execution._resume_source = checkpoint
        execution._last_checkpoint_sequence = checkpoint.sequence
        execution._resume_limit_exceeded = execution_limit_exceeded
        execution._resume_cursor_pending = bool(pending)
        execution._resume_tool_batch = bool(pending)
        execution._skip_assistant_once = bool(pending)
        execution._resume_complete = checkpoint.state.phase is HarnessPhase.AFTER_MODEL and not assistants[-1].tool_calls if checkpoint.state.phase is HarnessPhase.AFTER_MODEL else False
        return execution

    async def _stream(self, request: HarnessRequest, state: HarnessState, *, resume_tool_batch: bool = False, skip_assistant_once: bool = False) -> AsyncIterator[RuntimeEvent]:
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
                _skip_assistant_message=skip_assistant_once,
                _resume_tool_batch=resume_tool_batch,
            ):
                if event.metadata.get("resume_internal"):
                    continue
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
