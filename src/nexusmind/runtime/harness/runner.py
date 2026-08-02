from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from uuid import uuid4
from collections.abc import AsyncIterator
from nexusmind.models.base import ChatModel
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.harness.runner_impl import _LegacyHarnessRuntime
from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.harness.state import HarnessState, HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.harness.checkpoint import CheckpointBoundary, HarnessCheckpoint, HarnessStateSnapshot
from nexusmind.runtime.policy import ApprovalProvider, ToolApprovalSummarizer, ToolPolicy
from nexusmind.tools.executor import ToolExecutorProtocol

class HarnessExecution:
    def __init__(self, runner: "HarnessRunner", request: HarnessRequest) -> None:
        self._runner = runner
        self._request = request
        self.state = HarnessState(messages=list(request.messages))
        self.stop_reason: StopReason | None = None

    def create_checkpoint(self, run_id: str, sequence: int, boundary: CheckpointBoundary) -> HarnessCheckpoint:
        return HarnessCheckpoint.create(
            state=self.state,
            run_id=run_id,
            sequence=sequence,
            boundary=boundary,
            stop_reason=self.stop_reason,
        )

    async def stream(self) -> AsyncIterator[RuntimeEvent]:
        try:
          async for event in self._runner._stream(self._request, self.state):
            if event.type.value == "run_completed":
                self.stop_reason = StopReason.MODEL_COMPLETED
                self.state.stop_reason = self.stop_reason
            elif event.type.value == "model_failed" and self.stop_reason is None:
                self.stop_reason = StopReason.MODEL_FAILED
                self.state.stop_reason = self.stop_reason
            elif event.type.value == "run_failed" and self.stop_reason is None:
                reason = event.metadata.get("stop_reason")
                if reason in StopReason._value2member_map_:
                    self.stop_reason = StopReason(reason)
                elif event.error == "Agent loop limit exceeded":
                    self.stop_reason = StopReason.LIMIT_EXCEEDED
                elif event.metadata.get("tool_execution_started"):
                    self.stop_reason = StopReason.TOOL_FAILED
                else:
                    self.stop_reason = StopReason.RUNTIME_ERROR
                self.state.stop_reason = self.stop_reason
                self.state.status = HarnessStatus.FAILED
            yield event
        except asyncio.CancelledError:
            self.state.status = HarnessStatus.CANCELLED
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
            ):
                if event.type.value == "run_completed":
                    state.status = HarnessStatus.COMPLETED
                elif event.type.value == "run_failed":
                    state.status = HarnessStatus.FAILED
                yield event
        except asyncio.CancelledError:
            state.status = HarnessStatus.CANCELLED
            raise
