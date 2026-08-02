from __future__ import annotations
import asyncio
from collections.abc import AsyncIterator
from nexusmind.models.base import ChatModel
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.harness.runner_impl import _LegacyHarnessRuntime
from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.harness.state import HarnessState, HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.policy import ApprovalProvider, ToolApprovalSummarizer, ToolPolicy
from nexusmind.tools.executor import ToolExecutorProtocol

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
        limits = request.limits or self._default_limits
        runtime = _LegacyHarnessRuntime(
            self._model,
            tool_executor=self._tool_executor,
            limits=limits,
            tool_policy=self._tool_policy,
            approval_provider=self._approval_provider,
            approval_summarizer=self._approval_summarizer,
        )
        self.state = HarnessState(messages=list(request.messages))
        self.stop_reason = None
        user = next((m.content for m in reversed(request.messages) if m.role.value == "user"), "")
        try:
            async for event in runtime.stream_user_message(
                user or "",
                tools=list(request.tools),
                _initial_messages=list(request.messages),
            ):
                if event.type.value == "model_started":
                    self.state.model_turns += 1
                elif event.type.value == "tool_call_completed":
                    self.state.tool_calls_total += 1
                elif event.type.value == "tool_result":
                    self.state.executed_tool_call_ids.add(event.tool_result.call_id)
                elif event.type.value == "run_completed":
                    self.state.status = HarnessStatus.COMPLETED
                    self.stop_reason = StopReason.MODEL_COMPLETED
                elif event.type.value == "run_failed":
                    self.state.status = HarnessStatus.FAILED
                    self.stop_reason = StopReason.RUNTIME_ERROR
                elif event.type.value == "model_failed":
                    self.stop_reason = StopReason.MODEL_FAILED
                yield event
        except asyncio.CancelledError:
            self.state.status = HarnessStatus.CANCELLED
            self.stop_reason = StopReason.CANCELLED
            raise
