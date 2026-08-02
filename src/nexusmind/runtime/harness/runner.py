from __future__ import annotations
from collections.abc import AsyncIterator
from nexusmind.models.base import ChatModel
from nexusmind.runtime.chat import AgentLoopLimits, _LegacyHarnessRuntime
from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.policy import ApprovalProvider, ToolApprovalSummarizer, ToolPolicy
from nexusmind.tools.executor import ToolExecutorProtocol

class HarnessRunner:
    """Bounded, provider-neutral execution boundary."""
    def __init__(self, model: ChatModel, tool_executor: ToolExecutorProtocol | None = None,
                 limits: AgentLoopLimits | None = None, tool_policy: ToolPolicy | None = None,
                 approval_provider: ApprovalProvider | None = None,
                 approval_summarizer: ToolApprovalSummarizer | None = None) -> None:
        self.limits = limits or AgentLoopLimits()
        self._runtime = _LegacyHarnessRuntime(model, tool_executor=tool_executor, limits=self.limits,
                                    tool_policy=tool_policy, approval_provider=approval_provider,
                                    approval_summarizer=approval_summarizer)

    async def stream(self, request: HarnessRequest) -> AsyncIterator[RuntimeEvent]:
        system = next((m.content for m in request.messages if m.role.value == "system"), None)
        user = next((m.content for m in reversed(request.messages) if m.role.value == "user"), "")
        async for event in self._runtime.stream_user_message(user or "", system_prompt=system, tools=list(request.tools)):
            yield event
