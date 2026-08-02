"""Chat protocol adapter for the provider-neutral agent harness."""

from __future__ import annotations

from collections.abc import AsyncIterator

from nexusmind.models.base import ChatModel
from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.harness.runner_impl import ToolExecutionCancelled
from nexusmind.runtime.harness.runner import HarnessRunner
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.runtime.policy import ApprovalProvider, ToolApprovalSummarizer, ToolPolicy
from nexusmind.tools.contracts import ToolDefinition
from nexusmind.tools.executor import ToolExecutorProtocol

AgentLoopLimits = HarnessLimits


class ChatRuntime:
    """Translate chat input into a HarnessRequest and forward its events."""

    def __init__(
        self,
        model: ChatModel,
        tool_executor: ToolExecutorProtocol | None = None,
        limits: HarnessLimits | None = None,
        tool_policy: ToolPolicy | None = None,
        approval_provider: ApprovalProvider | None = None,
        approval_summarizer: ToolApprovalSummarizer | None = None,
    ) -> None:
        self._harness = HarnessRunner(
            model,
            tool_executor=tool_executor,
            limits=limits,
            tool_policy=tool_policy,
            approval_provider=approval_provider,
            approval_summarizer=approval_summarizer,
        )

    async def stream_user_message(
        self,
        content: str,
        *,
        system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(Message(role=MessageRole.USER, content=content))
        request = HarnessRequest(
            messages=tuple(messages),
            tools=tuple(tools or ()),
            limits=self._harness.limits,
        )
        async for event in self._harness.stream(request):
            yield event
