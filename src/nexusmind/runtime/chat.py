"""Chat protocol adapter for the provider-neutral agent harness."""

from __future__ import annotations

from collections.abc import AsyncIterator

from nexusmind.models.base import ChatModel
from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.harness.runner_impl import ToolExecutionCancelled
from nexusmind.runtime.harness.runner import HarnessRunner
from nexusmind.runtime.harness.checkpoint_store import CheckpointStore
from nexusmind.runtime.harness.checkpointing import CheckpointCoordinator
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
        checkpoint_store: CheckpointStore | None = None,
        checkpoint_run_id: str | None = None,
        checkpoint_sequence: int | None = None,
        save_terminal_checkpoint: bool = True,
    ) -> None:
        self._harness = HarnessRunner(
            model,
            tool_executor=tool_executor,
            limits=limits,
            tool_policy=tool_policy,
            approval_provider=approval_provider,
            approval_summarizer=approval_summarizer,
        )
        if checkpoint_store is not None and not checkpoint_run_id:
            raise ValueError("checkpoint_run_id is required when automatic checkpointing is enabled")
        if checkpoint_store is None and checkpoint_run_id is not None:
            raise ValueError("checkpoint_store is required when checkpoint_run_id is provided")
        self._checkpoint_store = checkpoint_store
        self._checkpoint_run_id = checkpoint_run_id
        self._checkpoint_sequence = checkpoint_sequence
        self._save_terminal_checkpoint = save_terminal_checkpoint

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
        execution = self._harness.create_execution(request)
        stream = execution.stream()
        if self._checkpoint_store is not None:
            stream = CheckpointCoordinator(
                execution,
                self._checkpoint_store,
                run_id=self._checkpoint_run_id,
                start_sequence=self._checkpoint_sequence,
                save_terminal=self._save_terminal_checkpoint,
            ).stream()
        try:
            async for event in stream:
                yield event
        finally:
            # Preserve the compatibility snapshot exposed by HarnessRunner.
            self._harness.state = execution.state
            self._harness.stop_reason = execution.stop_reason
