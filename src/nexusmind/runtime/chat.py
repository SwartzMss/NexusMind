"""Chat protocol adapter for the provider-neutral agent harness."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta

from nexusmind.models.base import ChatModel
from nexusmind.runtime.events import RuntimeEvent
from nexusmind.runtime.harness.context import HarnessRequest
from nexusmind.runtime.harness.limits import HarnessLimits
from nexusmind.runtime.harness.runner_impl import ToolExecutionCancelled
from nexusmind.runtime.harness.runner import HarnessRunner
from nexusmind.runtime.harness.checkpoint_store import CheckpointStore
from nexusmind.runtime.harness.checkpointing import CheckpointCoordinator
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.runtime.lease_guarding import LeaseGuardedChatModel, LeaseGuardedToolExecutor
from nexusmind.runtime.leases import (
    RunLeaseCoordinator,
    RunLeaseOwnershipGuard,
    RunLeaseStore,
)
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
        lease_store: RunLeaseStore | None = None,
        lease_run_id: str | None = None,
        lease_owner_id: str | None = None,
        lease_ttl: timedelta = timedelta(seconds=30),
        lease_heartbeat_interval: timedelta | None = None,
        lease_clock: Callable[[], datetime] | None = None,
        lease_release_timeout: timedelta = timedelta(seconds=10),
    ) -> None:
        if checkpoint_store is not None and not checkpoint_run_id:
            raise ValueError("checkpoint_run_id is required when automatic checkpointing is enabled")
        if checkpoint_store is None and checkpoint_run_id is not None:
            raise ValueError("checkpoint_store is required when checkpoint_run_id is provided")
        if lease_store is not None and not lease_run_id:
            raise ValueError("lease_run_id is required when run leasing is enabled")
        if lease_store is None and lease_run_id is not None:
            raise ValueError("lease_store is required when lease_run_id is provided")
        if checkpoint_run_id is not None and lease_run_id is not None and checkpoint_run_id != lease_run_id:
            raise ValueError("Lease and checkpoint run IDs must match")
        self._lease_guard = RunLeaseOwnershipGuard(clock=lease_clock) if lease_store is not None else None
        guarded_model = LeaseGuardedChatModel(model, self._lease_guard) if self._lease_guard is not None else model
        guarded_executor = (
            LeaseGuardedToolExecutor(tool_executor, self._lease_guard)
            if self._lease_guard is not None and tool_executor is not None
            else tool_executor
        )
        self._harness = HarnessRunner(
            guarded_model,
            tool_executor=guarded_executor,
            limits=limits,
            tool_policy=tool_policy,
            approval_provider=approval_provider,
            approval_summarizer=approval_summarizer,
        )
        self._checkpoint_store = checkpoint_store
        self._checkpoint_run_id = checkpoint_run_id
        self._checkpoint_sequence = checkpoint_sequence
        self._save_terminal_checkpoint = save_terminal_checkpoint
        self._lease_store = lease_store
        self._lease_run_id = lease_run_id
        self._lease_owner_id = lease_owner_id
        self._lease_ttl = lease_ttl
        self._lease_heartbeat_interval = lease_heartbeat_interval
        self._lease_release_timeout = lease_release_timeout
        self._lease_coordinator: RunLeaseCoordinator | None = None

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
        if self._lease_store is not None:
            self._lease_coordinator = RunLeaseCoordinator(
                self._lease_store,
                run_id=self._lease_run_id,
                owner_id=self._lease_owner_id,
                ttl=self._lease_ttl,
                heartbeat_interval=self._lease_heartbeat_interval,
                lease_release_timeout=self._lease_release_timeout,
                guard=self._lease_guard,
            )
            stream = self._lease_coordinator.stream(stream)
        try:
            async for event in stream:
                yield event
        finally:
            # Preserve the compatibility snapshot exposed by HarnessRunner.
            self._harness.state = execution.state
            self._harness.stop_reason = execution.stop_reason
