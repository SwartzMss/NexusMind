"""Model and tool adapters that fence calls with execution ownership."""

from __future__ import annotations

import asyncio

from nexusmind.models.base import ChatModel
from nexusmind.runtime.leases import ExecutionOwnershipGuard
from nexusmind.tools.contracts import ToolCall, ToolResult, ToolResultBudget, ToolResultRequirements
from nexusmind.tools.executor import ToolExecutorProtocol


class LeaseGuardedChatModel(ChatModel):
    """Fence each provider invocation with the shared ownership proof."""

    def __init__(self, delegate: ChatModel, guard: ExecutionOwnershipGuard) -> None:
        self._delegate = delegate
        self._guard = guard

    async def stream(self, messages, tools=None):
        self._guard.assert_owned()
        async for event in self._delegate.stream(messages, tools=tools):
            yield event


class LeaseGuardedToolExecutor:
    """Fence each tool invocation after policy/approval and before execution."""

    def __init__(self, delegate: ToolExecutorProtocol, guard: ExecutionOwnershipGuard) -> None:
        self._delegate = delegate
        self._guard = guard

    def definition(self, name: str):
        return self._delegate.definition(name)

    def result_requirements(self, call: ToolCall) -> ToolResultRequirements:
        return self._delegate.result_requirements(call)

    async def execute(self, call: ToolCall) -> ToolResult:
        self._guard.assert_owned()
        try:
            return await self._delegate.execute(call)
        except asyncio.CancelledError:
            self._mark_execution_uncertain()
            raise

    async def execute_with_result_budget(
        self,
        call: ToolCall,
        *,
        result_budget: ToolResultBudget,
    ) -> ToolResult:
        self._guard.assert_owned()
        try:
            return await self._delegate.execute_with_result_budget(call, result_budget=result_budget)
        except asyncio.CancelledError:
            self._mark_execution_uncertain()
            raise

    def _mark_execution_uncertain(self) -> None:
        mark = getattr(self._guard, "mark_execution_uncertain", None)
        if callable(mark):
            mark()
