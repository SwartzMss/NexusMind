from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from nexusmind.tools.contracts import ToolCall, ToolDefinition, ToolRiskLevel


class ToolPolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ApprovalDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class ToolPolicyContext:
    run_id: str | None
    model_turn: int
    tool_call_index: int
    tool_definition: ToolDefinition


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    call_id: str
    tool_name: str
    risk_level: ToolRiskLevel
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if type(self.metadata) is not dict:
            raise TypeError("ApprovalRequest metadata must be a dict")
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


@dataclass(frozen=True, slots=True)
class ToolApproval:
    request_id: str
    call_id: str
    tool_name: str
    risk_level: ToolRiskLevel
    summary: str
    decision: ApprovalDecision | None = None


class ToolPolicy(Protocol):
    async def evaluate(self, call: ToolCall, context: ToolPolicyContext) -> ToolPolicyDecision:
        ...


class ApprovalProvider(Protocol):
    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


class ToolApprovalSummarizer(Protocol):
    def summarize(self, call: ToolCall, definition: ToolDefinition) -> str:
        ...


class DefaultToolPolicy:
    async def evaluate(self, call: ToolCall, context: ToolPolicyContext) -> ToolPolicyDecision:
        if context.tool_definition.risk_level == ToolRiskLevel.READ_ONLY:
            return ToolPolicyDecision.ALLOW
        return ToolPolicyDecision.REQUIRE_APPROVAL


class DefaultToolApprovalSummarizer:
    def __init__(self, max_length: int = 160) -> None:
        self._max_length = max_length

    def summarize(self, call: ToolCall, definition: ToolDefinition) -> str:
        summary = f"Approve tool {definition.name}"
        if len(summary) <= self._max_length:
            return summary
        return summary[: self._max_length].rstrip()


def new_approval_request_id() -> str:
    return f"approval_{uuid4().hex}"
