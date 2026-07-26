from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _default_input_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


class ToolRiskLevel(str, Enum):
    UNSPECIFIED = "unspecified"
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    COMMAND_EXECUTION = "command_execution"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=_default_input_schema)
    risk_level: ToolRiskLevel = ToolRiskLevel.UNSPECIFIED


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(repr=False)


class ToolErrorCode(str, Enum):
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"


@dataclass(frozen=True, slots=True)
class ToolError:
    code: ToolErrorCode
    message: str = field(repr=False)
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    output: Any | None = field(default=None, repr=False)
    error: ToolError | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
