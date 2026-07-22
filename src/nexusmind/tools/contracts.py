from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


DEFAULT_INPUT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_INPUT_SCHEMA))


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class ToolErrorCode(str, Enum):
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"


@dataclass(frozen=True, slots=True)
class ToolError:
    code: ToolErrorCode
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    output: Any | None = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

