from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any


def _default_input_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


class ToolRiskLevel(str, Enum):
    UNSPECIFIED = "unspecified"
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    LOCAL_EXEC = "local_exec"
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


@dataclass(frozen=True, slots=True)
class ToolResultBudget:
    max_bytes: int
    max_nodes: int
    max_depth: int

    def satisfies(self, requirements: ToolResultRequirements) -> bool:
        return (
            self.max_bytes >= requirements.min_bytes
            and self.max_nodes >= requirements.min_nodes
            and self.max_depth >= requirements.min_depth
        )


@dataclass(frozen=True, slots=True)
class ToolResultRequirements:
    min_bytes: int
    min_nodes: int
    min_depth: int


def json_result_requirements(payload: Any) -> ToolResultRequirements:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    nodes, depth = json_shape(payload)
    return ToolResultRequirements(min_bytes=len(encoded), min_nodes=nodes, min_depth=depth)


def json_shape(value: Any, *, depth: int = 0) -> tuple[int, int]:
    if isinstance(value, dict):
        children = [json_shape(item, depth=depth + 1) for item in value.values()]
    elif isinstance(value, list):
        children = [json_shape(item, depth=depth + 1) for item in value]
    else:
        children = []
    return 1 + sum(item[0] for item in children), max([depth, *(item[1] for item in children)])
