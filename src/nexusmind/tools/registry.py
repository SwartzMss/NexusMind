from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

from jsonschema.exceptions import SchemaError
from jsonschema.validators import Draft202012Validator

from nexusmind.tools.base import Tool
from nexusmind.tools.contracts import ToolDefinition, ToolRiskLevel

_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class ToolRegistryError(ValueError):
    pass


class ToolNotFoundError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    tool: Tool
    definition: ToolDefinition
    validator: Draft202012Validator


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(self, tool: Tool) -> None:
        definition = _copy_tool_definition(tool.definition)
        validate_tool_definition(definition)
        if definition.name in self._tools:
            raise ToolRegistryError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = _RegisteredTool(
            tool=tool,
            definition=definition,
            validator=Draft202012Validator(definition.input_schema),
        )

    def register_many(self, tools: list[Tool]) -> None:
        staged: dict[str, _RegisteredTool] = {}
        for tool in tools:
            definition = _copy_tool_definition(tool.definition)
            validate_tool_definition(definition)
            if definition.name in self._tools or definition.name in staged:
                raise ToolRegistryError(f"Tool already registered: {definition.name}")
            staged[definition.name] = _RegisteredTool(
                tool=tool,
                definition=definition,
                validator=Draft202012Validator(definition.input_schema),
            )
        self._tools.update(staged)

    def get(self, name: str) -> Tool:
        return self._get_registered(name).tool

    def _get_registered(self, name: str) -> _RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    def list_definitions(self) -> list[ToolDefinition]:
        return [_copy_tool_definition(self._tools[name].definition) for name in sorted(self._tools)]


def validate_tool_definition(definition: ToolDefinition) -> None:
    if not definition.name or not _TOOL_NAME_RE.fullmatch(definition.name):
        raise ToolRegistryError("Tool name must start with a letter and contain only letters, digits, '_' or '-'")
    if not isinstance(definition.risk_level, ToolRiskLevel):
        raise ToolRegistryError("Tool risk_level must be a ToolRiskLevel")
    schema = definition.input_schema
    if not isinstance(schema, dict):
        raise ToolRegistryError("Tool input_schema must be a JSON Schema object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ToolRegistryError(f"Invalid input_schema for tool {definition.name}") from exc
    if schema.get("type") != "object":
        raise ToolRegistryError("Tool input_schema must describe a JSON object")


def _copy_tool_definition(definition: ToolDefinition) -> ToolDefinition:
    return ToolDefinition(
        name=definition.name,
        description=definition.description,
        input_schema=deepcopy(definition.input_schema),
        risk_level=definition.risk_level,
    )
