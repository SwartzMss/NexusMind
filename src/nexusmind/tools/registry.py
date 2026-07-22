from __future__ import annotations

import re

from jsonschema.exceptions import SchemaError
from jsonschema.validators import Draft202012Validator

from nexusmind.tools.base import Tool
from nexusmind.tools.contracts import ToolDefinition

_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class ToolRegistryError(ValueError):
    pass


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        definition = tool.definition
        validate_tool_definition(definition)
        if definition.name in self._tools:
            raise ToolRegistryError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    def list_definitions(self) -> list[ToolDefinition]:
        return [self._tools[name].definition for name in sorted(self._tools)]


def validate_tool_definition(definition: ToolDefinition) -> None:
    if not definition.name or not _TOOL_NAME_RE.fullmatch(definition.name):
        raise ToolRegistryError("Tool name must start with a letter and contain only letters, digits, '_' or '-'")
    schema = definition.input_schema
    if not isinstance(schema, dict):
        raise ToolRegistryError("Tool input_schema must be a JSON Schema object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ToolRegistryError(f"Invalid input_schema for tool {definition.name}") from exc
    if schema.get("type") != "object":
        raise ToolRegistryError("Tool input_schema must describe a JSON object")

