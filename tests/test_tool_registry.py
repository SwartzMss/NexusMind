import pytest

from nexusmind.tools import ToolDefinition, ToolNotFoundError, ToolRegistry, ToolRegistryError
from nexusmind.tools.builtin import EchoTool


class StaticTool:
    def __init__(self, name: str) -> None:
        self._definition = ToolDefinition(name=name, description=f"{name} tool")

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def invoke(self, arguments):
        return arguments


def test_registry_registers_and_gets_tool() -> None:
    registry = ToolRegistry()
    tool = EchoTool()

    registry.register(tool)

    assert registry.get("echo") is tool
    assert registry.contains("echo")


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    with pytest.raises(ToolRegistryError):
        registry.register(EchoTool())


def test_registry_get_unknown_tool_is_explicit() -> None:
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().get("missing")


def test_registry_lists_definitions_in_deterministic_order() -> None:
    registry = ToolRegistry()
    registry.register(StaticTool("zeta"))
    registry.register(StaticTool("alpha"))

    assert [definition.name for definition in registry.list_definitions()] == ["alpha", "zeta"]


def test_registry_list_definitions_returns_isolated_copies() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    first = registry.list_definitions()[0]
    first.input_schema["required"] = []
    first.input_schema["additionalProperties"] = True

    second = registry.list_definitions()[0]

    assert second.input_schema["required"] == ["text"]
    assert second.input_schema["additionalProperties"] is False
