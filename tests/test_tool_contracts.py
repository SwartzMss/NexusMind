import pytest

from nexusmind.tools import ToolDefinition, ToolRegistry, ToolRegistryError
from nexusmind.tools.builtin import EchoTool


def test_valid_schema_can_be_registered() -> None:
    registry = ToolRegistry()

    registry.register(EchoTool())

    assert registry.contains("echo")


def test_default_input_schema_is_independent_between_definitions() -> None:
    first = ToolDefinition(name="first")
    second = ToolDefinition(name="second")

    first.input_schema["properties"]["text"] = {"type": "string"}

    assert second.input_schema["properties"] == {}


def test_default_input_schema_nested_state_is_not_reused() -> None:
    first = ToolDefinition(name="first")
    second = ToolDefinition(name="second")

    assert first.input_schema is not second.input_schema
    assert first.input_schema["properties"] is not second.input_schema["properties"]


@pytest.mark.parametrize(
    "definition",
    [
        ToolDefinition(name="", input_schema={"type": "object"}),
        ToolDefinition(name="1bad", input_schema={"type": "object"}),
        ToolDefinition(name="bad name", input_schema={"type": "object"}),
    ],
)
def test_invalid_tool_names_are_rejected(definition: ToolDefinition) -> None:
    class BadNameTool:
        @property
        def definition(self) -> ToolDefinition:
            return definition

        async def invoke(self, arguments):
            return None

    with pytest.raises(ToolRegistryError):
        ToolRegistry().register(BadNameTool())


def test_invalid_schema_is_rejected() -> None:
    class BadSchemaTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name="bad_schema", input_schema={"type": "not-a-json-schema-type"})

        async def invoke(self, arguments):
            return None

    with pytest.raises(ToolRegistryError):
        ToolRegistry().register(BadSchemaTool())


def test_non_object_schema_is_rejected() -> None:
    class ArraySchemaTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(name="array_schema", input_schema={"type": "array"})

        async def invoke(self, arguments):
            return None

    with pytest.raises(ToolRegistryError):
        ToolRegistry().register(ArraySchemaTool())
