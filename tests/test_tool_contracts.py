import pytest

from nexusmind.tools import ToolDefinition, ToolRegistry, ToolRegistryError
from nexusmind.tools.builtin import EchoTool


def test_valid_schema_can_be_registered() -> None:
    registry = ToolRegistry()

    registry.register(EchoTool())

    assert registry.contains("echo")


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

