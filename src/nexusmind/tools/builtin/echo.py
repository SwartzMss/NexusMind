from __future__ import annotations

from typing import Any

from nexusmind.tools.contracts import ToolDefinition


class EchoTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Return the provided text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, str]:
        return {"text": arguments["text"]}

