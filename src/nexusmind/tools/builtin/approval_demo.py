from __future__ import annotations

from typing import Any

from nexusmind.tools.contracts import ToolDefinition, ToolRiskLevel


class ApprovalDemoTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="approval_demo",
            description="Demonstrate approval flow without modifying local state.",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.LOCAL_WRITE,
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, str]:
        return {"message": arguments["message"]}
