from __future__ import annotations

from typing import Any

import json

from nexusmind.tools.contracts import ToolDefinition, ToolResultBudget, ToolResultRequirements, ToolRiskLevel


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

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        payload = {"ok": True, "output": {"message": arguments["message"]}}
        return ToolResultRequirements(
            min_bytes=len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
            min_nodes=4,
            min_depth=2,
        )

    async def invoke_with_result_budget(
        self,
        arguments: dict[str, Any],
        *,
        result_budget: ToolResultBudget,
    ) -> dict[str, str]:
        return await self.invoke(arguments)
