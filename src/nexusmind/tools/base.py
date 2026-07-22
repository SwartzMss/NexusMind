from __future__ import annotations

from typing import Any, Protocol

from nexusmind.tools.contracts import ToolDefinition


class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition:
        ...

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        ...

