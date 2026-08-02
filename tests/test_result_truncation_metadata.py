import asyncio

from nexusmind.tools import ToolExecutor, ToolRegistry
from nexusmind.tools.contracts import ToolDefinition
from nexusmind.tools.base import Tool

class TruncatedTool(Tool):
    definition = ToolDefinition(name="truncated", description="", input_schema={"type":"object"})
    async def invoke(self, arguments):
        return {"text": "partial", "truncated": True}

def test_executor_records_result_truncated_metadata():
    registry = ToolRegistry(); registry.register(TruncatedTool())
    result = asyncio.run(ToolExecutor(registry).execute_with_result_budget(__import__('nexusmind.tools.contracts', fromlist=['ToolCall']).ToolCall(id="c1", name="truncated", arguments={}), result_budget=None))
    assert result.metadata["result_truncated"] is True
