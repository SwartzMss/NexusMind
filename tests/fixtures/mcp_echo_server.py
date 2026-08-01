from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP("nexusmind-test-echo")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def echo(text: str) -> dict[str, str]:
    return {"text": text}


if __name__ == "__main__":
    mcp.run()

