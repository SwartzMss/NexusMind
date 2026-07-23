from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nexusmind-test-echo")


@mcp.tool()
def echo(text: str) -> dict[str, str]:
    return {"text": text}


if __name__ == "__main__":
    mcp.run()

