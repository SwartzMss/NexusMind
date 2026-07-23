from __future__ import annotations


class MCPError(Exception):
    """Base class for NexusMind MCP integration failures."""


class MCPConfigError(MCPError, ValueError):
    pass


class MCPConnectionError(MCPError):
    pass


class MCPDiscoveryError(MCPError):
    pass


class MCPToolCallError(MCPError):
    pass

