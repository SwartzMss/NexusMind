from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from nexusmind.mcp.errors import MCPConfigError
from nexusmind.mcp.limits import MAX_MCP_CLIENTS_PER_GROUP

_SERVER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_MAX_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class MCPStdioServerConfig:
    server_id: str
    command: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    connect_timeout: float = 10.0
    request_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not _SERVER_ID_RE.fullmatch(self.server_id):
            raise MCPConfigError("MCP server_id must start with a letter and contain only letters, digits, '_' or '-'")
        if not self.command:
            raise MCPConfigError("MCP stdio server command is required")
        if not all(isinstance(arg, str) for arg in self.args):
            raise MCPConfigError("MCP stdio server args must be strings")
        if self.cwd is not None and not isinstance(self.cwd, str):
            raise MCPConfigError("MCP stdio server cwd must be a string or null")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in self.env.items()):
            raise MCPConfigError("MCP stdio server env must contain string keys and values")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))
        _validate_timeout("connect_timeout", self.connect_timeout)
        _validate_timeout("request_timeout", self.request_timeout)


def load_mcp_server_config(path: str | Path, server_id: str) -> MCPStdioServerConfig:
    raw = _read_mcp_config(path)
    return _server_config_from_raw(raw, server_id)


def load_mcp_server_configs(path: str | Path, server_ids: Iterable[str]) -> dict[str, MCPStdioServerConfig]:
    requested = _limited_server_id_tuple(server_ids)
    raw = _read_mcp_config(path)
    return {server_id: _server_config_from_raw(raw, server_id) for server_id in requested}


def _limited_server_id_tuple(server_ids: Iterable[str]) -> tuple[str, ...]:
    requested: set[str] = set()
    consumed = 0
    for server_id in server_ids:
        consumed += 1
        if consumed > MAX_MCP_CLIENTS_PER_GROUP:
            raise MCPConfigError("MCP config references too many servers")
        if type(server_id) is not str:
            raise MCPConfigError("MCP server_id must be a string")
        if not _SERVER_ID_RE.fullmatch(server_id):
            raise MCPConfigError("MCP server_id is invalid")
        requested.add(server_id)
        if len(requested) > MAX_MCP_CLIENTS_PER_GROUP:
            raise MCPConfigError("MCP config references too many servers")
    return tuple(sorted(requested))


def _read_mcp_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw_bytes = handle.read(_MAX_CONFIG_BYTES + 1)
    except FileNotFoundError as exc:
        raise MCPConfigError(f"MCP config file not found: {config_path}") from exc
    except OSError as exc:
        raise MCPConfigError(f"MCP config file is not readable: {config_path}") from exc
    if len(raw_bytes) > _MAX_CONFIG_BYTES:
        raise MCPConfigError("MCP config file is too large")
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MCPConfigError(f"MCP config file is not valid UTF-8: {config_path}") from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise MCPConfigError(f"MCP config file is not valid JSON: {config_path}") from exc
    if not isinstance(raw, dict):
        raise MCPConfigError("MCP config must be a JSON object")
    return raw


def _server_config_from_raw(raw: dict[str, Any], server_id: str) -> MCPStdioServerConfig:
    servers = raw.get("servers")
    if not isinstance(servers, dict):
        raise MCPConfigError("MCP config must contain a servers object")
    server = servers.get(server_id)
    if not isinstance(server, dict):
        raise MCPConfigError(f"MCP server not found: {server_id}")
    if server.get("transport") != "stdio":
        raise MCPConfigError("Only MCP stdio transport is supported")
    command = server.get("command")
    if not isinstance(command, str) or not command:
        raise MCPConfigError("MCP stdio server command is required")
    args = _string_tuple(server.get("args", []), "args")
    env = _string_dict(server.get("env", {}), "env")
    cwd = server.get("cwd")
    connect_timeout = _number(server.get("connect_timeout", 10.0), "connect_timeout")
    request_timeout = _number(server.get("request_timeout", 30.0), "request_timeout")
    if cwd is not None and not isinstance(cwd, str):
        raise MCPConfigError("MCP stdio server cwd must be a string or null")
    return MCPStdioServerConfig(
        server_id=server_id,
        command=_expand_env(command),
        args=tuple(_expand_env(arg) for arg in args),
        cwd=_expand_env(cwd) if cwd else cwd,
        env={key: _expand_env(value) for key, value in env.items()},
        connect_timeout=connect_timeout,
        request_timeout=request_timeout,
    )


def _validate_timeout(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise MCPConfigError(f"MCP {name} must be a finite positive number")


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise MCPConfigError(f"MCP stdio server {name} must be a list of strings")
    return tuple(value)


def _string_dict(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise MCPConfigError(f"MCP stdio server {name} must contain string keys and values")
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise MCPConfigError(f"MCP {name} must be a number")
    return float(value)


def _expand_env(value: str) -> str:
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if not match:
        return value
    name = match.group(1)
    if name not in os.environ:
        raise MCPConfigError(f"Missing environment variable referenced by MCP config: {name}")
    return os.environ[name]

