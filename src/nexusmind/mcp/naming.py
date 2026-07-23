from __future__ import annotations

import hashlib
import re

_MAX_TOOL_NAME = 64


def mcp_tool_local_name(server_id: str, remote_name: str) -> str:
    seed = f"{server_id}:{remote_name}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{server_id}__{remote_name}").strip("_")
    if not normalized or not normalized[0].isalpha():
        normalized = f"tool_{normalized}"
    suffix = f"_{digest}"
    prefix = normalized[: _MAX_TOOL_NAME - len(suffix)].rstrip("_-")
    if not prefix or not prefix[0].isalpha():
        prefix = "tool"
    return f"{prefix}{suffix}"
