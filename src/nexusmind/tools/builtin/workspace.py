from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any

from nexusmind.tools.contracts import ToolDefinition, ToolRiskLevel
from nexusmind.workspace import Workspace, WorkspaceEncodingError, WorkspaceLimitError, WorkspacePathError, workspace_relative_path

MAX_LIST_ENTRIES = 1000
MAX_LIST_DEPTH = 6
MAX_LIST_OUTPUT_BYTES = 128 * 1024
MAX_READ_FILE_BYTES = 2 * 1024 * 1024
MAX_READ_LINES = 500
MAX_READ_CONTENT_BYTES = 64 * 1024
MAX_SEARCH_FILES = 1000
MAX_SEARCH_FILE_BYTES = 1024 * 1024
MAX_SEARCH_TOTAL_BYTES = 16 * 1024 * 1024
MAX_SEARCH_MATCHES = 200
MAX_SEARCH_LINE_CHARS = 500
MAX_SEARCH_OUTPUT_BYTES = 128 * 1024
MAX_SEARCH_ENTRIES_VISITED = 10000
MAX_SEARCH_DIRECTORIES_VISITED = 1000
MAX_SEARCH_FILES_CONSIDERED = 1000
MAX_WORKSPACE_PATH_CHARS = 4096


class ListFilesTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_files",
            description="List files and directories inside the configured workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": MAX_WORKSPACE_PATH_CHARS, "default": "."},
                    "recursive": {"type": "boolean", "default": False},
                    "max_depth": {"type": "integer", "minimum": 1, "maximum": MAX_LIST_DEPTH, "default": 3},
                },
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested = arguments.get("path", ".")
        root = self._workspace.resolve_existing_directory(requested)
        result_path = workspace_relative_path(self._workspace, root)
        recursive = bool(arguments.get("recursive", False))
        max_depth = int(arguments.get("max_depth", 3))
        entries: list[dict[str, Any]] = []
        truncated = False
        pending = deque([(root, 0)])
        while pending and len(entries) < MAX_LIST_ENTRIES:
            directory, depth = pending.popleft()
            children, children_truncated = _limited_sorted_children(directory, MAX_LIST_ENTRIES + 1)
            if children_truncated:
                truncated = True
            for child in children:
                if len(entries) >= MAX_LIST_ENTRIES:
                    truncated = True
                    break
                entry = _list_entry(self._workspace, child)
                entries.append(entry)
                if _json_output_bytes({"path": result_path, "entries": entries, "truncated": truncated}) > MAX_LIST_OUTPUT_BYTES:
                    entries.pop()
                    truncated = True
                    break
                if recursive and depth + 1 < max_depth and entry["type"] == "directory":
                    pending.append((child, depth + 1))
            if not recursive or _json_output_bytes({"entries": entries}) > MAX_LIST_OUTPUT_BYTES:
                break
        if pending:
            truncated = True
        entries.sort(key=lambda entry: entry["path"])
        result = {"path": result_path, "entries": entries, "truncated": truncated}
        _ensure_json_output_limit(result, MAX_LIST_OUTPUT_BYTES)
        return result


class ReadFileTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file inside the configured workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": MAX_WORKSPACE_PATH_CHARS},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": MAX_READ_LINES, "default": 200},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested = arguments["path"]
        path = self._workspace.resolve_existing_file(requested)
        if path.stat().st_size > MAX_READ_FILE_BYTES:
            raise WorkspaceLimitError("Workspace file exceeds the size limit")
        raw = _read_limited(path, MAX_READ_FILE_BYTES)
        if b"\x00" in raw:
            raise WorkspaceEncodingError("Workspace file is not valid UTF-8")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceEncodingError("Workspace file is not valid UTF-8") from exc
        start_line = int(arguments.get("start_line", 1))
        max_lines = int(arguments.get("max_lines", 200))
        lines = text.splitlines(keepends=True)
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        content = ""
        truncated = start_line - 1 + max_lines < len(lines)
        for line in selected:
            encoded_size = len((content + line).encode("utf-8"))
            if encoded_size > MAX_READ_CONTENT_BYTES:
                truncated = True
                break
            content += line
        end_line = start_line + len(content.splitlines()) - 1 if content else start_line - 1
        return {
            "path": workspace_relative_path(self._workspace, path),
            "start_line": start_line,
            "end_line": end_line,
            "content": content,
            "truncated": truncated,
        }


class SearchTextTool:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_text",
            description="Search UTF-8 workspace files for a literal text query.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 256},
                    "path": {"type": "string", "maxLength": MAX_WORKSPACE_PATH_CHARS, "default": "."},
                    "case_sensitive": {"type": "boolean", "default": True},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.READ_ONLY,
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"]
        requested = arguments.get("path", ".")
        root = self._workspace.resolve_existing_directory(requested)
        result_path = workspace_relative_path(self._workspace, root)
        case_sensitive = bool(arguments.get("case_sensitive", True))
        needle = query if case_sensitive else query.casefold()
        files_scanned = 0
        files_considered = 0
        directories_visited = 0
        entries_visited = 0
        bytes_scanned = 0
        matches: list[dict[str, Any]] = []
        incomplete = False
        stop_due_to_limit = False
        pending = deque([root])
        while pending and not stop_due_to_limit:
            if directories_visited >= MAX_SEARCH_DIRECTORIES_VISITED:
                stop_due_to_limit = True
                break
            directory = pending.popleft()
            directories_visited += 1
            try:
                children, children_truncated = _limited_sorted_children(directory, MAX_SEARCH_ENTRIES_VISITED - entries_visited + 1)
            except OSError:
                incomplete = True
                continue
            if children_truncated:
                stop_due_to_limit = True
            for child in children:
                entries_visited += 1
                if entries_visited > MAX_SEARCH_ENTRIES_VISITED:
                    stop_due_to_limit = True
                    break
                try:
                    if child.is_symlink():
                        continue
                    if child.is_dir():
                        pending.append(child)
                        continue
                    if not child.is_file():
                        continue
                except OSError:
                    incomplete = True
                    continue
                files_considered += 1
                if files_considered > MAX_SEARCH_FILES_CONSIDERED or files_scanned >= MAX_SEARCH_FILES:
                    stop_due_to_limit = True
                    break
                try:
                    size = child.stat().st_size
                except OSError:
                    incomplete = True
                    continue
                if size > MAX_SEARCH_FILE_BYTES:
                    continue
                remaining_bytes = MAX_SEARCH_TOTAL_BYTES - bytes_scanned
                if size > remaining_bytes:
                    stop_due_to_limit = True
                    break
                try:
                    raw = _read_limited(child, min(MAX_SEARCH_FILE_BYTES, remaining_bytes))
                except OSError:
                    incomplete = True
                    continue
                bytes_scanned += len(raw)
                if b"\x00" in raw:
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                files_scanned += 1
                haystack_lines = text.splitlines()
                for line_number, line in enumerate(haystack_lines, start=1):
                    comparable = line if case_sensitive else line.casefold()
                    if needle not in comparable:
                        continue
                    match = {
                        "path": workspace_relative_path(self._workspace, child),
                        "line": line_number,
                        "text": line[:MAX_SEARCH_LINE_CHARS],
                    }
                    matches.append(match)
                    if len(matches) > MAX_SEARCH_MATCHES or _json_output_bytes({"matches": matches}) > MAX_SEARCH_OUTPUT_BYTES:
                        matches.pop()
                        stop_due_to_limit = True
                        break
                if stop_due_to_limit:
                    break
            if stop_due_to_limit:
                break
            if bytes_scanned >= MAX_SEARCH_TOTAL_BYTES:
                stop_due_to_limit = True
                break
        if pending:
            stop_due_to_limit = True
        matches.sort(key=lambda item: (item["path"], item["line"]))
        result = {
            "query": query,
            "path": result_path,
            "matches": matches,
            "files_scanned": files_scanned,
            "bytes_scanned": bytes_scanned,
            "truncated": incomplete or stop_due_to_limit,
        }
        while matches and _json_output_bytes(result) > MAX_SEARCH_OUTPUT_BYTES:
            matches.pop()
            result["truncated"] = True
        _ensure_json_output_limit(result, MAX_SEARCH_OUTPUT_BYTES)
        return result


def _limited_sorted_children(directory: Path, limit: int) -> tuple[list[Path], bool]:
    children: list[Path] = []
    for child in directory.iterdir():
        children.append(child)
        if len(children) >= limit:
            break
    truncated = len(children) >= limit
    if truncated:
        children = children[: limit - 1]
    return sorted(children, key=lambda path: path.name), truncated


def _list_entry(workspace: Workspace, path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": workspace_relative_path(workspace, path)}
    if path.is_symlink():
        entry["type"] = "symlink"
    elif path.is_dir():
        entry["type"] = "directory"
    elif path.is_file():
        entry["type"] = "file"
        try:
            entry["size"] = path.stat().st_size
        except OSError:
            entry["size"] = 0
    else:
        entry["type"] = "file"
    return entry


def _read_limited(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise WorkspaceLimitError("Workspace file exceeds the size limit")
    return data


def _json_output_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _ensure_json_output_limit(value: Any, max_bytes: int) -> None:
    if _json_output_bytes(value) > max_bytes:
        raise WorkspaceLimitError("Workspace tool output exceeds the size limit")
