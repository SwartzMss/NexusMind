from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from nexusmind.tools.contracts import (
    ToolDefinition,
    ToolResultBudget,
    ToolResultRequirements,
    ToolRiskLevel,
    json_result_requirements,
)
from nexusmind.workspace import (
    Workspace,
    WorkspaceConflictError,
    WorkspaceEncodingError,
    WorkspaceLimitError,
    WorkspacePathError,
    WorkspaceWriteError,
    resolve_workspace_create_target,
    workspace_relative_path,
)

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
MAX_WRITE_CONTENT_CHARS = 262144
MAX_WRITE_CONTENT_BYTES = 256 * 1024
MAX_REPLACE_TEXT_CHARS = 65536
MAX_REPLACE_OCCURRENCES = 100
MAX_REPLACE_RESULT_BYTES = 2 * 1024 * 1024
SHA256_RE = "^[0-9a-f]{64}$"


@dataclass(frozen=True, slots=True)
class WorkspaceWriteLimits:
    max_successful_mutations: int = 16
    max_total_bytes_written: int = 8 * 1024 * 1024


class WorkspaceWriteBudget:
    def __init__(self, limits: WorkspaceWriteLimits | None = None) -> None:
        self._limits = limits or WorkspaceWriteLimits()
        self.successful_mutations = 0
        self.total_bytes_written = 0

    def check(self, bytes_to_write: int) -> None:
        if self.successful_mutations >= self._limits.max_successful_mutations:
            raise WorkspaceLimitError("Workspace write budget exceeded")
        if self.total_bytes_written + bytes_to_write > self._limits.max_total_bytes_written:
            raise WorkspaceLimitError("Workspace write budget exceeded")

    def record_success(self, bytes_written: int) -> None:
        self.successful_mutations += 1
        self.total_bytes_written += bytes_written


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

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        return _workspace_result_requirements(
            {"path": _normalized_request_path(arguments.get("path", ".")), "entries": [], "truncated": True}
        )

    async def invoke_with_result_budget(
        self, arguments: dict[str, Any], *, result_budget: ToolResultBudget
    ) -> dict[str, Any]:
        result = await self.invoke(arguments)
        while result["entries"] and not _workspace_output_fits(result, result_budget):
            result["entries"].pop()
            result["truncated"] = True
        _require_workspace_output_fits(result, result_budget)
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
            "sha256": _sha256_hex(raw),
            "size": len(raw),
            "content": content,
            "truncated": truncated,
        }

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        start_line = int(arguments.get("start_line", 1))
        return _workspace_result_requirements(
            {
                "path": _normalized_request_path(arguments["path"]),
                "start_line": start_line,
                "end_line": start_line - 1,
                "sha256": "f" * 64,
                "size": MAX_READ_FILE_BYTES,
                "content": "",
                "truncated": True,
            }
        )

    async def invoke_with_result_budget(
        self, arguments: dict[str, Any], *, result_budget: ToolResultBudget
    ) -> dict[str, Any]:
        result = await self.invoke(arguments)
        content = result["content"]
        if not _workspace_output_fits(result, result_budget):
            low, high = 0, len(content)
            while low < high:
                middle = (low + high + 1) // 2
                candidate = {**result, "content": content[:middle], "truncated": True}
                if _workspace_output_fits(candidate, result_budget):
                    low = middle
                else:
                    high = middle - 1
            result["content"] = content[:low]
            result["end_line"] = (
                result["start_line"] + len(result["content"].splitlines()) - 1
                if result["content"]
                else result["start_line"] - 1
            )
            result["truncated"] = True
        _require_workspace_output_fits(result, result_budget)
        return result


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

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        return _workspace_result_requirements(
            {
                "query": arguments["query"],
                "path": _normalized_request_path(arguments.get("path", ".")),
                "matches": [],
                "files_scanned": MAX_SEARCH_FILES,
                "bytes_scanned": MAX_SEARCH_TOTAL_BYTES,
                "truncated": True,
            }
        )

    async def invoke_with_result_budget(
        self, arguments: dict[str, Any], *, result_budget: ToolResultBudget
    ) -> dict[str, Any]:
        result = await self.invoke(arguments)
        while result["matches"] and not _workspace_output_fits(result, result_budget):
            result["matches"].pop()
            result["truncated"] = True
        _require_workspace_output_fits(result, result_budget)
        return result


class WriteFileTool:
    def __init__(self, workspace: Workspace, budget: WorkspaceWriteBudget) -> None:
        self._workspace = workspace
        self._budget = budget

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="Create or replace a UTF-8 workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": MAX_WORKSPACE_PATH_CHARS},
                    "mode": {"type": "string", "enum": ["create", "replace"]},
                    "content": {"type": "string", "maxLength": MAX_WRITE_CONTENT_CHARS},
                    "expected_sha256": {"type": "string", "pattern": SHA256_RE},
                },
                "required": ["path", "mode", "content"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.LOCAL_WRITE,
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        content_bytes = _encode_write_content(arguments["content"], max_bytes=MAX_WRITE_CONTENT_BYTES)
        mode = arguments["mode"]
        if mode == "create":
            if "expected_sha256" in arguments:
                raise WorkspaceConflictError("Workspace file already exists")
            target = resolve_workspace_create_target(self._workspace, arguments["path"])
            if target.path.exists() or target.path.is_symlink():
                raise WorkspaceConflictError("Workspace file already exists")
            self._budget.check(len(content_bytes))
            cleanup_warning = _create_file_exclusive(target.path, content_bytes)
            self._budget.record_success(len(content_bytes))
            result = {
                "path": target.relative_path,
                "operation": "create",
                "previous_sha256": None,
                "sha256": _sha256_hex(content_bytes),
                "bytes_written": len(content_bytes),
                "committed": True,
            }
            if cleanup_warning:
                result["cleanup_warning"] = True
            return result
        expected_sha = arguments.get("expected_sha256")
        if type(expected_sha) is not str:
            raise WorkspaceConflictError("Workspace file changed since it was read")
        path = self._workspace.resolve_existing_file(arguments["path"])
        relative_path = workspace_relative_path(self._workspace, path)
        previous_raw = _read_existing_utf8_bytes(path)
        previous_sha = _sha256_hex(previous_raw)
        if previous_sha != expected_sha:
            raise WorkspaceConflictError("Workspace file changed since it was read")
        self._budget.check(len(content_bytes))
        _atomic_replace_file(path, content_bytes, expected_sha256=expected_sha)
        self._budget.record_success(len(content_bytes))
        return {
            "path": relative_path,
            "operation": "replace",
            "previous_sha256": previous_sha,
            "sha256": _sha256_hex(content_bytes),
            "bytes_written": len(content_bytes),
        }

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        path = _normalized_request_path(arguments["path"])
        if arguments["mode"] == "create":
            output = {
                "path": path,
                "operation": "create",
                "previous_sha256": None,
                "sha256": "f" * 64,
                "bytes_written": MAX_WRITE_CONTENT_BYTES,
                "committed": True,
                "cleanup_warning": True,
            }
        else:
            output = {
                "path": path,
                "operation": "replace",
                "previous_sha256": "f" * 64,
                "sha256": "f" * 64,
                "bytes_written": MAX_WRITE_CONTENT_BYTES,
            }
        return _workspace_result_requirements(output)

    async def invoke_with_result_budget(
        self,
        arguments: dict[str, Any],
        *,
        result_budget: ToolResultBudget,
    ) -> dict[str, Any]:
        return await self.invoke(arguments)


class ReplaceTextTool:
    def __init__(self, workspace: Workspace, budget: WorkspaceWriteBudget) -> None:
        self._workspace = workspace
        self._budget = budget

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="replace_text",
            description="Replace exact UTF-8 text inside a workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": MAX_WORKSPACE_PATH_CHARS},
                    "expected_sha256": {"type": "string", "pattern": SHA256_RE},
                    "old_text": {"type": "string", "minLength": 1, "maxLength": MAX_REPLACE_TEXT_CHARS},
                    "new_text": {"type": "string", "maxLength": MAX_REPLACE_TEXT_CHARS},
                    "expected_occurrences": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_REPLACE_OCCURRENCES,
                        "default": 1,
                    },
                },
                "required": ["path", "expected_sha256", "old_text", "new_text"],
                "additionalProperties": False,
            },
            risk_level=ToolRiskLevel.LOCAL_WRITE,
        )

    async def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        old_text = arguments["old_text"]
        new_text = arguments["new_text"]
        if old_text == new_text:
            raise WorkspaceWriteError("Workspace replacement text is unchanged")
        path = self._workspace.resolve_existing_file(arguments["path"])
        relative_path = workspace_relative_path(self._workspace, path)
        previous_raw = _read_existing_utf8_bytes(path)
        previous_sha = _sha256_hex(previous_raw)
        if previous_sha != arguments["expected_sha256"]:
            raise WorkspaceConflictError("Workspace file changed since it was read")
        text = previous_raw.decode("utf-8")
        expected_occurrences = int(arguments.get("expected_occurrences", 1))
        occurrences = text.count(old_text)
        if occurrences != expected_occurrences:
            raise WorkspaceWriteError("Workspace replacement count does not match")
        result_text = text.replace(old_text, new_text)
        result_bytes = _encode_write_content(result_text, max_bytes=MAX_REPLACE_RESULT_BYTES)
        self._budget.check(len(result_bytes))
        _atomic_replace_file(path, result_bytes, expected_sha256=arguments["expected_sha256"])
        self._budget.record_success(len(result_bytes))
        return {
            "path": relative_path,
            "operation": "replace_text",
            "previous_sha256": previous_sha,
            "sha256": _sha256_hex(result_bytes),
            "replacements": occurrences,
            "bytes_written": len(result_bytes),
        }

    def result_requirements(self, arguments: dict[str, Any]) -> ToolResultRequirements:
        return _workspace_result_requirements(
            {
                "path": _normalized_request_path(arguments["path"]),
                "operation": "replace_text",
                "previous_sha256": "f" * 64,
                "sha256": "f" * 64,
                "replacements": MAX_REPLACE_OCCURRENCES,
                "bytes_written": MAX_REPLACE_RESULT_BYTES,
            }
        )

    async def invoke_with_result_budget(
        self,
        arguments: dict[str, Any],
        *,
        result_budget: ToolResultBudget,
    ) -> dict[str, Any]:
        return await self.invoke(arguments)


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


def _read_existing_utf8_bytes(path: Path) -> bytes:
    raw = _read_limited(path, MAX_READ_FILE_BYTES)
    if b"\x00" in raw:
        raise WorkspaceEncodingError("Workspace file is not valid UTF-8")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspaceEncodingError("Workspace file is not valid UTF-8") from exc
    return raw


def _encode_write_content(content: str, *, max_bytes: int) -> bytes:
    raw = content.encode("utf-8")
    if b"\x00" in raw:
        raise WorkspaceEncodingError("Workspace file is not valid UTF-8")
    if len(raw) > max_bytes:
        raise WorkspaceLimitError("Workspace write exceeds the size limit")
    return raw


def _create_file_exclusive(path: Path, content: bytes) -> bool:
    temp_name: str | None = None
    original_error: BaseException | None = None
    committed = False
    cleanup_warning = False
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".nexusmind-", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, path)
            committed = True
        except FileExistsError as exc:
            raise WorkspaceConflictError("Workspace file already exists") from exc
    except FileExistsError as exc:
        original_error = WorkspaceConflictError("Workspace file already exists")
        original_error.__cause__ = exc
    except OSError as exc:
        original_error = WorkspaceWriteError("Workspace file could not be committed")
        original_error.__cause__ = exc
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError as cleanup_exc:
                if committed:
                    cleanup_warning = True
                    temp_name = None
                    return cleanup_warning
                cleanup_error = WorkspaceWriteError("Workspace temporary file could not be cleaned up")
                cleanup_error.__cause__ = cleanup_exc
                if original_error is not None:
                    cleanup_error.__context__ = original_error
                raise cleanup_error
    if original_error is not None:
        raise original_error
    return cleanup_warning


def _atomic_replace_file(path: Path, content: bytes, *, expected_sha256: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temp_name: str | None = None
    original_error: BaseException | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".nexusmind-", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        if _sha256_hex(_read_existing_utf8_bytes(path)) != expected_sha256:
            raise WorkspaceConflictError("Workspace file changed since it was read")
        os.replace(temp_name, path)
        temp_name = None
    except WorkspaceConflictError as exc:
        original_error = exc
    except OSError as exc:
        original_error = WorkspaceWriteError("Workspace file could not be committed")
        original_error.__cause__ = exc
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError as cleanup_exc:
                cleanup_error = WorkspaceWriteError("Workspace temporary file could not be cleaned up")
                cleanup_error.__cause__ = cleanup_exc
                if original_error is not None:
                    cleanup_error.__context__ = original_error
                raise cleanup_error
    if original_error is not None:
        raise original_error


def _sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_output_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _workspace_result_requirements(output: dict[str, Any]) -> ToolResultRequirements:
    return json_result_requirements({"ok": True, "output": output})


def _workspace_output_fits(output: dict[str, Any], budget: ToolResultBudget) -> bool:
    return budget.satisfies(_workspace_result_requirements(output))


def _require_workspace_output_fits(output: dict[str, Any], budget: ToolResultBudget) -> None:
    if not _workspace_output_fits(output, budget):
        raise WorkspaceLimitError("Workspace tool result budget is too small")


def _normalized_request_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    return "/".join(parts) or "."


def _ensure_json_output_limit(value: Any, max_bytes: int) -> None:
    if _json_output_bytes(value) > max_bytes:
        raise WorkspaceLimitError("Workspace tool output exceeds the size limit")
