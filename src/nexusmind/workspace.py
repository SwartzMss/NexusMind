from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Literal


class WorkspaceError(Exception):
    pass


class WorkspacePathError(WorkspaceError):
    pass


class WorkspaceLimitError(WorkspaceError):
    pass


class WorkspaceEncodingError(WorkspaceError):
    pass


class WorkspaceConflictError(WorkspaceError):
    pass


class WorkspaceWriteError(WorkspaceError):
    pass


@dataclass(frozen=True, slots=True)
class WorkspaceWriteTarget:
    path: Path
    relative_path: str
    parent: Path


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path = field(repr=False)

    def __post_init__(self) -> None:
        try:
            root = self.root.resolve(strict=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise WorkspacePathError("Workspace root does not exist") from exc
        if not root.is_dir():
            raise WorkspacePathError("Workspace root is not a directory")
        object.__setattr__(self, "root", root)

    def __repr__(self) -> str:
        return "Workspace(root=<configured>)"

    def resolve_existing_file(self, relative_path: str) -> Path:
        return resolve_workspace_path(self, relative_path, expected_type="file")

    def resolve_existing_directory(self, relative_path: str) -> Path:
        return resolve_workspace_path(self, relative_path, expected_type="directory")


def resolve_workspace_path(
    workspace: Workspace,
    relative_path: str,
    *,
    expected_type: Literal["file", "directory", "any"],
) -> Path:
    if type(relative_path) is not str:
        raise WorkspacePathError("Workspace path must be a string")
    _validate_relative_path_text(relative_path)
    parts = _relative_parts(relative_path)
    current = workspace.root
    for part in parts:
        current = current / part
        try:
            if current.is_symlink():
                raise WorkspacePathError("Workspace path contains a symlink")
            current = current.resolve(strict=True)
        except WorkspacePathError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise WorkspacePathError("Workspace path does not exist") from exc
        _ensure_inside(workspace.root, current)
    if expected_type == "file" and not current.is_file():
        raise WorkspacePathError("Workspace path is not a regular file")
    if expected_type == "directory" and not current.is_dir():
        raise WorkspacePathError("Workspace path is not a directory")
    return current


def workspace_relative_path(workspace: Workspace, path: Path) -> str:
    try:
        relative = path.relative_to(workspace.root)
    except ValueError as exc:
        raise WorkspacePathError("Workspace path is outside the configured root") from exc
    text = relative.as_posix()
    return "." if text == "." else text


def resolve_workspace_create_target(workspace: Workspace, relative_path: str) -> WorkspaceWriteTarget:
    if type(relative_path) is not str:
        raise WorkspacePathError("Workspace path must be a string")
    _validate_relative_path_text(relative_path)
    parts = _relative_parts(relative_path)
    if not parts:
        raise WorkspacePathError("Workspace path is not a regular file")
    parent = workspace.root
    for part in parts[:-1]:
        parent = parent / part
        try:
            if parent.is_symlink():
                raise WorkspacePathError("Workspace path contains a symlink")
            parent = parent.resolve(strict=True)
        except WorkspacePathError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            raise WorkspacePathError("Workspace path does not exist") from exc
        _ensure_inside(workspace.root, parent)
        if not parent.is_dir():
            raise WorkspacePathError("Workspace path is not a directory")
    target = parent / parts[-1]
    if target.is_symlink():
        raise WorkspacePathError("Workspace path contains a symlink")
    try:
        resolved_parent = parent.resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise WorkspacePathError("Workspace path does not exist") from exc
    _ensure_inside(workspace.root, resolved_parent)
    relative = "/".join(parts)
    return WorkspaceWriteTarget(path=target, relative_path=relative, parent=resolved_parent)


def _validate_relative_path_text(relative_path: str) -> None:
    if "\x00" in relative_path:
        raise WorkspacePathError("Workspace path is invalid")
    if relative_path in {"", "."}:
        return
    lowered = relative_path.lower()
    if lowered.startswith("file://"):
        raise WorkspacePathError("Workspace path must be relative")
    if relative_path.startswith("~"):
        raise WorkspacePathError("Workspace path must be relative")
    path = Path(relative_path)
    windows = PureWindowsPath(relative_path)
    if path.is_absolute() or windows.is_absolute() or windows.drive:
        raise WorkspacePathError("Workspace path must be relative")
    if relative_path.startswith("\\\\") or relative_path.startswith("//"):
        raise WorkspacePathError("Workspace path must be relative")


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if relative_path in {"", "."}:
        return ()
    normalized = relative_path.replace("\\", "/")
    parts: list[str] = []
    for part in normalized.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise WorkspacePathError("Workspace path is outside the configured root")
        parts.append(part)
    return tuple(parts)


def _ensure_inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError("Workspace path is outside the configured root") from exc
