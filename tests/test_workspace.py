from __future__ import annotations

from pathlib import Path

import pytest

from nexusmind.workspace import Workspace, WorkspacePathError, resolve_workspace_path


def test_workspace_root_must_exist_and_be_directory(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    assert workspace.root.is_absolute()
    assert "tmp" not in repr(workspace)

    with pytest.raises(WorkspacePathError):
        Workspace(tmp_path / "missing")

    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(WorkspacePathError):
        Workspace(file_path)


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "src/../../secret.txt",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "C:relative-path",
        "\\\\server\\share",
        "~/secret.txt",
        "file:///etc/passwd",
        "bad\x00path",
    ],
)
def test_workspace_rejects_escaping_and_special_paths(tmp_path: Path, path: str) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError):
        resolve_workspace_path(workspace, path, expected_type="any")


def test_workspace_rejects_type_mismatch(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError):
        workspace.resolve_existing_file("src")
    with pytest.raises(WorkspacePathError):
        workspace.resolve_existing_directory("src/main.py")


def test_workspace_rejects_symlink_components(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError):
        workspace.resolve_existing_directory("link")
