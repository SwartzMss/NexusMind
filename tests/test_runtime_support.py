from __future__ import annotations

from pathlib import Path

import pytest

from nexusmind.runtime_support import RuntimeLayoutError, create_runtime_layout, resolve_runtime_root


def test_resolve_runtime_root_uses_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEXUSMIND_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert resolve_runtime_root() == tmp_path / ".nexusmind"


def test_resolve_runtime_root_accepts_absolute_override(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "managed"
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", str(root))

    assert resolve_runtime_root() == root


def test_resolve_runtime_root_rejects_relative_override(monkeypatch) -> None:
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", "relative/runtime")

    with pytest.raises(RuntimeLayoutError, match="absolute"):
        resolve_runtime_root()


def test_create_runtime_layout_creates_stable_directories(tmp_path: Path) -> None:
    layout = create_runtime_layout(tmp_path / "runtime")

    assert layout.root == tmp_path / "runtime"
    assert layout.data_dir.is_dir()
    assert layout.logs_dir.is_dir()
    assert layout.config_dir.is_dir()
    assert layout.models_dir.is_dir()
    assert layout.log_file == layout.logs_dir / "nexusmind.log"
