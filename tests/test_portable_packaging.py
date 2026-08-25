from __future__ import annotations

from pathlib import Path


def test_pyinstaller_spec_targets_desktop_entry_and_onedir() -> None:
    text = Path("packaging/nexusmind.spec").read_text(encoding="utf-8")

    assert "src/nexusmind/desktop.py" in text.replace("\\", "/")
    assert "COLLECT(" in text
    assert "console=True" in text


def test_portable_script_builds_smoke_tests_and_archives() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")

    assert "PyInstaller" in text
    assert "nexusmind.exe" in text
    assert "--help" in text
    assert "NEXUSMIND_RUNTIME_DIR" in text
    assert "Compress-Archive" in text
