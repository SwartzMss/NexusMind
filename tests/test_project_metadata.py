from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - test runner compatibility
    import tomli as tomllib


def test_supported_python_range_matches_deterministic_unicode_policy() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.11,<3.14"
