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


def test_project_exposes_desktop_entry_and_packaging_extra() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert project["scripts"]["nexusmind"] == "nexusmind.desktop:main"
    assert any(requirement.startswith("pyinstaller") for requirement in project["optional-dependencies"]["packaging"])


def test_readme_documents_windows_portable_runtime() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    for required in (
        "NEXUSMIND_RUNTIME_DIR",
        ".nexusmind\\",
        "nexusmind.log",
        "build-portable.ps1",
        "nexusmind-windows-portable.zip",
        "不会自动迁移",
        "删除整个解压目录",
    ):
        assert required in readme
    assert "%USERPROFILE%\\.nexusmind\\" not in readme


def test_readme_documents_path_only_knowledge_base_creation() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    assert "nexusmind create ./security-kb\n" in readme
    assert '--name "Security Notes"' not in readme
    assert 'display_name="Security Notes"' not in readme


def test_search_and_diagnostics_ranking_contract_is_documented() -> None:
    root = Path(__file__).parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    architecture = (root / "docs" / "architecture.md").read_text(encoding="utf-8")

    assert "document-aware" in readme
    assert "raw backend ranking" in readme
    assert "document-aware" in architecture
    assert "diagnose" in architecture
    assert "raw backend ranking" in architecture
    assert "backend capacity" in architecture
    assert "median absolute deviation" in architecture


def test_diversification_benchmark_uses_lf_on_every_platform() -> None:
    attributes = (Path(__file__).parents[1] / ".gitattributes").read_text(
        encoding="utf-8"
    ).splitlines()

    assert "evals/knowledge/diversification/corpus/*.md text eol=lf" in attributes
    assert "evals/knowledge/diversification.md text eol=lf" in attributes
