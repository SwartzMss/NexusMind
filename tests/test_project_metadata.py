from __future__ import annotations

from pathlib import Path
import tomllib


def test_supported_python_range_matches_deterministic_unicode_policy() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == ">=3.11,<3.14"


def test_project_exposes_only_the_supported_cli_entrypoint() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert project["scripts"] == {"nexusmind": "nexusmind.runtime_entrypoint:main"}
    assert any(requirement.startswith("pyinstaller") for requirement in project["optional-dependencies"]["packaging"])
    assert "firecrawl-anydoc==0.2.4" in project["dependencies"]


def test_current_product_contract_has_no_removed_ui_references() -> None:
    root = Path(__file__).parents[1]
    current_contract_files = (
        "README.md",
        "docs/architecture.md",
        "docs/releases/v0.1.0.md",
        "pyproject.toml",
        "requirements.txt",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        "packaging/nexusmind.spec",
    )
    forbidden_references = ("knowledge_base_ui", "nexusmind-kb", "tkinter")

    for relative_path in current_contract_files:
        text = (root / relative_path).read_text(encoding="utf-8")
        assert not any(reference in text for reference in forbidden_references), relative_path


def test_readme_documents_windows_portable_runtime() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    for required in (
        "NEXUSMIND_RUNTIME_DIR",
        "独立的 `.nexusmind`",
        "logs\\nexusmind.log",
        "build-portable.ps1",
        "nexusmind-windows-portable.zip",
        "不执行自动迁移或修复",
        "没有公开 GitHub Release",
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
