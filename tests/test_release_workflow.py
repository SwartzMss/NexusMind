from __future__ import annotations

from pathlib import Path

import pytest


WORKFLOW = Path(".github/workflows/release.yml")


def test_release_workflow_is_tag_driven_and_gated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for marker in (
        "tags:",
        '- "v*"',
        "needs: [tests, python-package, windows-portable]",
        "contents: write",
    ):
        assert marker in text


def test_release_workflow_runs_the_supported_test_matrix() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python-version: [\"3.11\", \"3.12\", \"3.13\"]" in text
    assert "python -m pytest -vv" in text


def test_release_workflow_verifies_python_metadata_and_clean_installs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for marker in (
        "GITHUB_REF_NAME",
        "pyproject.toml",
        '>=3.11,<3.14',
        "MIT",
        "python -m build",
        "venv-wheel",
        "venv-sdist",
        "python -m pip check",
        'bin/nexusmind" --help',
        "importlib.metadata",
        "nexusmind-kb",
    ):
        assert marker in text


def test_release_workflow_publishes_only_exact_verified_artifacts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for marker in (
        "nexusmind-0.1.0-py3-none-any.whl",
        "nexusmind-0.1.0.tar.gz",
        "nexusmind-windows-portable.zip",
        "scripts/build-portable.ps1",
        "gh release create",
        "--verify-tag",
        "docs/releases/v0.1.0.md",
    ):
        assert marker in text


def test_release_workflow_is_valid_yaml_when_pyyaml_is_available() -> None:
    yaml = pytest.importorskip("yaml")

    parsed = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    assert parsed["on"]["push"]["tags"] == ["v*"]
    assert set(parsed["jobs"]) == {
        "tests",
        "python-package",
        "windows-portable",
        "publish",
    }
