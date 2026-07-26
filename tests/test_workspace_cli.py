from __future__ import annotations

from pathlib import Path

import pytest

from nexusmind import cli


def test_builtin_registry_only_includes_workspace_tools_when_configured(tmp_path: Path) -> None:
    plain = cli.build_builtin_tool_registry()
    with_workspace = cli.build_builtin_tool_registry(workspace=cli.Workspace(tmp_path))

    assert not plain.contains("list_files")
    assert not plain.contains("read_file")
    assert not plain.contains("search_text")
    assert with_workspace.contains("list_files")
    assert with_workspace.contains("read_file")
    assert with_workspace.contains("search_text")


def test_skill_run_requires_workspace_before_model_config(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "instructions.md").write_text("Review files.\n", encoding="utf-8")
    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "review"
description = "Review"
instructions_file = "instructions.md"
allowed_tools = ["builtin:read_file"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = cli.main(["skill", "run", "review", "--skills-dir", str(skills_dir), "hello"])

    assert code == 2
    assert "workspace tool references require --workspace" in capsys.readouterr().err
