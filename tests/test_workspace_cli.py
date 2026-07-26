from __future__ import annotations

from pathlib import Path

import pytest

from nexusmind import cli


def test_builtin_registry_only_includes_workspace_tools_when_configured(tmp_path: Path) -> None:
    plain = cli.build_builtin_tool_registry()
    with_workspace = cli.build_builtin_tool_registry(workspace=cli.Workspace(tmp_path))
    with_write = cli.build_builtin_tool_registry(workspace=cli.Workspace(tmp_path), enable_workspace_write=True)

    assert not plain.contains("list_files")
    assert not plain.contains("read_file")
    assert not plain.contains("search_text")
    assert with_workspace.contains("list_files")
    assert with_workspace.contains("read_file")
    assert with_workspace.contains("search_text")
    assert not with_workspace.contains("write_file")
    assert not with_workspace.contains("replace_text")
    assert with_write.contains("write_file")
    assert with_write.contains("replace_text")


def test_workspace_write_requires_workspace(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["chat", "--workspace-write", "hello"])

    assert code == 2
    assert "--workspace-write requires --workspace" in capsys.readouterr().err


def test_workspace_write_requires_workspace_before_model_config(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"model_config": False}

    def load_model_config():
        called["model_config"] = True
        raise AssertionError("model config should not be loaded")

    monkeypatch.setattr(cli, "load_model_config_from_env", load_model_config)

    code = cli.main(["chat", "--workspace-write", "hello"])

    assert code == 2
    assert called["model_config"] is False
    assert "--workspace-write requires --workspace" in capsys.readouterr().err


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


def test_skill_run_requires_workspace_write_before_model_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "fix"
    skill_dir.mkdir(parents=True)
    (skill_dir / "instructions.md").write_text("Fix files.\n", encoding="utf-8")
    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "fix"
description = "Fix"
instructions_file = "instructions.md"
allowed_tools = ["builtin:write_file"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = cli.main(["skill", "run", "fix", "--skills-dir", str(skills_dir), "--workspace", str(tmp_path), "hello"])

    assert code == 2
    assert "workspace write tool references require --workspace-write" in capsys.readouterr().err


def test_workspace_approval_summary_does_not_include_content() -> None:
    summary = cli.CLIApprovalSummarizer().summarize(
        cli.ToolCall(
            id="call_1",
            name="write_file",
            arguments={"path": "safe.txt", "mode": "replace", "content": "secret-value", "expected_sha256": "a" * 64},
        ),
        cli.ToolDefinition(name="write_file"),
    )

    assert "safe.txt" in summary
    assert "secret-value" not in summary
    assert "aaaaaaaaaaaaa" not in summary


def test_workspace_approval_summary_normalizes_path_and_keeps_target(tmp_path: Path) -> None:
    workspace = cli.Workspace(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "critical.py").write_text("x", encoding="utf-8")
    long_prefix = "./" * 100

    summary = cli.CLIApprovalSummarizer(workspace=workspace).summarize(
        cli.ToolCall(
            id="call_1",
            name="write_file",
            arguments={"path": f"{long_prefix}src/critical.py", "mode": "replace", "content": "y", "expected_sha256": "b" * 64},
        ),
        cli.ToolDefinition(name="write_file"),
    )

    assert "src/critical.py" in summary
    assert str(tmp_path) not in summary


def test_workspace_approval_summary_distinguishes_same_prefix_different_basename(tmp_path: Path) -> None:
    workspace = cli.Workspace(tmp_path)
    (tmp_path / "very").mkdir()
    (tmp_path / "very" / "long").mkdir()
    prefix = "very/long/" + ("nested/" * 20)
    parent = tmp_path / "very" / "long"
    for _ in range(20):
        parent = parent / "nested"
        parent.mkdir()
    (parent / "alpha.py").write_text("x", encoding="utf-8")
    (parent / "beta.py").write_text("x", encoding="utf-8")
    summarizer = cli.CLIApprovalSummarizer(workspace=workspace)

    alpha = summarizer.summarize(
        cli.ToolCall(id="1", name="replace_text", arguments={"path": f"{prefix}alpha.py", "old_text": "x", "new_text": "y", "expected_sha256": "c" * 64}),
        cli.ToolDefinition(name="replace_text"),
    )
    beta = summarizer.summarize(
        cli.ToolCall(id="2", name="replace_text", arguments={"path": f"{prefix}beta.py", "old_text": "x", "new_text": "y", "expected_sha256": "c" * 64}),
        cli.ToolDefinition(name="replace_text"),
    )

    assert "alpha.py" in alpha
    assert "beta.py" in beta
    assert alpha != beta
