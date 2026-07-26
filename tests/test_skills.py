import os
from pathlib import Path

from nexusmind.runtime.chat import AgentLoopLimits
from nexusmind.skills.loader import SkillDefinition, SkillError, discover_skills, load_skill
from nexusmind.skills.resolver import build_skill_loop_limits, resolve_skill_tool_references
from nexusmind.tools.builtin import EchoTool
from nexusmind.tools.contracts import ToolDefinition, ToolRiskLevel
from nexusmind.tools.registry import ToolRegistry


def _write_skill(root: Path, name: str = "code-review", *, allowed_tools='["builtin:echo"]', limits: str = "") -> Path:
    skill_dir = root / name
    skill_dir.mkdir()
    (skill_dir / "skill.toml").write_text(
        f"""
schema_version = 1
name = "{name}"
description = "Review code"
instructions_file = "instructions.md"
allowed_tools = {allowed_tools}
{limits}
""".strip(),
        encoding="utf-8",
    )
    (skill_dir / "instructions.md").write_text("Review carefully.", encoding="utf-8")
    return skill_dir


def test_load_skill_returns_immutable_snapshot_without_repr_instructions(tmp_path) -> None:
    skill_dir = _write_skill(tmp_path)

    skill = load_skill(skill_dir)

    assert skill.name == "code-review"
    assert skill.instructions == "Review carefully."
    assert skill.allowed_tools == ("builtin:echo",)
    assert "Review carefully" not in repr(skill)
    try:
        skill.allowed_tools += ("builtin:approval_demo",)
    except Exception:
        pass
    else:
        raise AssertionError("expected immutable SkillDefinition")


def test_load_skill_rejects_unknown_fields_and_duplicate_tools(tmp_path) -> None:
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "code-review"
description = "Review code"
instructions_file = "instructions.md"
allowed_tools = ["builtin:echo", "builtin:echo"]
risk_level = "read_only"
""".strip(),
        encoding="utf-8",
    )

    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "unknown manifest field" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_load_skill_wraps_missing_files_and_malformed_toml(tmp_path) -> None:
    missing = tmp_path / "missing"
    try:
        load_skill(missing)
    except SkillError as exc:
        assert "Skill error:" in str(exc)
    else:
        raise AssertionError("expected SkillError")

    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "Skill error:" in str(exc)
    else:
        raise AssertionError("expected SkillError")

    (skill_dir / "skill.toml").write_text("schema_version = [", encoding="utf-8")
    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "invalid manifest TOML" in str(exc)
        assert "AttributeError" not in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_load_skill_wraps_missing_instructions(tmp_path) -> None:
    skill_dir = tmp_path / "missing-instructions"
    skill_dir.mkdir()
    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "missing-instructions"
description = "Review code"
instructions_file = "instructions.md"
allowed_tools = []
""".strip(),
        encoding="utf-8",
    )

    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "Skill error:" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_load_skill_rejects_nul_instruction_path_as_skill_error(tmp_path) -> None:
    skill_dir = tmp_path / "nul-path"
    skill_dir.mkdir()
    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "nul-path"
description = "Review code"
instructions_file = "\\u0000"
allowed_tools = []
""".strip(),
        encoding="utf-8",
    )

    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "NUL" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_load_skill_rejects_path_escape_and_invalid_utf8(tmp_path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    skill_dir = _write_skill(tmp_path)
    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "code-review"
description = "Review code"
instructions_file = "../outside.md"
allowed_tools = []
""".strip(),
        encoding="utf-8",
    )
    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("expected SkillError")

    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "code-review"
description = "Review code"
instructions_file = "instructions.md"
allowed_tools = []
""".strip(),
        encoding="utf-8",
    )
    (skill_dir / "instructions.md").write_bytes(b"\xff")
    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "valid UTF-8" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_load_skill_rejects_symlink_escape(tmp_path) -> None:
    if not hasattr(os, "symlink"):
        return
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    skill_dir = _write_skill(tmp_path)
    link = skill_dir / "link.md"
    try:
        os.symlink(outside, link)
    except OSError:
        return
    (skill_dir / "skill.toml").write_text(
        """
schema_version = 1
name = "code-review"
description = "Review code"
instructions_file = "link.md"
allowed_tools = []
""".strip(),
        encoding="utf-8",
    )

    try:
        load_skill(skill_dir)
    except SkillError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_discover_skills_rejects_missing_root_too_many_entries_and_symlink_escape(tmp_path) -> None:
    try:
        discover_skills(tmp_path / "missing")
    except SkillError as exc:
        assert "Skill error:" in str(exc)
    else:
        raise AssertionError("expected SkillError")

    many = tmp_path / "many"
    many.mkdir()
    for index in range(257):
        (many / f"entry-{index}").mkdir()
    try:
        discover_skills(many)
    except SkillError as exc:
        assert "too many entries" in str(exc)
    else:
        raise AssertionError("expected SkillError")

    if not hasattr(os, "symlink"):
        return
    root = tmp_path / "root"
    outside_root = tmp_path / "outside-root"
    root.mkdir()
    outside_root.mkdir()
    _write_skill(outside_root, "external")
    try:
        os.symlink(outside_root / "external", root / "external")
    except OSError:
        return
    try:
        discover_skills(root)
    except SkillError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_discover_skills_sorts_and_rejects_duplicate_names(tmp_path) -> None:
    _write_skill(tmp_path, "zeta")
    duplicate = _write_skill(tmp_path, "alpha")
    (duplicate / "skill.toml").write_text(
        (duplicate / "skill.toml").read_text(encoding="utf-8").replace('name = "alpha"', 'name = "zeta"'),
        encoding="utf-8",
    )

    try:
        discover_skills(tmp_path)
    except SkillError as exc:
        assert "duplicate skill name" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_resolve_builtin_tool_reference_and_limits(tmp_path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        limits="[limits]\nmax_model_turns = 4\nmax_tool_calls_total = 8",
    )
    skill = load_skill(skill_dir)
    registry = ToolRegistry()
    registry.register(EchoTool())

    tools = resolve_skill_tool_references(skill, registry)
    limits = build_skill_loop_limits(skill, AgentLoopLimits())

    assert [tool.name for tool in tools] == ["echo"]
    assert limits.max_model_turns == 4
    assert limits.max_tool_calls_total == 8
    assert limits.max_json_depth == AgentLoopLimits().max_json_depth


def test_skill_allows_mcp_remote_names_that_adapter_can_normalize(tmp_path) -> None:
    for remote_name in ("filesystem.read", "repo/search", "1password_lookup", "namespace:read", "report%2Fread"):
        skill_name = f"skill-{remote_name.replace('.', '-').replace('/', '-').replace('_', '-')}"
        skill_name = skill_name.replace(":", "-").replace("%", "-").lower()
        skill_dir = _write_skill(tmp_path, skill_name, allowed_tools=f'["mcp:demo:{remote_name}"]')
        skill = load_skill(skill_dir)
        assert skill.allowed_tools == (f"mcp:demo:{remote_name}",)


def test_resolver_preserves_percent_sequences_in_mcp_remote_names(tmp_path) -> None:
    from nexusmind.mcp.naming import mcp_tool_local_name

    class LiteralPercentTool:
        @property
        def definition(self):
            return ToolDefinition(
                name=mcp_tool_local_name("demo", "report%2Fread"),
                input_schema={"type": "object", "properties": {}},
                risk_level=ToolRiskLevel.UNSPECIFIED,
            )

        async def invoke(self, arguments):
            return {}

    skill = SkillDefinition(
        1,
        "x",
        "desc",
        tmp_path,
        "instructions",
        ("mcp:demo:report%2Fread",),
    )
    registry = ToolRegistry()
    registry.register(LiteralPercentTool())

    assert [tool.name for tool in resolve_skill_tool_references(skill, registry)] == [
        mcp_tool_local_name("demo", "report%2Fread")
    ]


def test_limits_cannot_expand_host_defaults(tmp_path) -> None:
    skill_dir = _write_skill(tmp_path, limits="[limits]\nmax_model_turns = 999")
    skill = load_skill(skill_dir)

    try:
        build_skill_loop_limits(skill, AgentLoopLimits())
    except SkillError as exc:
        assert "exceeds host default" in str(exc)
    else:
        raise AssertionError("expected SkillError")


def test_missing_tool_reference_fails_without_arguments(tmp_path) -> None:
    skill = SkillDefinition(
        1,
        "x",
        "desc",
        tmp_path,
        "secret instructions",
        ("builtin:missing",),
    )
    try:
        resolve_skill_tool_references(skill, ToolRegistry())
    except SkillError as exc:
        assert "secret instructions" not in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected SkillError")
