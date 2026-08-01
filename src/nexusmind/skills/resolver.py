from __future__ import annotations

from dataclasses import replace

from nexusmind.mcp.limits import MAX_MCP_CLIENTS_PER_GROUP
from nexusmind.mcp.naming import mcp_tool_local_name
from nexusmind.runtime.chat import AgentLoopLimits
from nexusmind.skills.loader import SkillDefinition, SkillError
from nexusmind.tools.contracts import ToolDefinition
from nexusmind.tools.registry import ToolRegistry

MAX_MCP_SERVERS_PER_SKILL = MAX_MCP_CLIENTS_PER_GROUP
WORKSPACE_READ_TOOL_REFERENCES = frozenset(
    {
        "builtin:list_files",
        "builtin:read_file",
        "builtin:search_text",
    }
)
WORKSPACE_WRITE_TOOL_REFERENCES = frozenset(
    {
        "builtin:write_file",
        "builtin:replace_text",
    }
)
WORKSPACE_EXEC_TOOL_REFERENCES = frozenset({"builtin:run_command"})
WORKSPACE_TOOL_REFERENCES = WORKSPACE_READ_TOOL_REFERENCES | WORKSPACE_WRITE_TOOL_REFERENCES | WORKSPACE_EXEC_TOOL_REFERENCES


def resolve_skill_tool_references(skill: SkillDefinition, registry: ToolRegistry) -> list[ToolDefinition]:
    resolved: list[ToolDefinition] = []
    local_names: set[str] = set()
    for reference in skill.allowed_tools:
        local_name = _local_name_for_reference(reference)
        if local_name in local_names:
            raise SkillError("Skill error: duplicate resolved tool reference")
        if not registry.contains(local_name):
            raise SkillError(f"Skill error: tool reference not found: {reference}")
        definition = registry.definition(local_name)
        if definition.name != local_name:
            raise SkillError("Skill error: resolved tool definition is inconsistent")
        local_names.add(local_name)
        resolved.append(definition)
    return resolved


def validate_builtin_skill_tool_references(skill: SkillDefinition, registry: ToolRegistry) -> None:
    seen: set[str] = set()
    for reference in skill.allowed_tools:
        if not reference.startswith("builtin:"):
            continue
        local_name = reference.removeprefix("builtin:")
        if local_name in seen:
            raise SkillError("Skill error: duplicate resolved tool reference")
        if not registry.contains(local_name):
            raise SkillError(f"Skill error: tool reference not found: {reference}")
        seen.add(local_name)


def build_skill_loop_limits(skill: SkillDefinition, base: AgentLoopLimits | None = None) -> AgentLoopLimits:
    base = base or AgentLoopLimits()
    if skill.max_model_turns is not None and skill.max_model_turns > base.max_model_turns:
        raise SkillError("Skill error: max_model_turns exceeds host default")
    if skill.max_tool_calls_total is not None and skill.max_tool_calls_total > base.max_tool_calls_total:
        raise SkillError("Skill error: max_tool_calls_total exceeds host default")
    return replace(
        base,
        max_model_turns=skill.max_model_turns or base.max_model_turns,
        max_tool_calls_total=skill.max_tool_calls_total or base.max_tool_calls_total,
    )


def skill_requires_mcp(skill: SkillDefinition) -> bool:
    return any(reference.startswith("mcp:") for reference in skill.allowed_tools)


def skill_requires_workspace(skill: SkillDefinition) -> bool:
    return any(reference in WORKSPACE_TOOL_REFERENCES for reference in skill.allowed_tools)


def skill_requires_workspace_write(skill: SkillDefinition) -> bool:
    return any(reference in WORKSPACE_WRITE_TOOL_REFERENCES for reference in skill.allowed_tools)


def skill_requires_workspace_exec(skill: SkillDefinition) -> bool:
    return any(reference in WORKSPACE_EXEC_TOOL_REFERENCES for reference in skill.allowed_tools)


def skill_mcp_server_ids(skill: SkillDefinition) -> tuple[str, ...]:
    server_ids: set[str] = set()
    for reference in skill.allowed_tools:
        if reference.startswith("mcp:"):
            _, server_id, _ = reference.split(":", 2)
            server_ids.add(server_id)
            if len(server_ids) > MAX_MCP_SERVERS_PER_SKILL:
                raise SkillError("Skill error: too many MCP servers referenced")
    return tuple(sorted(server_ids))


def validate_skill_mcp_server_id(skill: SkillDefinition, server_id: str) -> None:
    server_ids = skill_mcp_server_ids(skill)
    mismatched = sorted(item for item in server_ids if item != server_id)
    if mismatched:
        raise SkillError(f"Skill error: MCP tool reference uses a different server_id: {mismatched[0]}")


def _local_name_for_reference(reference: str) -> str:
    if reference.startswith("builtin:"):
        return reference.removeprefix("builtin:")
    if reference.startswith("mcp:"):
        _, server_id, remote_name = reference.split(":", 2)
        return mcp_tool_local_name(server_id, remote_name)
    raise SkillError("Skill error: unsupported tool reference")
