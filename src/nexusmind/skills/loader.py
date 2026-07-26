from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python < 3.11
    import tomli as tomllib

from nexusmind.mcp.limits import MAX_MCP_CLIENTS_PER_GROUP

_MANIFEST_NAME = "skill.toml"
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_BUILTIN_TOOL_REF_RE = re.compile(r"^builtin:[A-Za-z][A-Za-z0-9_-]{0,63}$")
_MCP_SERVER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_MANIFEST_MAX_BYTES = 64 * 1024
_INSTRUCTIONS_MAX_BYTES = 256 * 1024
_DESCRIPTION_MAX_CHARS = 512
_MAX_SKILL_DIRS = 256
_TOP_LEVEL_FIELDS = {"schema_version", "name", "description", "instructions_file", "allowed_tools", "limits"}
_LIMIT_FIELDS = {"max_model_turns", "max_tool_calls_total"}


class SkillError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    schema_version: int
    name: str
    description: str
    source_dir: Path = field(repr=False)
    instructions: str = field(repr=False)
    allowed_tools: tuple[str, ...] = ()
    max_model_turns: int | None = None
    max_tool_calls_total: int | None = None


def load_skill(skill_dir: str | Path) -> SkillDefinition:
    source_dir = _resolve_existing_path(Path(skill_dir), "skill path")
    if not _is_dir(source_dir, "skill path"):
        raise SkillError("Skill error: skill path is not a directory")
    manifest_path = _resolve_inside(source_dir, _MANIFEST_NAME)
    manifest = _read_manifest(manifest_path)
    _validate_manifest_shape(manifest)

    instructions_file = manifest["instructions_file"]
    instructions_path = _resolve_inside(source_dir, instructions_file)
    instructions = _read_text_file(instructions_path, max_bytes=_INSTRUCTIONS_MAX_BYTES, label="instructions")
    if not instructions.strip():
        raise SkillError("Skill error: instructions must be non-empty")

    allowed_tools = tuple(manifest.get("allowed_tools", ()))
    limits = manifest.get("limits", {})
    return SkillDefinition(
        schema_version=manifest["schema_version"],
        name=manifest["name"],
        description=manifest["description"],
        source_dir=source_dir,
        instructions=instructions,
        allowed_tools=allowed_tools,
        max_model_turns=limits.get("max_model_turns"),
        max_tool_calls_total=limits.get("max_tool_calls_total"),
    )


def discover_skills(root_dir: str | Path) -> list[SkillDefinition]:
    root = _resolve_existing_path(Path(root_dir), "skills directory")
    if not _is_dir(root, "skills directory"):
        raise SkillError("Skill error: skills directory is not a directory")
    skills: list[SkillDefinition] = []
    names: set[str] = set()
    entries = _list_limited_directory(root)
    for entry in entries:
        if not _is_dir(entry, "skill path"):
            continue
        resolved_entry = _resolve_existing_path(entry, "skill path")
        _ensure_inside(root, resolved_entry)
        skill = load_skill(entry)
        if skill.name in names:
            raise SkillError(f"Skill error: duplicate skill name: {skill.name}")
        names.add(skill.name)
        skills.append(skill)
    return sorted(skills, key=lambda skill: skill.name)


def _read_manifest(path: Path) -> dict[str, object]:
    raw = _read_bytes(path, max_bytes=_MANIFEST_MAX_BYTES, label="manifest")
    if b"\x00" in raw:
        raise SkillError("Skill error: manifest contains NUL byte")
    try:
        manifest = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise SkillError("Skill error: manifest is not valid UTF-8") from None
    except tomllib.TOMLDecodeError as exc:
        raise SkillError(f"Skill error: invalid manifest TOML: {str(exc)}") from None
    if not isinstance(manifest, dict):
        raise SkillError("Skill error: manifest must be a TOML table")
    return manifest


def _validate_manifest_shape(manifest: dict[str, object]) -> None:
    unknown = set(manifest) - _TOP_LEVEL_FIELDS
    if unknown:
        raise SkillError(f"Skill error: unknown manifest field: {sorted(unknown)[0]}")
    for field_name in ("schema_version", "name", "description", "instructions_file"):
        if field_name not in manifest:
            raise SkillError(f"Skill error: missing required manifest field: {field_name}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise SkillError("Skill error: schema_version must be integer 1")
    if type(manifest["name"]) is not str or not _NAME_RE.fullmatch(manifest["name"]):
        raise SkillError("Skill error: name is invalid")
    if type(manifest["description"]) is not str or not manifest["description"].strip():
        raise SkillError("Skill error: description must be non-empty")
    if len(manifest["description"]) > _DESCRIPTION_MAX_CHARS:
        raise SkillError("Skill error: description is too long")
    if type(manifest["instructions_file"]) is not str or not manifest["instructions_file"]:
        raise SkillError("Skill error: instructions_file must be a non-empty string")
    _validate_allowed_tools(manifest.get("allowed_tools", []))
    _validate_limits(manifest.get("limits", {}))


def _validate_allowed_tools(value: object) -> None:
    if not isinstance(value, list):
        raise SkillError("Skill error: allowed_tools must be a list")
    seen: set[str] = set()
    mcp_server_ids: set[str] = set()
    for item in value:
        if type(item) is not str or not _valid_tool_reference(item):
            raise SkillError("Skill error: allowed_tools contains an invalid tool reference")
        if item in seen:
            raise SkillError(f"Skill error: duplicate tool reference: {item}")
        seen.add(item)
        if item.startswith("mcp:"):
            _, server_id, _ = item.split(":", 2)
            mcp_server_ids.add(server_id)
            if len(mcp_server_ids) > MAX_MCP_CLIENTS_PER_GROUP:
                raise SkillError("Skill error: too many MCP servers referenced")


def _validate_limits(value: object) -> None:
    if not isinstance(value, dict):
        raise SkillError("Skill error: limits must be a table")
    unknown = set(value) - _LIMIT_FIELDS
    if unknown:
        raise SkillError(f"Skill error: unknown limit field: {sorted(unknown)[0]}")
    for field_name, limit in value.items():
        if type(limit) is not int or limit <= 0:
            raise SkillError(f"Skill error: {field_name} must be a positive integer")


def _resolve_inside(root: Path, relative: str) -> Path:
    if "\x00" in relative:
        raise SkillError("Skill error: skill file path contains NUL byte")
    candidate = _resolve_existing_path(root / relative, "skill file")
    _ensure_inside(root, candidate)
    if not _is_file(candidate, "skill file"):
        raise SkillError("Skill error: skill file is not a regular file")
    return candidate


def _read_text_file(path: Path, *, max_bytes: int, label: str) -> str:
    raw = _read_bytes(path, max_bytes=max_bytes, label=label)
    if b"\x00" in raw:
        raise SkillError(f"Skill error: {label} contains NUL byte")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillError(f"Skill error: {label} is not valid UTF-8") from None


def _read_bytes(path: Path, *, max_bytes: int, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise SkillError(f"Skill error: could not read {label} file") from exc
    if len(raw) > max_bytes:
        raise SkillError(f"Skill error: {label} file is too large")
    return raw


def _resolve_existing_path(path: Path, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SkillError(f"Skill error: {label} does not exist or is not accessible") from exc


def _list_limited_directory(root: Path) -> list[Path]:
    entries: list[Path] = []
    try:
        for entry in root.iterdir():
            entries.append(entry)
            if len(entries) > _MAX_SKILL_DIRS:
                raise SkillError("Skill error: too many entries in skills directory")
    except SkillError:
        raise
    except OSError as exc:
        raise SkillError("Skill error: could not read skills directory") from exc
    return sorted(entries, key=lambda path: path.name)


def _ensure_inside(root: Path, candidate: Path) -> None:
    try:
        candidate.relative_to(root)
    except ValueError:
        raise SkillError("Skill error: skill file escapes its directory") from None


def _is_dir(path: Path, label: str) -> bool:
    try:
        return path.is_dir()
    except OSError as exc:
        raise SkillError(f"Skill error: could not inspect {label}") from exc


def _is_file(path: Path, label: str) -> bool:
    try:
        return path.is_file()
    except OSError as exc:
        raise SkillError(f"Skill error: could not inspect {label}") from exc


def _valid_tool_reference(reference: str) -> bool:
    if _BUILTIN_TOOL_REF_RE.fullmatch(reference):
        return True
    if not reference.startswith("mcp:"):
        return False
    parts = reference.split(":", 2)
    if len(parts) != 3:
        return False
    _, server_id, remote_ref = parts
    if not _MCP_SERVER_ID_RE.fullmatch(server_id) or not remote_ref:
        return False
    return not _CONTROL_CHARS_RE.search(remote_ref)
