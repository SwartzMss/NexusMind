"""Strict persistent configuration contracts for user-facing knowledge bases."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, replace
import json
import os
from pathlib import Path
from typing import ClassVar, TypeAlias
from uuid import NAMESPACE_URL, uuid4, uuid5


class KnowledgeBaseError(Exception):
    """Base class for controlled knowledge-base failures."""


class KnowledgeBaseConfigError(KnowledgeBaseError):
    """Raised when knowledge-base configuration is invalid."""


class KnowledgeBaseSourceError(KnowledgeBaseError):
    """Raised when a registered source cannot be used."""


class KnowledgeBasePersistenceError(KnowledgeBaseError):
    """Raised when knowledge-base persistent state cannot be accessed."""


class KnowledgeBaseClosedError(KnowledgeBaseError):
    """Raised when a closed knowledge base is used."""


_AUTO_SOURCE_ID = object()


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeBaseLimits:
    max_manifest_bytes: int = 1_000_000
    max_sources: int = 1_000
    max_knowledge_base_id_chars: int = 256
    max_display_name_chars: int = 1_024
    max_source_id_chars: int = 256
    max_path_chars: int = 32_768

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _require_non_empty_text(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise KnowledgeBaseConfigError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalFileSourceConfig:
    config_version: ClassVar[str] = "1"
    type: ClassVar[str] = "local_file"
    source_id: str = _AUTO_SOURCE_ID  # type: ignore[assignment]
    path: str
    _source_id_was_auto: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.path, "path")
        source_id_was_auto = self.source_id is _AUTO_SOURCE_ID
        if source_id_was_auto:
            object.__setattr__(
                self,
                "source_id",
                str(
                    uuid5(
                        NAMESPACE_URL,
                        f"nexusmind-source:{self.type}:{_path_identity(self.path)}",
                    )
                ),
            )
        object.__setattr__(self, "_source_id_was_auto", source_id_was_auto)
        _require_non_empty_text(self.source_id, "source_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalDirectorySourceConfig:
    config_version: ClassVar[str] = "1"
    type: ClassVar[str] = "local_directory"
    source_id: str = _AUTO_SOURCE_ID  # type: ignore[assignment]
    path: str
    _source_id_was_auto: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty_text(self.path, "path")
        source_id_was_auto = self.source_id is _AUTO_SOURCE_ID
        if source_id_was_auto:
            object.__setattr__(
                self,
                "source_id",
                str(
                    uuid5(
                        NAMESPACE_URL,
                        f"nexusmind-source:{self.type}:{_path_identity(self.path)}",
                    )
                ),
            )
        object.__setattr__(self, "_source_id_was_auto", source_id_was_auto)
        _require_non_empty_text(self.source_id, "source_id")


RegisteredSourceConfig: TypeAlias = LocalFileSourceConfig | LocalDirectorySourceConfig


def _normalized_path(path: str) -> str:
    if "\0" in path:
        raise KnowledgeBaseConfigError("source path cannot be normalized")
    try:
        return str(Path(path).resolve(strict=False))
    except (OSError, RuntimeError, ValueError) as exc:
        raise KnowledgeBaseConfigError("source path cannot be normalized") from exc


def _path_identity(path: str) -> str:
    return os.path.normcase(_normalized_path(path))


@dataclass(frozen=True, slots=True, kw_only=True)
class KnowledgeBaseManifest:
    format_version: ClassVar[str] = "2"
    knowledge_base_id: str
    display_name: str | None = None
    sources: tuple[RegisteredSourceConfig, ...] = ()
    retired_sources: tuple[RegisteredSourceConfig, ...] = ()
    limits: InitVar[KnowledgeBaseLimits | None] = None

    def __post_init__(self, limits: KnowledgeBaseLimits | None) -> None:
        active_limits = limits if limits is not None else KnowledgeBaseLimits()
        if not isinstance(active_limits, KnowledgeBaseLimits):
            raise KnowledgeBaseConfigError("limits must be KnowledgeBaseLimits")
        knowledge_base_id = _require_non_empty_text(
            self.knowledge_base_id, "knowledge_base_id"
        )
        if len(knowledge_base_id) > active_limits.max_knowledge_base_id_chars:
            raise KnowledgeBaseConfigError("knowledge_base_id exceeds configured limit")
        if self.display_name is not None:
            display_name = _require_non_empty_text(self.display_name, "display_name")
            if len(display_name) > active_limits.max_display_name_chars:
                raise KnowledgeBaseConfigError("display_name exceeds configured limit")
        if type(self.sources) is not tuple or type(self.retired_sources) is not tuple:
            raise KnowledgeBaseConfigError("source collections must be exact tuples")
        if len(self.sources) > active_limits.max_sources:
            raise KnowledgeBaseConfigError("source count exceeds configured limit")
        normalized_active: list[RegisteredSourceConfig] = []
        normalized_retired: list[RegisteredSourceConfig] = []
        seen: set[str] = set()
        for item, destination in (
            *((item, normalized_active) for item in self.sources),
            *((item, normalized_retired) for item in self.retired_sources),
        ):
            if type(item) not in (LocalFileSourceConfig, LocalDirectorySourceConfig):
                raise KnowledgeBaseConfigError("sources contain an unsupported member")
            source_id = _require_non_empty_text(item.source_id, "source_id")
            _require_non_empty_text(item.path, "path")
            if source_id in seen:
                raise KnowledgeBaseConfigError("source identifiers must be unique")
            seen.add(source_id)
            if len(source_id) > active_limits.max_source_id_chars:
                raise KnowledgeBaseConfigError("source_id exceeds configured limit")
            path = _normalized_path(item.path)
            if len(path) > active_limits.max_path_chars:
                raise KnowledgeBaseConfigError("path exceeds configured limit")
            destination.append(replace(item, path=path))
        object.__setattr__(self, "sources", tuple(sorted(normalized_active, key=lambda item: item.source_id)))
        object.__setattr__(self, "retired_sources", tuple(sorted(normalized_retired, key=lambda item: item.source_id)))


_ROOT_KEYS_V1 = frozenset({"format_version", "knowledge_base_id", "display_name", "sources"})
_ROOT_KEYS_V2 = _ROOT_KEYS_V1 | {"retired_sources"}
_SOURCE_KEYS = frozenset({"config_version", "source_id", "type", "path"})
_READ_CHUNK_BYTES = 64 * 1024


def _manifest_mapping(manifest: KnowledgeBaseManifest) -> dict[str, object]:
    if type(manifest) is not KnowledgeBaseManifest:
        raise KnowledgeBaseConfigError("manifest must be KnowledgeBaseManifest")
    return {
        "format_version": manifest.format_version,
        "knowledge_base_id": manifest.knowledge_base_id,
        "display_name": manifest.display_name,
        "sources": [
            {
                "config_version": item.config_version,
                "source_id": item.source_id,
                "type": item.type,
                "path": item.path,
            }
            for item in manifest.sources
        ],
        "retired_sources": [
            {
                "config_version": item.config_version,
                "source_id": item.source_id,
                "type": item.type,
                "path": item.path,
            }
            for item in manifest.retired_sources
        ],
    }


def encode_manifest(manifest: KnowledgeBaseManifest, limits: KnowledgeBaseLimits) -> bytes:
    """Encode a manifest into its deterministic v1 representation."""
    if not isinstance(limits, KnowledgeBaseLimits):
        raise KnowledgeBaseConfigError("limits must be KnowledgeBaseLimits")
    if type(manifest) is not KnowledgeBaseManifest:
        raise KnowledgeBaseConfigError("manifest must be KnowledgeBaseManifest")
    validated = KnowledgeBaseManifest(
        knowledge_base_id=manifest.knowledge_base_id,
        display_name=manifest.display_name,
        sources=manifest.sources,
        retired_sources=manifest.retired_sources,
        limits=limits,
    )
    data = (
        json.dumps(
            _manifest_mapping(validated),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if len(data) > limits.max_manifest_bytes:
        raise KnowledgeBaseConfigError("manifest size exceeds configured limit")
    return data


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise KnowledgeBaseConfigError("JSON object contains duplicate keys")
        result[key] = value
    return result


def _require_exact_keys(value: dict[str, object], expected: frozenset[str], layer: str) -> None:
    if frozenset(value) != expected:
        raise KnowledgeBaseConfigError(f"{layer} object has invalid keys")


def _decode_source(value: object) -> RegisteredSourceConfig:
    if type(value) is not dict:
        raise KnowledgeBaseConfigError("source entry must be an object")
    item: dict[str, object] = value
    _require_exact_keys(item, _SOURCE_KEYS, "source")
    if item["config_version"] != "1" or type(item["config_version"]) is not str:
        raise KnowledgeBaseConfigError("unsupported source config version")
    source_id = _require_non_empty_text(item["source_id"], "source_id")
    if type(item["path"]) is not str:
        raise KnowledgeBaseConfigError("path must be text")
    path = item["path"]
    if not Path(path).is_absolute() or path != _normalized_path(path):
        raise KnowledgeBaseConfigError("persisted source path must be normalized and absolute")
    source_type = item["type"]
    if type(source_type) is not str:
        raise KnowledgeBaseConfigError("source type must be text")
    source_classes = {
        LocalFileSourceConfig.type: LocalFileSourceConfig,
        LocalDirectorySourceConfig.type: LocalDirectorySourceConfig,
    }
    source_class = source_classes.get(source_type)
    if source_class is None:
        raise KnowledgeBaseConfigError("unsupported source type")
    return source_class(source_id=source_id, path=path)


def decode_manifest(data: bytes, limits: KnowledgeBaseLimits) -> KnowledgeBaseManifest:
    """Strictly decode manifest bytes, rejecting schema ambiguity."""
    if type(data) is not bytes:
        raise KnowledgeBaseConfigError("manifest data must be bytes")
    if not isinstance(limits, KnowledgeBaseLimits):
        raise KnowledgeBaseConfigError("limits must be KnowledgeBaseLimits")
    if len(data) > limits.max_manifest_bytes:
        raise KnowledgeBaseConfigError("manifest size exceeds configured limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KnowledgeBaseConfigError("manifest is not valid UTF-8") from exc
    try:
        value = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except KnowledgeBaseConfigError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise KnowledgeBaseConfigError("manifest is not valid JSON") from exc
    if type(value) is not dict:
        raise KnowledgeBaseConfigError("manifest root must be an object")
    root: dict[str, object] = value
    version = root.get("format_version")
    if version == "1":
        _require_exact_keys(root, _ROOT_KEYS_V1, "manifest")
    elif version == "2":
        _require_exact_keys(root, _ROOT_KEYS_V2, "manifest")
    else:
        raise KnowledgeBaseConfigError("unsupported manifest format version")
    if type(root["knowledge_base_id"]) is not str:
        raise KnowledgeBaseConfigError("knowledge_base_id must be text")
    if root["display_name"] is not None and type(root["display_name"]) is not str:
        raise KnowledgeBaseConfigError("display_name must be text or null")
    if type(root["sources"]) is not list:
        raise KnowledgeBaseConfigError("sources must be an array")
    if version == "2" and type(root["retired_sources"]) is not list:
        raise KnowledgeBaseConfigError("retired_sources must be an array")
    sources = tuple(_decode_source(item) for item in root["sources"])
    retired_sources = (
        tuple(_decode_source(item) for item in root["retired_sources"])
        if version == "2"
        else ()
    )
    return KnowledgeBaseManifest(
        knowledge_base_id=root["knowledge_base_id"],
        display_name=root["display_name"],
        sources=sources,
        retired_sources=retired_sources,
        limits=limits,
    )


def read_manifest(path: str | os.PathLike[str], limits: KnowledgeBaseLimits) -> KnowledgeBaseManifest:
    """Read and strictly decode a manifest file."""
    if not isinstance(limits, KnowledgeBaseLimits):
        raise KnowledgeBaseConfigError("limits must be KnowledgeBaseLimits")
    try:
        with Path(path).open("rb") as stream:
            remaining = limits.max_manifest_bytes + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = stream.read(min(remaining, _READ_CHUNK_BYTES))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
    except (OSError, TypeError, ValueError) as exc:
        raise KnowledgeBasePersistenceError("unable to read knowledge-base manifest") from exc
    return decode_manifest(data, limits)


def write_manifest(
    path: str | os.PathLike[str],
    manifest: KnowledgeBaseManifest,
    limits: KnowledgeBaseLimits,
) -> None:
    """Atomically replace a manifest using a same-directory temporary file."""
    data = encode_manifest(manifest, limits)
    try:
        destination = Path(path)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    except (TypeError, ValueError, OSError) as exc:
        raise KnowledgeBasePersistenceError("unable to write knowledge-base manifest") from exc
    created = False
    try:
        with temporary.open("xb") as stream:
            created = True
            view = memoryview(data)
            while view:
                written = stream.write(view)
                if written is None or written <= 0:
                    raise OSError("incomplete manifest write")
                view = view[written:]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        created = False
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except OSError as exc:
        raise KnowledgeBasePersistenceError("unable to write knowledge-base manifest") from exc
    finally:
        if created:
            try:
                temporary.unlink()
            except OSError:
                pass


__all__ = [
    "KnowledgeBaseClosedError",
    "KnowledgeBaseConfigError",
    "KnowledgeBaseError",
    "KnowledgeBaseLimits",
    "KnowledgeBaseManifest",
    "KnowledgeBasePersistenceError",
    "KnowledgeBaseSourceError",
    "LocalDirectorySourceConfig",
    "LocalFileSourceConfig",
    "RegisteredSourceConfig",
    "decode_manifest",
    "encode_manifest",
    "read_manifest",
    "write_manifest",
]
