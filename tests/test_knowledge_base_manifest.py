from __future__ import annotations

import json
import io
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import nexusmind.knowledge_base_manifest as manifest_module

from nexusmind import (
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBaseError,
    KnowledgeBaseLimits,
    KnowledgeBaseManifest,
    KnowledgeBasePersistenceError,
    KnowledgeBaseSourceError,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
)
from nexusmind.knowledge_base_manifest import (
    decode_manifest,
    encode_manifest,
    read_manifest,
    write_manifest,
)


LIMIT_DEFAULTS = {
    "max_manifest_bytes": 1_000_000,
    "max_sources": 1_000,
    "max_knowledge_base_id_chars": 256,
    "max_source_id_chars": 256,
    "max_path_chars": 32_768,
}

ABSOLUTE_BASE = Path.cwd().resolve() / "manifest-test-data"


def source(
    name: str = "source", path: str | None = None
) -> LocalFileSourceConfig:
    return LocalFileSourceConfig(
        path=str(ABSOLUTE_BASE / name) if path is None else path
    )


def manifest(**changes: object) -> KnowledgeBaseManifest:
    values: dict[str, object] = {
        "knowledge_base_id": "kb",
        "sources": (),
    }
    values.update(changes)
    return KnowledgeBaseManifest(**values)  # type: ignore[arg-type]


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def test_public_error_hierarchy_and_default_limits() -> None:
    assert issubclass(KnowledgeBaseConfigError, KnowledgeBaseError)
    assert issubclass(KnowledgeBaseSourceError, KnowledgeBaseError)
    assert issubclass(KnowledgeBasePersistenceError, KnowledgeBaseError)
    assert issubclass(KnowledgeBaseClosedError, KnowledgeBaseError)
    assert {
        field: getattr(KnowledgeBaseLimits(), field)
        for field in KnowledgeBaseLimits.__dataclass_fields__
    } == LIMIT_DEFAULTS


@pytest.mark.parametrize("field", LIMIT_DEFAULTS)
@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "1"])
def test_limits_require_positive_plain_integers(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        KnowledgeBaseLimits(**{field: value})


@pytest.mark.parametrize("source_type", [LocalFileSourceConfig, LocalDirectorySourceConfig])
def test_source_contracts_are_frozen_and_own_fixed_discriminators(source_type: type) -> None:
    config = source_type(path="relative/path")
    assert config.config_version == "1"
    assert config.type == ("local_file" if source_type is LocalFileSourceConfig else "local_directory")
    with pytest.raises(FrozenInstanceError):
        config.path = "elsewhere"
    with pytest.raises(TypeError):
        source_type(path="/tmp", config_version="2")
    with pytest.raises(TypeError):
        source_type(path="/tmp", type="wrong")
    with pytest.raises(TypeError):
        source_type(source_id="docs", path="/tmp")


@pytest.mark.parametrize("source_type", [LocalFileSourceConfig, LocalDirectorySourceConfig])
def test_automatic_source_id_is_stable_for_normalized_path(source_type: type) -> None:
    direct = source_type(path=str(ABSOLUTE_BASE / "source"))
    equivalent = source_type(path=str(ABSOLUTE_BASE / "missing" / ".." / "source"))

    assert direct.source_id == equivalent.source_id


def test_automatic_source_id_distinguishes_file_and_directory_types() -> None:
    path = str(ABSOLUTE_BASE / "source")

    assert LocalFileSourceConfig(path=path).source_id != LocalDirectorySourceConfig(
        path=path
    ).source_id


def test_automatic_source_id_uses_platform_path_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        manifest_module.os.path,
        "normcase",
        lambda path: path.replace("/", "\\").lower(),
    )

    upper = LocalFileSourceConfig(path=str(ABSOLUTE_BASE / "Future.md"))
    lower = LocalFileSourceConfig(path=str(ABSOLUTE_BASE / "future.md"))

    assert upper.source_id == lower.source_id


@pytest.mark.parametrize("source_type", [LocalFileSourceConfig, LocalDirectorySourceConfig])
@pytest.mark.parametrize("path", ["", " ", "\t", "\r\n", 1, None, b"/tmp"])
def test_source_path_must_be_non_empty_text(source_type: type, path: object) -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        source_type(path=path)  # type: ignore[arg-type]


@pytest.mark.parametrize("knowledge_base_id", ["", " ", "\t", "\r\n", 1, None])
def test_manifest_id_must_be_non_empty_text(knowledge_base_id: object) -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(knowledge_base_id=knowledge_base_id)


def test_manifest_requires_exact_tuple_supported_unique_sources_and_sorts() -> None:
    a = source("a", str(ABSOLUTE_BASE / "a"))
    b = LocalDirectorySourceConfig(path=str(ABSOLUTE_BASE / "b"))
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(sources=[a])
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(sources=(object(),))
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(sources=(a, a))
    with pytest.raises(KnowledgeBaseConfigError, match="source paths must be unique"):
        manifest(
            sources=(
                LocalFileSourceConfig(path=str(ABSOLUTE_BASE / "shared")),
                LocalDirectorySourceConfig(path=str(ABSOLUTE_BASE / "shared")),
            )
        )
    value = manifest(sources=(b, a))
    assert set(value.sources) == {a, b}
    assert tuple(item.source_id for item in value.sources) == tuple(
        sorted((a.source_id, b.source_id))
    )
def test_manifest_normalizes_paths_and_enforces_all_configured_bounds(tmp_path: Path) -> None:
    relative = LocalFileSourceConfig(path="relative/../file.txt")
    value = KnowledgeBaseManifest(
        knowledge_base_id="kb",
        sources=(relative,),
        limits=KnowledgeBaseLimits(max_sources=1, max_path_chars=32_768),
    )
    assert Path(value.sources[0].path).is_absolute()
    assert value.sources[0].path == str(Path(relative.path).resolve(strict=False))

    cases = [
        ({"knowledge_base_id": "xx"}, KnowledgeBaseLimits(max_knowledge_base_id_chars=1)),
        ({"sources": (source("xx"),)}, KnowledgeBaseLimits(max_source_id_chars=35)),
        ({"sources": (source(path="/too-long"),)}, KnowledgeBaseLimits(max_path_chars=1)),
        ({"sources": (source("a"), source("b"))}, KnowledgeBaseLimits(max_sources=1)),
    ]
    for changes, limits in cases:
        with pytest.raises(KnowledgeBaseConfigError):
            KnowledgeBaseManifest(limits=limits, **({"knowledge_base_id": "kb", "sources": ()} | changes))


def test_codec_is_exact_utf8_deterministic_and_order_independent() -> None:
    limits = KnowledgeBaseLimits()
    empty = manifest()
    assert encode_manifest(empty, limits) == (
        b'{"format_version":"1","knowledge_base_id":"kb","sources":[]}\n'
    )
    assert decode_manifest(encode_manifest(empty, limits), limits) == empty

    a = source("a", str(ABSOLUTE_BASE / "\u4e2d"))
    b = LocalDirectorySourceConfig(path=str(ABSOLUTE_BASE / "b"))
    first = manifest(sources=(b, a))
    second = manifest(sources=(a, b))
    assert encode_manifest(first, limits) == encode_manifest(second, limits)
    assert b"\\u" not in encode_manifest(first, limits)
    encoded_sources = []
    for item in first.sources:
        encoded_sources.append(
            '{"config_version":"1","path":%s,"source_id":"%s","type":"%s"}'
            % (json.dumps(item.path, ensure_ascii=False), item.source_id, item.type)
        )
    exact_two_source_json = (
        '{"format_version":"1","knowledge_base_id":"kb","sources":['
        + ",".join(encoded_sources)
        + "]}\n"
    ).encode("utf-8")
    assert encode_manifest(first, limits) == exact_two_source_json
    assert decode_manifest(encode_manifest(first, limits), limits) == first


def test_encode_rederives_source_id_from_type_and_normalized_path() -> None:
    limits = KnowledgeBaseLimits()
    invalid_source = source()
    object.__setattr__(invalid_source, "source_id", "tampered")
    invalid_manifest = manifest()
    object.__setattr__(invalid_manifest, "sources", (invalid_source,))

    decoded = decode_manifest(encode_manifest(invalid_manifest, limits), limits)

    assert decoded.sources == (source(),)


@pytest.mark.parametrize(
    "data",
    [
        b"\xff",
        b"not-json",
        b"[]",
        b"null",
    ],
)
def test_decode_rejects_invalid_utf8_json_and_non_object_roots(data: bytes) -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        decode_manifest(data, KnowledgeBaseLimits())


def test_decode_checks_byte_limit_before_decoding() -> None:
    with pytest.raises(KnowledgeBaseConfigError, match="size"):
        decode_manifest(b"\xff\xff", KnowledgeBaseLimits(max_manifest_bytes=1))


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"remove": "knowledge_base_id"},
        {"display_name": None},
        {"format_version": "3"},
        {"format_version": 1},
        {"knowledge_base_id": 1},
        {"sources": None},
        {"sources": {}},
    ],
)
def test_decode_rejects_root_schema_violations(change: dict[str, object]) -> None:
    value: dict[str, object] = {
        "format_version": "1",
        "knowledge_base_id": "kb",
        "sources": [],
    }
    removed = change.pop("remove", None)
    if removed:
        del value[str(removed)]
    value.update(change)
    with pytest.raises(KnowledgeBaseConfigError):
        decode_manifest(encoded(value), KnowledgeBaseLimits())


def test_decode_rejects_source_id_not_derived_from_type_and_path() -> None:
    value = {
        "format_version": "1",
        "knowledge_base_id": "kb",
        "sources": [
            {
                "config_version": "1",
                "source_id": "manual",
                "type": "local_file",
                "path": str(ABSOLUTE_BASE / "source"),
            }
        ],
    }

    with pytest.raises(KnowledgeBaseConfigError, match="source_id"):
        decode_manifest(encoded(value), KnowledgeBaseLimits())


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"remove": "path"},
        {"config_version": "2"},
        {"config_version": 1},
        {"source_id": 1},
        {"source_id": ""},
        {"type": "remote"},
        {"type": 1},
        {"path": 1},
        {"path": ""},
        {"path": "relative"},
    ],
)
def test_decode_rejects_source_schema_violations(change: dict[str, object]) -> None:
    path = str(ABSOLUTE_BASE / "docs")
    item: dict[str, object] = {
        "config_version": "1",
        "source_id": LocalFileSourceConfig(path=path).source_id,
        "type": "local_file",
        "path": path,
    }
    removed = change.pop("remove", None)
    if removed:
        del item[str(removed)]
    item.update(change)
    root = {"format_version": "1", "knowledge_base_id": "kb", "sources": [item]}
    with pytest.raises(KnowledgeBaseConfigError):
        decode_manifest(encoded(root), KnowledgeBaseLimits())


def test_decode_rejects_duplicate_source_ids_and_oversized_decoded_values() -> None:
    path = str(ABSOLUTE_BASE / "a")
    item = {"config_version": "1", "source_id": LocalFileSourceConfig(path=path).source_id, "type": "local_file", "path": path}
    root = {"format_version": "1", "knowledge_base_id": "kb", "sources": [item, item]}
    with pytest.raises(KnowledgeBaseConfigError):
        decode_manifest(encoded(root), KnowledgeBaseLimits())
    with pytest.raises(KnowledgeBaseConfigError):
        decode_manifest(encoded(root | {"knowledge_base_id": "xx", "sources": []}), KnowledgeBaseLimits(max_knowledge_base_id_chars=1))


def test_encode_checks_byte_limit() -> None:
    with pytest.raises(KnowledgeBaseConfigError, match="size"):
        encode_manifest(manifest(), KnowledgeBaseLimits(max_manifest_bytes=1))


def test_encode_revalidates_manifest_against_the_supplied_limits() -> None:
    value = manifest(knowledge_base_id="long")
    with pytest.raises(KnowledgeBaseConfigError):
        encode_manifest(value, KnowledgeBaseLimits(max_knowledge_base_id_chars=1))


def test_path_normalization_failures_are_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_resolve(self: Path, *, strict: bool = False) -> Path:
        raise OSError("private path detail")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    with pytest.raises(KnowledgeBaseConfigError) as raised:
        manifest(sources=(source(),))
    assert "private path detail" not in str(raised.value)


def test_embedded_nul_path_is_controlled_during_construction_and_decode() -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(sources=(source(path="bad\0path"),))

    item = {
        "config_version": "1",
        "source_id": "docs",
        "type": "local_file",
        "path": str(ABSOLUTE_BASE) + "\0bad",
    }
    root = {
        "format_version": "1",
        "knowledge_base_id": "kb",
        "sources": [item],
    }
    with pytest.raises(KnowledgeBaseConfigError):
        decode_manifest(encoded(root), KnowledgeBaseLimits())


def test_read_manifest_reads_at_most_limit_plus_one_byte(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[int] = []

    class TrackingBytesIO(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested.append(size)
            return super().read(size)

    stream = TrackingBytesIO(b"x" * 20)
    monkeypatch.setattr(Path, "open", lambda self, mode: stream)
    with pytest.raises(KnowledgeBaseConfigError, match="size"):
        read_manifest("ignored", KnowledgeBaseLimits(max_manifest_bytes=10))
    assert requested == [11]


def test_read_manifest_handles_very_large_valid_byte_limit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    value = manifest()
    path.write_bytes(encode_manifest(value, KnowledgeBaseLimits()))

    assert read_manifest(
        path, KnowledgeBaseLimits(max_manifest_bytes=sys.maxsize)
    ) == value


class _RecordingBinaryFile(io.BytesIO):
    def __init__(self, events: list[str], *, short_write: bool = False, fail: str | None = None):
        super().__init__()
        self.events = events
        self.short_write = short_write
        self.fail = fail

    def write(self, data: object) -> int:
        self.events.append("write")
        if self.fail == "write":
            raise OSError("private write detail")
        raw = bytes(data)
        if self.short_write and len(raw) > 1:
            raw = raw[: max(1, len(raw) // 2)]
        return super().write(raw)

    def flush(self) -> None:
        self.events.append("flush")
        if self.fail == "flush":
            raise OSError("private flush detail")
        super().flush()

    def fileno(self) -> int:
        return 123

    def close(self) -> None:
        self.events.append("close")
        # Keep BytesIO inspectable after the context manager.


def test_atomic_writer_uses_exclusive_temp_and_completes_short_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    stream = _RecordingBinaryFile(events, short_write=True)
    opened: list[tuple[Path, str]] = []
    replaced: list[tuple[Path, Path]] = []
    monkeypatch.setattr(Path, "open", lambda path, mode: (opened.append((path, mode)) or stream))
    monkeypatch.setattr(os, "fsync", lambda fd: events.append(f"fsync:{fd}"))
    monkeypatch.setattr(os, "replace", lambda old, new: replaced.append((old, new)))
    monkeypatch.setattr(os, "open", lambda path, flags: 456)
    monkeypatch.setattr(os, "close", lambda fd: events.append(f"closefd:{fd}"))

    destination = tmp_path / "manifest.json"
    value = manifest()
    write_manifest(destination, value, KnowledgeBaseLimits())

    assert opened[0][1] == "xb"
    assert opened[0][0].parent == destination.parent
    assert opened[0][0] != destination
    assert bytes(stream.getbuffer()) == encode_manifest(value, KnowledgeBaseLimits())
    assert replaced == [(opened[0][0], destination)]
    assert events.index("flush") < events.index("fsync:123") < events.index("close")
    assert "fsync:456" in events


@pytest.mark.parametrize("failure", ["write", "flush", "fsync", "replace"])
def test_atomic_writer_cleans_up_after_each_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    events: list[str] = []
    stream = _RecordingBinaryFile(events, fail=failure if failure in {"write", "flush"} else None)
    temporary: list[Path] = []
    unlinked: list[Path] = []

    def open_temp(path: Path, mode: str) -> _RecordingBinaryFile:
        temporary.append(path)
        return stream

    def fsync(fd: int) -> None:
        if failure == "fsync":
            raise OSError("private fsync detail")

    def replace(old: Path, new: Path) -> None:
        if failure == "replace":
            raise OSError("private replace detail")

    monkeypatch.setattr(Path, "open", open_temp)
    monkeypatch.setattr(Path, "unlink", lambda path: unlinked.append(path))
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(KnowledgeBasePersistenceError) as raised:
        write_manifest(tmp_path / "manifest.json", manifest(), KnowledgeBaseLimits())
    assert "private" not in str(raised.value)
    assert unlinked == temporary


def test_write_and_read_manifest_atomically_without_leaving_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    value = manifest()
    write_manifest(path, value, KnowledgeBaseLimits())
    assert path.read_bytes() == encode_manifest(value, KnowledgeBaseLimits())
    assert read_manifest(path, KnowledgeBaseLimits()) == value
    assert list(tmp_path.iterdir()) == [path]


def test_persistence_errors_are_controlled_and_do_not_expose_paths(tmp_path: Path) -> None:
    secret = tmp_path / "secret-name" / "manifest.json"
    with pytest.raises(KnowledgeBasePersistenceError) as raised:
        write_manifest(secret, manifest(), KnowledgeBaseLimits())
    assert "secret-name" not in str(raised.value)
    with pytest.raises(KnowledgeBasePersistenceError) as raised:
        read_manifest(secret, KnowledgeBaseLimits())
    assert "secret-name" not in str(raised.value)
