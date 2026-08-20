from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

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
    "max_display_name_chars": 1_024,
    "max_source_id_chars": 256,
    "max_path_chars": 32_768,
}


def source(source_id: str = "source", path: str = "/tmp/source") -> LocalFileSourceConfig:
    return LocalFileSourceConfig(source_id=source_id, path=path)


def manifest(**changes: object) -> KnowledgeBaseManifest:
    values: dict[str, object] = {
        "knowledge_base_id": "kb",
        "display_name": None,
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
    config = source_type(source_id="docs", path="relative/path")
    assert config.config_version == "1"
    assert config.type == ("local_file" if source_type is LocalFileSourceConfig else "local_directory")
    with pytest.raises(FrozenInstanceError):
        config.path = "elsewhere"
    with pytest.raises(TypeError):
        source_type(source_id="docs", path="/tmp", config_version="2")
    with pytest.raises(TypeError):
        source_type(source_id="docs", path="/tmp", type="wrong")


@pytest.mark.parametrize("source_id", ["", 1, None])
def test_source_id_must_be_non_empty_text(source_id: object) -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        LocalFileSourceConfig(source_id=source_id, path="/tmp")  # type: ignore[arg-type]


@pytest.mark.parametrize("path", [1, None, b"/tmp"])
def test_source_path_must_be_text(path: object) -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        LocalFileSourceConfig(source_id="docs", path=path)  # type: ignore[arg-type]


@pytest.mark.parametrize("knowledge_base_id", ["", 1, None])
def test_manifest_id_must_be_non_empty_text(knowledge_base_id: object) -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(knowledge_base_id=knowledge_base_id)


@pytest.mark.parametrize("display_name", ["", 1, False])
def test_display_name_must_be_none_or_non_empty_text(display_name: object) -> None:
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(display_name=display_name)


def test_manifest_requires_exact_tuple_supported_unique_sources_and_sorts() -> None:
    a = source("a", "/tmp/a")
    b = LocalDirectorySourceConfig(source_id="b", path="/tmp/b")
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(sources=[a])
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(sources=(object(),))
    with pytest.raises(KnowledgeBaseConfigError):
        manifest(sources=(a, source("a", "/tmp/other")))
    value = manifest(sources=(b, a))
    assert value.sources == (a, b)
    with pytest.raises(FrozenInstanceError):
        value.display_name = "changed"


def test_manifest_normalizes_paths_and_enforces_all_configured_bounds(tmp_path: Path) -> None:
    relative = LocalFileSourceConfig(source_id="s", path="relative/../file.txt")
    value = KnowledgeBaseManifest(
        knowledge_base_id="kb",
        display_name="name",
        sources=(relative,),
        limits=KnowledgeBaseLimits(max_sources=1, max_path_chars=32_768),
    )
    assert Path(value.sources[0].path).is_absolute()
    assert value.sources[0].path == str(Path(relative.path).resolve(strict=False))

    cases = [
        ({"knowledge_base_id": "xx"}, KnowledgeBaseLimits(max_knowledge_base_id_chars=1)),
        ({"display_name": "xx"}, KnowledgeBaseLimits(max_display_name_chars=1)),
        ({"sources": (source("xx"),)}, KnowledgeBaseLimits(max_source_id_chars=1)),
        ({"sources": (source(path="/too-long"),)}, KnowledgeBaseLimits(max_path_chars=1)),
        ({"sources": (source("a"), source("b"))}, KnowledgeBaseLimits(max_sources=1)),
    ]
    for changes, limits in cases:
        with pytest.raises(KnowledgeBaseConfigError):
            KnowledgeBaseManifest(limits=limits, **({"knowledge_base_id": "kb", "display_name": None, "sources": ()} | changes))


def test_codec_is_exact_utf8_deterministic_and_order_independent() -> None:
    limits = KnowledgeBaseLimits()
    empty = manifest()
    assert encode_manifest(empty, limits) == (
        b'{"display_name":null,"format_version":"1",'
        b'"knowledge_base_id":"kb","sources":[]}\n'
    )
    assert decode_manifest(encode_manifest(empty, limits), limits) == empty

    a = source("a", "/tmp/\u4e2d")
    b = LocalDirectorySourceConfig(source_id="b", path="/tmp/b")
    first = manifest(display_name="\u77e5\u8bc6", sources=(b, a))
    second = manifest(display_name="\u77e5\u8bc6", sources=(a, b))
    assert encode_manifest(first, limits) == encode_manifest(second, limits)
    assert b"\\u" not in encode_manifest(first, limits)
    assert decode_manifest(encode_manifest(first, limits), limits) == first


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
        {"remove": "display_name"},
        {"format_version": "2"},
        {"format_version": 1},
        {"knowledge_base_id": 1},
        {"display_name": 1},
        {"sources": None},
        {"sources": {}},
    ],
)
def test_decode_rejects_root_schema_violations(change: dict[str, object]) -> None:
    value: dict[str, object] = {
        "format_version": "1",
        "knowledge_base_id": "kb",
        "display_name": None,
        "sources": [],
    }
    removed = change.pop("remove", None)
    if removed:
        del value[str(removed)]
    value.update(change)
    with pytest.raises(KnowledgeBaseConfigError):
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
        {"path": "relative"},
    ],
)
def test_decode_rejects_source_schema_violations(change: dict[str, object]) -> None:
    item: dict[str, object] = {
        "config_version": "1",
        "source_id": "docs",
        "type": "local_file",
        "path": "/tmp/docs",
    }
    removed = change.pop("remove", None)
    if removed:
        del item[str(removed)]
    item.update(change)
    root = {"format_version": "1", "knowledge_base_id": "kb", "display_name": None, "sources": [item]}
    with pytest.raises(KnowledgeBaseConfigError):
        decode_manifest(encoded(root), KnowledgeBaseLimits())


def test_decode_rejects_duplicate_source_ids_and_oversized_decoded_values() -> None:
    item = {"config_version": "1", "source_id": "a", "type": "local_file", "path": "/tmp/a"}
    root = {"format_version": "1", "knowledge_base_id": "kb", "display_name": None, "sources": [item, item]}
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


def test_write_and_read_manifest_atomically_without_leaving_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    value = manifest(display_name="Knowledge")
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
