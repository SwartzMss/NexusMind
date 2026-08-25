from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import pytest

from nexusmind.runtime_support import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    RuntimeLayoutError,
    configure_runtime_logging,
    create_runtime_layout,
    resolve_runtime_root,
    runtime_operation,
)


def test_resolve_runtime_root_uses_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEXUSMIND_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert resolve_runtime_root() == tmp_path / ".nexusmind"


def test_resolve_runtime_root_accepts_absolute_override(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "managed"
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", str(root))

    assert resolve_runtime_root() == root


def test_resolve_runtime_root_uses_frozen_executable_directory(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "portable" / "nexusmind.exe"
    monkeypatch.delenv("NEXUSMIND_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert resolve_runtime_root() == executable.parent / ".nexusmind"


def test_absolute_override_wins_over_frozen_default(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "managed"
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", str(override))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "portable" / "nexusmind.exe"))

    assert resolve_runtime_root() == override


def test_resolve_runtime_root_rejects_relative_override(monkeypatch) -> None:
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", "relative/runtime")

    with pytest.raises(RuntimeLayoutError, match="absolute"):
        resolve_runtime_root()


def test_create_runtime_layout_creates_stable_directories(tmp_path: Path) -> None:
    layout = create_runtime_layout(tmp_path / "runtime")

    assert layout.root == tmp_path / "runtime"
    assert layout.data_dir.is_dir()
    assert layout.logs_dir.is_dir()
    assert layout.config_dir.is_dir()
    assert layout.models_dir.is_dir()
    assert layout.log_file == layout.logs_dir / "nexusmind.log"


def test_configure_runtime_logging_writes_bounded_json_without_message_payload(tmp_path: Path) -> None:
    layout = create_runtime_layout(tmp_path / "runtime")
    logger = configure_runtime_logging(layout)

    logger.info(
        "secret runtime message",
        extra={"event": "startup", "command": "query", "question": "private question"},
    )
    for handler in logger.handlers:
        handler.flush()

    record = json.loads(layout.log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "startup"
    assert record["command"] == "query"
    assert "secret runtime message" not in record.values()
    assert "private question" not in record.values()
    handler = logger.handlers[0]
    assert handler.maxBytes == LOG_MAX_BYTES
    assert handler.backupCount == LOG_BACKUP_COUNT


def test_runtime_operation_logs_start_completion_and_safe_failure(caplog) -> None:
    logger = logging.getLogger("nexusmind.operation-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with runtime_operation(logger, "sync", source_id="docs") as operation:
            operation["document_count"] = 3

    assert [record.event for record in caplog.records] == ["sync_started", "sync_completed"]
    assert caplog.records[-1].document_count == 3

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger.name), pytest.raises(RuntimeError, match="secret payload"):
        with runtime_operation(logger, "query"):
            raise RuntimeError("secret payload")

    assert caplog.records[-1].event == "query_failed"
    assert caplog.records[-1].error_type == "RuntimeError"
    assert "secret payload" not in caplog.records[-1].getMessage()
