from __future__ import annotations

import json
import logging
from uuid import UUID

import pytest

from nexusmind import (
    cli,
)


def _capture_runtime_logs(monkeypatch, caplog) -> None:
    logger = logging.getLogger("nexusmind.runtime")
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=logger.name)


def test_cli_create_add_sync_search_inspect_and_remove(tmp_path, capsys, caplog, monkeypatch) -> None:
    _capture_runtime_logs(monkeypatch, caplog)
    source = tmp_path / "notes"
    source.mkdir()
    (source / "security.md").write_text("密钥轮换需要记录新的版本。", encoding="utf-8")
    root = tmp_path / "kb"

    assert cli.main(["create", str(root)]) == 0
    assert cli.main(["source", "add", str(source), "--knowledge-base", str(root)]) == 0
    assert cli.main(["source", "list", "--knowledge-base", str(root), "--json"]) == 0
    sources = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert set(sources[0]) == {"config_version", "source_id", "type", "path"}
    assert sources[0]["config_version"] == "1"
    assert sources[0]["type"] == "local_directory"
    assert sources[0]["path"] == str(source.resolve())
    UUID(sources[0]["source_id"])

    assert cli.main(["sync", "--knowledge-base", str(root), "--json"]) == 0
    assert [record.event for record in caplog.records[-2:]] == ["sync_started", "sync_completed"]
    assert caplog.records[-1].document_count == 1
    caplog.clear()
    assert cli.main(["search", "密钥轮换", "--knowledge-base", str(root), "--json"]) == 0
    results = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert results[0]["document"]["logical_path"] == "security.md"
    assert [record.event for record in caplog.records] == ["search_started", "search_completed"]
    assert caplog.records[-1].result_count == 1
    assert all("密钥轮换" not in record.getMessage() for record in caplog.records)

    assert cli.main(["inspect", "--knowledge-base", str(root), "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out.splitlines()[-1])
    UUID(inspection["status"]["knowledge_base_id"])
    assert "display_name" not in inspection["status"]
    assert inspection["status"]["document_count"] == 1

    assert cli.main(["inspect", "--knowledge-base", str(root)]) == 0
    plain_inspection = capsys.readouterr().out
    assert f"KnowledgeBase: {root.resolve()}" in plain_inspection
    assert inspection["status"]["knowledge_base_id"] not in plain_inspection

    assert cli.main(["diagnose", "密钥轮换", "--knowledge-base", str(root), "--json"]) == 0
    diagnostics = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert diagnostics["results"]

    assert cli.main(["source", "remove", str(source), "--knowledge-base", str(root)]) == 0


def test_cli_help_exposes_only_knowledge_commands(capsys) -> None:
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    for command in ("create", "source", "sync", "search", "query", "inspect", "diagnose"):
        assert command in output
    for removed in ("chat", "tools", "runs", "mcp", "skill"):
        assert removed not in output


def test_create_help_has_no_name_option(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli._parser().parse_args(["create", "--help"])
    assert raised.value.code == 0
    assert "--name" not in capsys.readouterr().out


def test_create_rejects_name_option(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli._parser().parse_args(["create", "kb", "--name", "Name"])
    assert raised.value.code == 2
    assert "unrecognized arguments: --name Name" in capsys.readouterr().err


def test_cli_source_remove_rejects_internal_id_selector(capsys) -> None:
    try:
        cli.main(["source", "remove", "--id", "internal-id"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--id must not be accepted")

    assert "unrecognized arguments: --id" in capsys.readouterr().err
