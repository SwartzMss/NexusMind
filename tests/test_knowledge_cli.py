from __future__ import annotations

import json
import logging

from nexusmind import cli


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

    assert cli.main(["create", str(root), "--id", "security", "--name", "Security"]) == 0
    assert cli.main(["source", "add", "--knowledge-base", str(root), "--id", "docs", "--path", str(source), "--type", "directory"]) == 0
    assert cli.main(["source", "list", "--knowledge-base", str(root), "--json"]) == 0
    sources = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert sources[0]["source_id"] == "docs"

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
    assert inspection["status"]["document_count"] == 1

    assert cli.main(["diagnose", "密钥轮换", "--knowledge-base", str(root), "--json"]) == 0
    diagnostics = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert diagnostics["results"]

    assert cli.main(["source", "remove", "--knowledge-base", str(root), "--id", "docs"]) == 0


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
