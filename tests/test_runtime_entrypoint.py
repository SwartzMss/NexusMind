from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from nexusmind import runtime_entrypoint


def test_runtime_entrypoint_initializes_runtime_and_returns_cli_status(monkeypatch, tmp_path) -> None:
    layout = SimpleNamespace(log_file=tmp_path / "nexusmind.log")
    logger = logging.getLogger("runtime-entrypoint-test-success")
    monkeypatch.setattr(runtime_entrypoint, "create_runtime_layout", lambda: layout)
    monkeypatch.setattr(runtime_entrypoint, "configure_runtime_logging", lambda value: logger)
    monkeypatch.setattr(runtime_entrypoint.cli, "main", lambda argv: 7)

    assert runtime_entrypoint.main(["--help"]) == 7


def test_runtime_entrypoint_logs_unexpected_exception_and_points_to_log(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    layout = SimpleNamespace(log_file=tmp_path / "nexusmind.log")
    logger = logging.getLogger("runtime-entrypoint-test-failure")
    monkeypatch.setattr(runtime_entrypoint, "create_runtime_layout", lambda: layout)
    monkeypatch.setattr(runtime_entrypoint, "configure_runtime_logging", lambda value: logger)

    def explode(argv):
        raise RuntimeError("secret")

    monkeypatch.setattr(runtime_entrypoint.cli, "main", explode)
    with caplog.at_level(logging.ERROR, logger=logger.name):
        assert runtime_entrypoint.main(["query", "secret question"]) == 1

    error = capsys.readouterr().err
    assert "unexpectedly" in error
    assert str(layout.log_file) in error
    assert "secret" not in error.replace("unexpectedly", "")
    assert caplog.records[-1].event == "runtime_failed"


def test_runtime_entrypoint_does_not_swallow_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    layout = SimpleNamespace(log_file=tmp_path / "nexusmind.log")
    monkeypatch.setattr(runtime_entrypoint, "create_runtime_layout", lambda: layout)
    monkeypatch.setattr(
        runtime_entrypoint,
        "configure_runtime_logging",
        lambda value: logging.getLogger("runtime-entrypoint-test-interrupt"),
    )

    def interrupt(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_entrypoint.cli, "main", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runtime_entrypoint.main([])


def test_runtime_entrypoint_is_the_only_executable_module() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1]
    assert root.joinpath("src/nexusmind/runtime_entrypoint.py").is_file()
    assert not root.joinpath("src/nexusmind/desktop.py").exists()
