from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from nexusmind import desktop


def test_desktop_main_initializes_runtime_and_returns_cli_status(monkeypatch, tmp_path) -> None:
    layout = SimpleNamespace(log_file=tmp_path / "nexusmind.log")
    logger = logging.getLogger("desktop-test-success")
    monkeypatch.setattr(desktop, "create_runtime_layout", lambda: layout)
    monkeypatch.setattr(desktop, "configure_runtime_logging", lambda value: logger)
    monkeypatch.setattr(desktop.cli, "main", lambda argv: 7)

    assert desktop.main(["--help"]) == 7


def test_desktop_main_logs_unexpected_exception_and_points_to_log(monkeypatch, tmp_path, capsys, caplog) -> None:
    layout = SimpleNamespace(log_file=tmp_path / "nexusmind.log")
    logger = logging.getLogger("desktop-test-failure")
    monkeypatch.setattr(desktop, "create_runtime_layout", lambda: layout)
    monkeypatch.setattr(desktop, "configure_runtime_logging", lambda value: logger)

    def explode(argv):
        raise RuntimeError("secret")

    monkeypatch.setattr(desktop.cli, "main", explode)
    with caplog.at_level(logging.ERROR, logger=logger.name):
        assert desktop.main(["query", "secret question"]) == 1

    error = capsys.readouterr().err
    assert "unexpectedly" in error
    assert str(layout.log_file) in error
    assert "secret" not in error.replace("unexpectedly", "")
    assert caplog.records[-1].event == "runtime_failed"


def test_desktop_main_does_not_swallow_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    layout = SimpleNamespace(log_file=tmp_path / "nexusmind.log")
    monkeypatch.setattr(desktop, "create_runtime_layout", lambda: layout)
    monkeypatch.setattr(
        desktop,
        "configure_runtime_logging",
        lambda value: logging.getLogger("desktop-test-interrupt"),
    )

    def interrupt(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(desktop.cli, "main", interrupt)
    with pytest.raises(KeyboardInterrupt):
        desktop.main([])
