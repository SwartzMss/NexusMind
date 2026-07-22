import os

from nexusmind import cli


def test_tools_list_outputs_builtin_tools_without_model_env(monkeypatch, capsys) -> None:
    _clear_model_env(monkeypatch)

    assert cli.main(["tools", "list"]) == 0

    captured = capsys.readouterr()
    assert "echo" in captured.out
    assert captured.err == ""


def test_tools_call_echo_outputs_json_without_model_env(monkeypatch, capsys) -> None:
    _clear_model_env(monkeypatch)

    assert cli.main(["tools", "call", "echo", '{"text":"hello"}']) == 0

    captured = capsys.readouterr()
    assert captured.out == '{"text": "hello"}\n'
    assert captured.err == ""


def test_tools_call_invalid_json_returns_nonzero(monkeypatch, capsys) -> None:
    _clear_model_env(monkeypatch)

    assert cli.main(["tools", "call", "echo", "{bad"]) == 2

    captured = capsys.readouterr()
    assert "Invalid JSON arguments" in captured.err


def test_tools_call_validation_error_returns_nonzero(monkeypatch, capsys) -> None:
    _clear_model_env(monkeypatch)

    assert cli.main(["tools", "call", "echo", "{}"]) == 2

    captured = capsys.readouterr()
    assert "INVALID_ARGUMENTS" in captured.err


def _clear_model_env(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("NEXUSMIND_MODEL_"):
            monkeypatch.delenv(name, raising=False)

