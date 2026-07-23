import json

import pytest

from nexusmind.mcp.config import MCPConfigError, MCPStdioServerConfig, load_mcp_server_config


def test_loads_valid_stdio_config(tmp_path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "demo": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["server.py"],
                        "cwd": None,
                        "env": {"TOKEN": "secret"},
                        "connect_timeout": 1,
                        "request_timeout": 2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_mcp_server_config(path, "demo")

    assert config.server_id == "demo"
    assert config.command == "python"
    assert config.args == ("server.py",)
    assert config.connect_timeout == 1
    assert config.request_timeout == 2


def test_stdio_config_repr_does_not_include_env() -> None:
    config = MCPStdioServerConfig(server_id="demo", command="python", env={"TOKEN": "sk-live-secret"})

    assert "sk-live-secret" not in repr(config)


@pytest.mark.parametrize(
    "server",
    [
        {"transport": "stdio"},
        {"transport": "http", "command": "python"},
        {"transport": "stdio", "command": "python", "connect_timeout": 0},
        {"transport": "stdio", "command": "python", "args": "server.py"},
    ],
)
def test_invalid_config_reports_controlled_error(tmp_path, server) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"servers": {"demo": server}}), encoding="utf-8")

    with pytest.raises(MCPConfigError):
        load_mcp_server_config(path, "demo")


def test_missing_and_damaged_config_report_controlled_error(tmp_path) -> None:
    with pytest.raises(MCPConfigError):
        load_mcp_server_config(tmp_path / "missing.json", "demo")

    damaged = tmp_path / "damaged.json"
    damaged.write_text("{bad", encoding="utf-8")
    with pytest.raises(MCPConfigError):
        load_mcp_server_config(damaged, "demo")


def test_env_reference_must_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"servers": {"demo": {"transport": "stdio", "command": "${MISSING_SECRET}"}}}),
        encoding="utf-8",
    )

    with pytest.raises(MCPConfigError):
        load_mcp_server_config(path, "demo")

