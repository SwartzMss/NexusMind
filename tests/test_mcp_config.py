import json
from pathlib import Path

import pytest

from nexusmind.mcp import MAX_MCP_CLIENTS_PER_GROUP
from nexusmind.mcp.config import MCPConfigError, MCPStdioServerConfig, load_mcp_server_config, load_mcp_server_configs


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


def test_stdio_config_env_is_immutable_snapshot() -> None:
    env = {"TOKEN": "original"}
    config = MCPStdioServerConfig(server_id="demo", command="python", env=env)
    env["TOKEN"] = "changed"

    assert config.env["TOKEN"] == "original"
    with pytest.raises(TypeError):
        config.env["TOKEN"] = "changed-again"


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

    with pytest.raises(MCPConfigError):
        load_mcp_server_config(tmp_path, "demo")

    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(MCPConfigError):
        load_mcp_server_config(invalid_utf8, "demo")

    too_large = tmp_path / "too-large.json"
    too_large.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(MCPConfigError, match="too large"):
        load_mcp_server_config(too_large, "demo")


def test_env_reference_must_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"servers": {"demo": {"transport": "stdio", "command": "${MISSING_SECRET}"}}}),
        encoding="utf-8",
    )

    with pytest.raises(MCPConfigError):
        load_mcp_server_config(path, "demo")


def test_load_mcp_server_configs_loads_only_requested_servers_sorted(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "zeta": {"transport": "stdio", "command": "python", "args": ["z.py"]},
                    "alpha": {"transport": "stdio", "command": "python", "args": ["a.py"]},
                    "unused": {"transport": "stdio", "command": "${MISSING_SECRET}"},
                }
            }
        ),
        encoding="utf-8",
    )

    configs = load_mcp_server_configs(path, ["zeta", "alpha", "alpha"])

    assert list(configs) == ["alpha", "zeta"]
    assert configs["alpha"].args == ("a.py",)
    assert configs["zeta"].args == ("z.py",)


def test_load_mcp_server_configs_fails_before_any_client_for_missing_or_bad_required_server(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "demo": {"transport": "stdio", "command": "python"},
                    "bad": {"transport": "stdio", "command": "${MISSING_SECRET}"},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MCPConfigError, match="MCP server not found"):
        load_mcp_server_configs(path, ["demo", "missing"])
    with pytest.raises(MCPConfigError, match="Missing environment variable"):
        load_mcp_server_configs(path, ["bad"])


def test_load_mcp_server_configs_reads_config_once(tmp_path, monkeypatch) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "alpha": {"transport": "stdio", "command": "python"},
                    "zeta": {"transport": "stdio", "command": "python"},
                }
            }
        ),
        encoding="utf-8",
    )
    reads = {"count": 0}
    original_open = Path.open

    def counting_open(self, *args, **kwargs):
        reads["count"] += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)

    load_mcp_server_configs(path, ["alpha", "zeta"])

    assert reads == {"count": 1}


def test_load_mcp_server_configs_rejects_too_many_servers_before_reading_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("not json", encoding="utf-8")
    reads = {"count": 0}

    def counting_open(self, *args, **kwargs):
        reads["count"] += 1
        raise AssertionError("config must not be read")

    monkeypatch.setattr(Path, "open", counting_open)

    with pytest.raises(MCPConfigError, match="too many servers"):
        load_mcp_server_configs(path, (f"server{i}" for i in range(MAX_MCP_CLIENTS_PER_GROUP + 1)))

    assert reads == {"count": 0}


def test_load_mcp_server_configs_limits_consumption_of_repeated_iterables(tmp_path, monkeypatch) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("not json", encoding="utf-8")
    yielded = {"count": 0}

    def repeated_forever():
        while True:
            yielded["count"] += 1
            yield "demo"

    def fail_open(self, *args, **kwargs):
        raise AssertionError("config must not be read")

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(MCPConfigError, match="too many servers"):
        load_mcp_server_configs(path, repeated_forever())

    assert yielded == {"count": MAX_MCP_CLIENTS_PER_GROUP + 1}


@pytest.mark.parametrize("server_ids", [["demo", []], ["demo", 1], ["demo", "bad id"]])
def test_load_mcp_server_configs_rejects_invalid_server_id_inputs_before_reading_file(tmp_path, monkeypatch, server_ids) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("not json", encoding="utf-8")

    def fail_open(self, *args, **kwargs):
        raise AssertionError("config must not be read")

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(MCPConfigError):
        load_mcp_server_configs(path, server_ids)

