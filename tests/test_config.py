from pathlib import Path

from nexusmind.config import ConfigError, ModelConfig, load_model_config_from_env


def _clear_env(monkeypatch) -> None:
    monkeypatch.delenv("NEXUSMIND_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("NEXUSMIND_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("NEXUSMIND_MODEL_NAME", raising=False)
    monkeypatch.delenv("NEXUSMIND_MODEL_TIMEOUT", raising=False)


def test_load_model_config_reports_missing_values_without_dotenv(monkeypatch) -> None:
    _clear_env(monkeypatch)

    try:
        load_model_config_from_env(load_dotenv_file=False)
    except ConfigError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ConfigError")

    assert "NEXUSMIND_MODEL_BASE_URL" in message
    assert "NEXUSMIND_MODEL_API_KEY" in message
    assert "NEXUSMIND_MODEL_NAME" in message


def test_load_model_config_reads_dotenv_file(monkeypatch, tmp_path: Path) -> None:
    _clear_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    tmp_path.joinpath(".env").write_text(
        "\n".join(
            [
                "NEXUSMIND_MODEL_BASE_URL=https://provider.test/v1",
                "NEXUSMIND_MODEL_API_KEY=sk-test-secret",
                "NEXUSMIND_MODEL_NAME=test-model",
                "NEXUSMIND_MODEL_TIMEOUT=12.5",
            ]
        ),
        encoding="utf-8",
    )

    assert load_model_config_from_env() == ModelConfig(
        base_url="https://provider.test/v1",
        api_key="sk-test-secret",
        model="test-model",
        timeout=12.5,
    )
