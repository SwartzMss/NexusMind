from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0


def load_model_config_from_env() -> ModelConfig:
    load_dotenv()

    base_url = os.getenv("NEXUSMIND_MODEL_BASE_URL", "").strip()
    api_key = os.getenv("NEXUSMIND_MODEL_API_KEY", "").strip()
    model = os.getenv("NEXUSMIND_MODEL_NAME", "").strip()
    timeout_raw = os.getenv("NEXUSMIND_MODEL_TIMEOUT", "60").strip()

    missing = [
        name
        for name, value in {
            "NEXUSMIND_MODEL_BASE_URL": base_url,
            "NEXUSMIND_MODEL_API_KEY": api_key,
            "NEXUSMIND_MODEL_NAME": model,
        }.items()
        if not value
    ]
    if missing:
        raise ConfigError(f"Missing required configuration: {', '.join(missing)}")

    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ConfigError("NEXUSMIND_MODEL_TIMEOUT must be a number") from exc
    if timeout <= 0:
        raise ConfigError("NEXUSMIND_MODEL_TIMEOUT must be greater than 0")

    return ModelConfig(base_url=base_url, api_key=api_key, model=model, timeout=timeout)

