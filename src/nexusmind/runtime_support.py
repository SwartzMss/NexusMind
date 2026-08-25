"""Desktop runtime directory and diagnostic logging support."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


class RuntimeLayoutError(RuntimeError):
    """Raised when the mutable runtime layout cannot be resolved or created."""


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    root: Path
    data_dir: Path
    logs_dir: Path
    config_dir: Path
    models_dir: Path
    log_file: Path


def resolve_runtime_root() -> Path:
    """Resolve the absolute mutable-data root without depending on cwd."""
    configured = os.getenv("NEXUSMIND_RUNTIME_DIR", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".nexusmind"
    if not root.is_absolute():
        raise RuntimeLayoutError("NEXUSMIND_RUNTIME_DIR must be an absolute path")
    return root


def create_runtime_layout(root: Path | None = None) -> RuntimeLayout:
    """Create and return NexusMind's stable mutable runtime directories."""
    selected = resolve_runtime_root() if root is None else root
    if not selected.is_absolute():
        raise RuntimeLayoutError("Runtime root must be an absolute path")
    data = selected / "data"
    logs = selected / "logs"
    config = selected / "config"
    models = selected / "models"
    try:
        for path in (selected, data, logs, config, models):
            path.mkdir(parents=True, exist_ok=True)
            if not path.is_dir():
                raise RuntimeLayoutError(f"Runtime path is not a directory: {path}")
    except OSError as exc:
        raise RuntimeLayoutError("NexusMind runtime directories could not be created") from exc
    return RuntimeLayout(selected, data, logs, config, models, logs / "nexusmind.log")
