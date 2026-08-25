"""Desktop runtime directory and diagnostic logging support."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time
import traceback
from collections.abc import Iterator


LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
SAFE_LOG_FIELDS = frozenset(
    {
        "command",
        "source_id",
        "document_count",
        "result_count",
        "citation_count",
        "duration_ms",
        "exit_code",
        "error_type",
        "python_frozen",
    }
)


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


class JsonLogFormatter(logging.Formatter):
    """Format a deliberately small allowlist of diagnostic fields as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "runtime_event"),
        }
        for field in SAFE_LOG_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info is not None:
            payload["traceback"] = "".join(traceback.format_tb(record.exc_info[2])).rstrip()
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


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


def configure_runtime_logging(layout: RuntimeLayout) -> logging.Logger:
    """Configure the dedicated bounded runtime logger."""
    logger = logging.getLogger("nexusmind.runtime")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for existing in tuple(logger.handlers):
        existing.close()
        logger.removeHandler(existing)
    handler = RotatingFileHandler(
        layout.log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    return logger


@contextmanager
def runtime_operation(logger: logging.Logger, name: str, **fields: object) -> Iterator[dict[str, object]]:
    """Log a content-safe operation lifecycle and expose completion counters."""
    started = time.monotonic()
    logger.info("operation started", extra={"event": f"{name}_started", **fields})
    outcome: dict[str, object] = {}
    try:
        yield outcome
    except Exception as exc:
        logger.error(
            "operation failed",
            extra={
                "event": f"{name}_failed",
                **fields,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "error_type": type(exc).__name__,
            },
        )
        raise
    logger.info(
        "operation completed",
        extra={
            "event": f"{name}_completed",
            **fields,
            **outcome,
            "duration_ms": round((time.monotonic() - started) * 1000),
        },
    )
