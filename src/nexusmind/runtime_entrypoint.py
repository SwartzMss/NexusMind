"""Diagnosable executable boundary for installed and frozen NexusMind CLIs."""

from __future__ import annotations

import sys

from nexusmind import cli
from nexusmind.runtime_support import RuntimeLayoutError, configure_runtime_logging, create_runtime_layout


def main(argv: list[str] | None = None) -> int:
    """Initialize runtime services, dispatch the CLI, and contain unknown errors."""
    try:
        layout = create_runtime_layout()
    except RuntimeLayoutError as exc:
        print(f"NexusMind could not initialize its runtime directory: {exc}", file=sys.stderr)
        return 1
    try:
        logger = configure_runtime_logging(layout)
    except OSError:
        print("NexusMind could not initialize runtime logging.", file=sys.stderr)
        return 1
    logger.info(
        "runtime startup",
        extra={"event": "startup", "python_frozen": bool(getattr(sys, "frozen", False))},
    )
    try:
        status = cli.main(argv)
    except Exception as exc:
        logger.exception(
            "unexpected runtime failure",
            extra={"event": "runtime_failed", "error_type": type(exc).__name__},
        )
        print(f"NexusMind failed unexpectedly. Diagnostic log: {layout.log_file}", file=sys.stderr)
        return 1
    logger.info("runtime exit", extra={"event": "shutdown", "exit_code": status})
    return status


if __name__ == "__main__":
    raise SystemExit(main())
