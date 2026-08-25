# Windows Portable Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship NexusMind as a diagnosable Windows `onedir` portable CLI whose Python runtime and dependencies are bundled while mutable data and logs remain under a stable user runtime root.

**Architecture:** Add a platform-neutral runtime-support module for path creation, bounded JSON logging, and operation lifecycle events, plus a thin desktop executable boundary that wraps the existing CLI. Route the installed console script and PyInstaller entry through that boundary, instrument existing sync/search/query handlers without logging user content, and build a portable ZIP with a PowerShell script plus a Windows CI packaging job.

**Tech Stack:** Python 3.11–3.13, standard-library `logging`/`pathlib`, PyInstaller `onedir`, PowerShell, pytest, GitHub Actions.

---

## File Structure

- Create `src/nexusmind/runtime_support.py`: runtime-root resolution, stable directory creation, JSON log formatting/configuration, and reusable lifecycle logging.
- Create `src/nexusmind/desktop.py`: executable initialization and unexpected-exception boundary.
- Modify `src/nexusmind/cli.py`: add content-safe sync/search/query lifecycle events.
- Modify `pyproject.toml`: route `nexusmind` through the desktop boundary and add the packaging extra.
- Create `tests/test_runtime_support.py`: runtime-layout and structured-logging unit tests.
- Create `tests/test_desktop.py`: executable-boundary unit tests.
- Modify `tests/test_knowledge_cli.py` and `tests/test_knowledge_query_cli.py`: verify operation lifecycle events and failure diagnostics.
- Create `packaging/nexusmind.spec`: reproducible PyInstaller `onedir` definition.
- Create `scripts/build-portable.ps1`: build, smoke test, and ZIP orchestration.
- Create `tests/test_portable_packaging.py`: deterministic checks for packaging entry points and configuration.
- Modify `.github/workflows/ci.yml`: build and smoke-test the portable artifact on Windows.
- Modify `README.md`: document runtime paths, logging, diagnostics, and portable build/use.

### Task 1: Stable Runtime Layout

**Files:**
- Create: `src/nexusmind/runtime_support.py`
- Create: `tests/test_runtime_support.py`

- [ ] **Step 1: Write failing runtime-layout tests**

```python
from pathlib import Path
import pytest

from nexusmind.runtime_support import RuntimeLayoutError, create_runtime_layout, resolve_runtime_root


def test_resolve_runtime_root_uses_user_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEXUSMIND_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert resolve_runtime_root() == tmp_path / ".nexusmind"


def test_resolve_runtime_root_accepts_absolute_override(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "managed"
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", str(root))
    assert resolve_runtime_root() == root


def test_resolve_runtime_root_rejects_relative_override(monkeypatch) -> None:
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", "relative/runtime")
    with pytest.raises(RuntimeLayoutError, match="absolute"):
        resolve_runtime_root()


def test_create_runtime_layout_creates_stable_directories(tmp_path: Path) -> None:
    layout = create_runtime_layout(tmp_path / "runtime")
    assert layout.root == tmp_path / "runtime"
    assert layout.data_dir.is_dir()
    assert layout.logs_dir.is_dir()
    assert layout.config_dir.is_dir()
    assert layout.models_dir.is_dir()
    assert layout.log_file == layout.logs_dir / "nexusmind.log"
```

- [ ] **Step 2: Run the focused tests and verify the missing module failure**

Run: `python -m pytest tests/test_runtime_support.py -vv`

Expected: collection fails with `ModuleNotFoundError: No module named 'nexusmind.runtime_support'`.

- [ ] **Step 3: Implement runtime-root validation and directory creation**

```python
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


class RuntimeLayoutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    root: Path
    data_dir: Path
    logs_dir: Path
    config_dir: Path
    models_dir: Path
    log_file: Path


def resolve_runtime_root() -> Path:
    configured = os.getenv("NEXUSMIND_RUNTIME_DIR", "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / ".nexusmind"
    if not root.is_absolute():
        raise RuntimeLayoutError("NEXUSMIND_RUNTIME_DIR must be an absolute path")
    return root


def create_runtime_layout(root: Path | None = None) -> RuntimeLayout:
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
```

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_runtime_support.py -vv`

Expected: 4 passed.

- [ ] **Step 5: Commit the runtime layout**

```bash
git add src/nexusmind/runtime_support.py tests/test_runtime_support.py
git commit -m "feat: add stable desktop runtime layout"
```

### Task 2: Bounded Structured Runtime Logging

**Files:**
- Modify: `src/nexusmind/runtime_support.py`
- Modify: `tests/test_runtime_support.py`

- [ ] **Step 1: Add failing JSON logging and lifecycle tests**

```python
import json
import logging

from nexusmind.runtime_support import configure_runtime_logging, runtime_operation


def test_configure_runtime_logging_writes_json_without_message_payload(tmp_path: Path) -> None:
    layout = create_runtime_layout(tmp_path / "runtime")
    logger = configure_runtime_logging(layout)
    logger.info("runtime event", extra={"event": "startup", "command": "query"})
    for handler in logger.handlers:
        handler.flush()
    record = json.loads(layout.log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert record["event"] == "startup"
    assert record["command"] == "query"
    assert "runtime event" not in record.values()


def test_runtime_operation_logs_start_completion_and_safe_failure(caplog) -> None:
    logger = logging.getLogger("nexusmind.runtime.test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        with runtime_operation(logger, "sync", source_id="docs") as operation:
            operation["document_count"] = 3
    assert [record.event for record in caplog.records] == ["sync_started", "sync_completed"]
    assert caplog.records[-1].document_count == 3

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logger.name), pytest.raises(RuntimeError, match="secret payload"):
        with runtime_operation(logger, "query"):
            raise RuntimeError("secret payload")
    assert caplog.records[-1].event == "query_failed"
    assert caplog.records[-1].error_type == "RuntimeError"
    assert "secret payload" not in caplog.records[-1].getMessage()
```

- [ ] **Step 2: Verify tests fail for missing logging APIs**

Run: `python -m pytest tests/test_runtime_support.py -vv`

Expected: import failure for `configure_runtime_logging` and `runtime_operation`.

- [ ] **Step 3: Implement JSON formatting, bounded rotation, and lifecycle context**

Add a `JsonLogFormatter` that emits only `timestamp`, `level`, `logger`, `event`, and an explicit allowlist of diagnostic extras; do not serialize `record.msg`, exception messages, arbitrary `record.__dict__`, prompts, questions, or document content. Add:

```python
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
SAFE_LOG_FIELDS = frozenset({
    "command", "source_id", "document_count", "result_count", "citation_count",
    "duration_ms", "exit_code", "error_type", "python_frozen",
})


def configure_runtime_logging(layout: RuntimeLayout) -> logging.Logger:
    logger = logging.getLogger("nexusmind.runtime")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
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
def runtime_operation(logger: logging.Logger, name: str, **fields: object):
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
```

- [ ] **Step 4: Run logging tests and the full runtime-support file**

Run: `python -m pytest tests/test_runtime_support.py -vv`

Expected: all tests pass, including JSON parsing and lifecycle failure redaction.

- [ ] **Step 5: Commit structured logging**

```bash
git add src/nexusmind/runtime_support.py tests/test_runtime_support.py
git commit -m "feat: add bounded structured runtime logging"
```

### Task 3: Executable Boundary and CLI Operation Diagnostics

**Files:**
- Create: `src/nexusmind/desktop.py`
- Create: `tests/test_desktop.py`
- Modify: `src/nexusmind/cli.py`
- Modify: `tests/test_knowledge_cli.py`
- Modify: `tests/test_knowledge_query_cli.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing desktop-boundary tests**

```python
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
    monkeypatch.setattr(desktop.cli, "main", lambda argv: (_ for _ in ()).throw(RuntimeError("secret")))
    with caplog.at_level(logging.ERROR, logger=logger.name):
        assert desktop.main(["query", "secret question"]) == 1
    error = capsys.readouterr().err
    assert "unexpectedly" in error
    assert str(layout.log_file) in error
    assert "secret" not in error
    assert caplog.records[-1].event == "runtime_failed"


def test_desktop_main_does_not_swallow_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    layout = SimpleNamespace(log_file=tmp_path / "nexusmind.log")
    monkeypatch.setattr(desktop, "create_runtime_layout", lambda: layout)
    monkeypatch.setattr(desktop, "configure_runtime_logging", lambda value: logging.getLogger("desktop-test-interrupt"))
    monkeypatch.setattr(desktop.cli, "main", lambda argv: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        desktop.main([])
```

- [ ] **Step 2: Add failing CLI lifecycle assertions**

Use `caplog` in `tests/test_knowledge_cli.py` to assert successful `sync` emits `sync_started` and `sync_completed`, successful `search` emits `search_started` and `search_completed`, and the records contain counts but not the query text. Use `caplog` in `tests/test_knowledge_query_cli.py` to assert successful `query` emits `query_started` and `query_completed` with `citation_count == 1` but not `Binder UID?`.

- [ ] **Step 3: Run the focused tests and verify failures**

Run: `python -m pytest tests/test_desktop.py tests/test_knowledge_cli.py tests/test_knowledge_query_cli.py -vv`

Expected: desktop import fails and lifecycle event assertions fail.

- [ ] **Step 4: Implement the executable boundary**

```python
from __future__ import annotations

import sys

from nexusmind import cli
from nexusmind.runtime_support import RuntimeLayoutError, configure_runtime_logging, create_runtime_layout


def main(argv: list[str] | None = None) -> int:
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
    logger.info("runtime startup", extra={"event": "startup", "python_frozen": bool(getattr(sys, "frozen", False))})
    try:
        status = cli.main(argv)
    except Exception as exc:
        logger.exception("unexpected runtime failure", extra={"event": "runtime_failed", "error_type": type(exc).__name__})
        print(f"NexusMind failed unexpectedly. Diagnostic log: {layout.log_file}", file=sys.stderr)
        return 1
    logger.info("runtime exit", extra={"event": "shutdown", "exit_code": status})
    return status


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Instrument CLI sync/search/query without logging content**

Wrap `_sync`, `_search`, and `_query` bodies with `runtime_operation(logging.getLogger("nexusmind.runtime"), ...)`. Use only `source_id`, numeric `document_count`, `result_count`, and `citation_count`; never pass `args.query`, `args.question`, paths, answer text, document text, or exception messages to logging extras.

- [ ] **Step 6: Route the installed CLI through the desktop boundary**

Change `pyproject.toml` to:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8,<9",
]
packaging = [
  "pyinstaller>=6.10,<7",
]

[project.scripts]
nexusmind = "nexusmind.desktop:main"
nexusmind-kb = "nexusmind.knowledge_base_ui:main"
```

- [ ] **Step 7: Run focused and CLI regression tests**

Run: `python -m pytest tests/test_desktop.py tests/test_runtime_support.py tests/test_knowledge_cli.py tests/test_knowledge_query_cli.py -vv`

Expected: all focused tests pass and existing CLI output assertions remain unchanged.

- [ ] **Step 8: Commit the executable diagnostics boundary**

```bash
git add src/nexusmind/desktop.py src/nexusmind/cli.py pyproject.toml tests/test_desktop.py tests/test_knowledge_cli.py tests/test_knowledge_query_cli.py
git commit -m "feat: add diagnosable desktop CLI boundary"
```

### Task 4: PyInstaller Portable Build

**Files:**
- Create: `packaging/nexusmind.spec`
- Create: `scripts/build-portable.ps1`
- Create: `tests/test_portable_packaging.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing packaging-contract tests**

```python
from pathlib import Path


def test_pyinstaller_spec_targets_desktop_entry_and_onedir() -> None:
    text = Path("packaging/nexusmind.spec").read_text(encoding="utf-8")
    assert "src/nexusmind/desktop.py" in text.replace("\\", "/")
    assert "COLLECT(" in text
    assert "console=True" in text


def test_portable_script_builds_smoke_tests_and_archives() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")
    assert "PyInstaller" in text
    assert "nexusmind.exe" in text
    assert "--help" in text
    assert "NEXUSMIND_RUNTIME_DIR" in text
    assert "Compress-Archive" in text
```

- [ ] **Step 2: Run packaging tests and verify missing-file failures**

Run: `python -m pytest tests/test_portable_packaging.py -vv`

Expected: both tests fail with `FileNotFoundError`.

- [ ] **Step 3: Add the `onedir` PyInstaller spec**

Create `packaging/nexusmind.spec` with repository-root path resolution, `Analysis` targeting `src/nexusmind/desktop.py`, collected `nexusmind` submodules, package metadata/data required by `httpx`, `python-dotenv`, and certificate handling, `EXE(..., console=True, name="nexusmind")`, and `COLLECT(..., name="nexusmind")`. Do not set one-file mode and do not embed runtime user directories.

- [ ] **Step 4: Add the PowerShell build script**

The script must use `$ErrorActionPreference = "Stop"`, resolve the repository root from `$PSScriptRoot`, invoke `python -m PyInstaller packaging/nexusmind.spec --noconfirm --clean`, set `NEXUSMIND_RUNTIME_DIR` to an absolute `build/smoke-runtime` path, execute `dist/nexusmind/nexusmind.exe --help`, verify `logs/nexusmind.log` exists, and run `Compress-Archive -Path dist/nexusmind -DestinationPath dist/nexusmind-windows-portable.zip -Force`.

- [ ] **Step 5: Keep generated build directories ignored**

Ensure `.gitignore` contains `dist/`, `build/`, and PyInstaller-generated `*.spec` files are not globally ignored because the checked-in spec is source. Existing ignore entries should be reused without duplication.

- [ ] **Step 6: Run packaging contract tests**

Run: `python -m pytest tests/test_portable_packaging.py -vv`

Expected: 2 passed.

- [ ] **Step 7: Commit portable packaging support**

```bash
git add packaging/nexusmind.spec scripts/build-portable.ps1 tests/test_portable_packaging.py .gitignore
git commit -m "build: add Windows portable runtime package"
```

### Task 5: Windows CI and User Documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `tests/test_project_metadata.py`

- [ ] **Step 1: Add failing metadata/documentation assertions**

Extend `tests/test_project_metadata.py` to assert that `pyproject.toml` exposes the `packaging` extra and routes `nexusmind` to `nexusmind.desktop:main`; assert README contains `NEXUSMIND_RUNTIME_DIR`, `%USERPROFILE%\.nexusmind`, `nexusmind.log`, `build-portable.ps1`, and `nexusmind-windows-portable.zip`.

- [ ] **Step 2: Run metadata tests and verify README assertions fail**

Run: `python -m pytest tests/test_project_metadata.py -vv`

Expected: failure for missing portable runtime documentation.

- [ ] **Step 3: Add a dedicated Windows packaging job**

Append a `portable-package` job to `.github/workflows/ci.yml` that runs on `windows-latest`, uses Python 3.13, installs `.[dev,packaging]`, runs the focused runtime/desktop/packaging tests, executes `./scripts/build-portable.ps1`, verifies the ZIP, and uploads it with pinned `actions/upload-artifact` v4. Keep the existing Python 3.11–3.13 test matrix unchanged.

- [ ] **Step 4: Document runtime operation and portable distribution**

Add a README section describing the four runtime directories, the absolute environment override, the rotating structured log location and sensitive-data policy, friendly failure behavior, the `onedir` ZIP contents, the PowerShell build command `./scripts/build-portable.ps1`, and how to launch `nexusmind.exe --help`. Explicitly state that user data and models are not bundled and that installer/GUI/auto-update are outside this artifact.

- [ ] **Step 5: Run metadata and focused tests**

Run: `python -m pytest tests/test_project_metadata.py tests/test_runtime_support.py tests/test_desktop.py tests/test_portable_packaging.py -vv`

Expected: all tests pass.

- [ ] **Step 6: Commit CI and documentation**

```bash
git add .github/workflows/ci.yml README.md tests/test_project_metadata.py
git commit -m "docs: explain Windows portable runtime"
```

### Task 6: Full Verification and PR Preparation

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -vv`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Compile all Python sources**

Run: `python -m compileall -q src`

Expected: exit code 0 and no output.

- [ ] **Step 3: Validate dependency metadata**

Run: `python -m pip check`

Expected: `No broken requirements found.`

- [ ] **Step 4: Validate the patch**

Run: `git diff --check origin/main...HEAD && git status --short`

Expected: no whitespace errors; only intentional changes are present and the worktree is clean after commits.

- [ ] **Step 5: Review acceptance evidence**

Confirm the unit tests prove runtime directory separation, log creation/redaction, sync/search/query lifecycle diagnostics, and unexpected exception handling. Confirm the checked-in Windows workflow supplies the platform-specific no-system-Python executable smoke test and portable ZIP creation that cannot be executed natively on a non-Windows development host.

- [ ] **Step 6: Push and create the PR**

Push `agent/issue-101-windows-portable-runtime`, then create a PR targeting `main` with `Closes #101`, a concise implementation summary, local verification commands/results, and a note that the Windows package smoke test runs in CI.
