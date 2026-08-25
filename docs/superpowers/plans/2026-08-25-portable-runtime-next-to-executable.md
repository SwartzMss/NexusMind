# Portable Runtime Next to Executable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a frozen Windows `nexusmind.exe` default to a `.nexusmind` runtime directory beside the executable while preserving environment overrides and source-development behavior.

**Architecture:** Extend the existing runtime-root resolver with a PyInstaller-frozen branch based on `sys.frozen` and `sys.executable`, after the existing absolute environment override. Update the real Windows packaging smoke test to prove the location is executable-relative rather than cwd-relative, then align documentation and metadata tests.

**Tech Stack:** Python 3.11–3.13, `pathlib`, PyInstaller, PowerShell, pytest, GitHub Actions.

---

### Task 1: Frozen Runtime Root Resolution

**Files:**
- Modify: `tests/test_runtime_support.py`
- Modify: `src/nexusmind/runtime_support.py`

- [ ] **Step 1: Write failing frozen-resolution tests**

Add tests that simulate PyInstaller without requiring a frozen process:

```python
import sys


def test_resolve_runtime_root_uses_frozen_executable_directory(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "portable" / "nexusmind.exe"
    monkeypatch.delenv("NEXUSMIND_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    assert resolve_runtime_root() == executable.parent / ".nexusmind"


def test_absolute_override_wins_over_frozen_default(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "managed"
    monkeypatch.setenv("NEXUSMIND_RUNTIME_DIR", str(override))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "portable" / "nexusmind.exe"))

    assert resolve_runtime_root() == override
```

- [ ] **Step 2: Run the focused tests and verify the frozen default fails**

Run: `python -m pytest tests/test_runtime_support.py -vv`

Expected: the frozen-default test reports the user-home path instead of the executable-adjacent path.

- [ ] **Step 3: Implement the frozen-aware precedence**

Update `resolve_runtime_root()` to keep the override first and add the frozen branch:

```python
import sys


def resolve_runtime_root() -> Path:
    """Resolve the absolute mutable-data root without depending on cwd."""
    configured = os.getenv("NEXUSMIND_RUNTIME_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    elif getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent / ".nexusmind"
    else:
        root = Path.home() / ".nexusmind"
    if not root.is_absolute():
        raise RuntimeLayoutError("NEXUSMIND_RUNTIME_DIR must be an absolute path")
    return root
```

- [ ] **Step 4: Run runtime-support and desktop tests**

Run: `python -m pytest tests/test_runtime_support.py tests/test_desktop.py -vv`

Expected: all tests pass; non-frozen home behavior and absolute override behavior remain green.

- [ ] **Step 5: Commit runtime resolution**

```bash
git add src/nexusmind/runtime_support.py tests/test_runtime_support.py
git commit -m "feat: colocate frozen runtime with executable"
```

### Task 2: Executable-Relative Windows Smoke Test

**Files:**
- Modify: `tests/test_portable_packaging.py`
- Modify: `scripts/build-portable.ps1`

- [ ] **Step 1: Write a failing packaging-script contract test**

Replace the old assertion that the script sets `NEXUSMIND_RUNTIME_DIR` with assertions that it clears the override, launches from an external working directory, and checks the adjacent log:

```python
def test_portable_script_smoke_tests_executable_relative_runtime() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")

    assert "Remove-Item Env:NEXUSMIND_RUNTIME_DIR" in text
    assert "Push-Location $smokeWorkingDirectory" in text
    assert 'Join-Path $bundlePath ".nexusmind\\logs\\nexusmind.log"' in text
```

- [ ] **Step 2: Run the contract test and verify it fails**

Run: `python -m pytest tests/test_portable_packaging.py -vv`

Expected: failure because the current script sets an explicit smoke runtime under `build/`.

- [ ] **Step 3: Update the PowerShell smoke test**

Set `$smokeWorkingDirectory` to an absolute directory under `build`, create it,
remove `Env:NEXUSMIND_RUNTIME_DIR` if present, `Push-Location` into that external
directory only while invoking `$executablePath --help`, then check:

```powershell
$smokeLog = Join-Path $bundlePath ".nexusmind\logs\nexusmind.log"
```

Keep the existing executable, exit-code, log-file, ZIP, and cleanup checks.

- [ ] **Step 4: Run packaging tests**

Run: `python -m pytest tests/test_portable_packaging.py -vv`

Expected: all tests pass.

- [ ] **Step 5: Commit the Windows acceptance check**

```bash
git add scripts/build-portable.ps1 tests/test_portable_packaging.py
git commit -m "test: verify adjacent portable runtime layout"
```

### Task 3: Documentation and Metadata Contract

**Files:**
- Modify: `tests/test_project_metadata.py`
- Modify: `README.md`

- [ ] **Step 1: Change the README contract test first**

Require the adjacent layout and migration warning instead of the old profile path:

```python
for required in (
    "nexusmind\\.nexusmind\\",
    "NEXUSMIND_RUNTIME_DIR",
    "nexusmind.log",
    "不会自动迁移",
    "删除整个解压目录",
):
    assert required in readme
assert r"%USERPROFILE%\.nexusmind\" not in readme
```

- [ ] **Step 2: Run metadata tests and verify documentation failure**

Run: `python -m pytest tests/test_project_metadata.py -vv`

Expected: failure because README still documents `%USERPROFILE%\.nexusmind` as the portable default.

- [ ] **Step 3: Update portable runtime documentation**

Change the README tree to show `.nexusmind` beside `nexusmind.exe` and
`_internal`. State that the executable directory must be writable, the override
still accepts an absolute path, existing profile data is not migrated
automatically, and deleting/replacing the extracted directory can delete its
adjacent data unless `.nexusmind` is preserved.

- [ ] **Step 4: Run metadata and focused regression tests**

Run: `python -m pytest tests/test_project_metadata.py tests/test_runtime_support.py tests/test_portable_packaging.py -vv`

Expected: all tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md tests/test_project_metadata.py
git commit -m "docs: explain executable-adjacent runtime data"
```

### Task 4: Full Verification and PR

**Files:**
- Verify: all changed files

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -vv`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Compile source and check dependencies**

Run: `python -m compileall -q src && python -m pip check`

Expected: compile exits 0; pip reports no broken requirements.

- [ ] **Step 3: Validate the complete patch**

Run: `git diff --check origin/main...HEAD && git status --short`

Expected: no whitespace errors and a clean worktree after commits.

- [ ] **Step 4: Push and create the PR**

Push `agent/portable-runtime-next-to-executable` and create a PR targeting
`main`. Summarize the frozen-only default change, compatibility behavior,
Windows smoke test, and upgrade warning. Include local verification results.
