# NexusMind 0.1.0 First Public Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish verified Python and Windows portable artifacts from a `v0.1.0` tag through a gated GitHub Release workflow.

**Architecture:** A tag-driven workflow runs the supported test matrix, builds and clean-installs Python distributions, builds and exercises the Windows portable archive, then publishes only artifacts produced by successful upstream jobs. Repository contract tests pin the workflow, packaging smoke path, README guidance, and release notes without adding runtime behavior.

**Tech Stack:** Python 3.11-3.13, pytest, Hatchling/`build`, PowerShell, PyInstaller, GitHub Actions, GitHub CLI

---

## File Structure

- Create `.github/workflows/release.yml`: tag trigger, test/build/smoke jobs, gated publication.
- Modify `scripts/build-portable.ps1`: extracted-ZIP, multi-process KnowledgeBase E2E.
- Create `docs/releases/v0.1.0.md`: release notes consumed by the workflow.
- Modify `README.md`: artifact selection, short quick start, explicit constraints.
- Create `tests/test_release_workflow.py`: release workflow contract tests.
- Modify `tests/test_portable_packaging.py`: portable E2E script contract tests.
- Create `tests/test_release_documentation.py`: README and release-note contract tests.

### Task 1: Windows portable multi-process E2E

**Files:**
- Modify: `tests/test_portable_packaging.py`
- Modify: `scripts/build-portable.ps1`

- [ ] **Step 1: Write the failing test**

Append this packaging contract:

```python
def test_portable_script_runs_extracted_archive_knowledge_base_e2e() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")
    required = (
        "Expand-Archive", "smoke-fixture.md", '"create"',
        '"source", "add"', '"sync"', '"search"', '"inspect"',
        '"--json"', "ConvertFrom-Json", "document_count",
    )
    for marker in required:
        assert marker in text
    assert text.index("Compress-Archive") < text.index("Expand-Archive")
```

- [ ] **Step 2: Verify RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_portable_packaging.py::test_portable_script_runs_extracted_archive_knowledge_base_e2e -q`

Expected: FAIL because the script does not extract or exercise the ZIP.

- [ ] **Step 3: Add the E2E implementation**

After archiving, expand into a clean `build/release-smoke` root, create a UTF-8
Markdown fixture with a unique token, and invoke the extracted executable in five
separate processes. Use this helper and assertions:

```powershell
function Invoke-PortableCommand {
    param([string[]]$Arguments)
    $output = & $smokeExecutable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Portable command failed: $($Arguments -join ' ')" }
    return ($output -join "`n")
}
Invoke-PortableCommand @("create", $knowledgeBasePath, "--name", "Release Smoke")
Invoke-PortableCommand @("source", "add", $fixturePath, "--knowledge-base", $knowledgeBasePath)
Invoke-PortableCommand @("sync", "--knowledge-base", $knowledgeBasePath, "--json")
$search = Invoke-PortableCommand @("search", "release-smoke-token", "--knowledge-base", $knowledgeBasePath, "--json") | ConvertFrom-Json
$inspection = Invoke-PortableCommand @("inspect", "--knowledge-base", $knowledgeBasePath, "--json") | ConvertFrom-Json
if ($search.Count -lt 1 -or $inspection.status.document_count -ne 1) {
    throw "Portable end-to-end smoke test did not reopen canonical state"
}
```

Keep all fixture and KnowledgeBase paths outside the bundle and clean the smoke root
in `finally` so no smoke state enters the ZIP.

- [ ] **Step 4: Verify GREEN**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_portable_packaging.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add tests/test_portable_packaging.py scripts/build-portable.ps1 && git commit -m "test: smoke test portable release lifecycle"`

### Task 2: Gated tag release workflow

**Files:**
- Create: `tests/test_release_workflow.py`
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write failing workflow tests**

```python
from pathlib import Path

WORKFLOW = Path(".github/workflows/release.yml")

def test_release_workflow_is_tag_driven_and_gated() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for marker in ('tags:', '- "v*"', 'needs: [tests, python-package, windows-portable]', 'contents: write'):
        assert marker in text

def test_release_workflow_verifies_and_publishes_exact_artifacts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for marker in (
        "nexusmind-0.1.0-py3-none-any.whl", "nexusmind-0.1.0.tar.gz",
        "nexusmind-windows-portable.zip", ">=3.11,<3.14", "license",
        "python -m pip check", "nexusmind --help", "nexusmind-kb --help",
        "gh release create", "docs/releases/v0.1.0.md",
    ):
        assert marker in text
```

Add a YAML parse test using `yaml.BaseLoader` when PyYAML is available, preventing
YAML 1.1 from coercing the `on` key.

- [ ] **Step 2: Verify RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_release_workflow.py -q`

Expected: FAIL because the workflow is absent.

- [ ] **Step 3: Implement the workflow**

Start with this job graph:

```yaml
on:
  push:
    tags: ["v*"]
permissions:
  contents: read
jobs:
  tests: {}
  python-package: {}
  windows-portable: {}
  publish:
    needs: [tests, python-package, windows-portable]
    permissions:
      contents: write
```

Use the same pinned action SHAs as `ci.yml`. The test job runs all tests on Python
3.11, 3.12, and 3.13. The Python job verifies the tag, `pyproject.toml` version,
Python range, and MIT license; runs `python -m build`; asserts exact filenames; and
clean-installs wheel and sdist into separate new virtual environments. Each clean
environment checks installed version/license metadata, `nexusmind --help`,
`nexusmind-kb --help`, and `python -m pip check`. Only verified archives are
uploaded. The Windows job installs `.[dev,packaging]`, runs packaging tests, builds
the portable ZIP, and uploads only the smoke-tested ZIP.

The publish job downloads all three files, rejects missing or extra files, and runs:

```bash
gh release create "$GITHUB_REF_NAME" release-artifacts/* --verify-tag --title "NexusMind $GITHUB_REF_NAME" --notes-file docs/releases/v0.1.0.md
```

- [ ] **Step 4: Verify GREEN**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_release_workflow.py tests/test_project_metadata.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add tests/test_release_workflow.py .github/workflows/release.yml && git commit -m "ci: add verified 0.1.0 release workflow"`

### Task 3: First-user documentation and release notes

**Files:**
- Create: `tests/test_release_documentation.py`
- Modify: `README.md`
- Create: `docs/releases/v0.1.0.md`

- [ ] **Step 1: Write failing documentation tests**

```python
from pathlib import Path

def test_readme_top_covers_first_release_download_and_cli_path() -> None:
    top = Path("README.md").read_text(encoding="utf-8")[:5000]
    for marker in (
        "nexusmind-windows-portable.zip", "nexusmind-0.1.0-py3-none-any.whl",
        "nexusmind create", "nexusmind source add", "nexusmind sync",
        "nexusmind search", "nexusmind inspect", ".txt", ".md", ".markdown",
    ):
        assert marker in top

def test_release_notes_cover_capabilities_and_constraints() -> None:
    notes = Path("docs/releases/v0.1.0.md").read_text(encoding="utf-8")
    for marker in (
        "BM25", "Semantic", "Hybrid-RRF", "reranking", "validated citations",
        "immutable document version history", "manifest/SQLite persistence v1",
        "strict UTF-8", "no Git source adapter", "no background synchronization",
        "no cloud-hosted KnowledgeBase", "rebuilt rather than persisted",
    ):
        assert marker in notes
```

- [ ] **Step 2: Verify RED**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_release_documentation.py -q`

Expected: FAIL because the README top lacks release artifact guidance and notes are absent.

- [ ] **Step 3: Implement documentation**

At the README top add, in order: a one-paragraph product description, artifact
selection, a runnable five-command CLI path using a UTF-8 fixture, and a compact
supported/not-supported list. Preserve the existing detailed CLI, API, storage,
runtime, and build documentation below it. Write `docs/releases/v0.1.0.md` with
“Highlights” and “First release constraints” sections covering every item in issue
#107 without claiming new features.

- [ ] **Step 4: Verify GREEN**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_release_documentation.py tests/test_project_metadata.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add tests/test_release_documentation.py README.md docs/releases/v0.1.0.md && git commit -m "docs: add 0.1.0 release quick start"`

### Task 4: Integrate, verify, and open the PR

**Files:**
- Modify only files needed for release-blocking defects discovered by verification.

- [ ] **Step 1: Run the full suite**

Run: `/home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest -q`

Expected: PASS with only intentional existing skips.

- [ ] **Step 2: Check repository hygiene**

Run: `git diff --check origin/main...HEAD` and `git status --short`.

Expected: no whitespace errors and no uncommitted files.

- [ ] **Step 3: Review the complete diff**

Run: `git diff --stat origin/main...HEAD` and `git diff origin/main...HEAD`.

Expected: only design/plan, workflow, portable smoke, tests, README, and release
notes change; no Knowledge Runtime feature surface changes.

- [ ] **Step 4: Push and open the pull request**

Run: `git push -u origin codex/issue-107-first-public-release`.

Then create a PR titled `Prepare NexusMind 0.1.0 first public release` targeting
`main`. The body summarizes the gated release flow and test evidence, notes that the
real PyInstaller run occurs on Windows CI, and contains `Closes #107`.
