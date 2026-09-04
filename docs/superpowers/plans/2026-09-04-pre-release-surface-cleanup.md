# Pre-release Surface Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Remove the unreleased Tkinter UI and Python <3.11 compatibility baggage while preserving the nexusmind CLI, Python API, runtime resilience, and persistence safety.

**Architecture:** Keep src/nexusmind/desktop.py as the CLI/runtime boundary and the PyInstaller entry unchanged. Delete the UI module/tests, expose only the nexusmind script, remove tomli, and align current docs, release validation, packaging tests, and metadata tests. Historical docs/superpowers plans/specs are out of scope.

**Tech Stack:** Python 3.11–3.13, pytest, Hatchling, PyInstaller, GitHub Actions YAML, Markdown.

---

## File map

- Delete: src/nexusmind/knowledge_base_ui.py and tests/test_knowledge_base_ui.py
- Modify: pyproject.toml, requirements.txt, .github/workflows/release.yml
- Modify: tests/test_project_metadata.py, tests/test_release_workflow.py, tests/test_portable_packaging.py
- Modify: README.md, docs/architecture.md, docs/releases/v0.1.0.md
- Create: none

Historical docs/superpowers plans/specs remain unchanged.

## Task 1: Add failing contract tests

**Files:** tests/test_project_metadata.py, tests/test_release_workflow.py, tests/test_portable_packaging.py

- [ ] **Step 1:** Replace the test-side tomllib try/except with:

~~~
import tomllib
~~~

Rename the script test to test_project_exposes_only_the_supported_cli_entrypoint and assert:

~~~
assert project["scripts"] == {"nexusmind": "nexusmind.desktop:main"}
~~~

- [ ] **Step 2:** Add a stale-reference test that reads exactly README.md, docs/architecture.md, docs/releases/v0.1.0.md, pyproject.toml, requirements.txt, .github/workflows/ci.yml, .github/workflows/release.yml, and packaging/nexusmind.spec. Assert none contains knowledge_base_ui, nexusmind-kb, or tkinter. Exclude tests and historical docs so the check cannot self-match.

- [ ] **Step 3:** Make release fixtures create only nexusmind and require nexusmind --help while rejecting nexusmind-kb. Rename test_pyinstaller_spec_targets_desktop_entry_and_onedir to test_pyinstaller_spec_targets_cli_runtime_entry_and_onedir; keep the desktop.py, COLLECT(, and console=True assertions.

- [ ] **Step 4:** Run:

~~~
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_project_metadata.py tests/test_release_workflow.py tests/test_portable_packaging.py -q
~~~

Expected: old-contract failures identify metadata/workflow/document references. Python 3.10 may additionally fail embedded release code at import tomllib; this is an unsupported-environment limitation.

## Task 2: Remove the UI and unsupported dependency

**Files:** src/nexusmind/knowledge_base_ui.py, tests/test_knowledge_base_ui.py, pyproject.toml, requirements.txt

- [ ] **Step 1:** Delete only the UI module and UI-only test. Keep desktop.py, cli.py, tests/test_desktop.py, and all core KnowledgeBase/retrieval/query/evaluation modules.

- [ ] **Step 2:** Delete tomli>=2,<3; python_version < '3.11' from requirements.txt and its quoted equivalent from pyproject.toml. Keep requires-python >=3.11,<3.14 and runtime/provider/extraction/query-expansion/platform/persistence fallback behavior.

- [ ] **Step 3:** Verify and commit:

~~~
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -c "import tomllib; import nexusmind.desktop; print('supported imports ok')"
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_desktop.py tests/test_knowledge_cli.py tests/test_knowledge_query_cli.py -q
git add src/nexusmind/knowledge_base_ui.py tests/test_knowledge_base_ui.py pyproject.toml requirements.txt
git commit -m "refactor: remove unreleased desktop UI surface"
~~~

## Task 3: Align release validation and current documentation

**Files:** .github/workflows/release.yml, tests/test_release_workflow.py, tests/test_project_metadata.py, README.md, docs/architecture.md, docs/releases/v0.1.0.md

- [ ] **Step 1:** In both release clean-install snippets change the entry-point loop from ("nexusmind", "nexusmind-kb") to ("nexusmind",). Preserve version, license, Requires-Python, artifact, publication, and portable build checks.

- [ ] **Step 2:** Remove README desktop/Tkinter instructions and nexusmind-kb usage; rename the install heading to 开发环境安装; change the capability statement to CLI + Python API. Replace the architecture product-entry row with desktop.py / cli.py. Update v0.1.0 notes to mention only CLI + Python API and nexusmind.

- [ ] **Step 3:** Run and commit:

~~~
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_project_metadata.py tests/test_release_workflow.py tests/test_portable_packaging.py -q
git diff --check
git add .github/workflows/release.yml tests/test_release_workflow.py tests/test_project_metadata.py README.md docs/architecture.md docs/releases/v0.1.0.md
git commit -m "build: publish the cli-only first-release surface"
~~~

Expected on Python 3.11–3.13: all selected tests pass.

## Task 4: Audit retained behavior and verify

- [ ] **Step 1:** Run:

~~~
rg -n "knowledge_base_ui|nexusmind-kb|tkinter|python_version < ['\"]3\\.11|import tomli|ModuleNotFoundError.*tomllib" --glob '!docs/superpowers/**' --glob '!*.pyc' .
~~~

No current product/release reference may remain. Review fallback, legacy, migration, compatible, and platform matches individually. Retain query-expansion/extraction fallback, optional backend/provider degradation, Windows path/runtime handling, manifest/SQLite validation, and document-version safety.

- [ ] **Step 2:** Run retained-safety tests:

~~~
PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_query_expansion.py tests/test_document_extraction.py tests/test_knowledge_base_manifest.py tests/test_knowledge_store.py tests/test_knowledge_versioning.py tests/test_runtime_support.py -q
~~~

- [ ] **Step 3:** Run the non-release suite, compileall, git diff --check origin/main...HEAD, and the full suite. Record the available Python 3.10 release-workflow failures at stdlib tomllib as environment-only; never restore tomli. Supported CI remains Python 3.11, 3.12, and 3.13.

- [ ] **Step 4:** Prepare the PR audit table: UI/nexusmind-kb removed because no public release contract exists; tomli shim removed because support starts at 3.11; list retained resilience, platform, and persistence behavior.

## Task 5: Review and create the PR

- [ ] **Step 1:** Use superpowers:requesting-code-review against origin/main and the final branch HEAD. Check issue #132 acceptance criteria, deletion scope, single-entrypoint packaging, Python matrix, and retained safety.

- [ ] **Step 2:** With clean status, rerun focused contract tests and the non-release suite.

- [ ] **Step 3:** Push agent/issue-132-pre-release-cleanup and create a PR titled refactor: remove desktop UI and pre-release compatibility baggage, with Closes #132, the audit table, retained-items rationale, and test results in the body.

- [ ] **Step 4:** Report the PR URL and pending CI status; leave the worktree available for review.

