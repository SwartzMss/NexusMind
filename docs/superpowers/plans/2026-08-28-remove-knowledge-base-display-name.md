# Remove KnowledgeBase Display Name Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the KnowledgeBase-level display name from every public creation, inspection, desktop, persistence, documentation, and test contract while preserving path-based workflows and Knowledge Source display names.

**Architecture:** Keep root path as the public locator and generated `knowledge_base_id` as the internal identity. Redefine the unreleased strict manifest v1 to contain only `format_version`, `knowledge_base_id`, and `sources`; remove the field end-to-end rather than adding compatibility or replacement-name behavior.

**Tech Stack:** Python 3.11+, argparse, dataclasses, tkinter/ttk, JSON manifest persistence, pytest, Markdown.

---

### Task 1: Remove the field from the strict manifest v1 contract

**Files:**
- Modify: `tests/test_knowledge_base_manifest.py`
- Modify: `src/nexusmind/knowledge_base_manifest.py`

- [ ] **Step 1: Rewrite manifest tests for the three-key root contract**

Remove `max_display_name_chars` and `display_name` from manifest fixtures, delete display-name validation/limit tests, and change the canonical encoding assertion to:

```python
assert encode_manifest(manifest()) == (
    b'{"format_version":"1","knowledge_base_id":"kb","sources":[]}\n'
)
```

Add an explicit strict-schema case:

```python
def test_decode_rejects_removed_display_name_field() -> None:
    root = {
        "format_version": "1",
        "knowledge_base_id": "kb",
        "display_name": None,
        "sources": [],
    }
    with pytest.raises(KnowledgeBaseConfigError, match="manifest fields"):
        decode_manifest(json.dumps(root).encode())
```

- [ ] **Step 2: Run the manifest tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_base_manifest.py -q`

Expected: failures show encoded JSON still contains `display_name`, the dataclass still exposes it, and decoding still accepts it.

- [ ] **Step 3: Remove the field from the manifest implementation**

In `KnowledgeBaseLimits`, delete `max_display_name_chars`. In `KnowledgeBaseManifest`, delete `display_name` and its validation. Set the exact root keys and encoder payload to:

```python
_ROOT_KEYS = frozenset({"format_version", "knowledge_base_id", "sources"})

root = {
    "format_version": manifest.format_version,
    "knowledge_base_id": manifest.knowledge_base_id,
    "sources": [_encode_source(source) for source in manifest.sources],
}
```

Construct decoded manifests without a display name:

```python
return KnowledgeBaseManifest(
    knowledge_base_id=root["knowledge_base_id"],
    sources=tuple(sources),
    limits=limits,
)
```

- [ ] **Step 4: Run manifest tests and verify GREEN**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_base_manifest.py -q`

Expected: all manifest tests pass.

- [ ] **Step 5: Commit the persistence contract change**

```bash
git add src/nexusmind/knowledge_base_manifest.py tests/test_knowledge_base_manifest.py
git commit -m "refactor: remove knowledge base name from manifest"
```

### Task 2: Remove the field from the Python API and inspection values

**Files:**
- Modify: `tests/test_knowledge_base.py`
- Modify: `tests/test_knowledge_base_diagnostics.py`
- Modify: `src/nexusmind/knowledge_base.py`
- Modify: `src/nexusmind/knowledge_inspection.py`

- [ ] **Step 1: Update public API and status tests**

Change creation tests to call:

```python
kb = KnowledgeBase.create(str(root), knowledge_base_id="security")
assert kb.status().knowledge_base_id == "security"
assert not hasattr(kb.status(), "display_name")
```

Add rejection coverage:

```python
def test_create_rejects_removed_display_name(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="display_name"):
        KnowledgeBase.create(str(tmp_path / "kb"), display_name="Name")  # type: ignore[call-arg]
```

Remove KnowledgeBase-level display-name validation cases and expected diagnostic fields. Do not alter `KnowledgeSource(display_name=...)` fixtures.

- [ ] **Step 2: Run focused API tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_base.py tests/test_knowledge_base_diagnostics.py -q`

Expected: failures show `create` still accepts the keyword and status/manifest construction still uses the field.

- [ ] **Step 3: Remove the API and status plumbing**

Change the factory signature to omit the parameter:

```python
@classmethod
def create(
    cls,
    root: str,
    *,
    knowledge_base_id: str | None = None,
    index_factory: Callable[[], KnowledgeIndex] | None = None,
    limits: KnowledgeBaseLimits | None = None,
) -> KnowledgeBase:
```

Construct `KnowledgeBaseManifest` with only its internal ID, sources, and limits. Remove `display_name` from `KnowledgeBaseStatus` and every status/inspection constructor in `knowledge_base.py`:

```python
@dataclass(frozen=True, slots=True)
class KnowledgeBaseStatus:
    knowledge_base_id: str
    source_count: int
    document_count: int
```

Preserve all source-level `inspection.source.display_name` flows.

- [ ] **Step 4: Run API tests and dependent inspection tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_base.py tests/test_knowledge_base_diagnostics.py tests/test_knowledge_inspection.py tests/test_knowledge_diagnostics.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the public Python contract change**

```bash
git add src/nexusmind/knowledge_base.py src/nexusmind/knowledge_inspection.py tests/test_knowledge_base.py tests/test_knowledge_base_diagnostics.py
git commit -m "refactor: remove knowledge base display name API"
```

### Task 3: Make CLI creation path-only

**Files:**
- Modify: `tests/test_knowledge_cli.py`
- Modify: `src/nexusmind/cli.py`

- [ ] **Step 1: Add path-only and removed-option CLI tests**

Change the successful create invocation to:

```python
assert cli.main(["create", str(root)]) == 0
```

Add parser assertions:

```python
def test_create_help_has_no_name_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli._parser().parse_args(["create", "--help"])
    assert raised.value.code == 0
    assert "--name" not in capsys.readouterr().out


def test_create_rejects_name_option() -> None:
    with pytest.raises(SystemExit) as raised:
        cli._parser().parse_args(["create", "kb", "--name", "Name"])
    assert raised.value.code == 2
```

Update inspection JSON assertions to require `"display_name" not in inspection["status"]`.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_cli.py -q`

Expected: help still lists `--name`, parsing accepts it, and create still forwards it.

- [ ] **Step 3: Remove CLI parser and rendering references**

Define creation with only the positional path:

```python
create = subparsers.add_parser("create")
create.add_argument("path")
```

Call `KnowledgeBase.create(args.path)` and remove KnowledgeBase status display-name rendering. Keep `Path(args.knowledge_base)` wherever it reports the actual root path, but do not treat its basename as a replacement name.

- [ ] **Step 4: Run CLI workflow tests and verify GREEN**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_cli.py tests/test_knowledge_query_cli.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the CLI contract change**

```bash
git add src/nexusmind/cli.py tests/test_knowledge_cli.py
git commit -m "refactor: make knowledge base creation path-only"
```

### Task 4: Remove the desktop display-name workflow

**Files:**
- Modify: `tests/test_knowledge_base_ui.py`
- Modify: `src/nexusmind/knowledge_base_ui.py`

- [ ] **Step 1: Rewrite controller and layout tests**

Change controller creation coverage so the fake receives only the root:

```python
controller.create("/tmp/kb")
assert created == [{"root": "/tmp/kb"}]
```

Replace the display-name layout test with:

```python
def test_create_form_only_asks_for_root() -> None:
    app, entries = build_app_fixture()
    assert hasattr(app, "root")
    assert not hasattr(app, "display_name")
    assert len(entries) == 1
```

Update status fixtures and assertions so KnowledgeBase status has no display-name member. Leave source rows and their display names unchanged.

- [ ] **Step 2: Run desktop tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_base_ui.py tests/test_desktop.py -q`

Expected: controller, layout, and status rendering tests fail against the existing name field.

- [ ] **Step 3: Remove the controller/UI field and status label**

Change the controller method to:

```python
def create(self, root: str) -> None:
    self._run(lambda: self._create(root))
```

Delete the `StringVar`, label, entry, and `_on_create` argument forwarding for `display_name`. Remove KnowledgeBase name rendering from status refresh; retain root, counts, and source display names.

- [ ] **Step 4: Run desktop tests and verify GREEN**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_knowledge_base_ui.py tests/test_desktop.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the desktop change**

```bash
git add src/nexusmind/knowledge_base_ui.py tests/test_knowledge_base_ui.py
git commit -m "refactor: remove knowledge base name from desktop"
```

### Task 5: Update public and release-facing documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/releases/v0.1.0.md`
- Modify: `docs/superpowers/specs/2026-08-27-first-release-schema-design.md`
- Modify: `docs/superpowers/plans/2026-08-27-first-release-schema.md`
- Modify: `tests/test_project_metadata.py`
- Modify: `tests/test_release_documentation.py`

- [ ] **Step 1: Add or update documentation contract assertions**

In project metadata coverage, assert the current public examples are path-only:

```python
readme = (ROOT / "README.md").read_text(encoding="utf-8")
assert "nexusmind create ./security-kb\n" in readme
assert '--name "Security Notes"' not in readme
assert 'display_name="Security Notes"' not in readme
```

Where release/schema tests assert manifest fields, require:

```python
{"format_version", "knowledge_base_id", "sources"}
```

- [ ] **Step 2: Run documentation/metadata tests and verify RED**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_metadata.py tests/test_release_documentation.py -q`

Expected: stale README and schema text fail the new assertions.

- [ ] **Step 3: Update documentation precisely**

Use path-only examples:

```text
nexusmind create ./security-kb
```

Use path-only Python creation:

```python
kb = KnowledgeBase.create("./security-kb")
```

Describe the manifest v1 root contract as `format_version`, generated `knowledge_base_id`, and `sources`. Remove statements that a KnowledgeBase has an optional display name. Keep all documentation about Knowledge Source display names.

- [ ] **Step 4: Scan for stale KnowledgeBase-level references**

Run:

```bash
rg -n 'create .*--name|KnowledgeBase\.create\([^\n]*display_name|knowledge_base.*display_name|display_name.*knowledge_base' README.md docs src tests
```

Expected: no KnowledgeBase-level matches; inspect any remaining matches and retain only Knowledge Source uses or the deliberate removed-field rejection tests/spec history.

- [ ] **Step 5: Run metadata tests and commit documentation**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_metadata.py tests/test_release_documentation.py -q`

Expected: all selected tests pass.

```bash
git add README.md docs tests/test_project_metadata.py tests/test_release_documentation.py
git commit -m "docs: document path-only knowledge base creation"
```

### Task 6: Complete regression verification and prepare the PR

**Files:**
- Modify only if a regression reveals a missing issue-scoped update.

- [ ] **Step 1: Run all KnowledgeBase-facing regression tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_knowledge_base_manifest.py \
  tests/test_knowledge_base.py \
  tests/test_knowledge_base_diagnostics.py \
  tests/test_knowledge_cli.py \
  tests/test_knowledge_query_cli.py \
  tests/test_knowledge_base_ui.py \
  tests/test_desktop.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the full locally supported suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q --ignore=tests/test_release_workflow.py`

Expected: all tests pass, with only existing declared skips. The ignored file requires Python 3.11 `tomllib`, unavailable in the local Python 3.10 environment.

- [ ] **Step 3: Verify diff hygiene and strict scope**

Run:

```bash
git diff origin/main --check
git status --short
git diff --stat origin/main
rg -n 'display_name' src/nexusmind README.md docs
```

Expected: no whitespace errors; only intended files are changed; remaining source matches belong only to Knowledge Sources.

- [ ] **Step 4: Request code review and address verified findings**

Invoke `superpowers:requesting-code-review`. Review the complete `origin/main..HEAD` diff against issue 109 and the design spec. Apply only findings supported by code or tests, then rerun Steps 1-3.

- [ ] **Step 5: Push and create the pull request**

```bash
git push -u origin codex/issue-109-remove-kb-display-name
gh pr create --repo SwartzMss/NexusMind --base main \
  --head codex/issue-109-remove-kb-display-name \
  --title "Remove KnowledgeBase display name from public workflow" \
  --body-file /tmp/issue-109-pr-body.md
```

The PR body must summarize the path-only contract, strict manifest v1 change,
desktop removal, documentation updates, verification results, local Python 3.10
release-test limitation, and include `Closes #109`.
