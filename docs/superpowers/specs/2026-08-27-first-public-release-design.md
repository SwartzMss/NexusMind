# NexusMind 0.1.0 First Public Release Design

## Goal

Turn the current NexusMind repository state into the first public `0.1.0`
distribution without adding Knowledge Runtime features. A pushed `v0.1.0` tag must
produce verified Python and Windows portable artifacts and publish them together in
a GitHub Release.

## Release Architecture

Add `.github/workflows/release.yml`, triggered only by version tags matching `v*`.
The workflow separates verification and construction into independent jobs and has
a final publication job depend on all of them:

1. Run the complete test suite on the supported Python versions.
2. Build the wheel and sdist, validate their metadata, and clean-install the wheel
   and sdist in isolated environments.
3. Build the Windows portable ZIP and smoke-test the extracted archive through the
   real CLI in multiple process invocations.
4. Download only the artifacts uploaded by successful jobs and create the GitHub
   Release with the checked-in `v0.1.0` notes.

The publication job uses GitHub CLI provided by the runner rather than introducing
another third-party release action. It receives `contents: write`; build and test
jobs retain read-only repository permissions.

## Version and Package Contract

The release workflow derives the expected version by removing the leading `v` from
the tag. Before publication it verifies that:

- the tag version and `[project].version` in `pyproject.toml` are both `0.1.0`;
- `requires-python` remains `>=3.11,<3.14`;
- the declared license is MIT;
- the generated filenames are
  `nexusmind-0.1.0-py3-none-any.whl`, `nexusmind-0.1.0.tar.gz`, and
  `nexusmind-windows-portable.zip`;
- installed package metadata reports the expected version and license;
- `nexusmind` and `nexusmind-kb` are installed and respond to `--help`;
- `python -m pip check` succeeds in each clean installation.

Use the standard Python `build` package to create the wheel and sdist. Each archive
is installed into its own new virtual environment so validation cannot import from
the source checkout or reuse dependencies from the build environment.

## Windows Portable End-to-End Test

Extend the portable build script so the archive is tested after creation from a
fresh extraction directory. The test creates a small strict UTF-8 Markdown fixture
and invokes the extracted `nexusmind.exe` as separate processes for:

```text
create -> source add -> sync -> search -> inspect
```

Commands use JSON output where available so the script can assert canonical state
instead of relying only on exit codes. The later `search` and `inspect` invocations
reopen the KnowledgeBase produced by earlier processes, proving persistence across
process boundaries. Search output must contain fixture content, and inspection must
report the synchronized source and document. Temporary smoke data is outside the
bundle and is never included in the published ZIP.

## Documentation

Restructure the top of `README.md` into a first-release path that immediately
answers what NexusMind is, which artifact to download, and how to create, populate,
sync, search, and inspect a KnowledgeBase. Keep the runnable quick start short and
move detailed explanations below it. Explicitly list the supported local input
types and first-release constraints, including the absence of Git ingestion,
background synchronization, cloud KnowledgeBases, and persisted derived indexes.

Add `docs/releases/v0.1.0.md` with concise release notes covering only capabilities
that already ship: local file/directory sources, explicit synchronization, lexical
and optional advanced retrieval, citation-validated answers, diagnostics, immutable
document history, strict persistence v1, Windows portable runtime, CLI, and Python
API. The same notes are consumed by the release workflow.

## Test Strategy

Follow test-driven development for repository changes:

- extend packaging tests to require a real extracted-archive E2E smoke flow;
- add release-workflow contract tests for tag triggering, permissions, dependency
  gates, exact artifacts, clean installs, metadata checks, and release publication;
- add documentation contract tests for the first-user quick start, artifact
  guidance, shipped capability notes, and explicit constraints;
- run focused tests after each change, then run the entire existing suite;
- validate workflow YAML syntax and PowerShell syntax where local tools permit;
- rely on the Windows GitHub runner for the real PyInstaller build and portable E2E
  that cannot execute on the local Linux workspace.

## Failure and Publication Safety

Every shell uses fail-fast behavior. Build artifacts are uploaded only after their
own metadata or E2E checks pass. The publication job declares `needs` on all test
and build jobs, checks the exact downloaded filenames, and creates the release only
after those checks. A mismatched tag, metadata value, installation, entrypoint,
portable command, or expected output fails the workflow before GitHub Release
creation.

## Non-Goals

This work does not add source adapters, document parsers, watchers, persisted
derived indexes, installers, updaters, services, GUI changes, cloud features, or an
agent/workflow runtime. Any discovered defect is changed only when it blocks the
specified release flow.
