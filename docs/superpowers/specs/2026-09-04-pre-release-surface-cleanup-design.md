# Pre-release Surface Cleanup Design

## Goal

Reduce NexusMind to its intended first-release surface: the `nexusmind` CLI,
the Python API, and the core KnowledgeBase/retrieval/query/evaluation runtime.
Remove the unreleased Tkinter product interface and compatibility baggage while
preserving runtime resilience, supported-platform behavior, and persistence
safety.

## Scope and boundaries

This change is intentionally limited to the current product and release
contract. It will remove the Tkinter UI implementation and its UI-only tests,
the `nexusmind-kb` console script, the Python <3.11 `tomli` shim, and assertions
that publish either obsolete surface. It will update the current README,
architecture documentation, release notes, packaging metadata tests, release
workflow tests, and portable packaging expectations as needed.

The existing `src/nexusmind/desktop.py` path remains the CLI/runtime entry
point. Its name is historical, but renaming it would add entrypoint and
packaging churn without improving the first-release contract. Historical
`docs/superpowers/plans/` and `docs/superpowers/specs/` are records of prior
engineering work and are explicitly out of scope, even when they mention the
old UI or compatibility choices.

## Audit classification

The audit uses these classifications before deleting anything:

| Classification | Treatment in this change |
| --- | --- |
| Compatibility for an unreleased interface | Remove or simplify |
| Runtime resilience / graceful degradation | Keep |
| Persistence or data safety | Keep unless proven unnecessary |
| Current supported-platform behavior | Keep |

Items identified for removal are:

- `src/nexusmind/knowledge_base_ui.py` and `tests/test_knowledge_base_ui.py`;
- the `nexusmind-kb` project script and release clean-install assertions;
- `tomli` dependency declarations and the test-side `tomllib` fallback;
- current documentation that presents a desktop interface as a product entry
  point.

Items intentionally retained include query-expansion fallback, document
extraction fallback, Windows path identity and runtime-layout handling,
optional backend/provider degradation, and manifest/SQLite integrity and
current-document safety checks. These are behavior needed by supported
runtimes or data safety, not compatibility aliases.

## Resulting architecture

The executable boundary remains:

```text
nexusmind console script
        |
        v
nexusmind.desktop.main
        |
        +-- runtime_support (runtime directory and logging)
        +-- cli.main (CLI commands)
                    |
                    +-- KnowledgeBase / retrieval / query / evaluation APIs
```

There is no GUI adapter or second console entry point. The portable PyInstaller
bundle continues to target `src/nexusmind/desktop.py` and produces the
`nexusmind` executable.

## Packaging and release contract

`pyproject.toml` and `requirements.txt` will contain only dependencies needed
by the supported Python 3.11–3.13 matrix. The project will expose only
`nexusmind = "nexusmind.desktop:main"`. Release workflow clean-install checks
will verify only that entry point; wheel and sdist metadata checks remain
unchanged otherwise.

README and current release notes will describe CLI + Python API usage and will
not mention Tkinter, `nexusmind-kb`, or a desktop product interface. Current
architecture navigation will list `cli.py` and `desktop.py` as the executable
boundary. Historical plan/spec documents remain untouched.

## Testing strategy

Tests will be changed with the contract:

- delete UI-only tests together with the deleted module;
- assert project metadata exposes only `nexusmind` and has no `tomli` shim;
- assert release workflow verifies only `nexusmind` in clean installs;
- keep `tests/test_desktop.py` and portable packaging tests focused on the
  retained CLI/runtime boundary;
- add a repository-facing stale-reference check for current product/release
  files, while excluding historical superpowers records;
- run the full supported test suite on the available environment and the
  existing Python 3.11/3.12/3.13 CI matrix.

The acceptance signal is zero stale current-contract references, unchanged
core retrieval/query behavior, and a passing supported test matrix.
