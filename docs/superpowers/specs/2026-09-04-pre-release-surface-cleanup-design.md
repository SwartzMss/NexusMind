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

The executable boundary is named `src/nexusmind/runtime_entrypoint.py` so the
first release does not publish a historical `desktop.py` module. Historical
`docs/superpowers/plans/` and `docs/superpowers/specs/` records outside this
current design and plan are out of scope, even when they mention the old UI or
compatibility choices.

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

The public first-release context API is also explicit: `build_context()` takes
`retrieval_limit=10` and `max_passages=10` as separate controls and no longer
accepts `limit`. `WhitespaceLexicalAnalyzer` is retained only as a deterministic
benchmark comparison control. `Chunk` requires explicit `heading_path`,
`section_title`, and `source_location`; plain-text chunkers pass empty values
explicitly, while structure-aware chunkers populate them for context expansion
and provenance.

## Resulting architecture

The executable boundary remains:

```text
nexusmind console script
        |
        v
nexusmind.runtime_entrypoint.main
        |
        +-- runtime_support (runtime directory and logging)
        +-- cli.main (CLI commands)
                    |
                    +-- KnowledgeBase / retrieval / query / evaluation APIs
```

There is no GUI adapter or second console entry point. The portable PyInstaller
bundle targets `src/nexusmind/runtime_entrypoint.py` and produces the
`nexusmind` executable.

## Packaging and release contract

`pyproject.toml` and `requirements.txt` will contain only dependencies needed
by the supported Python 3.11–3.13 matrix. The project will expose only
`nexusmind = "nexusmind.runtime_entrypoint:main"`. Release workflow clean-install checks
will verify only that entry point; wheel and sdist metadata checks remain
unchanged otherwise.

README and current release notes will describe CLI + Python API usage and will
not mention Tkinter, `nexusmind-kb`, or a desktop product interface. Current
architecture navigation will list `cli.py` and `runtime_entrypoint.py` as the executable
boundary. Historical plan/spec documents remain untouched.

## Testing strategy

Tests will be changed with the contract:

- delete UI-only tests together with the deleted module;
- assert project metadata exposes only `nexusmind` and has no `tomli` shim;
- assert release workflow verifies only `nexusmind` in clean installs;
- keep `tests/test_runtime_entrypoint.py` and portable packaging tests focused on the
  retained CLI/runtime boundary;
- add a repository-facing stale-reference check for current product/release
  files, while excluding historical superpowers records;
- run the full supported test suite on the available environment and the
  existing Python 3.11/3.12/3.13 CI matrix.

The acceptance signal is zero stale current-contract references, unchanged
core retrieval/query behavior, and a passing supported test matrix.
