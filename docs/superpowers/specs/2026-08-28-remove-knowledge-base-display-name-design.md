# Remove KnowledgeBase Display Name Design

## Context

The public workflow identifies a KnowledgeBase by its root path while persistence
uses a generated `knowledge_base_id`. The separate KnowledgeBase-level
`display_name` is not used for lookup, synchronization, retrieval, query, or
persistence addressing, so it adds a public concept without changing behavior.

The previously published `v0.1.0` GitHub Release and tag have been removed. There
is therefore no released manifest v1 compatibility contract to preserve. This
change will redefine the unreleased manifest v1 directly instead of introducing
a migration or a v2 schema.

## Public Contract

- The complete CLI creation command is `nexusmind create <path>`.
- `create --name` is absent from help and rejected by argument parsing.
- `KnowledgeBase.create(...)` has no `display_name` parameter.
- KnowledgeBase inspection and status values do not expose `display_name`.
- The desktop creation workflow asks only for the root directory.
- The root path remains the user-facing locator, and the generated
  `knowledge_base_id` remains the internal persistence identity.

Knowledge Source and document display names are separate domain concepts and
remain unchanged.

## Persistence Contract

Manifest format v1 remains format v1. Its root keys become exactly:

- `format_version`
- `knowledge_base_id`
- `sources`

Encoding omits `display_name`. Decoding rejects it as an unexpected root field,
consistent with the existing strict schema policy. `KnowledgeBaseManifest` and
`KnowledgeBaseLimits` remove their KnowledgeBase display-name members and
validation. SQLite source rows keep their own `display_name` column because it
belongs to Knowledge Sources rather than the KnowledgeBase.

## Component Changes

### CLI and Python API

The CLI parser removes `--name`, and its create handler calls
`KnowledgeBase.create(path)` without a name. The `KnowledgeBase.create` signature
and all status/inspection construction paths remove the field. Plain and JSON
inspection output continue to expose the root path and internal ID according to
their current contracts.

### Desktop UI

The create controller accepts only a root path. The create form removes the
display-name variable, label, and input. Status rendering uses existing path and
state information and does not substitute a directory basename as a synthetic
name, because the issue explicitly excludes new basename-derived semantics.

### Documentation

README creation examples, Python API examples, architecture descriptions, and
release-facing documents use path-only creation and describe the strict v1 root
fields without `display_name`.

## Error Handling

Argument parsing provides the normal error for the removed `--name` option.
Python callers passing `display_name` receive the normal unexpected-keyword
`TypeError`. Manifests containing the removed field receive the existing strict
unexpected-field configuration error. No fallback aliases or silent compatibility
paths are introduced.

## Testing

Tests will be changed before implementation to establish these contracts:

- CLI help lacks `--name`, path-only creation succeeds, and `--name` is rejected.
- `KnowledgeBase.create` accepts path-only creation and rejects `display_name`.
- status and inspection objects/JSON omit the field.
- desktop creation contains no name input and passes only the root path.
- manifest v1 round trips with the exact three-key root contract and rejects the
  removed field.
- existing create/open/sync/search/query/inspect workflows continue to pass.
- documentation and release metadata checks contain no stale KnowledgeBase-level
  name examples.

The full suite will be run under the supported Python version in CI. Local
verification will run all available tests; release-workflow tests that import the
Python 3.11 standard-library `tomllib` cannot pass in the current Python 3.10-only
environment and will be reported separately from issue-specific results.

## Non-goals

This change does not add registries, aliases, recent-project lists, launchers,
name-based lookup, basename-derived identity, or changes to source/document
identity, synchronization, retrieval, or query behavior.
