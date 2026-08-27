# First-Release Schema Simplification Design

## Context

NexusMind is preparing its first public `0.1.0` release. Development-time
manifest, SQLite, snapshot, CLI, and Python API shapes have not shipped and do
not require backward compatibility. The first release should establish one
strict baseline instead of carrying migrations for prerelease states.

This work is developed on a separate branch based on PR #105 so that review of
#105 can continue without scope expansion. The cleanup PR depends on #105 and
must be rebased onto `main` after #105 merges.

## Goals

- Publish one manifest schema and one SQLite schema, both numbered version 1.
- Make local source identifiers internal and deterministically generated.
- Preserve current source removal/re-addition document history without
  tombstones.
- Reject incomplete snapshots instead of synthesizing legacy history.
- Expose one knowledge-answering API through `KnowledgeBase.query()`.
- Remove CLI and documentation paths that exist only to repair prerelease data.

## Non-goals

- Migrating manifests, databases, or snapshots created by prerelease builds.
- Changing path normalization, Windows case identity, document version-chain
  semantics, persistence atomicity, or CLI JSON contracts.
- Removing schema version fields or strict rejection of unknown schemas.

## Manifest and source identity

`KnowledgeBaseManifest.format_version` becomes `"1"`. Its only root fields are
`format_version`, `knowledge_base_id`, `display_name`, and `sources`.
`retired_sources` and all tombstone matching, consumption, quota, and lifecycle
logic are removed.

`LocalFileSourceConfig` and `LocalDirectorySourceConfig` accept only `path`.
Their `source_id` is an `init=False` field derived from source type and
platform-correct path identity. Persisted source entries continue to contain
`source_id` as an integrity value. Decoding reconstructs the source from its
path and rejects the entry unless the persisted ID equals the derived ID.

Because source IDs remain stable for the same source type and path, removing
and later re-adding a source reconnects to retained document versions without a
tombstone.

## SQLite and snapshots

The complete schema containing `sources`, `documents`, and
`document_versions` becomes SQLite schema version `"1"`. Store creation builds
that schema directly. Existing non-empty databases must match it exactly;
there is no migration branch for a history-free schema.

`KnowledgeSnapshot.document_versions` remains an empty tuple by default so an
empty snapshot is concise. A snapshot containing documents but no versions is
invalid. Restore validates and rejects it rather than synthesizing root
versions. Snapshots produced by current collections and stores always include
coherent versions.

## Public API and CLI

`KnowledgeBase.answer()` is removed. Callers use `KnowledgeBase.query()` and
read its `.answer` value. The answer generator and lower-level answer contracts
remain unchanged.

The CLI removes `source remove --id`, `_AmbiguousSourcePathError`, and legacy
duplicate-path recovery messages. Source add, list, sync, and remove remain
path-based. JSON source output retains the exact four-key contract:
`config_version`, `source_id`, `type`, and `path`.

## Errors and validation

- Unknown manifest and SQLite versions continue to fail closed.
- Malformed manifests, databases, and snapshots continue to produce their
  existing controlled error families.
- Duplicate platform path identities remain rejected.
- A persisted source ID that does not match its type and path is a
  `KnowledgeBaseConfigError`.

## Tests and documentation

Tests that manufacture explicit source IDs will derive IDs from source configs
or assert relationships without fixed IDs. Migration, legacy tombstone,
duplicate-repair, and answer-alias tests are removed or replaced with strict
first-release contract tests. Existing Windows path-identity, version-chain,
atomicity, persistence-integrity, CLI JSON, and portable-package coverage must
continue to pass.

README and architecture text will describe only the first-release contracts;
they will not mention old manifests, old runtime layouts, legacy IDs, or data
migration.
