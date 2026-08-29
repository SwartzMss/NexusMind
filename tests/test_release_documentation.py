from __future__ import annotations

from pathlib import Path


def test_readme_top_covers_source_install_and_cli_path() -> None:
    top = Path("README.md").read_text(encoding="utf-8")[:5000]

    for marker in (
        "没有公开的 GitHub Release",
        "从源码安装",
        'python -m pip install -e ".[dev]"',
        "nexusmind create",
        "nexusmind source add",
        "nexusmind sync",
        "nexusmind search",
        "nexusmind inspect",
        ".txt",
        ".md",
        ".markdown",
    ):
        assert marker in top

    assert "nexusmind-0.1.0-py3-none-any.whl" not in top


def test_release_notes_cover_capabilities_and_constraints() -> None:
    notes = Path("docs/releases/v0.1.0.md").read_text(encoding="utf-8")

    for marker in (
        "BM25",
        "Semantic",
        "Hybrid-RRF",
        "reranking",
        "validated citations",
        "immutable document version history",
        "manifest/SQLite persistence v1",
        "strict UTF-8",
        "no Git source adapter",
        "no background synchronization",
        "no cloud-hosted KnowledgeBase",
        "rebuilt rather than persisted",
    ):
        assert marker in notes


def test_release_facing_schema_docs_use_three_field_manifest_root() -> None:
    root = Path("docs")
    architecture = root.joinpath("architecture.md").read_text(encoding="utf-8")
    release_notes = root.joinpath("releases/v0.1.0.md").read_text(encoding="utf-8")
    schema_design = root.joinpath(
        "superpowers/specs/2026-08-27-first-release-schema-design.md"
    ).read_text(encoding="utf-8")
    schema_plan = root.joinpath(
        "superpowers/plans/2026-08-27-first-release-schema.md"
    ).read_text(encoding="utf-8")

    assert "`format_version`, `knowledge_base_id`, and `sources`" in schema_design
    assert '"format_version", "knowledge_base_id", "sources"' in schema_plan
    assert "`display_name`" not in schema_design
    assert "root path" in architecture
    assert "`format_version`, `knowledge_base_id`, and `sources`" in architecture
    assert "root path" in release_notes
