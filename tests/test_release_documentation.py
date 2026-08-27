from __future__ import annotations

from pathlib import Path


def test_readme_top_covers_first_release_download_and_cli_path() -> None:
    top = Path("README.md").read_text(encoding="utf-8")[:5000]

    for marker in (
        "nexusmind-windows-portable.zip",
        "nexusmind-0.1.0-py3-none-any.whl",
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
