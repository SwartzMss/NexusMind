from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nexusmind import (
    Chunk,
    Document,
    KnowledgeBaseConfigError,
    KnowledgeBaseStatus,
    KnowledgeSearchResult,
    KnowledgeSource,
    KnowledgeSyncResult,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    SearchHit,
)
from nexusmind.knowledge_base_ui import (
    MAX_SEARCH_LIMIT,
    KnowledgeBaseUIController,
)


@dataclass
class FakeKnowledgeBase:
    sources: list[LocalFileSourceConfig | LocalDirectorySourceConfig] = field(
        default_factory=list
    )
    canonical_source_count: int = 0
    document_count: int = 0
    calls: list[object] = field(default_factory=list)
    search_results: tuple[KnowledgeSearchResult, ...] = ()
    on_sync: object | None = None

    def status(self) -> KnowledgeBaseStatus:
        self.calls.append("status")
        return KnowledgeBaseStatus(
            "kb", "Docs", len(self.sources), self.canonical_source_count, self.document_count
        )

    def list_sources(self):
        self.calls.append("list_sources")
        return tuple(self.sources)

    def add_source(self, config):
        self.calls.append(("add_source", config))
        self.sources.append(config)

    def remove_source(self, source_id: str) -> None:
        self.calls.append(("remove_source", source_id))
        self.sources = [item for item in self.sources if item.source_id != source_id]

    def sync(self):
        self.calls.append("sync")
        if callable(self.on_sync):
            self.on_sync()
        self.canonical_source_count = len(self.sources)
        return tuple(_sync_result(item.source_id) for item in self.sources)

    def sync_source(self, source_id: str):
        self.calls.append(("sync_source", source_id))
        return _sync_result(source_id)

    def search(self, query: str, *, limit: int = 10):
        self.calls.append(("search", query, limit))
        return self.search_results[:limit]

    def close(self) -> None:
        self.calls.append("close")


def _sync_result(source_id: str) -> KnowledgeSyncResult:
    return KnowledgeSyncResult(source_id, 1, 2, 3, 4, 5)


def _controller(fake: FakeKnowledgeBase) -> KnowledgeBaseUIController:
    controller = KnowledgeBaseUIController(open_existing=lambda root: fake)
    controller.open("selected-by-user")
    fake.calls.clear()
    return controller


def _search_result(source_id: str, path: str, score: float) -> KnowledgeSearchResult:
    document = Document(source_id=source_id, logical_path=path, content="canonical text")
    chunk = Chunk(document.document_id, f"chunk-{source_id}", f"match from {path}", 0, 5)
    return KnowledgeSearchResult(
        source=KnowledgeSource(
            source_id=source_id, source_type="fixture", display_name=source_id
        ),
        document=document,
        hit=SearchHit(chunk, score, ("match",)),
    )


def test_create_and_open_delegate_to_injected_knowledge_base_boundary() -> None:
    created = FakeKnowledgeBase()
    opened = FakeKnowledgeBase()
    calls: list[object] = []

    def create(root: str, **kwargs):
        calls.append(("create", root, kwargs))
        return created

    def open_existing(root: str):
        calls.append(("open", root))
        return opened

    controller = KnowledgeBaseUIController(create=create, open_existing=open_existing)
    controller.create("new-root", "my-kb", "My KB")
    controller.open("existing-root")

    assert calls == [
        (
            "create",
            "new-root",
            {"knowledge_base_id": "my-kb", "display_name": "My KB"},
        ),
        ("open", "existing-root"),
    ]
    assert created.calls[-1] == "close"
    assert controller.view.status == KnowledgeBaseStatus("kb", "Docs", 0, 0, 0)


def test_file_and_directory_registration_do_not_implicitly_sync() -> None:
    fake = FakeKnowledgeBase()
    controller = _controller(fake)

    controller.add_file("file", "notes.md")
    controller.add_directory("directory", "docs")

    additions = [call[1] for call in fake.calls if isinstance(call, tuple)]
    assert additions == [
        LocalFileSourceConfig(source_id="file", path="notes.md"),
        LocalDirectorySourceConfig(source_id="directory", path="docs"),
    ]
    assert "sync" not in fake.calls
    assert controller.view.status.registered_source_count == 2


def test_sync_all_and_single_source_render_exact_counters() -> None:
    fake = FakeKnowledgeBase(
        sources=[LocalFileSourceConfig(source_id="one", path="one.md")]
    )
    controller = _controller(fake)

    controller.sync_all()
    assert fake.calls[0] == "sync"
    assert controller.view.sync_results[0].documents_added == 1
    assert controller.view.sync_results[0].documents_updated == 2
    assert controller.view.sync_results[0].documents_unchanged == 3
    assert controller.view.sync_results[0].documents_removed == 4
    assert controller.view.sync_results[0].chunks_indexed == 5

    controller.sync_source("one")
    assert ("sync_source", "one") in fake.calls
    assert tuple(item.source_id for item in controller.view.sync_results) == ("one",)


def test_source_removal_delegates_and_refreshes_status() -> None:
    fake = FakeKnowledgeBase(
        sources=[LocalFileSourceConfig(source_id="one", path="one.md")]
    )
    controller = _controller(fake)

    controller.remove_source("one")

    assert ("remove_source", "one") in fake.calls
    assert controller.view.sources == ()
    assert controller.view.status.registered_source_count == 0


def test_status_is_rendered_only_from_knowledge_base_status() -> None:
    fake = FakeKnowledgeBase(
        sources=[LocalFileSourceConfig(source_id="one", path="one.md")],
        canonical_source_count=1,
        document_count=7,
    )
    controller = _controller(fake)

    assert controller.view.status == KnowledgeBaseStatus("kb", "Docs", 1, 1, 7)


def test_search_preserves_order_limit_and_provenance() -> None:
    fake = FakeKnowledgeBase(
        search_results=(
            _search_result("second", "b.md", 9.0),
            _search_result("first", "a.md", 8.0),
        )
    )
    controller = _controller(fake)

    controller.search("needle", limit=2)

    assert fake.calls == [("search", "needle", 2)]
    assert [item.source_id for item in controller.view.search_results] == [
        "second",
        "first",
    ]
    assert controller.view.search_results[0].logical_path == "b.md"
    assert controller.view.search_results[0].score == 9.0
    assert controller.view.search_results[0].snippet == "match from b.md"
    assert controller.view.search_results[0].chunk_id == "chunk-second"


@pytest.mark.parametrize("query,limit", [("", 10), (" ", 10), ("query", 0), ("query", MAX_SEARCH_LIMIT + 1)])
def test_search_rejects_invalid_ui_inputs_without_calling_runtime(
    query: str, limit: int
) -> None:
    fake = FakeKnowledgeBase()
    controller = _controller(fake)

    controller.search(query, limit)

    assert fake.calls == []
    assert controller.view.error is not None


def test_controlled_errors_do_not_render_exception_text_or_tracebacks() -> None:
    private = "private-path query document provider-secret"

    def fail(root: str):
        raise KnowledgeBaseConfigError(private)

    controller = KnowledgeBaseUIController(open_existing=fail)
    controller.open("also-private")

    assert controller.view.error == "The supplied KnowledgeBase configuration is invalid."
    assert private not in controller.view.error
    assert "Traceback" not in controller.view.error
    assert "also-private" not in controller.view.error


def test_duplicate_local_mutation_is_rejected_while_sync_is_active() -> None:
    fake = FakeKnowledgeBase(
        sources=[LocalFileSourceConfig(source_id="one", path="one.md")]
    )
    controller = _controller(fake)
    fake.on_sync = lambda: controller.remove_source("one")

    controller.sync_all()

    assert "sync" in fake.calls
    assert ("remove_source", "one") not in fake.calls
    assert controller.view.mutation_active is False
