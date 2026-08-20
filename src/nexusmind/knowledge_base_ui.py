"""Minimal local desktop UI as a thin adapter over :class:`KnowledgeBase`."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock, Thread
from typing import Callable, Protocol

from .knowledge_base import KnowledgeBase, KnowledgeBaseStatus
from .knowledge_base_manifest import (
    KnowledgeBaseClosedError,
    KnowledgeBaseConfigError,
    KnowledgeBaseError,
    KnowledgeBasePersistenceError,
    KnowledgeBaseSourceError,
    LocalDirectorySourceConfig,
    LocalFileSourceConfig,
    RegisteredSourceConfig,
)


DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
MAX_SNIPPET_CHARS = 1_000


class KnowledgeBaseLike(Protocol):
    def status(self) -> KnowledgeBaseStatus: ...
    def list_sources(self) -> tuple[RegisteredSourceConfig, ...]: ...
    def add_source(self, config: RegisteredSourceConfig) -> None: ...
    def remove_source(self, source_id: str) -> None: ...
    def sync(self) -> tuple[object, ...]: ...
    def sync_source(self, source_id: str) -> object: ...
    def search(
        self, query: str, *, limit: int = 10
    ) -> tuple[object, ...]: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceView:
    source_id: str
    source_type: str
    path: str


@dataclass(frozen=True, slots=True)
class SyncView:
    source_id: str
    documents_added: int
    documents_updated: int
    documents_unchanged: int
    documents_removed: int
    chunks_indexed: int


@dataclass(frozen=True, slots=True)
class SearchResultView:
    source_id: str
    logical_path: str
    score: float
    snippet: str
    chunk_id: str


@dataclass(frozen=True, slots=True)
class KnowledgeBaseUIView:
    status: KnowledgeBaseStatus | None = None
    sources: tuple[SourceView, ...] = ()
    sync_results: tuple[SyncView, ...] = ()
    search_results: tuple[SearchResultView, ...] = ()
    notice: str | None = None
    error: str | None = None
    mutation_active: bool = False


def _safe_error(error: BaseException) -> str:
    """Map failures to bounded product messages without reflecting exception text."""

    if isinstance(error, KnowledgeBaseConfigError):
        return "The supplied KnowledgeBase configuration is invalid."
    if isinstance(error, KnowledgeBaseSourceError):
        return "The KnowledgeBase source operation could not be completed."
    if isinstance(error, KnowledgeBasePersistenceError):
        return "The KnowledgeBase storage operation could not be completed."
    if isinstance(error, KnowledgeBaseClosedError):
        return "The KnowledgeBase is closed; open it again to continue."
    if isinstance(error, KnowledgeBaseError):
        return "The KnowledgeBase operation could not be completed."
    return "The application could not complete the operation."


class KnowledgeBaseUIController:
    """Headless, injectable application boundary used by the Tk window."""

    def __init__(
        self,
        *,
        create: Callable[..., KnowledgeBaseLike] = KnowledgeBase.create,
        open_existing: Callable[..., KnowledgeBaseLike] = KnowledgeBase.open,
    ) -> None:
        self._create = create
        self._open_existing = open_existing
        self._knowledge_base: KnowledgeBaseLike | None = None
        self._mutation_lock = Lock()
        self._view = KnowledgeBaseUIView()

    @property
    def view(self) -> KnowledgeBaseUIView:
        return self._view

    def create(self, root: str, knowledge_base_id: str, display_name: str = "") -> None:
        self._replace_knowledge_base(
            lambda: self._create(
                root,
                knowledge_base_id=knowledge_base_id,
                display_name=display_name or None,
            ),
            "KnowledgeBase created.",
        )

    def open(self, root: str) -> None:
        self._replace_knowledge_base(
            lambda: self._open_existing(root), "KnowledgeBase opened."
        )

    def close(self) -> None:
        knowledge_base = self._knowledge_base
        self._knowledge_base = None
        if knowledge_base is not None:
            try:
                knowledge_base.close()
            except Exception:
                pass
        self._view = KnowledgeBaseUIView(notice="KnowledgeBase closed.")

    def refresh(self) -> None:
        knowledge_base = self._require_knowledge_base()
        if knowledge_base is None:
            return
        try:
            self._view = KnowledgeBaseUIView(
                status=knowledge_base.status(),
                sources=tuple(
                    SourceView(item.source_id, item.type, item.path)
                    for item in knowledge_base.list_sources()
                ),
                sync_results=self._view.sync_results,
                search_results=self._view.search_results,
                notice=self._view.notice,
                mutation_active=self._view.mutation_active,
            )
        except Exception as error:
            self._set_error(error)

    def add_file(self, source_id: str, path: str) -> None:
        self._mutate(
            lambda kb: kb.add_source(
                LocalFileSourceConfig(source_id=source_id, path=path)
            ),
            "File source registered. Sync explicitly when ready.",
        )

    def add_directory(self, source_id: str, path: str) -> None:
        self._mutate(
            lambda kb: kb.add_source(
                LocalDirectorySourceConfig(source_id=source_id, path=path)
            ),
            "Directory source registered. Sync explicitly when ready.",
        )

    def remove_source(self, source_id: str) -> None:
        self._mutate(
            lambda kb: kb.remove_source(source_id),
            "Source registration and canonical content removed.",
        )

    def sync_all(self) -> None:
        self._mutate(
            lambda kb: kb.sync(),
            "All registered sources synchronized.",
            render_sync=True,
        )

    def sync_source(self, source_id: str) -> None:
        self._mutate(
            lambda kb: (kb.sync_source(source_id),),
            "Selected source synchronized.",
            render_sync=True,
        )

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
        knowledge_base = self._require_knowledge_base()
        if knowledge_base is None:
            return
        if type(query) is not str or not query.strip():
            self._view = self._with_error("Enter a non-blank search query.")
            return
        if type(limit) is not int or not 1 <= limit <= MAX_SEARCH_LIMIT:
            self._view = self._with_error(
                f"Result limit must be between 1 and {MAX_SEARCH_LIMIT}."
            )
            return
        try:
            results = knowledge_base.search(query, limit=limit)
            rendered = tuple(self._render_search_result(item) for item in results)
            self._view = KnowledgeBaseUIView(
                status=self._view.status,
                sources=self._view.sources,
                sync_results=self._view.sync_results,
                search_results=rendered,
                notice=f"Search completed with {len(rendered)} result(s).",
            )
        except Exception as error:
            self._set_error(error)

    def _replace_knowledge_base(
        self, build: Callable[[], KnowledgeBaseLike], notice: str
    ) -> None:
        if not self._mutation_lock.acquire(blocking=False):
            self._view = self._with_error("A local mutation is already active.")
            return
        self._view = self._with_activity(True)
        try:
            candidate = build()
            previous = self._knowledge_base
            self._knowledge_base = candidate
            if previous is not None:
                try:
                    previous.close()
                except Exception:
                    pass
            self._view = KnowledgeBaseUIView(notice=notice)
            self.refresh()
        except Exception as error:
            self._set_error(error)
        finally:
            self._mutation_lock.release()
            self._view = self._with_activity(False)

    def _mutate(
        self,
        operation: Callable[[KnowledgeBaseLike], object],
        notice: str,
        *,
        render_sync: bool = False,
    ) -> None:
        knowledge_base = self._require_knowledge_base()
        if knowledge_base is None:
            return
        if not self._mutation_lock.acquire(blocking=False):
            self._view = self._with_error("A local mutation is already active.")
            return
        self._view = self._with_activity(True)
        try:
            outcome = operation(knowledge_base)
            sync_results = self._view.sync_results
            if render_sync and type(outcome) is tuple:
                sync_results = tuple(self._render_sync_result(item) for item in outcome)
            self._view = KnowledgeBaseUIView(
                sync_results=sync_results,
                search_results=self._view.search_results,
                notice=notice,
                mutation_active=True,
            )
            self.refresh()
        except Exception as error:
            self._set_error(error)
        finally:
            self._mutation_lock.release()
            self._view = self._with_activity(False)

    def _require_knowledge_base(self) -> KnowledgeBaseLike | None:
        if self._knowledge_base is None:
            self._view = self._with_error("Create or open a KnowledgeBase first.")
            return None
        return self._knowledge_base

    def _set_error(self, error: BaseException) -> None:
        self._view = self._with_error(_safe_error(error))

    def _with_error(self, message: str) -> KnowledgeBaseUIView:
        return KnowledgeBaseUIView(
            status=self._view.status,
            sources=self._view.sources,
            sync_results=self._view.sync_results,
            search_results=self._view.search_results,
            error=message,
            mutation_active=self._view.mutation_active,
        )

    def _with_activity(self, active: bool) -> KnowledgeBaseUIView:
        return KnowledgeBaseUIView(
            status=self._view.status,
            sources=self._view.sources,
            sync_results=self._view.sync_results,
            search_results=self._view.search_results,
            notice=self._view.notice,
            error=self._view.error,
            mutation_active=active,
        )

    @staticmethod
    def _render_sync_result(result: object) -> SyncView:
        return SyncView(
            getattr(result, "source_id"),
            getattr(result, "documents_added"),
            getattr(result, "documents_updated"),
            getattr(result, "documents_unchanged"),
            getattr(result, "documents_removed"),
            getattr(result, "chunks_indexed"),
        )

    @staticmethod
    def _render_search_result(result: object) -> SearchResultView:
        hit = getattr(result, "hit")
        source = getattr(result, "source")
        document = getattr(result, "document")
        content = hit.chunk.content
        snippet = (
            content
            if len(content) <= MAX_SNIPPET_CHARS
            else content[: MAX_SNIPPET_CHARS - 1] + "…"
        )
        return SearchResultView(
            source_id=source.source_id,
            logical_path=document.logical_path,
            score=hit.score,
            snippet=snippet,
            chunk_id=hit.chunk.chunk_id,
        )


class KnowledgeBaseTkApp:
    """Small Tk window; business operations stay in the injected controller."""

    def __init__(
        self,
        controller: KnowledgeBaseUIController | None = None,
        *,
        _tk: object | None = None,
        _ttk: object | None = None,
        _filedialog: object | None = None,
        _root: object | None = None,
    ) -> None:
        if _tk is None or _ttk is None or _filedialog is None:
            import tkinter as tk
            from tkinter import filedialog, ttk

            _tk = tk
            _ttk = ttk
            _filedialog = filedialog

        self._tk = _tk
        self._ttk = _ttk
        self._filedialog = _filedialog
        self._controller = controller or KnowledgeBaseUIController()
        self.root = _tk.Tk() if _root is None else _root
        self.root.title("NexusMind KnowledgeBase")
        self.root.geometry("1000x720")
        self._busy = False
        self._close_requested = False
        self._worker: Thread | None = None
        self._operation_buttons: list[object] = []

        shell = _ttk.Frame(self.root, padding=12)
        shell.pack(fill="both", expand=True)
        lifecycle = _ttk.LabelFrame(shell, text="KnowledgeBase", padding=8)
        lifecycle.pack(fill="x")
        self.root_path = _tk.StringVar()
        self.kb_id = _tk.StringVar()
        self.display_name = _tk.StringVar()
        self.source_id = _tk.StringVar()
        self.search_query = _tk.StringVar()
        self.search_limit = _tk.StringVar(value=str(DEFAULT_SEARCH_LIMIT))
        self.message = _tk.StringVar(value="Create or open a KnowledgeBase.")
        self.status_text = _tk.StringVar(value="No KnowledgeBase open")

        _ttk.Label(lifecycle, text="Destination:").grid(row=0, column=0, sticky="w")
        _ttk.Entry(lifecycle, textvariable=self.root_path, width=55).grid(
            row=0, column=1, sticky="ew"
        )
        _ttk.Button(
            lifecycle, text="Choose directory", command=self._choose_root
        ).grid(row=0, column=2)
        _ttk.Label(lifecycle, text="ID:").grid(row=1, column=0, sticky="w")
        _ttk.Entry(lifecycle, textvariable=self.kb_id, width=30).grid(
            row=1, column=1, sticky="ew"
        )
        _ttk.Label(lifecycle, text="Display name:").grid(
            row=2, column=0, sticky="w"
        )
        _ttk.Entry(lifecycle, textvariable=self.display_name, width=30).grid(
            row=2, column=1, sticky="ew"
        )
        self._button(lifecycle, "Create", self._start_create).grid(row=3, column=1, sticky="e")
        self._button(lifecycle, "Open", self._start_open).grid(row=3, column=2)
        _ttk.Label(shell, textvariable=self.status_text).pack(fill="x", pady=(8, 0))
        _ttk.Label(shell, textvariable=self.message, foreground="#8b1a1a").pack(fill="x")

        sources = _ttk.LabelFrame(shell, text="Sources", padding=8)
        sources.pack(fill="both", expand=True, pady=8)
        controls = _ttk.Frame(sources)
        controls.pack(fill="x")
        _ttk.Entry(controls, textvariable=self.source_id, width=24).pack(side="left")
        self._button(controls, "Add file", lambda: self._pick_source(False)).pack(side="left")
        self._button(controls, "Add directory", lambda: self._pick_source(True)).pack(side="left")
        self._button(controls, "Sync all", lambda: self._background(self._controller.sync_all)).pack(side="left")
        self._button(controls, "Sync selected", lambda: self._selected_action(self._controller.sync_source)).pack(side="left")
        self._button(controls, "Remove selected", lambda: self._selected_action(self._controller.remove_source)).pack(side="left")
        self.source_list = _tk.Listbox(sources, height=8)
        self.source_list.pack(fill="both", expand=True)
        self.sync_text = _tk.Text(sources, height=4, state="disabled")
        self.sync_text.pack(fill="x")

        search = _ttk.LabelFrame(shell, text="Search", padding=8)
        search.pack(fill="both", expand=True)
        _ttk.Entry(search, textvariable=self.search_query).pack(side="top", fill="x")
        _ttk.Spinbox(
            search,
            from_=1,
            to=MAX_SEARCH_LIMIT,
            textvariable=self.search_limit,
            width=6,
        ).pack(anchor="w")
        self._button(search, "Search", self._start_search).pack(anchor="w")
        self.results = _tk.Text(search, height=12, state="disabled", wrap="word")
        self.results.pack(fill="both", expand=True)
        lifecycle.columnconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _button(self, parent: object, text: str, command: Callable[[], None]):
        button = self._ttk.Button(parent, text=text, command=command)
        self._operation_buttons.append(button)
        return button

    def _choose_root(self) -> None:
        selected = self._filedialog.askdirectory()
        if selected:
            self.root_path.set(selected)

    def _start_create(self) -> None:
        root = self.root_path.get()
        knowledge_base_id = self.kb_id.get()
        display_name = self.display_name.get()
        self._background(
            lambda: self._controller.create(root, knowledge_base_id, display_name)
        )

    def _start_open(self) -> None:
        root = self.root_path.get()
        self._background(lambda: self._controller.open(root))

    def _pick_source(self, directory: bool) -> None:
        selected = (
            self._filedialog.askdirectory()
            if directory
            else self._filedialog.askopenfilename(
                filetypes=[("Text documents", "*.txt *.md *.markdown")]
            )
        )
        if not selected:
            return
        source_id = self.source_id.get()
        operation = self._controller.add_directory if directory else self._controller.add_file
        self._background(lambda: operation(source_id, selected))

    def _selected_action(self, operation: Callable[[str], None]) -> None:
        selection = self.source_list.curselection()
        if not selection:
            self.message.set("Select a registered source first.")
            return
        source_id = self._controller.view.sources[selection[0]].source_id
        self._background(lambda: operation(source_id))

    def _start_search(self) -> None:
        try:
            limit = int(self.search_limit.get())
        except ValueError:
            limit = 0
        query = self.search_query.get()
        self._background(lambda: self._controller.search(query, limit))

    def _background(self, operation: Callable[[], None]) -> None:
        if self._busy:
            self.message.set("Another KnowledgeBase operation is already active.")
            return
        self._busy = True
        self._set_operations_enabled(False)
        self.message.set("Working…")

        def run() -> None:
            operation()

        self._worker = Thread(target=run)
        self._worker.start()
        self.root.after(25, self._poll_worker)

    def _poll_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.root.after(25, self._poll_worker)
            return
        self._worker = None
        self._busy = False
        if self._close_requested:
            self._finish_close()
            return
        self._set_operations_enabled(True)
        self._render()

    def _set_operations_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self._operation_buttons:
            button.configure(state=state)

    def _render(self) -> None:
        view = self._controller.view
        if view.status is None:
            self.status_text.set("No KnowledgeBase open")
        else:
            display = view.status.display_name or "—"
            self.status_text.set(
                f"ID: {view.status.knowledge_base_id} | Name: {display} | "
                f"Registered: {view.status.registered_source_count} | "
                f"Canonical: {view.status.canonical_source_count} | "
                f"Documents: {view.status.document_count}"
            )
        self.message.set(view.error or view.notice or "Ready.")
        self.source_list.delete(0, self._tk.END)
        for source in view.sources:
            self.source_list.insert(
                self._tk.END, f"{source.source_id} | {source.source_type} | {source.path}"
            )
        sync_lines = [
            f"{item.source_id}: +{item.documents_added} ~{item.documents_updated} "
            f"={item.documents_unchanged} -{item.documents_removed}; chunks {item.chunks_indexed}"
            for item in view.sync_results
        ]
        result_lines = [
            f"[{item.score:.6g}] {item.source_id} / {item.logical_path}\n"
            f"chunk: {item.chunk_id}\n{item.snippet}\n"
            for item in view.search_results
        ]
        self._replace_text(self.sync_text, "\n".join(sync_lines))
        self._replace_text(self.results, "\n".join(result_lines))

    @staticmethod
    def _replace_text(widget: object, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _close(self) -> None:
        if self._busy:
            self._close_requested = True
            self._set_operations_enabled(False)
            self.message.set("Finishing current operation before closing…")
            return
        self._finish_close()

    def _finish_close(self) -> None:
        self._controller.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    KnowledgeBaseTkApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
