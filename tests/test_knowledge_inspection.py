from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexusmind import (
    Chunk,
    Document,
    KnowledgeChunkInspection,
    KnowledgeCollection,
    KnowledgeCollectionError,
    KnowledgeDocumentInspection,
    KnowledgeSource,
    TextChunker,
)


@dataclass
class _Adapter:
    source_id: str
    documents: tuple[Document, ...]

    def source(self) -> KnowledgeSource:
        return KnowledgeSource(
            source_id=self.source_id,
            source_type="test",
            display_name=self.source_id,
            metadata={"owner": self.source_id},
        )

    def load_documents(self) -> tuple[Document, ...]:
        return self.documents


class _FixedChunker:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.returned: object | None = None
        self.fail = False
        self.mutate = False

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        self.calls.append(document.document_id)
        if self.fail:
            raise RuntimeError("private chunker failure")
        if self.mutate:
            document.metadata["tag"] = "mutated"
        if self.returned is not None:
            return self.returned  # type: ignore[return-value]
        return tuple(
            Chunk(
                document_id=document.document_id,
                chunk_id=f"{document.document_id}:{start}",
                content=document.content[start:end],
                start_offset=start,
                end_offset=end,
            )
            for start, end in ((0, 10), (10, 20))
            if start < len(document.content)
        )


class _AlternatingChunker:
    def __init__(self) -> None:
        self.calls = 0

    def chunk(self, document: Document) -> tuple[Chunk, ...]:
        self.calls += 1
        if not document.content:
            return ()
        return (
            Chunk(
                document.document_id,
                f"chunk-{1 if self.calls % 2 else 2}",
                document.content,
                0,
                len(document.content),
            ),
        )


def _document(
    source_id: str = "docs", logical_path: str = "a.txt", content: str = "abcdefghijklmnopqrst"
) -> Document:
    return Document(
        source_id=source_id,
        logical_path=logical_path,
        content=content,
        metadata={"tag": "canonical"},
    )


def _collection(
    chunker: _FixedChunker, *documents: Document
) -> KnowledgeCollection:
    collection = KnowledgeCollection(chunker=chunker)
    by_source: dict[str, list[Document]] = {}
    for document in documents:
        by_source.setdefault(document.source_id, []).append(document)
    for source_id, source_documents in by_source.items():
        collection.sync(_Adapter(source_id, tuple(source_documents)))
    chunker.calls.clear()
    return collection


def test_inspection_models_are_frozen_slotted_and_validate_public_fields() -> None:
    chunk = KnowledgeChunkInspection(1, "chunk", 0, 3, 3, "abc")
    source = KnowledgeSource(source_id="docs", source_type="test", display_name="Docs")
    document = _document()
    inspection = KnowledgeDocumentInspection(source, document, (chunk,))

    assert not hasattr(chunk, "__dict__")
    assert not hasattr(inspection, "__dict__")
    with pytest.raises((AttributeError, TypeError)):
        chunk.ordinal = 2  # type: ignore[misc]

    bad_chunks = (
        (True, "chunk", 0, 3, 3, "abc"),
        (0, "chunk", 0, 3, 3, "abc"),
        (1, "", 0, 3, 3, "abc"),
        (1, "chunk", True, 3, 3, "abc"),
        (1, "chunk", -1, 3, 4, "abc"),
        (1, "chunk", 0, 0, 0, ""),
        (1, "chunk", 3, 2, -1, ""),
        (1, "chunk", 0, 3, 2, "abc"),
        (1, "chunk", 0, 3, 3, 1),
        (1, "chunk", 0, 3, 3, "abcd"),
    )
    for values in bad_chunks:
        with pytest.raises((TypeError, ValueError)):
            KnowledgeChunkInspection(*values)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="source"):
        KnowledgeDocumentInspection(object(), document, ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="document"):
        KnowledgeDocumentInspection(source, object(), ())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        KnowledgeDocumentInspection(source, document, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="KnowledgeChunkInspection"):
        KnowledgeDocumentInspection(source, document, (object(),))  # type: ignore[arg-type]


def test_document_inspection_requires_exact_source_and_document_types() -> None:
    class _SourceSubclass(KnowledgeSource):
        pass

    class _DocumentSubclass(Document):
        pass

    source = KnowledgeSource(source_id="docs", source_type="test", display_name="Docs")
    document = _document()
    source_subclass = _SourceSubclass(
        source_id="docs", source_type="test", display_name="Docs"
    )
    document_subclass = _DocumentSubclass(
        source_id="docs", logical_path="a.txt", content=document.content
    )

    with pytest.raises(TypeError, match="source"):
        KnowledgeDocumentInspection(source_subclass, document, ())
    with pytest.raises(TypeError, match="document"):
        KnowledgeDocumentInspection(source, document_subclass, ())


def test_inspect_document_returns_exact_offsets_previews_and_detached_metadata() -> None:
    chunker = _FixedChunker()
    document = _document()
    collection = _collection(chunker, document)

    inspection = collection.inspect_document(document.document_id, preview_chars=8)

    assert inspection.source.source_id == document.source_id
    assert inspection.document == document
    assert [
        (item.ordinal, item.start_offset, item.end_offset, item.character_count)
        for item in inspection.chunks
    ] == [(1, 0, 10, 10), (2, 10, 20, 10)]
    assert [item.preview for item in inspection.chunks] == ["abcdefgh", "klmnopqr"]
    assert chunker.calls == [document.document_id, document.document_id]

    inspection.source.metadata["owner"] = "external"
    inspection.document.metadata["tag"] = "external"
    again = collection.inspect_document(document.document_id)
    assert again.source.metadata == {"owner": "docs"}
    assert again.document.metadata == {"tag": "canonical"}


def test_inspect_document_normalizes_accepted_canonical_subclasses() -> None:
    class _SourceSubclass(KnowledgeSource):
        pass

    class _DocumentSubclass(Document):
        pass

    class _SubclassAdapter(_Adapter):
        def source(self) -> KnowledgeSource:
            return _SourceSubclass(
                source_id=self.source_id,
                source_type="test",
                display_name="Subclass source",
                logical_location="memory://docs",
                metadata={"nested": {"owner": "canonical"}},
            )

    document = _DocumentSubclass(
        source_id="docs",
        logical_path="subclass.txt",
        content="abcdefghijklmnopqrst",
        content_type="text/custom",
        metadata={"nested": {"tag": "canonical"}},
    )
    chunker = _FixedChunker()
    collection = KnowledgeCollection(chunker=chunker)
    collection.sync(_SubclassAdapter("docs", (document,)))

    inspection = collection.inspect_document(document.document_id)

    assert type(inspection.source) is KnowledgeSource
    assert type(inspection.document) is Document
    assert (
        inspection.source.source_id,
        inspection.source.display_name,
        inspection.source.logical_location,
        inspection.source.metadata,
    ) == (
        "docs",
        "Subclass source",
        "memory://docs",
        {"nested": {"owner": "canonical"}},
    )
    assert (
        inspection.document.source_id,
        inspection.document.logical_path,
        inspection.document.content,
        inspection.document.content_type,
        inspection.document.metadata,
        inspection.document.document_id,
        inspection.document.content_hash,
    ) == (
        document.source_id,
        document.logical_path,
        document.content,
        document.content_type,
        document.metadata,
        document.document_id,
        document.content_hash,
    )
    inspection.source.metadata["nested"]["owner"] = "external"
    inspection.document.metadata["nested"]["tag"] = "external"
    again = collection.inspect_document(document.document_id)
    assert again.source.metadata == {"nested": {"owner": "canonical"}}
    assert again.document.metadata == {"nested": {"tag": "canonical"}}


def test_inspect_documents_is_stable_handles_empty_documents_and_verifies_each() -> None:
    chunker = _FixedChunker()
    documents = (
        _document("z", "b.txt", ""),
        _document("a", "z.txt", "abcdefghij"),
        _document("a", "a.txt", "abcdefghijklmnopqrst"),
    )
    collection = _collection(chunker, *documents)

    first = collection.inspect_documents(preview_chars=4)
    first_calls = tuple(chunker.calls)
    chunker.calls.clear()
    second = collection.inspect_documents(preview_chars=4)

    expected_ids = tuple(
        document.document_id
        for document in sorted(documents, key=lambda value: (value.source_id, value.document_id))
    )
    assert tuple(item.document.document_id for item in first) == expected_ids
    assert first == second
    expected_calls = tuple(
        document_id for document_id in expected_ids for _ in range(2)
    )
    assert first_calls == expected_calls
    assert tuple(chunker.calls) == expected_calls
    assert next(item for item in first if not item.document.content).chunks == ()


def test_inspection_rejects_identical_consecutive_starts_but_accepts_real_overlap() -> None:
    chunker = _FixedChunker()
    document = _document(content="abcdefghij")
    collection = _collection(chunker, document)
    chunker.returned = (
        Chunk(document.document_id, "one", "abc", 0, 3),
        Chunk(document.document_id, "two", "abcde", 0, 5),
    )

    with pytest.raises(KnowledgeCollectionError, match="order"):
        collection.inspect_document(document.document_id)

    overlapping = KnowledgeCollection(chunker=TextChunker(chunk_size=5, overlap=2))
    overlapping.sync(_Adapter("docs", (document,)))
    inspection = overlapping.inspect_document(document.document_id)
    assert [(chunk.start_offset, chunk.end_offset) for chunk in inspection.chunks] == [
        (0, 5),
        (3, 8),
        (6, 10),
    ]


@pytest.mark.parametrize("operation", ["inspect_document", "inspect_documents"])
def test_inspection_rejects_nondeterministic_valid_chunk_tuples(operation: str) -> None:
    chunker = _AlternatingChunker()
    document = _document(content="alpha")
    collection = KnowledgeCollection(chunker=chunker)
    collection.sync(_Adapter("docs", (document,)))

    with pytest.raises(KnowledgeCollectionError, match="deterministic"):
        if operation == "inspect_document":
            collection.inspect_document(document.document_id)
        else:
            collection.inspect_documents()

    assert chunker.calls == 3


@pytest.mark.parametrize("document_id", [None, "", "  ", True, 1])
def test_inspect_document_requires_exact_nonblank_document_id(document_id: object) -> None:
    document = _document()
    collection = _collection(_FixedChunker(), document)
    with pytest.raises(ValueError, match="document_id"):
        collection.inspect_document(document_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("preview_chars", [True, 0, -1, 1.0, "1"])
def test_inspection_requires_exact_positive_preview_bound(preview_chars: object) -> None:
    document = _document()
    collection = _collection(_FixedChunker(), document)
    with pytest.raises((TypeError, ValueError), match="preview_chars"):
        collection.inspect_document(
            document.document_id, preview_chars=preview_chars  # type: ignore[arg-type]
        )
    with pytest.raises((TypeError, ValueError), match="preview_chars"):
        collection.inspect_documents(preview_chars=preview_chars)  # type: ignore[arg-type]


def test_inspection_rejects_unknown_document_and_redacts_chunker_failures() -> None:
    chunker = _FixedChunker()
    document = _document()
    collection = _collection(chunker, document)
    with pytest.raises(KnowledgeCollectionError, match="unknown document_id"):
        collection.inspect_document("missing")

    chunker.fail = True
    with pytest.raises(KnowledgeCollectionError, match="chunker failed") as caught:
        collection.inspect_document(document.document_id)
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    ("returned", "message"),
    [
        ([], "tuple"),
        ((object(),), "Chunk"),
        ((Chunk("wrong", "one", "abc", 0, 3),), "document_id"),
        (
            (
                Chunk("DOC", "same", "abc", 0, 3),
                Chunk("DOC", "same", "def", 3, 6),
            ),
            "duplicate",
        ),
        (
            (
                Chunk("DOC", "one", "def", 3, 6),
                Chunk("DOC", "two", "abc", 0, 3),
            ),
            "order",
        ),
        ((Chunk("DOC", "one", "abc", -1, 3),), "offset"),
        ((Chunk("DOC", "one", "wrong", 0, 3),), "content"),
    ],
)
def test_inspection_rejects_malformed_chunker_output(
    returned: object, message: str
) -> None:
    chunker = _FixedChunker()
    document = _document()
    collection = _collection(chunker, document)
    if type(returned) is tuple:
        fixed = []
        for item in returned:
            if type(item) is Chunk:
                fixed.append(
                    Chunk(
                        document.document_id if item.document_id == "DOC" else item.document_id,
                        item.chunk_id,
                        item.content,
                        item.start_offset,
                        item.end_offset,
                    )
                )
            else:
                fixed.append(item)
        returned = tuple(fixed)
    chunker.returned = returned

    with pytest.raises(KnowledgeCollectionError, match=message):
        collection.inspect_document(document.document_id)


def test_inspection_is_read_only_even_when_chunker_mutates_its_input() -> None:
    chunker = _FixedChunker()
    document = _document(content="searchable content!!")
    collection = _collection(chunker, document)
    before_snapshot = collection.snapshot()
    before_search = collection.search("searchable")
    chunker.mutate = True

    inspection = collection.inspect_document(document.document_id)

    assert inspection.document.metadata == {"tag": "canonical"}
    assert collection.snapshot() == before_snapshot
    assert collection.search("searchable") == before_search
