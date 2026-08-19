from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
import json
import math

import httpx
import pytest

from nexusmind import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingValidationError,
    EmbeddingVector,
    OpenAICompatibleEmbeddingProvider,
)


class _FixtureProvider:
    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((index + 1, 1)) for index, _ in enumerate(texts))

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector((1, 1))


class _ChunkedStream(httpx.SyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self.chunks = chunks
        self.yielded = 0

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk


def test_embedding_contracts_are_public_and_values_become_floats() -> None:
    provider: EmbeddingProvider = _FixtureProvider()

    vector = provider.embed_query("query")

    assert issubclass(EmbeddingValidationError, EmbeddingError)
    assert issubclass(EmbeddingProviderError, EmbeddingError)
    assert vector == EmbeddingVector((1.0, 1.0))
    assert all(type(value) is float for value in vector.values)


def test_embedding_vector_is_frozen() -> None:
    vector = EmbeddingVector((1, 2))

    with pytest.raises(FrozenInstanceError):
        vector.values = (3.0, 4.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([1.0], "non-empty tuple"),
        ((), "non-empty tuple"),
        ((True,), "real numbers"),
        (("1",), "real numbers"),
        ((math.nan,), "finite"),
        ((math.inf,), "finite"),
        ((-math.inf,), "finite"),
        ((0,), "non-zero"),
        ((0.0, -0.0), "non-zero"),
    ],
)
def test_embedding_vector_rejects_invalid_values(
    values: object, message: str
) -> None:
    with pytest.raises(EmbeddingValidationError, match=message):
        EmbeddingVector(values)  # type: ignore[arg-type]


def test_openai_provider_batches_documents_and_orders_response_indexes() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://provider.test/v1/",
        api_key="secret-key",
        model="embed-model",
        transport=httpx.MockTransport(handler),
    )

    assert provider.embed_documents(("first", "second")) == (
        EmbeddingVector((1, 0)),
        EmbeddingVector((0, 1)),
    )
    assert len(requests) == 1
    assert requests[0].url == httpx.URL("https://provider.test/v1/embeddings")
    assert requests[0].headers["authorization"] == "Bearer secret-key"
    assert json.loads(requests[0].content) == {
        "model": "embed-model",
        "input": ["first", "second"],
    }


def test_openai_provider_separates_query_and_empty_document_batch() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3, 4]}]})

    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://provider.test/v1",
        api_key="key",
        model="model",
        transport=httpx.MockTransport(handler),
    )

    assert provider.embed_documents(()) == ()
    assert provider.embed_query("lookup") == EmbeddingVector((3, 4))
    assert len(requests) == 1
    assert json.loads(requests[0].content)["input"] == ["lookup"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": ""},
        {"base_url": " https://provider.test"},
        {"api_key": ""},
        {"model": ""},
        {"timeout": True},
        {"timeout": 0},
        {"timeout": math.inf},
    ],
)
def test_openai_provider_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    config: dict[str, object] = {
        "base_url": "https://provider.test/v1",
        "api_key": "key",
        "model": "model",
    }
    config.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        OpenAICompatibleEmbeddingProvider(**config)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"data": {}},
        {"data": []},
        {"data": [{"index": 1, "embedding": [1]}]},
        {"data": [{"index": True, "embedding": [1]}]},
        {"data": [{"index": 0}]},
        {"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]},
    ],
)
def test_openai_provider_rejects_hostile_response_shapes(payload: object) -> None:
    provider = _openai_provider(lambda _: httpx.Response(200, json=payload))

    with pytest.raises(EmbeddingProviderError, match="embedding provider response is invalid"):
        provider.embed_documents(("first",))


def test_openai_provider_rejects_inconsistent_batch_dimensions() -> None:
    provider = _openai_provider(
        lambda _: httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [1]},
            {"index": 1, "embedding": [1, 2]},
        ]})
    )

    with pytest.raises(EmbeddingProviderError, match="embedding provider response is invalid"):
        provider.embed_documents(("first", "second"))


def test_openai_provider_enforces_response_limit_while_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _ChunkedStream((b"12345678", b"abcd", b"unread"))
    provider = _openai_provider(lambda _: httpx.Response(200, stream=stream))
    monkeypatch.setattr(provider, "_MAX_RESPONSE_BYTES", 10)

    with pytest.raises(EmbeddingProviderError, match="response is too large"):
        provider.embed_query("query")

    assert stream.yielded == 2


def test_openai_provider_rejects_oversized_content_length_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _ChunkedStream((b"unread",))
    provider = _openai_provider(
        lambda _: httpx.Response(
            200,
            headers={"content-length": "11"},
            stream=stream,
        )
    )
    monkeypatch.setattr(provider, "_MAX_RESPONSE_BYTES", 10)

    with pytest.raises(EmbeddingProviderError, match="response is too large"):
        provider.embed_query("query")

    assert stream.yielded == 0


@pytest.mark.parametrize(
    "response_factory",
    [
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("sentinel-secret", request=request)),
        lambda _: httpx.Response(500, text="sentinel-secret full private query"),
        lambda _: httpx.Response(200, content=b"not-json sentinel-secret"),
    ],
)
def test_openai_provider_redacts_transport_http_and_json_failures(response_factory: object) -> None:
    provider = _openai_provider(response_factory)  # type: ignore[arg-type]

    with pytest.raises(EmbeddingProviderError) as caught:
        provider.embed_query("full private query")

    assert str(caught.value) == "embedding provider request failed"
    assert "sentinel-secret" not in str(caught.value)
    assert caught.value.__cause__ is not None


def _openai_provider(handler: object) -> OpenAICompatibleEmbeddingProvider:
    return OpenAICompatibleEmbeddingProvider(
        base_url="https://provider.test/v1",
        api_key="sentinel-secret",
        model="model",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )
