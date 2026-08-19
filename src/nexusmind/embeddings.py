"""Provider-neutral embedding contracts and validated vector values."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol

import httpx


class EmbeddingError(Exception):
    """Base class for controlled embedding failures."""


class EmbeddingValidationError(EmbeddingError):
    """Embedding input or output is structurally invalid."""


class EmbeddingProviderError(EmbeddingError):
    """An embedding provider request or response failed."""


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """An immutable, finite, non-zero embedding vector."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or not self.values:
            raise EmbeddingValidationError(
                "embedding values must be a non-empty tuple"
            )

        normalized: list[float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmbeddingValidationError(
                    "embedding values must be real numbers"
                )
            converted = float(value)
            if not isfinite(converted):
                raise EmbeddingValidationError("embedding values must be finite")
            normalized.append(converted)

        if not any(value != 0.0 for value in normalized):
            raise EmbeddingValidationError("embedding vector must be non-zero")
        object.__setattr__(self, "values", tuple(normalized))


class EmbeddingProvider(Protocol):
    """Synchronous provider contract with distinct document/query paths."""

    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]: ...

    def embed_query(self, text: str) -> EmbeddingVector: ...


class OpenAICompatibleEmbeddingProvider:
    """Synchronous, bounded adapter for an OpenAI-compatible embeddings API."""

    _MAX_BATCH_SIZE = 2_048
    _MAX_RESPONSE_BYTES = 64 * 1024 * 1024
    _MAX_DIMENSIONS = 65_536

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = self._plain_nonempty_string(base_url, "base_url")
        self._api_key = self._plain_nonempty_string(api_key, "api_key")
        self._model = self._plain_nonempty_string(model, "model")
        if isinstance(timeout, bool) or not isinstance(timeout, Real):
            raise TypeError("timeout must be a real number")
        converted_timeout = float(timeout)
        if not isfinite(converted_timeout) or converted_timeout <= 0:
            raise ValueError("timeout must be finite and greater than zero")
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            raise TypeError("transport must be an httpx BaseTransport")
        self._timeout = converted_timeout
        self._transport = transport

    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        self._validate_texts(texts)
        if not texts:
            return ()
        return self._embed(texts)

    def embed_query(self, text: str) -> EmbeddingVector:
        if type(text) is not str:
            raise TypeError("text must be a string")
        return self._embed((text,))[0]

    def _embed(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        try:
            with httpx.Client(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = client.post(
                    f"{self._base_url.rstrip('/')}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "input": list(texts)},
                )
                response.raise_for_status()
            if len(response.content) > self._MAX_RESPONSE_BYTES:
                raise EmbeddingProviderError("embedding provider response is invalid")
            payload = response.json()
            return self._parse_response(payload, len(texts))
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError("embedding provider request failed") from exc

    def _parse_response(
        self, payload: object, expected_count: int
    ) -> tuple[EmbeddingVector, ...]:
        try:
            if type(payload) is not dict:
                raise ValueError
            data = payload.get("data")
            if type(data) is not list or len(data) != expected_count:
                raise ValueError

            indexed: dict[int, EmbeddingVector] = {}
            dimension: int | None = None
            for item in data:
                if type(item) is not dict:
                    raise ValueError
                index = item.get("index")
                raw_embedding = item.get("embedding")
                if (
                    type(index) is not int
                    or not 0 <= index < expected_count
                    or index in indexed
                    or type(raw_embedding) is not list
                    or not raw_embedding
                    or len(raw_embedding) > self._MAX_DIMENSIONS
                ):
                    raise ValueError
                vector = EmbeddingVector(tuple(raw_embedding))
                if dimension is None:
                    dimension = len(vector.values)
                elif len(vector.values) != dimension:
                    raise ValueError
                indexed[index] = vector
            if set(indexed) != set(range(expected_count)):
                raise ValueError
            return tuple(indexed[index] for index in range(expected_count))
        except (KeyError, TypeError, ValueError, EmbeddingValidationError) as exc:
            raise EmbeddingProviderError(
                "embedding provider response is invalid"
            ) from exc

    @classmethod
    def _validate_texts(cls, texts: tuple[str, ...]) -> None:
        if type(texts) is not tuple:
            raise TypeError("texts must be a tuple")
        if len(texts) > cls._MAX_BATCH_SIZE:
            raise EmbeddingValidationError("embedding batch is too large")
        if any(type(text) is not str for text in texts):
            raise TypeError("embedding texts must be strings")

    @staticmethod
    def _plain_nonempty_string(value: object, name: str) -> str:
        if type(value) is not str:
            raise TypeError(f"{name} must be a string")
        if not value or value.strip() != value:
            raise ValueError(f"{name} must be a non-empty unpadded string")
        return value


__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingValidationError",
    "EmbeddingVector",
    "OpenAICompatibleEmbeddingProvider",
]
