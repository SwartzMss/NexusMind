"""Provider-neutral embedding contracts and validated vector values."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Protocol


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


__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingValidationError",
    "EmbeddingVector",
]
