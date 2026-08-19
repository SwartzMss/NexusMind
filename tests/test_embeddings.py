from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from nexusmind import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingValidationError,
    EmbeddingVector,
)


class _FixtureProvider:
    def embed_documents(
        self, texts: tuple[str, ...]
    ) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((index + 1, 1)) for index, _ in enumerate(texts))

    def embed_query(self, text: str) -> EmbeddingVector:
        return EmbeddingVector((1, 1))


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
