"""Bounded query planning for retrieval-oriented multi-query search."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Protocol, runtime_checkable

import httpx

from .config import ModelConfig

MAX_EXPANDED_QUERIES = 3

_TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"0[xX][0-9A-Fa-f]+"
    r"|[0-9][0-9A-Fa-f]{3,}(?=[^A-Za-z0-9]|$)"
    r"|[0-9]{3,}"
    r"|[A-Za-z][A-Za-z0-9]*(?:[_./:-][A-Za-z0-9_.:/-]+)+"
    r"|[A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*"
    r"|[A-Z][A-Z0-9]{1,}"
    r")(?![A-Za-z0-9])"
)


class QueryExpansionError(Exception):
    """A query expansion request failed or returned an invalid plan."""


@dataclass(frozen=True, slots=True)
class QueryExpansion:
    original_query: str
    expanded_queries: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.original_query) is not str or not self.original_query.strip():
            raise ValueError("original_query must be a non-empty string")
        if type(self.expanded_queries) is not tuple:
            raise TypeError("expanded_queries must be a tuple")
        if len(self.expanded_queries) > MAX_EXPANDED_QUERIES:
            raise ValueError("expanded_queries exceeds the supported limit")
        if any(type(query) is not str or not query.strip() for query in self.expanded_queries):
            raise ValueError("expanded_queries must contain non-empty strings")
        normalized = [query.strip() for query in self.expanded_queries]
        if len(set(normalized)) != len(normalized) or self.original_query.strip() in normalized:
            raise ValueError("expanded_queries must be unique and exclude the original query")
        identifiers = set(_TECHNICAL_IDENTIFIER_PATTERN.findall(self.original_query))
        if any(any(identifier not in query for identifier in identifiers) for query in normalized):
            raise ValueError("expanded_queries must preserve exact technical identifiers")
        object.__setattr__(self, "expanded_queries", tuple(normalized))


@runtime_checkable
class QueryExpander(Protocol):
    def expand(self, question: str) -> QueryExpansion: ...


class OpenAICompatibleQueryExpander:
    """Generate strict retrieval queries through an OpenAI-compatible endpoint."""

    def __init__(self, config: ModelConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        if type(config) is not ModelConfig:
            raise TypeError("config must be ModelConfig")
        if transport is not None and not isinstance(transport, httpx.BaseTransport):
            raise TypeError("transport must be an httpx BaseTransport")
        self._config = config
        self._transport = transport

    def expand(self, question: str) -> QueryExpansion:
        if type(question) is not str or not question.strip():
            raise ValueError("question must be a non-empty string")
        system = (
            "Generate retrieval queries for a technical knowledge base, not answers. "
            "Preserve every exact identifier, API/function/process/file name, acronym, and error code. "
            "Expand informal wording into likely technical terms and useful Chinese/English synonyms. "
            "Do not invent system-specific facts or merely paraphrase. Return exactly one JSON object "
            'with key "queries", an array of at most 3 concise unique strings; do not include the original query.'
        )
        request = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question:\n{question}"},
            ],
            "stream": False,
        }
        try:
            with httpx.Client(
                base_url=self._config.base_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                timeout=self._config.timeout,
                transport=self._transport,
            ) as client:
                response = client.post("chat/completions", json=request)
                if not 200 <= response.status_code < 300 or len(response.content) > 16_384:
                    raise QueryExpansionError("query expansion request failed")
            envelope = json.loads(response.content, parse_constant=_reject_constant)
            choices = envelope["choices"]
            if type(choices) is not list or len(choices) != 1:
                raise ValueError
            choice = choices[0]
            if type(choice) is not dict or choice.get("finish_reason") != "stop":
                raise ValueError
            content = choice["message"]["content"]
            payload = json.loads(content, parse_constant=_reject_constant)
            if type(payload) is not dict or set(payload) != {"queries"}:
                raise ValueError
            queries = payload["queries"]
            if type(queries) is not list or any(type(item) is not str for item in queries):
                raise ValueError
            return QueryExpansion(question, tuple(queries))
        except QueryExpansionError:
            raise
        except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError, RecursionError, UnicodeDecodeError):
            raise QueryExpansionError("query expansion failed") from None


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = [
    "MAX_EXPANDED_QUERIES",
    "OpenAICompatibleQueryExpander",
    "QueryExpander",
    "QueryExpansion",
    "QueryExpansionError",
]
