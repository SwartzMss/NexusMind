"""Explicit, ephemeral web-search provider boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx


DEFAULT_NEXUSSEARCH_URL = "http://127.0.0.1:8788"
MAX_WEB_SEARCH_RESULTS = 100


class WebSearchError(RuntimeError):
    """Base class for controlled web-search failures."""


class WebSearchTransportError(WebSearchError):
    """Raised when the search service cannot be reached in time."""


class WebSearchProtocolError(WebSearchError):
    """Raised when the search service returns an invalid response."""


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str
    engine: str | None = None
    published_at: str | None = None


class WebSearchProvider(Protocol):
    def search(self, query: str, *, limit: int = 10) -> tuple[WebSearchResult, ...]: ...


class NexusSearchProvider:
    """Synchronous client for the NexusSearch/SearXNG-compatible JSON API."""

    def __init__(
        self,
        base_url: str = DEFAULT_NEXUSSEARCH_URL,
        *,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport

    def search(self, query: str, *, limit: int = 10) -> tuple[WebSearchResult, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= limit <= MAX_WEB_SEARCH_RESULTS:
            raise ValueError(f"limit must be between 1 and {MAX_WEB_SEARCH_RESULTS}")
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.get(
                    f"{self._base_url}/search",
                    params={"q": query, "format": "json"},
                )
                response.raise_for_status()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise WebSearchTransportError("NexusSearch is unavailable or timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise WebSearchProtocolError(
                f"NexusSearch returned HTTP {exc.response.status_code}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise WebSearchProtocolError("NexusSearch returned malformed JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise WebSearchProtocolError("NexusSearch response must contain a results list")

        return tuple(_parse_result(item) for item in payload["results"][:limit])


def _parse_result(item: Any) -> WebSearchResult:
    if not isinstance(item, dict):
        raise WebSearchProtocolError("NexusSearch result must be an object")
    title = item.get("title")
    url = item.get("url")
    snippet = item.get("content", "")
    if not isinstance(title, str) or not title.strip():
        raise WebSearchProtocolError("NexusSearch result has an invalid title")
    if not isinstance(url, str) or not url.strip():
        raise WebSearchProtocolError("NexusSearch result has an invalid url")
    if snippet is None:
        snippet = ""
    if not isinstance(snippet, str):
        raise WebSearchProtocolError("NexusSearch result has invalid content")
    return WebSearchResult(
        title=title,
        url=url,
        snippet=snippet,
        engine=_normalize_engine(item),
        published_at=_optional_string(item.get("published_at", item.get("publishedDate"))),
    )


def _normalize_engine(item: dict[str, Any]) -> str | None:
    engine = _optional_string(item.get("engine"))
    if engine is not None:
        return engine
    engines = item.get("engines")
    if not isinstance(engines, list):
        return None
    normalized = tuple(value.strip() for value in engines if isinstance(value, str) and value.strip())
    return ", ".join(normalized) or None


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
