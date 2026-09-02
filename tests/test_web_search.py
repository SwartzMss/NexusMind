from __future__ import annotations

import httpx
import pytest

from nexusmind import cli
from nexusmind.web_search import (
    NexusSearchProvider,
    WebSearchProtocolError,
    WebSearchResult,
    WebSearchTransportError,
)


def _provider(handler) -> NexusSearchProvider:
    return NexusSearchProvider(transport=httpx.MockTransport(handler))


def test_search_parses_results_and_preserves_order_and_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "NVIDIA Rubin"
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json={"results": [
            {"title": "First", "url": "https://one.test", "content": "One", "engine": "brave", "publishedDate": "2026-01-02"},
            {"title": "Second", "url": "https://two.test", "content": "Two", "engines": ["bing", "google"]},
        ]})

    results = _provider(handler).search("NVIDIA Rubin", limit=1)
    assert len(results) == 1
    assert results[0].title == "First"
    assert results[0].snippet == "One"
    assert results[0].engine == "brave"
    assert results[0].published_at == "2026-01-02"


def test_search_normalizes_engines_and_allows_empty_results() -> None:
    result = _provider(lambda request: httpx.Response(200, json={"results": [
        {"title": "Result", "url": "https://example.test", "content": "", "engines": ["bing", "google"]}
    ]})).search("query")[0]
    assert result.engine == "bing, google"
    assert _provider(lambda request: httpx.Response(200, json={"results": []})).search("query") == ()


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"not json"),
    httpx.Response(200, json={}),
    httpx.Response(200, json={"results": "invalid"}),
    httpx.Response(200, json={"results": [{"title": "Missing URL"}]}),
    httpx.Response(503, json={"results": []}),
])
def test_search_rejects_malformed_or_unsuccessful_responses(response: httpx.Response) -> None:
    with pytest.raises(WebSearchProtocolError):
        _provider(lambda request: response).search("query")


@pytest.mark.parametrize("error", [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("timed out"),
])
def test_search_wraps_connection_and_timeout_failures(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(WebSearchTransportError):
        _provider(handler).search("query")


@pytest.mark.parametrize("query,limit", [("", 10), ("query", 0), ("query", 101)])
def test_search_validates_inputs(query: str, limit: int) -> None:
    with pytest.raises(ValueError):
        _provider(lambda request: httpx.Response(200, json={"results": []})).search(query, limit=limit)


def test_web_search_cli_renders_explicit_results(monkeypatch, capsys) -> None:
    def search(self, query: str, *, limit: int = 10):
        assert query == "NVIDIA Rubin"
        assert limit == 3
        return (WebSearchResult("Rubin", "https://example.test", "GPU", "brave"),)

    monkeypatch.setattr(NexusSearchProvider, "search", search)
    assert cli.main(["web-search", "NVIDIA Rubin", "--limit", "3"]) == 0
    assert capsys.readouterr().out == (
        "1. Rubin\n   https://example.test\n   GPU\n   engine: brave\n"
    )


def test_web_search_cli_reports_controlled_error(monkeypatch, capsys) -> None:
    def search(self, query: str, *, limit: int = 10):
        raise WebSearchTransportError("NexusSearch is unavailable or timed out")

    monkeypatch.setattr(NexusSearchProvider, "search", search)
    assert cli.main(["web-search", "query"]) == 1
    assert capsys.readouterr().err == (
        "Web search failed: NexusSearch is unavailable or timed out\n"
    )
