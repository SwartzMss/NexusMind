import asyncio
import json

import httpx

from nexusmind.config import ModelConfig
from nexusmind.models.base import ChatModelError, ToolDefinition
from nexusmind.models.openai_compatible import OpenAICompatibleChatModel
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole


def _config(timeout: float = 60) -> ModelConfig:
    return ModelConfig(
        base_url="https://provider.test/v1",
        api_key="sk-test-secret",
        model="test-model",
        timeout=timeout,
    )


def _sse(payload: object) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _collect(model: OpenAICompatibleChatModel):
    messages = [Message(role=MessageRole.USER, content="hello")]
    return asyncio.run(_collect_async(model, messages))


async def _collect_async(model: OpenAICompatibleChatModel, messages: list[Message]):
    return [event async for event in model.stream(messages)]


def test_adapter_streams_normal_sse_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://provider.test/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert "tools" not in body
        content = "".join(
            [
                _sse({"choices": [{"delta": {"content": "hello"}}]}),
                _sse({"choices": [{"delta": {"content": " world"}}]}),
                "data: [DONE]\n\n",
            ]
        )
        return httpx.Response(200, content=content)

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    events = _collect(model)

    assert [event.type for event in events] == [
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.TEXT_DELTA,
    ]
    assert "".join(event.text or "" for event in events) == "hello world"


def test_adapter_redacts_api_key_from_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "bad key sk-test-secret"}},
        )

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ChatModelError")

    assert "sk-test-secret" not in message
    assert "[REDACTED]" in message


def test_adapter_raises_on_stream_error_and_redacts_secret() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"error": {"message": "bad key sk-test-secret"}}))

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ChatModelError")

    assert "sk-test-secret" not in message
    assert "[REDACTED]" in message


def test_adapter_raises_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    model = OpenAICompatibleChatModel(_config(timeout=0.01), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model request timed out"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_ignores_malformed_json_non_object_json_and_empty_choices() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content="".join(
                [
                    "data: {malformed\n\n",
                    "data: []\n\n",
                    _sse({"choices": []}),
                    _sse({"choices": [{"delta": {"content": "ok"}}]}),
                ]
            ),
        )

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    events = _collect(model)

    assert "".join(event.text or "" for event in events) == "ok"


def test_adapter_rejects_tools_until_tool_call_events_are_supported() -> None:
    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(lambda request: httpx.Response(500)))

    async def collect():
        messages = [Message(role=MessageRole.USER, content="hello")]
        tools = [ToolDefinition(name="lookup", parameters={"type": "object", "properties": {}})]
        return [event async for event in model.stream(messages, tools=tools)]

    try:
        asyncio.run(collect())
    except ChatModelError as exc:
        assert "Tool calls are not supported" in str(exc)
    else:
        raise AssertionError("expected ChatModelError")
