import asyncio
import json

import httpx

from nexusmind.config import ModelConfig
from nexusmind.models.base import ChatModelError
from nexusmind.models.openai_compatible import OpenAICompatibleChatModel
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools import ToolDefinition


def _config(timeout: float = 60) -> ModelConfig:
    return ModelConfig(
        base_url="https://provider.test/v1",
        api_key="sk-test-secret",
        model="test-model",
        timeout=timeout,
    )


def _sse(payload: object) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("choices"), list):
        for choice in payload["choices"]:
            if isinstance(choice, dict):
                choice.setdefault("index", 0)
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
        RuntimeEventType.MODEL_TURN_COMPLETED,
    ]
    assert "".join(event.text or "" for event in events) == "hello world"


def test_adapter_redacts_api_key_from_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={"x-request-id": "req_123"},
            json={"error": {"message": "bad key sk-test-secret"}},
        )

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ChatModelError")

    assert message == "Model provider returned HTTP 401 (request_id=req_123)"
    assert "sk-test-secret" not in message


def test_adapter_http_error_does_not_echo_provider_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad argument sk-live-secret"}})

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ChatModelError")

    assert message == "Model provider returned HTTP 400"
    assert "sk-live-secret" not in message


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

    assert message == "Model stream returned a provider error"
    assert "sk-test-secret" not in message


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


def test_adapter_raises_on_malformed_sse_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="data: {malformed\n\n")

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream returned malformed JSON"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_rejects_non_object_json_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="data: []\n\n")

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream returned a non-object payload"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_raises_on_non_strict_sse_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content='data: {"choices":[{"delta":{"content":NaN},"finish_reason":"stop"}]}\n\n')

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream returned malformed JSON"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_raises_before_parsing_oversized_sse_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=("data: " + ("x" * (1024 * 1024 + 1)) + "\n").encode())

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream returned an oversized SSE event"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_allows_large_network_chunk_of_small_sse_lines() -> None:
    content = (_sse({"usage": {"padding": "x" * 64}}) * 15000) + "data: [DONE]\n\n"
    assert len(content.encode()) > 1024 * 1024
    model = OpenAICompatibleChatModel(
        _config(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
    )

    events = _collect(model)

    assert events[-1].type == RuntimeEventType.MODEL_TURN_COMPLETED


def test_adapter_rejects_invalid_utf8() -> None:
    content = b'data: {"choices":[{"index":0,"delta":{"content":"\xff"}}]}\n\n'
    model = OpenAICompatibleChatModel(
        _config(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
    )

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream returned invalid UTF-8"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_rejects_data_after_completion_signal() -> None:
    content = "".join(
        [
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            _sse({"choices": [{"delta": {"content": "late"}}]}),
        ]
    )
    model = OpenAICompatibleChatModel(
        _config(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
    )

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream returned data after completion"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_allows_done_after_finish_reason() -> None:
    content = _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}) + "data: [DONE]\n\n"
    model = OpenAICompatibleChatModel(
        _config(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)),
    )

    events = _collect(model)

    assert events[-1].finish_reason == "stop"


def test_adapter_rejects_invalid_error_field_and_choice_index() -> None:
    payloads = [
        {"error": True},
        {"error": []},
        {"choices": [{"index": 1, "delta": {}}]},
        {"choices": [{"delta": {}}]},
        {"choices": [{"index": "0", "delta": {}}]},
        {"choices": [{"index": False, "delta": {}}]},
    ]

    for payload in payloads:
        model = OpenAICompatibleChatModel(
            _config(),
            transport=httpx.MockTransport(
                lambda request, payload=payload: httpx.Response(
                    200,
                    content=f"data: {json.dumps(payload)}\n\n",
                )
            ),
        )
        try:
            _collect(model)
        except ChatModelError:
            pass
        else:
            raise AssertionError(f"expected ChatModelError for {payload!r}")


def test_adapter_ignores_empty_choices_and_usage_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content="".join(
                [
                    _sse({"choices": []}),
                    _sse({"usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}}),
                    _sse({"choices": [{"delta": {"content": "ok"}}]}),
                    "data: [DONE]\n\n",
                ]
            ),
        )

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    events = _collect(model)

    assert "".join(event.text or "" for event in events) == "ok"


def test_adapter_accepts_finish_reason_as_completion_signal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]}),
        )

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    events = _collect(model)

    assert "".join(event.text or "" for event in events) == "done"


def test_adapter_rejects_invalid_finish_reason_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"choices": [{"delta": {}, "finish_reason": False}]}))

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream returned invalid finish_reason"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_raises_when_stream_ends_after_partial_text_without_completion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse({"choices": [{"delta": {"content": "partial"}}]}))

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream ended before completion"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_raises_when_stream_is_empty() -> None:
    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    try:
        _collect(model)
    except ChatModelError as exc:
        assert str(exc) == "Model stream ended before completion"
    else:
        raise AssertionError("expected ChatModelError")


def test_adapter_sends_openai_tools_payload() -> None:
    original_schema = {"type": "object", "properties": {"text": {"type": "string"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup text",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                },
            }
        ]
        body["tools"][0]["function"]["parameters"]["properties"]["text"]["type"] = "integer"
        return httpx.Response(200, content="data: [DONE]\n\n")

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    async def collect():
        messages = [Message(role=MessageRole.USER, content="hello")]
        tools = [ToolDefinition(name="lookup", description="Lookup text", input_schema=original_schema)]
        return [event async for event in model.stream(messages, tools=tools)]

    events = asyncio.run(collect())

    assert events[-1].type == RuntimeEventType.MODEL_TURN_COMPLETED
    assert original_schema == {"type": "object", "properties": {"text": {"type": "string"}}}


def test_adapter_sends_multiple_tools_in_given_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert [tool["function"]["name"] for tool in body["tools"]] == ["first", "second"]
        return httpx.Response(200, content="data: [DONE]\n\n")

    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(handler))

    async def collect():
        messages = [Message(role=MessageRole.USER, content="hello")]
        tools = [
            ToolDefinition(name="first", input_schema={"type": "object", "properties": {}}),
            ToolDefinition(name="second", input_schema={"type": "object", "properties": {}}),
        ]
        return [event async for event in model.stream(messages, tools=tools)]

    asyncio.run(collect())
