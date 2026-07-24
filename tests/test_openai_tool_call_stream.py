import asyncio
import json

import httpx

from nexusmind.config import ModelConfig
from nexusmind.models.base import ChatModelError
from nexusmind.models.openai_compatible import OpenAICompatibleChatModel
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools import ToolDefinition


def _config() -> ModelConfig:
    return ModelConfig("https://provider.test/v1", "sk-test-secret", "test-model")


def _sse(payload: object) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("choices"), list):
        for choice in payload["choices"]:
            if isinstance(choice, dict):
                choice.setdefault("index", 0)
    return f"data: {json.dumps(payload)}\n\n"


async def _collect_with_content(content: str):
    model = OpenAICompatibleChatModel(_config(), transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)))
    messages = [Message(role=MessageRole.USER, content="hello")]
    tools = [ToolDefinition(name="echo", description="Echo", input_schema={"type": "object", "properties": {"text": {"type": "string"}}})]
    return [event async for event in model.stream(messages, tools=tools)]


def test_stream_parses_fragmented_tool_call_and_turn_completion() -> None:
    content = "".join(
        [
            _sse({"choices": [{"delta": {"content": "Need a tool."}, "finish_reason": None}]}),
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_",
                                        "type": "func",
                                        "function": {"name": "ec", "arguments": '{"te'},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "1",
                                        "type": "tion",
                                        "function": {"name": "ho", "arguments": 'xt":"hello"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
        ]
    )

    events = asyncio.run(_collect_with_content(content))

    assert [event.type for event in events] == [
        RuntimeEventType.MODEL_STARTED,
        RuntimeEventType.TEXT_DELTA,
        RuntimeEventType.TOOL_CALL_DELTA,
        RuntimeEventType.TOOL_CALL_DELTA,
        RuntimeEventType.TOOL_CALL_COMPLETED,
        RuntimeEventType.MODEL_TURN_COMPLETED,
    ]
    assert events[1].text == "Need a tool."
    assert events[2].tool_call_delta.arguments_fragment == '{"te'
    assert events[4].tool_call is not None
    assert events[4].tool_call.id == "call_1"
    assert events[4].tool_call.name == "echo"
    assert events[4].tool_call.arguments == {"text": "hello"}
    assert events[5].finish_reason == "tool_calls"


def test_stream_parses_parallel_interleaved_tool_calls_in_index_order() -> None:
    content = "".join(
        [
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 1, "id": "call_b", "type": "function", "function": {"name": "bravo", "arguments": '{"b"'}},
                                    {"index": 0, "id": "call_a", "type": "function", "function": {"name": "alpha", "arguments": '{"a"'}},
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": ":1}"}},
                                    {"index": 1, "function": {"arguments": ":2}"}},
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
        ]
    )

    events = asyncio.run(_collect_with_content(content))
    completed = [event.tool_call for event in events if event.type == RuntimeEventType.TOOL_CALL_COMPLETED]

    assert [call.name for call in completed] == ["alpha", "bravo"]
    assert [call.arguments for call in completed] == [{"a": 1}, {"b": 2}]


def test_stream_rejects_finish_reason_tool_calls_without_tool_call() -> None:
    content = _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})

    try:
        asyncio.run(_collect_with_content(content))
    except ChatModelError as exc:
        assert str(exc) == "Model stream ended without tool calls"
    else:
        raise AssertionError("expected ChatModelError")


def test_stream_rejects_partial_tool_call_before_eof() -> None:
    content = _sse(
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"arguments": '{"x"'}}]},
                    "finish_reason": None,
                }
            ]
        }
    )

    try:
        asyncio.run(_collect_with_content(content))
    except ChatModelError as exc:
        assert str(exc) == "Model stream ended before completion"
    else:
        raise AssertionError("expected ChatModelError")


def test_stream_rejects_invalid_tool_call_fields_and_legacy_function_call() -> None:
    bad_payloads = [
        {"choices": [{"delta": {"tool_calls": {}}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": None}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": "0"}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": True}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": {}}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": None}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": None}}]}, "finish_reason": None}]},
        {"choices": [{"delta": {"function_call": {"name": "echo"}}, "finish_reason": "stop"}]},
        {"choices": [{"delta": {}, "finish_reason": None}, {"delta": {}, "finish_reason": None}]},
        {"choices": {}},
        {"choices": ""},
        {"choices": 0},
        {"choices": False},
    ]

    for payload in bad_payloads:
        try:
            asyncio.run(_collect_with_content(_sse(payload)))
        except ChatModelError:
            pass
        else:
            raise AssertionError(f"expected ChatModelError for {payload!r}")


def test_provider_error_does_not_echo_secret_arguments() -> None:
    content = "".join(
        [
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "echo", "arguments": '{"token":"sk-live-secret"}'},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _sse({"error": {"message": "bad argument sk-live-secret"}}),
        ]
    )

    try:
        asyncio.run(_collect_with_content(content))
    except ChatModelError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected ChatModelError")

    assert message == "Model stream returned a provider error"
    assert "sk-live-secret" not in message
