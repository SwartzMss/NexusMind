from __future__ import annotations

import json
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import httpx

from nexusmind.config import ModelConfig
from nexusmind.models.base import ChatModel, ChatModelError
from nexusmind.models.tool_calls import ToolCallAssembler, ToolCallAssemblyError, ToolCallDelta
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolDefinition

_MAX_SSE_EVENT_BYTES = 1024 * 1024


class OpenAICompatibleChatModel(ChatModel):
    def __init__(self, config: ModelConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._transport = transport

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        yield RuntimeEvent(RuntimeEventType.MODEL_STARTED, metadata={"model": self._config.model})

        request = {
            "model": self._config.model,
            "messages": [_to_openai_message(message) for message in messages],
            "stream": True,
        }
        if tools:
            request["tools"] = [_to_openai_tool(tool) for tool in tools]

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self._config.timeout, transport=self._transport) as client:
                async with client.stream("POST", url, json=request, headers=headers) as response:
                    if response.status_code >= 400:
                        await response.aclose()
                        raise ChatModelError(_safe_http_error(response.status_code, response.headers))
                    completed = False
                    assembler = ToolCallAssembler()
                    finish_reason: str | None = None
                    async for line in _aiter_sse_lines(response):
                        chunk = _parse_sse_chunk(line, self._config.api_key)
                        if chunk.done:
                            completed = True
                            break
                        if completed:
                            if chunk.text or chunk.tool_call_deltas:
                                raise ChatModelError("Model stream returned data after completion")
                            if (
                                chunk.finish_reason is not None
                                and chunk.finish_reason != finish_reason
                            ):
                                raise ChatModelError(
                                    "Model stream returned conflicting finish reasons"
                                )
                            continue
                        completed = completed or chunk.completed
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if chunk.text:
                            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text=chunk.text)
                        for delta in chunk.tool_call_deltas:
                            try:
                                assembler.apply(delta)
                            except ToolCallAssemblyError as exc:
                                raise ChatModelError(str(exc)) from exc
                            yield RuntimeEvent(RuntimeEventType.TOOL_CALL_DELTA, tool_call_delta=delta)
                    if not completed:
                        raise ChatModelError("Model stream ended before completion")
                    if finish_reason == "tool_calls" and not assembler.has_tool_calls:
                        raise ChatModelError("Model stream ended without tool calls")
                    if assembler.has_tool_calls:
                        try:
                            for tool_call in assembler.finalize():
                                yield RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=tool_call)
                        except ToolCallAssemblyError as exc:
                            raise ChatModelError(str(exc)) from exc
                    yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason=_normalize_finish_reason(finish_reason))
        except ChatModelError:
            raise
        except httpx.TimeoutException as exc:
            raise ChatModelError("Model request timed out") from exc
        except httpx.HTTPError as exc:
            raise ChatModelError(f"Model request failed: {exc.__class__.__name__}") from exc


def _to_openai_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.role == MessageRole.TOOL and message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _to_openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": deepcopy(tool.input_schema),
        },
    }


@dataclass(frozen=True, slots=True)
class _SSEChunk:
    text: str | None = None
    completed: bool = False
    done: bool = False
    finish_reason: str | None = None
    tool_call_deltas: tuple[ToolCallDelta, ...] = ()


def _parse_sse_chunk(line: str, api_key: str) -> _SSEChunk:
    if not line.startswith("data:"):
        return _SSEChunk()
    data = line.removeprefix("data:").strip()
    if not data:
        return _SSEChunk()
    if data == "[DONE]":
        return _SSEChunk(completed=True, done=True)
    try:
        payload = json.loads(data, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError) as exc:
        raise ChatModelError("Model stream returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ChatModelError("Model stream returned a non-object payload")
    if "error" in payload and payload["error"] is not None:
        raise ChatModelError("Model stream returned a provider error")
    if "choices" not in payload:
        return _SSEChunk()
    choices = payload.get("choices")
    if not isinstance(choices, list):
        raise ChatModelError("Model stream returned invalid choices")
    if not choices:
        return _SSEChunk()
    if len(choices) != 1:
        raise ChatModelError("Model stream returned multiple choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ChatModelError("Model stream returned an invalid choice")
    choice_index = choice.get("index")
    if not isinstance(choice_index, int) or isinstance(choice_index, bool) or choice_index != 0:
        raise ChatModelError("Model stream returned an invalid choice index")
    raw_finish_reason = choice.get("finish_reason")
    if raw_finish_reason is not None and not isinstance(raw_finish_reason, str):
        raise ChatModelError("Model stream returned invalid finish_reason")
    finish_reason = raw_finish_reason
    completed = bool(finish_reason)
    delta = choice.get("delta", {})
    if not isinstance(delta, dict):
        raise ChatModelError("Model stream returned an invalid delta")
    if "function_call" in delta:
        raise ChatModelError("Legacy function_call streaming is not supported")
    content = delta.get("content")
    if content is not None and not isinstance(content, str):
        raise ChatModelError("Model stream returned invalid content")
    return _SSEChunk(
        text=content if isinstance(content, str) else None,
        completed=completed,
        finish_reason=finish_reason,
        tool_call_deltas=tuple(_parse_tool_call_deltas(delta)),
    )


def _parse_tool_call_deltas(delta: dict[str, Any]) -> list[ToolCallDelta]:
    if "tool_calls" not in delta:
        return []
    raw_tool_calls = delta["tool_calls"]
    if not isinstance(raw_tool_calls, list):
        raise ChatModelError("Model stream returned invalid tool_calls")
    parsed: list[ToolCallDelta] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            raise ChatModelError("Model stream returned an invalid tool call")
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ChatModelError("Model stream returned a tool call without a valid index")
        call_id = item.get("id", "")
        if not isinstance(call_id, str):
            raise ChatModelError("Model stream returned an invalid tool call id")
        type_name = item.get("type", "")
        if not isinstance(type_name, str):
            raise ChatModelError("Model stream returned an invalid tool call type")
        function = item.get("function", {})
        if not isinstance(function, dict):
            raise ChatModelError("Model stream returned an invalid tool call function")
        name = function.get("name", "")
        if not isinstance(name, str):
            raise ChatModelError("Model stream returned an invalid tool call name")
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            raise ChatModelError("Model stream returned invalid tool call arguments")
        parsed.append(
            ToolCallDelta(
                index=index,
                call_id_fragment=call_id,
                name_fragment=name,
                arguments_fragment=arguments,
                type_fragment=type_name,
            )
        )
    return parsed


def _normalize_finish_reason(finish_reason: str | None) -> str:
    if finish_reason in {"stop", "tool_calls", "length", "content_filter"}:
        return finish_reason
    return "unknown" if finish_reason else "null"


async def _aiter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        buffer.extend(chunk)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(buffer[:newline]).rstrip(b"\r")
            del buffer[: newline + 1]
            if len(raw_line) > _MAX_SSE_EVENT_BYTES:
                raise ChatModelError("Model stream returned an oversized SSE event")
            yield _decode_sse_line(raw_line)
        if len(buffer) > _MAX_SSE_EVENT_BYTES:
            raise ChatModelError("Model stream returned an oversized SSE event")
    if buffer:
        if len(buffer) > _MAX_SSE_EVENT_BYTES:
            raise ChatModelError("Model stream returned an oversized SSE event")
        yield _decode_sse_line(bytes(buffer).rstrip(b"\r"))


def _decode_sse_line(raw_line: bytes) -> str:
    try:
        return raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ChatModelError("Model stream returned invalid UTF-8") from exc


def _safe_http_error(status_code: int, headers: httpx.Headers) -> str:
    request_id = headers.get("x-request-id") or headers.get("request-id")
    if request_id:
        return f"Model provider returned HTTP {status_code} (request_id={request_id})"
    return f"Model provider returned HTTP {status_code}"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")

