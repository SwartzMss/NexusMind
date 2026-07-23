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
                        body = await response.aread()
                        raise ChatModelError(_safe_http_error(response.status_code, body, self._config.api_key))
                    completed = False
                    assembler = ToolCallAssembler()
                    finish_reason: str | None = None
                    async for line in response.aiter_lines():
                        chunk = _parse_sse_chunk(line, self._config.api_key)
                        completed = completed or chunk.completed
                        finish_reason = chunk.finish_reason or finish_reason
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
    finish_reason: str | None = None
    tool_call_deltas: tuple[ToolCallDelta, ...] = ()


def _parse_sse_chunk(line: str, api_key: str) -> _SSEChunk:
    if not line.startswith("data:"):
        return _SSEChunk()
    data = line.removeprefix("data:").strip()
    if not data:
        return _SSEChunk()
    if data == "[DONE]":
        return _SSEChunk(completed=True)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ChatModelError("Model stream returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ChatModelError("Model stream returned a non-object payload")
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        raise ChatModelError(_redact_secret(str(message or "Model stream returned an error"), api_key))
    if isinstance(error, str):
        raise ChatModelError(_redact_secret(error, api_key))
    choices = payload.get("choices") or []
    if not isinstance(choices, list):
        raise ChatModelError("Model stream returned invalid choices")
    if not choices:
        return _SSEChunk()
    if len(choices) != 1:
        raise ChatModelError("Model stream returned multiple choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ChatModelError("Model stream returned an invalid choice")
    raw_finish_reason = choice.get("finish_reason")
    finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else None
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
    raw_tool_calls = delta.get("tool_calls")
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise ChatModelError("Model stream returned invalid tool_calls")
    parsed: list[ToolCallDelta] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            raise ChatModelError("Model stream returned an invalid tool call")
        index = item.get("index")
        if not isinstance(index, int):
            raise ChatModelError("Model stream returned a tool call without a valid index")
        call_id = item.get("id", "")
        if call_id is None:
            call_id = ""
        if not isinstance(call_id, str):
            raise ChatModelError("Model stream returned an invalid tool call id")
        type_name = item.get("type", "")
        if type_name is None:
            type_name = ""
        if not isinstance(type_name, str):
            raise ChatModelError("Model stream returned an invalid tool call type")
        function = item.get("function", {})
        if function is None:
            function = {}
        if not isinstance(function, dict):
            raise ChatModelError("Model stream returned an invalid tool call function")
        name = function.get("name", "")
        if name is None:
            name = ""
        if not isinstance(name, str):
            raise ChatModelError("Model stream returned an invalid tool call name")
        arguments = function.get("arguments", "")
        if arguments is None:
            arguments = ""
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


def _safe_http_error(status_code: int, body: bytes, api_key: str) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
        message = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
            message = message or payload.get("message")
        if isinstance(message, str) and message:
            return _redact_secret(f"Model provider returned HTTP {status_code}: {message}", api_key)
    except json.JSONDecodeError:
        pass
    return f"Model provider returned HTTP {status_code}"


def _redact_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")

