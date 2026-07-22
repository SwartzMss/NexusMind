from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from nexusmind.config import ModelConfig
from nexusmind.models.base import ChatModel, ChatModelError, ToolDefinition
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole


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
            raise ChatModelError("Tool calls are not supported by the OpenAI-compatible adapter yet")

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
                    async for line in response.aiter_lines():
                        chunk = _parse_sse_chunk(line, self._config.api_key)
                        completed = completed or chunk.completed
                        if chunk.text:
                            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text=chunk.text)
                    if not completed:
                        raise ChatModelError("Model stream ended before completion")
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


@dataclass(frozen=True, slots=True)
class _SSEChunk:
    text: str | None = None
    completed: bool = False


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
        return _SSEChunk()
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        raise ChatModelError(_redact_secret(str(message or "Model stream returned an error"), api_key))
    if isinstance(error, str):
        raise ChatModelError(_redact_secret(error, api_key))
    choices = payload.get("choices") or []
    if not isinstance(choices, list):
        return _SSEChunk()
    if not choices:
        return _SSEChunk()
    choice = choices[0]
    if not isinstance(choice, dict):
        return _SSEChunk()
    completed = bool(choice.get("finish_reason"))
    delta = choice.get("delta", {})
    if not isinstance(delta, dict):
        return _SSEChunk(completed=completed)
    content = delta.get("content")
    return _SSEChunk(text=content if isinstance(content, str) else None, completed=completed)


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

