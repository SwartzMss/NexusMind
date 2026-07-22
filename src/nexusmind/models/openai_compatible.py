from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from nexusmind.config import ModelConfig
from nexusmind.models.base import ChatModel, ChatModelError, ToolDefinition
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.messages import Message, MessageRole


class OpenAICompatibleChatModel(ChatModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

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
            async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                async with client.stream("POST", url, json=request, headers=headers) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise ChatModelError(_safe_http_error(response.status_code, body))
                    async for line in response.aiter_lines():
                        delta = _parse_sse_text_delta(line)
                        if delta:
                            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text=delta)
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
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        },
    }


def _parse_sse_text_delta(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = payload.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("delta", {}).get("content")
    return content if isinstance(content, str) else None


def _safe_http_error(status_code: int, body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
        message = payload.get("error", {}).get("message") or payload.get("message")
        if isinstance(message, str) and message:
            return f"Model provider returned HTTP {status_code}: {message}"
    except json.JSONDecodeError:
        pass
    return f"Model provider returned HTTP {status_code}"

