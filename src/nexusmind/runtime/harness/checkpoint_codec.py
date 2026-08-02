from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from collections import OrderedDict

from nexusmind.runtime.harness.checkpoint import HarnessCheckpoint
from nexusmind.runtime.harness.state import HarnessPhase, HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolCall

class CheckpointDecodeError(ValueError):
    pass

MAX_CODEC_PAYLOAD_BYTES = 4 * 1024 * 1024

def _reject_duplicate_pairs(pairs):
    result = OrderedDict()
    for key, value in pairs:
        if key in result:
            raise CheckpointDecodeError("Checkpoint payload contains duplicate JSON fields")
        result[key] = value
    return result

def _message_to_dict(message: Message) -> dict[str, Any]:
    return {"role": message.role.value, "content": message.content, "name": message.name, "tool_call_id": message.tool_call_id, "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls], "metadata": message.metadata}

def checkpoint_to_json(checkpoint: HarnessCheckpoint) -> str:
    checkpoint.validate()
    state = checkpoint.state
    payload = {"schema_version": checkpoint.schema_version, "checkpoint_id": checkpoint.checkpoint_id, "run_id": checkpoint.run_id, "sequence": checkpoint.sequence, "boundary": checkpoint.boundary.value, "created_at": checkpoint.created_at, "state": {"messages": [_message_to_dict(m) for m in state.messages], "model_turns": state.model_turns, "tool_calls_total": state.tool_calls_total, "tool_argument_bytes_total": state.tool_argument_bytes_total, "tool_result_bytes_total": state.tool_result_bytes_total, "started_tool_call_ids": list(state.started_tool_call_ids), "executed_tool_call_ids": list(state.executed_tool_call_ids), "status": state.status.value, "stop_reason": state.stop_reason.value if state.stop_reason else None, "phase": state.phase.value}}
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))

def checkpoint_from_json(payload: str) -> HarnessCheckpoint:
    try:
        if type(payload) is not str:
            raise CheckpointDecodeError("Checkpoint payload must be text")
        if len(payload.encode("utf-8")) > MAX_CODEC_PAYLOAD_BYTES:
            raise CheckpointDecodeError("Checkpoint payload exceeds maximum size")
        data = json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
        if not isinstance(data, dict) or set(data) != {"schema_version", "checkpoint_id", "run_id", "sequence", "boundary", "created_at", "state"}:
            raise CheckpointDecodeError("Invalid checkpoint fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise CheckpointDecodeError("Unsupported checkpoint schema version")
        if type(data["sequence"]) is not int or data["sequence"] < 0:
            raise CheckpointDecodeError("Invalid checkpoint sequence")
        raw = data["state"]
        required = {"messages", "model_turns", "tool_calls_total", "tool_argument_bytes_total", "tool_result_bytes_total", "started_tool_call_ids", "executed_tool_call_ids", "status", "stop_reason", "phase"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise CheckpointDecodeError("Invalid checkpoint state fields")
        if type(raw["messages"]) is not list:
            raise CheckpointDecodeError("Checkpoint messages must be an array")
        for field in ("started_tool_call_ids", "executed_tool_call_ids"):
            if type(raw[field]) is not list:
                raise CheckpointDecodeError(f"Checkpoint {field} must be an array")
            if any(type(call_id) is not str or not call_id for call_id in raw[field]):
                raise CheckpointDecodeError(f"Checkpoint {field} contains invalid IDs")
        messages = []
        for item in raw["messages"]:
            if not isinstance(item, dict) or set(item) != {"role", "content", "name", "tool_call_id", "tool_calls", "metadata"}:
                raise CheckpointDecodeError("Invalid message fields")
            if type(item["tool_calls"]) is not list or not isinstance(item["metadata"], dict):
                raise CheckpointDecodeError("Invalid message value types")
            if item["content"] is not None and type(item["content"]) is not str:
                raise CheckpointDecodeError("Invalid message content type")
            if item["name"] is not None and type(item["name"]) is not str:
                raise CheckpointDecodeError("Invalid message name type")
            if item["tool_call_id"] is not None and type(item["tool_call_id"]) is not str:
                raise CheckpointDecodeError("Invalid tool call id type")
            calls_data = []
            for call in item["tool_calls"]:
                if not isinstance(call, dict) or set(call) != {"id", "name", "arguments"} or type(call["id"]) is not str or type(call["name"]) is not str or type(call["arguments"]) is not dict:
                    raise CheckpointDecodeError("Invalid tool call fields")
                calls_data.append(ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"]))
            calls = tuple(calls_data)
            messages.append(Message(role=MessageRole(item["role"]), content=item["content"], name=item["name"], tool_call_id=item["tool_call_id"], tool_calls=calls, metadata=item["metadata"]))
        from nexusmind.runtime.harness.checkpoint import HarnessStateSnapshot
        snapshot = HarnessStateSnapshot(tuple(messages), raw["model_turns"], raw["tool_calls_total"], raw["tool_argument_bytes_total"], raw["tool_result_bytes_total"], tuple(raw["started_tool_call_ids"]), tuple(raw["executed_tool_call_ids"]), HarnessStatus(raw["status"]), StopReason(raw["stop_reason"]) if raw["stop_reason"] is not None else None, HarnessPhase(raw["phase"]))
        checkpoint = HarnessCheckpoint(data["schema_version"], data["checkpoint_id"], data["run_id"], data["sequence"], __import__("nexusmind.runtime.harness.checkpoint", fromlist=["CheckpointBoundary"]).CheckpointBoundary(data["boundary"]), snapshot, data["created_at"])
        checkpoint.validate()
        return checkpoint
    except CheckpointDecodeError:
        raise
    except Exception as exc:
        raise CheckpointDecodeError("Invalid checkpoint payload") from exc
