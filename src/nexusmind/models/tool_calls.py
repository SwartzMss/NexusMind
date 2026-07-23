from __future__ import annotations

import json
from dataclasses import dataclass, field

from nexusmind.tools.contracts import ToolCall

MAX_TOOL_CALLS_PER_TURN = 32
MAX_CALL_ID_LENGTH = 256
MAX_TOOL_NAME_LENGTH = 256
MAX_ARGUMENTS_LENGTH_PER_CALL = 1024 * 1024
MAX_ARGUMENTS_LENGTH_TOTAL = 4 * 1024 * 1024


class ToolCallAssemblyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    index: int
    call_id_fragment: str = ""
    name_fragment: str = ""
    arguments_fragment: str = field(default="", repr=False)
    type_fragment: str = ""


@dataclass(slots=True)
class _PartialToolCall:
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    type_name: str = ""


class ToolCallAssembler:
    def __init__(self) -> None:
        self._partials: dict[int, _PartialToolCall] = {}
        self._total_arguments_length = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self._partials)

    def apply(self, delta: ToolCallDelta) -> None:
        if delta.index < 0:
            raise ToolCallAssemblyError("Model stream returned an invalid tool call index")
        if delta.index >= MAX_TOOL_CALLS_PER_TURN:
            raise ToolCallAssemblyError("Model stream exceeded the maximum tool call count")
        partial = self._partials.setdefault(delta.index, _PartialToolCall())
        if delta.call_id_fragment:
            partial.call_id += delta.call_id_fragment
            if len(partial.call_id) > MAX_CALL_ID_LENGTH:
                raise ToolCallAssemblyError("Model stream returned an overlong tool call id")
        if delta.name_fragment:
            partial.name += delta.name_fragment
            if len(partial.name) > MAX_TOOL_NAME_LENGTH:
                raise ToolCallAssemblyError("Model stream returned an overlong tool call name")
        if delta.type_fragment:
            partial.type_name += delta.type_fragment
            if partial.type_name != "function"[: len(partial.type_name)] and partial.type_name != "function":
                raise ToolCallAssemblyError("Model stream returned an unsupported tool call type")
        if delta.arguments_fragment:
            partial.arguments += delta.arguments_fragment
            self._total_arguments_length += len(delta.arguments_fragment)
            if len(partial.arguments) > MAX_ARGUMENTS_LENGTH_PER_CALL:
                raise ToolCallAssemblyError("Model stream returned overlong tool call arguments")
            if self._total_arguments_length > MAX_ARGUMENTS_LENGTH_TOTAL:
                raise ToolCallAssemblyError("Model stream returned too many tool call arguments")

    def finalize(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        seen_ids: set[str] = set()
        for index in sorted(self._partials):
            partial = self._partials[index]
            if not partial.call_id:
                raise ToolCallAssemblyError("Model stream ended with an incomplete tool call")
            if partial.call_id in seen_ids:
                raise ToolCallAssemblyError("Model stream returned duplicate tool call ids")
            seen_ids.add(partial.call_id)
            if partial.type_name and partial.type_name != "function":
                raise ToolCallAssemblyError("Model stream returned an unsupported tool call type")
            if not partial.name:
                raise ToolCallAssemblyError("Model stream ended with an incomplete tool call")
            raw_arguments = partial.arguments.strip()
            if not raw_arguments:
                arguments = {}
            else:
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise ToolCallAssemblyError("Model stream returned invalid tool call arguments") from exc
                if not isinstance(arguments, dict):
                    raise ToolCallAssemblyError("Model stream returned non-object tool call arguments")
            calls.append(ToolCall(id=partial.call_id, name=partial.name, arguments=arguments))
        return calls
