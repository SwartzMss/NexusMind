from __future__ import annotations

import argparse
import asyncio
import json
import sys

from nexusmind.config import ConfigError, load_model_config_from_env
from nexusmind.models.openai_compatible import OpenAICompatibleChatModel
from nexusmind.runtime.chat import ChatRuntime
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.tools import ToolCall, ToolErrorCode, ToolExecutor, ToolRegistry
from nexusmind.tools.builtin import EchoTool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexusmind")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("message", nargs="?")
    tools_parser = subparsers.add_parser("tools")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_subparsers.add_parser("list")
    tools_call_parser = tools_subparsers.add_parser("call")
    tools_call_parser.add_argument("name")
    tools_call_parser.add_argument("arguments")

    args = parser.parse_args(argv)
    if args.command == "chat":
        return asyncio.run(_chat(args.message))
    if args.command == "tools":
        return asyncio.run(_tools(args))
    return 2


async def _chat(message: str | None) -> int:
    if not message:
        message = input("> ").strip()
    if not message:
        print("No message provided.", file=sys.stderr)
        return 2

    try:
        config = load_model_config_from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    runtime = ChatRuntime(OpenAICompatibleChatModel(config))
    failed = False
    async for event in runtime.stream_user_message(message):
        if event.type == RuntimeEventType.TEXT_DELTA and event.text:
            print(event.text, end="", flush=True)
        elif event.type == RuntimeEventType.RUN_FAILED:
            failed = True
            print(f"\nModel error: {event.error}", file=sys.stderr)

    if not failed:
        print()
    return 1 if failed else 0


async def _tools(args: argparse.Namespace) -> int:
    registry = _build_builtin_tool_registry()
    if args.tools_command == "list":
        for definition in registry.list_definitions():
            description = definition.description or ""
            print(f"{definition.name}\t{description}")
        return 0
    if args.tools_command == "call":
        try:
            arguments = json.loads(args.arguments)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON arguments: {exc.msg}", file=sys.stderr)
            return 2
        if not isinstance(arguments, dict):
            print("Tool arguments must be a JSON object.", file=sys.stderr)
            return 2

        result = await ToolExecutor(registry).execute(ToolCall(id="cli-call-1", name=args.name, arguments=arguments))
        if result.error:
            print(f"{result.error.code.value}: {result.error.message}", file=sys.stderr)
            return 2 if result.error.code in {ToolErrorCode.TOOL_NOT_FOUND, ToolErrorCode.INVALID_ARGUMENTS} else 1
        print(json.dumps(result.output, ensure_ascii=False, sort_keys=True))
        return 0
    return 2


def _build_builtin_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


if __name__ == "__main__":
    raise SystemExit(main())

