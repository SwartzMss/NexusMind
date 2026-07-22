from __future__ import annotations

import argparse
import asyncio
import sys

from nexusmind.config import ConfigError, load_model_config_from_env
from nexusmind.models.openai_compatible import OpenAICompatibleChatModel
from nexusmind.runtime.chat import ChatRuntime
from nexusmind.runtime.events import RuntimeEventType


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexusmind")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("message", nargs="?")

    args = parser.parse_args(argv)
    if args.command == "chat":
        return asyncio.run(_chat(args.message))
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


if __name__ == "__main__":
    raise SystemExit(main())

