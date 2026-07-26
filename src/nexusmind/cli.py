from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys

from nexusmind.config import ConfigError, ModelConfig, load_model_config_from_env
from nexusmind.mcp import MCPError, MCPStdioClient, load_mcp_server_config, register_mcp_tools
from nexusmind.models.openai_compatible import OpenAICompatibleChatModel
from nexusmind.runtime.policy import ApprovalDecision, ApprovalRequest
from nexusmind.runtime.chat import ChatRuntime
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.tools import ToolCall, ToolErrorCode, ToolExecutor, ToolRegistry
from nexusmind.tools.builtin import ApprovalDemoTool, EchoTool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexusmind")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--mcp-config")
    chat_parser.add_argument("--mcp-server")
    chat_parser.add_argument("message", nargs="?")
    tools_parser = subparsers.add_parser("tools")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_subparsers.add_parser("list")
    tools_call_parser = tools_subparsers.add_parser("call")
    tools_call_parser.add_argument("name")
    tools_call_parser.add_argument("arguments")
    mcp_parser = subparsers.add_parser("mcp")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", required=True)
    mcp_tools_parser = mcp_subparsers.add_parser("tools")
    mcp_tools_parser.add_argument("--config", required=True)
    mcp_tools_parser.add_argument("--server", required=True)
    mcp_call_parser = mcp_subparsers.add_parser("call")
    mcp_call_parser.add_argument("--config", required=True)
    mcp_call_parser.add_argument("--server", required=True)
    mcp_call_parser.add_argument("--tool", required=True)
    mcp_call_parser.add_argument("--arguments", required=True)

    args = parser.parse_args(argv)
    if args.command == "chat":
        if bool(args.mcp_config) != bool(args.mcp_server):
            print("chat requires --mcp-config and --mcp-server together", file=sys.stderr)
            return 2
        return asyncio.run(_chat(args.message, mcp_config=args.mcp_config, mcp_server=args.mcp_server))
    if args.command == "tools":
        return asyncio.run(_tools(args))
    if args.command == "mcp":
        return asyncio.run(_mcp(args))
    return 2


async def _chat(message: str | None, *, mcp_config: str | None = None, mcp_server: str | None = None) -> int:
    if not message:
        message = input("> ").strip()
    if not message:
        print("No message provided.", file=sys.stderr)
        return 2

    try:
        model_config = load_model_config_from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    registry = _build_builtin_tool_registry()
    if mcp_config is None:
        return await _run_chat(message, registry, model_config=model_config)

    try:
        config = load_mcp_server_config(mcp_config, mcp_server)
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 1
    except Exception:
        print("MCP error: MCP chat tool setup failed", file=sys.stderr)
        return 1
    return await _run_chat_with_mcp(message, registry, config=config, model_config=model_config)


async def _run_chat_with_mcp(
    message: str,
    registry: ToolRegistry,
    *,
    config,
    model_config: ModelConfig,
) -> int:
    client = MCPStdioClient(config)
    try:
        await client.__aenter__()
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 1
    except Exception:
        print("MCP error: MCP chat tool setup failed", file=sys.stderr)
        return 1

    try:
        try:
            await register_mcp_tools(client, config.server_id, registry)
        except MCPError as exc:
            print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
            return_code = 1
        except Exception:
            print("MCP error: MCP chat tool setup failed", file=sys.stderr)
            return_code = 1
        else:
            return_code = await _run_chat(
                message,
                registry,
                model_config=model_config,
                executor_timeout=config.request_timeout,
            )
    except BaseException as exc:
        await client.__aexit__(type(exc), exc, exc.__traceback__)
        raise

    try:
        await client.__aexit__(None, None, None)
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 1
    return return_code


async def _run_chat(
    message: str,
    registry: ToolRegistry,
    *,
    model_config: ModelConfig,
    executor_timeout: float = 30.0,
) -> int:
    runtime = ChatRuntime(
        OpenAICompatibleChatModel(model_config),
        tool_executor=ToolExecutor(registry, timeout=executor_timeout),
        approval_provider=CLIApprovalProvider(),
    )
    tools = registry.list_definitions()
    failed = False
    async for event in runtime.stream_user_message(message, tools=tools):
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
        print(json.dumps(result.output, ensure_ascii=True, sort_keys=True))
        return 0
    return 2


def _build_builtin_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ApprovalDemoTool())
    registry.register(EchoTool())
    return registry


class CLIApprovalProvider:
    def __init__(self, input_stream=None, output_stream=None) -> None:
        self._input_stream = input_stream or sys.stdin
        self._output_stream = output_stream or sys.stderr

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        print("Tool approval required", file=self._output_stream)
        print(f"Tool: {_safe_cli_field(request.tool_name, max_length=120)}", file=self._output_stream)
        print(f"Risk: {request.risk_level.value}", file=self._output_stream)
        print(f"Action: {_safe_cli_field(request.summary, max_length=160)}", file=self._output_stream)
        print("", file=self._output_stream)
        print("[a] Allow once", file=self._output_stream)
        print("[d] Deny", file=self._output_stream)
        print("> ", end="", file=self._output_stream, flush=True)
        try:
            answer = await asyncio.to_thread(self._input_stream.readline)
        except Exception:
            return ApprovalDecision.DENY
        if not answer:
            return ApprovalDecision.DENY
        return ApprovalDecision.ALLOW_ONCE if answer.strip().lower() == "a" else ApprovalDecision.DENY


async def _mcp(args: argparse.Namespace) -> int:
    arguments = None
    if args.mcp_command == "call":
        try:
            arguments = json.loads(args.arguments)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON arguments: {exc.msg}", file=sys.stderr)
            return 2
        if not isinstance(arguments, dict):
            print("Tool arguments must be a JSON object.", file=sys.stderr)
            return 2
    try:
        config = load_mcp_server_config(args.config, args.server)
        async with MCPStdioClient(config) as client:
            registry = ToolRegistry()
            definitions = await register_mcp_tools(client, config.server_id, registry)
            if args.mcp_command == "tools":
                for definition in definitions:
                    remote_name = _remote_name_from_registry_tool(registry, definition.name)
                    description = _safe_cli_field(definition.description or "", max_length=160)
                    print(
                        json.dumps(
                            {
                                "name": definition.name,
                                "remote_name": _safe_cli_field(remote_name, max_length=160),
                                "description": description,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                return 0
            if args.mcp_command == "call":
                call = ToolCall(id="cli-mcp-call-1", name=args.tool, arguments=arguments)
                result = await ToolExecutor(registry, timeout=config.request_timeout).execute(call)
                if result.error:
                    print(f"{result.error.code.value}: {result.error.message}", file=sys.stderr)
                    return 2 if result.error.code in {ToolErrorCode.TOOL_NOT_FOUND, ToolErrorCode.INVALID_ARGUMENTS} else 1
                print(json.dumps(result.output, ensure_ascii=True, sort_keys=True))
                return 0
    except MCPError as exc:
        print(f"MCP error: {exc}", file=sys.stderr)
        return 1
    return 2


def _remote_name_from_registry_tool(registry: ToolRegistry, local_name: str) -> str:
    tool = registry.get(local_name)
    return str(getattr(tool, "remote_name", ""))


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _safe_cli_field(value: str, *, max_length: int) -> str:
    sanitized = _CONTROL_CHARS_RE.sub(" ", value)
    sanitized = " ".join(sanitized.split())
    if len(sanitized) <= max_length:
        return sanitized
    return sanitized[: max_length - 3].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())

