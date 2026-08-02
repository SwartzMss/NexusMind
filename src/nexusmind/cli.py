from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re
import sys

from nexusmind.commands import CommandConfig, CommandConfigError, RunCommandTool, command_profile_summary, load_command_config
from nexusmind.config import ConfigError, ModelConfig, load_model_config_from_env
from nexusmind.mcp import MCPClientGroup, MCPError, MCPStdioClient, load_mcp_server_config, load_mcp_server_configs, register_mcp_tools
from nexusmind.models.openai_compatible import OpenAICompatibleChatModel
from nexusmind.runtime.chat import AgentLoopLimits
from nexusmind.runtime.policy import ApprovalDecision, ApprovalRequest, DefaultToolApprovalSummarizer
from nexusmind.runtime.chat import ChatRuntime
from nexusmind.runtime.events import RuntimeEventType
from nexusmind.skills import SkillError, discover_skills, load_skill, resolve_skill_tool_references
from nexusmind.skills.resolver import (
    build_skill_loop_limits,
    skill_mcp_server_ids,
    skill_requires_mcp,
    skill_requires_workspace,
    skill_requires_workspace_exec,
    skill_requires_workspace_write,
    validate_builtin_skill_tool_references,
)
from nexusmind.tools import ToolCall, ToolErrorCode, ToolExecutor, ToolRegistry
from nexusmind.tools.contracts import ToolDefinition
from nexusmind.tools.builtin import ApprovalDemoTool, EchoTool, ListFilesTool, ReadFileTool, ReplaceTextTool, SearchTextTool, WriteFileTool
from nexusmind.tools.builtin.workspace import WorkspaceWriteBudget
from nexusmind.workspace import Workspace, WorkspaceError, resolve_workspace_create_target, resolve_workspace_path, workspace_relative_path
from nexusmind.persistence import SQLiteRunStore, StateStoreError, RunKind, RunStartContext, RunStatus, RunTraceEvent
from datetime import datetime, timezone

async def _best_effort_cancel(store, run_id) -> None:
    task = None
    try:
        if store and run_id:
            task = asyncio.create_task(store.finish_run(run_id, RunStatus.CANCELLED, error_code="cancelled"))
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
    except BaseException:
        if task is not None and not task.done():
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
    finally:
        if store:
            try: store.close()
            except Exception: pass

async def _best_effort_finish(store, run_id, status, **kwargs) -> bool:
    try:
        if store and run_id: await store.finish_run(run_id, status, **kwargs)
        return True
    except StateStoreError: return False

def _best_effort_close(store) -> None:
    if store:
        try: store.close()
        except StateStoreError: pass

def _truncate_utf8(value: str, limit: int) -> str:
    return value.encode("utf-8")[:limit].decode("utf-8", "ignore")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexusmind")
    subparsers = parser.add_subparsers(dest="command", required=True)
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--mcp-config")
    chat_parser.add_argument("--mcp-server")
    chat_parser.add_argument("--workspace")
    chat_parser.add_argument("--workspace-write", action="store_true")
    chat_parser.add_argument("--workspace-exec", action="store_true")
    chat_parser.add_argument("--command-config")
    chat_parser.add_argument("--state-db")
    chat_parser.add_argument("--record-content", action="store_true")
    chat_parser.add_argument("message", nargs="?")
    tools_parser = subparsers.add_parser("tools")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command", required=True)
    tools_subparsers.add_parser("list")
    tools_call_parser = tools_subparsers.add_parser("call")
    tools_call_parser.add_argument("name")
    tools_call_parser.add_argument("arguments")
    runs_parser = subparsers.add_parser("runs")
    runs_sub = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_sub.add_parser("list"); runs_list.add_argument("--state-db", required=True); runs_list.add_argument("--limit", type=int, default=20); runs_list.add_argument("--status"); runs_list.add_argument("--kind"); runs_list.add_argument("--skill")
    runs_show = runs_sub.add_parser("show"); runs_show.add_argument("run_id"); runs_show.add_argument("--state-db", required=True); runs_show.add_argument("--json", action="store_true")
    runs_prune = runs_sub.add_parser("prune"); runs_prune.add_argument("--state-db", required=True); runs_prune.add_argument("--older-than-days", type=int, required=True)
    runs_recover = runs_sub.add_parser("recover"); runs_recover.add_argument("--state-db", required=True)
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
    skill_parser = subparsers.add_parser("skill")
    skill_subparsers = skill_parser.add_subparsers(dest="skill_command", required=True)
    skill_list_parser = skill_subparsers.add_parser("list")
    skill_list_parser.add_argument("--skills-dir", default="./skills")
    skill_show_parser = skill_subparsers.add_parser("show")
    skill_show_parser.add_argument("name")
    skill_show_parser.add_argument("--skills-dir", default="./skills")
    skill_run_parser = skill_subparsers.add_parser("run")
    skill_run_parser.add_argument("name")
    skill_run_parser.add_argument("--skills-dir", default="./skills")
    skill_run_parser.add_argument("--mcp-config")
    skill_run_parser.add_argument("--workspace")
    skill_run_parser.add_argument("--workspace-write", action="store_true")
    skill_run_parser.add_argument("--workspace-exec", action="store_true")
    skill_run_parser.add_argument("--command-config")
    skill_run_parser.add_argument("--state-db")
    skill_run_parser.add_argument("--record-content", action="store_true")
    skill_run_parser.add_argument("message", nargs="*")

    parse_argv, separator_message = _split_skill_run_separator_message(argv)
    args, extra_args = parser.parse_known_args(parse_argv)
    if extra_args:
        if args.command == "skill" and getattr(args, "skill_command", None) == "run" and _skill_run_extra_args_are_message(extra_args):
            args.message.extend(_strip_argument_separator(extra_args))
        else:
            parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
    if separator_message and args.command == "skill" and getattr(args, "skill_command", None) == "run":
        args.message.extend(separator_message)
    if args.command == "chat":
        if bool(args.mcp_config) != bool(args.mcp_server):
            print("chat requires --mcp-config and --mcp-server together", file=sys.stderr)
            return 2
        return asyncio.run(
            _chat(
                args.message,
                mcp_config=args.mcp_config,
                mcp_server=args.mcp_server,
                workspace_path=args.workspace,
                enable_workspace_write=args.workspace_write,
                enable_workspace_exec=args.workspace_exec,
                command_config_path=args.command_config,
                state_db=args.state_db, record_content=args.record_content,
            )
        )
    if args.command == "tools":
        return asyncio.run(_tools(args))
    if args.command == "runs":
        return _runs(args)
    if args.command == "mcp":
        return asyncio.run(_mcp(args))
    if args.command == "skill":
        return asyncio.run(_skill(args))
    return 2


async def _chat(
    message: str | None,
    *,
    mcp_config: str | None = None,
    mcp_server: str | None = None,
    workspace_path: str | None = None,
    enable_workspace_write: bool = False,
    enable_workspace_exec: bool = False,
    command_config_path: str | None = None,
    state_db: str | None = None, record_content: bool = False,
) -> int:
    if not message:
        message = input("> ").strip()
    if not message:
        print("No message provided.", file=sys.stderr)
        return 2

    if enable_workspace_write and workspace_path is None:
        print("Workspace error: --workspace-write requires --workspace", file=sys.stderr)
        return 2
    if enable_workspace_exec and workspace_path is None:
        print("Command error: --workspace-exec requires --workspace", file=sys.stderr)
        return 2
    if enable_workspace_exec and not command_config_path:
        print("Command error: --workspace-exec requires --command-config", file=sys.stderr)
        return 2

    if state_db:
        preflight = None
        try:
            preflight = SQLiteRunStore(state_db); preflight.close()
        except StateStoreError:
            print("State error: Run store could not be initialized", file=sys.stderr); return 2
        finally:
            _best_effort_close(preflight)

    try:
        workspace = _build_workspace(workspace_path)
    except WorkspaceError as exc:
        print(f"Workspace error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 2
    try:
        command_config = _build_command_config(command_config_path, workspace) if enable_workspace_exec else None
    except CommandConfigError as exc:
        print(f"Command error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 2

    try:
        model_config = load_model_config_from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    registry = (
        _build_builtin_tool_registry()
        if workspace is None
        else build_builtin_tool_registry(
            workspace=workspace,
            enable_workspace_write=enable_workspace_write,
            command_config=command_config,
        )
    )
    if mcp_config is None:
        return await _run_chat(message, registry, model_config=model_config, workspace=workspace, state_db=state_db, record_content=record_content)

    try:
        config = load_mcp_server_config(mcp_config, mcp_server)
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 1
    except Exception:
        print("MCP error: MCP chat tool setup failed", file=sys.stderr)
        return 1
    return await _run_chat_with_mcp(message, registry, config=config, model_config=model_config, state_db=state_db, record_content=record_content)


async def _run_chat_with_mcp(
    message: str,
    registry: ToolRegistry,
    *,
    config,
    model_config: ModelConfig,
    state_db: str | None = None, record_content: bool = False,
) -> int:
    store = None; run_id = None; text_sink: list[str] = []; return_code = 1
    try:
        if state_db:
            store = SQLiteRunStore(state_db)
            run_id = await store.start_run(RunStartContext(RunKind.CHAT, model_name=getattr(model_config, "model", None), input_text=message, record_content=record_content))
    except StateStoreError:
        _best_effort_close(store)
        print("State error: Run store could not be initialized", file=sys.stderr); return 2
    client = MCPStdioClient(config)
    try:
        await client.__aenter__()
    except asyncio.CancelledError:
        await _best_effort_cancel(store, run_id)
        raise
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        if store and run_id:
            try: await store.finish_run(run_id, RunStatus.FAILED, error_code="mcp_start_failed", error_message="MCP server failed to start")
            except StateStoreError: pass
        _best_effort_close(store)
        return 1
    except Exception:
        print("MCP error: MCP chat tool setup failed", file=sys.stderr)
        if store and run_id:
            try: await store.finish_run(run_id, RunStatus.FAILED, error_code="mcp_start_failed", error_message="MCP server setup failed")
            except StateStoreError: pass
        _best_effort_close(store)
        return 1

    try:
        try:
            await register_mcp_tools(client, config.server_id, registry)
        except MCPError as exc:
            print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
            await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_register_failed", error_message="MCP tool registration failed")
            return_code = 1
        except Exception:
            print("MCP error: MCP chat tool setup failed", file=sys.stderr)
            await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_register_failed", error_message="MCP tool registration failed")
            return_code = 1
        else:
            return_code = await _run_chat(
                message,
                registry,
                model_config=model_config,
                executor_timeout=config.request_timeout,
                workspace=_registry_workspace(registry),
                state_db=state_db, record_content=record_content,
                external_store=store, external_run_id=run_id, final_text_sink=text_sink,
            )
    except BaseException as exc:
        try:
            await client.__aexit__(type(exc), exc, exc.__traceback__)
        except BaseException:
            pass
        if isinstance(exc, asyncio.CancelledError) and store and run_id:
            await _best_effort_cancel(store, run_id)
        _best_effort_close(store)
        raise

    try:
        await client.__aexit__(None, None, None)
    except asyncio.CancelledError:
        await _best_effort_cancel(store, run_id)
        raise
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_cleanup_failed", error_message="MCP server cleanup failed")
        _best_effort_close(store)
        return 1
    if store and run_id:
        try: await store.finish_run(run_id, RunStatus.FAILED if return_code else RunStatus.COMPLETED, final_text=''.join(text_sink) if record_content else None)
        except StateStoreError:
            print("State error: Run final status could not be persisted", file=sys.stderr); return_code = 1
        _best_effort_close(store)
    return return_code


async def _run_chat(
    message: str,
    registry: ToolRegistry,
    *,
    model_config: ModelConfig,
    executor_timeout: float = 30.0,
    system_prompt: str | None = None,
    tools: list[ToolDefinition] | None = None,
    limits: AgentLoopLimits | None = None,
    workspace: Workspace | None = None,
    state_db: str | None = None, record_content: bool = False, run_kind: RunKind = RunKind.CHAT,
    skill_name: str | None = None,
    external_store=None, external_run_id: str | None = None, final_text_sink=None,
) -> int:
    store = external_store; run_id = external_run_id; owns_store = store is None
    if state_db and store is None:
        try:
            store = SQLiteRunStore(state_db)
            run_id = await store.start_run(RunStartContext(run_kind, skill_name=skill_name, model_name=getattr(model_config, "model", None), input_text=message, record_content=record_content))
        except StateStoreError:
            _best_effort_close(store)
            print("State error: Run store could not be initialized", file=sys.stderr); return 2
    runtime = ChatRuntime(
        OpenAICompatibleChatModel(model_config),
        tool_executor=ToolExecutor(registry, timeout=executor_timeout),
        approval_provider=CLIApprovalProvider(),
        approval_summarizer=CLIApprovalSummarizer(workspace=workspace, command_config=_registry_command_config(registry)),
        limits=limits,
    )
    tools = registry.list_definitions() if tools is None else tools
    failed = False; failure_message = None; final_text = []; current_text = []; current_text_bytes = 0
    stream_kwargs = {"tools": tools}
    if system_prompt is not None:
        stream_kwargs["system_prompt"] = system_prompt
    try:
        async for event in runtime.stream_user_message(message, **stream_kwargs):
            if event.type == RuntimeEventType.MODEL_STARTED:
                current_text.clear(); current_text_bytes = 0
            if store and run_id and event.type not in {RuntimeEventType.RUN_STARTED, RuntimeEventType.RUN_COMPLETED, RuntimeEventType.RUN_FAILED, RuntimeEventType.TEXT_DELTA, RuntimeEventType.TOOL_CALL_DELTA}:
                try:
                    projected = project_runtime_event(event)
                    if event.tool_call is not None:
                        try: definition = registry.definition(event.tool_call.name); projected["risk_level"] = definition.risk_level.value
                        except Exception: projected["risk_level"] = "unspecified"
                        try:
                            tool = registry.get(event.tool_call.name)
                            if hasattr(tool, "server_id"): projected.update(server_id=tool.server_id, remote_tool_name=tool.remote_name)
                        except Exception: pass
                    if event.tool_result is not None:
                        try:
                            tool = registry.get(event.tool_result.name)
                            if hasattr(tool, "server_id"): projected.update(server_id=tool.server_id, remote_tool_name=tool.remote_name)
                        except Exception: pass
                    if event.type == RuntimeEventType.MODEL_TURN_COMPLETED: projected["text_bytes"] = current_text_bytes
                    await store.append_event(run_id, RunTraceEvent(event.type.value, datetime.now(timezone.utc), projected))
                except StateStoreError:
                    try: await store.finish_run(run_id, RunStatus.FAILED, error_code="trace_persist_failed", error_message="Run trace could not be persisted", trace_complete=False)
                    except StateStoreError: pass
                    message = "Run trace could not be persisted after tool execution" if event.type == RuntimeEventType.TOOL_RESULT else "Run trace could not be persisted"
                    print(message, file=sys.stderr); return 1
            if event.type == RuntimeEventType.TEXT_DELTA and event.text:
                current_text.append(event.text); current_text_bytes += len(event.text.encode())
                print(event.text, end="", flush=True)
            elif event.type == RuntimeEventType.MODEL_TURN_COMPLETED and event.finish_reason != "tool_calls":
                if record_content and store and run_id:
                    current=_truncate_utf8(''.join(current_text), 1024*1024); final_text.clear(); final_text.append(current)
                    if final_text_sink is not None: final_text_sink.clear(); final_text_sink.append(current)
            elif event.type == RuntimeEventType.RUN_FAILED:
                failed = True; failure_message = _safe_cli_field(event.error or "Runtime execution failed", max_length=1024); print(f"\nModel error: {failure_message}", file=sys.stderr)
        if not failed: print()
        if store and run_id and owns_store:
            try: await store.finish_run(run_id, RunStatus.FAILED if failed else RunStatus.COMPLETED, error_code="runtime_failed" if failed else None, error_message=failure_message, final_text=''.join(final_text) if record_content else None)
            except StateStoreError:
                print("State error: Run final status could not be persisted", file=sys.stderr); return 1
        return 1 if failed else 0
    except asyncio.CancelledError:
        await _best_effort_cancel(store, run_id)
        raise
    except Exception:
        await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="runtime_error", error_message="Runtime execution failed")
        raise
    finally:
        if owns_store: _best_effort_close(store)


async def _tools(args: argparse.Namespace) -> int:
    registry = build_builtin_tool_registry()
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

def _open_store(path: str):
    try: return SQLiteRunStore(path)
    except StateStoreError:
        print("State error: Run store could not be initialized", file=sys.stderr); return None

def project_runtime_event(event) -> dict:
    payload = {"text_bytes": len((event.text or '').encode()), "error": _safe_cli_field(event.error or '', max_length=1024)}
    if event.finish_reason: payload["finish_reason"] = event.finish_reason
    if event.tool_call is not None:
        call = event.tool_call; payload.update(call_id=call.id, tool_name=call.name, risk_level=event.metadata.get("risk_level", "unspecified"), argument_bytes=len(json.dumps(call.arguments, ensure_ascii=False).encode()))
    if event.tool_approval is not None:
        approval = event.tool_approval; payload.update(request_id=approval.request_id, call_id=approval.call_id, tool_name=approval.tool_name, risk_level=approval.risk_level.value, decision=getattr(approval.decision, 'value', approval.decision), summary=_safe_cli_field(approval.summary, max_length=512))
    if event.tool_result is not None:
        result = event.tool_result; output = result.output
        payload.update(call_id=result.call_id, tool_name=result.name, ok=result.error is None, result_bytes=len(json.dumps(output, ensure_ascii=False, default=str).encode()), result_truncated=bool(getattr(result, "metadata", {}).get("result_truncated", False)), error_code=getattr(result.error.code, 'value', None) if result.error else None)
        if result.name == "run_command" and isinstance(output, dict):
            payload.update(profile=output.get("profile"), cwd=output.get("cwd"), exit_code=output.get("exit_code"), timed_out=output.get("timed_out"), duration_ms=output.get("duration_ms"), stdout_bytes=len(str(output.get("stdout", "")).encode()), stderr_bytes=len(str(output.get("stderr", "")).encode()), stdout_truncated=output.get("stdout_truncated"), stderr_truncated=output.get("stderr_truncated"))
        if result.name in {"write_file", "replace_text"} and isinstance(output, dict):
            for key in ("path", "operation", "previous_sha256", "sha256", "bytes_written", "replacements", "committed", "cleanup_warning"):
                if key in output: payload[key] = output[key]
    return payload

def _runs(args: argparse.Namespace) -> int:
    store=_open_store(args.state_db)
    if store is None:return 2
    try:
        if args.runs_command == "list":
            print("RUN_ID\tSTATUS\tKIND\tSTARTED_AT\tSKILL")
            for row in store.list_runs(limit=args.limit,status=args.status,kind=args.kind,skill=args.skill): print("\t".join(str(x or "-") for x in row))
            return 0
        if args.runs_command == "show":
            item=store.show_run(args.run_id)
            if item is None: print("Run error: run not found",file=sys.stderr); return 2
            if args.json: print(json.dumps(item,default=str,ensure_ascii=False)); return 0
            print("Run ID\t"+str(item["run"]["id"])); print("Status\t"+str(item["run"]["status"])); print("Events\t"+str(len(item["events"])))
            for e in item["events"]:
                p=e["payload"]; summary=" ".join(f"{k}={p[k]}" for k in ("tool_name","risk_level","decision","path","committed","exit_code","server_id","remote_tool_name") if k in p)
                print(f"{e['sequence']}\t{e['event_type']}\t{e['occurred_at']}" + (f"\t{summary}" if summary else ""))
            return 0
        if args.runs_command == "recover":
            store.recover_abandoned(); print("Recovered abandoned runs"); return 0
        if args.runs_command == "prune":
            try: count=store.prune(args.older_than_days)
            except ValueError as exc: print(f"State error: {exc}", file=sys.stderr); return 2
            print(f"Deleted {count} run(s)"); return 0
    except StateStoreError:
        print("State error: Run store operation failed", file=sys.stderr); return 2
    finally: _best_effort_close(store)
    return 2


def build_builtin_tool_registry(
    *,
    workspace: Workspace | None = None,
    enable_workspace_write: bool = False,
    command_config: CommandConfig | None = None,
) -> ToolRegistry:
    if enable_workspace_write and workspace is None:
        raise WorkspaceError("--workspace-write requires --workspace")
    if command_config is not None and workspace is None:
        raise WorkspaceError("--workspace-exec requires --workspace")
    registry = ToolRegistry()
    tools = [ApprovalDemoTool(), EchoTool()]
    if workspace is not None:
        tools.extend([ListFilesTool(workspace), ReadFileTool(workspace), SearchTextTool(workspace)])
        if enable_workspace_write:
            budget = WorkspaceWriteBudget()
            tools.extend([WriteFileTool(workspace, budget), ReplaceTextTool(workspace, budget)])
        if command_config is not None:
            tools.append(RunCommandTool(command_config))
    registry.register_many(tools)
    return registry


def _build_builtin_tool_registry() -> ToolRegistry:
    return build_builtin_tool_registry()


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


class CLIApprovalSummarizer(DefaultToolApprovalSummarizer):
    def __init__(
        self,
        max_length: int = 160,
        *,
        workspace: Workspace | None = None,
        command_config: CommandConfig | None = None,
    ) -> None:
        super().__init__(max_length=max_length)
        self._workspace = workspace
        self._command_config = command_config

    def summarize(self, call: ToolCall, definition: ToolDefinition) -> str:
        if definition.name == "write_file":
            path = _approval_display_path(self._workspace, call.arguments.get("path"), create=True)
            mode = _safe_cli_field(str(call.arguments.get("mode", "")), max_length=20)
            content = call.arguments.get("content", "")
            size = len(content.encode("utf-8")) if isinstance(content, str) else 0
            sha = str(call.arguments.get("expected_sha256", ""))
            sha_prefix = sha[:12] if sha else "none"
            return f"Workspace {mode} {path}; bytes={size}; expected_sha256={sha_prefix}"
        if definition.name == "replace_text":
            path = _approval_display_path(self._workspace, call.arguments.get("path"), create=False)
            old_text = call.arguments.get("old_text", "")
            new_text = call.arguments.get("new_text", "")
            old_size = len(old_text.encode("utf-8")) if isinstance(old_text, str) else 0
            new_size = len(new_text.encode("utf-8")) if isinstance(new_text, str) else 0
            sha = str(call.arguments.get("expected_sha256", ""))
            occurrences = call.arguments.get("expected_occurrences", 1)
            return (
                f"Workspace replace_text {path}; old_bytes={old_size}; new_bytes={new_size}; "
                f"expected_sha256={sha[:12]}; expected_occurrences={occurrences}"
            )
        if definition.name == "run_command":
            profile_id = call.arguments.get("profile")
            if self._command_config is not None and isinstance(profile_id, str) and profile_id in self._command_config.profiles:
                return command_profile_summary(self._command_config.profiles[profile_id], max_length=160)
            return f"Run command profile={_safe_cli_field(str(profile_id), max_length=80)}"
        return super().summarize(call, definition)


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


async def _skill(args: argparse.Namespace) -> int:
    if args.skill_command == "list":
        try:
            for skill in discover_skills(args.skills_dir):
                print(f"{skill.name}\t{_safe_cli_field(skill.description, max_length=240)}")
        except SkillError as exc:
            print(_safe_cli_field(str(exc), max_length=240), file=sys.stderr)
            return 2
        return 0
    if args.skill_command == "show":
        try:
            skill = _find_skill(args.skills_dir, args.name)
        except SkillError as exc:
            print(_safe_cli_field(str(exc), max_length=240), file=sys.stderr)
            return 2
        print(f"name\t{skill.name}")
        print(f"description\t{_safe_cli_field(skill.description, max_length=240)}")
        print(f"schema_version\t{skill.schema_version}")
        print(f"source_dir\t{_path_relative_display(skill.source_dir, args.skills_dir)}")
        print(f"allowed_tools\t{','.join(skill.allowed_tools)}")
        print(f"max_model_turns\t{skill.max_model_turns or ''}")
        print(f"max_tool_calls_total\t{skill.max_tool_calls_total or ''}")
        return 0
    if args.skill_command == "run":
        return await _skill_run(args)
    return 2


async def _skill_run(args: argparse.Namespace) -> int:
    try:
        skill = _find_skill(args.skills_dir, args.name)
        limits = build_skill_loop_limits(skill)
    except SkillError as exc:
        print(_safe_cli_field(str(exc), max_length=240), file=sys.stderr)
        return 2
    if skill_requires_mcp(skill) and not args.mcp_config:
        print("Skill error: MCP tool references require --mcp-config", file=sys.stderr)
        return 2
    if skill_requires_workspace(skill) and not args.workspace:
        print("Skill error: workspace tool references require --workspace", file=sys.stderr)
        return 2
    if args.workspace_write and not args.workspace:
        print("Workspace error: --workspace-write requires --workspace", file=sys.stderr)
        return 2
    if skill_requires_workspace_write(skill) and not args.workspace_write:
        print("Skill error: workspace write tool references require --workspace-write", file=sys.stderr)
        return 2
    if args.workspace_exec and not args.workspace:
        print("Command error: --workspace-exec requires --workspace", file=sys.stderr)
        return 2
    if args.workspace_exec and not args.command_config:
        print("Command error: --workspace-exec requires --command-config", file=sys.stderr)
        return 2
    if skill_requires_workspace_exec(skill) and not args.workspace_exec:
        print("Skill error: command tool references require --workspace-exec", file=sys.stderr)
        return 2
    message = " ".join(args.message).strip()
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

    try:
        workspace = _build_workspace(args.workspace)
    except WorkspaceError as exc:
        print(f"Workspace error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 2
    if args.state_db:
        try:
            preflight = SQLiteRunStore(args.state_db); preflight.close()
        except StateStoreError:
            print("State error: Run store could not be initialized", file=sys.stderr); return 2
    try:
        command_config = _build_command_config(args.command_config, workspace) if args.workspace_exec else None
    except CommandConfigError as exc:
        print(f"Command error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 2

    registry = build_builtin_tool_registry(
        workspace=workspace,
        enable_workspace_write=args.workspace_write,
        command_config=command_config,
    )
    try:
        validate_builtin_skill_tool_references(skill, registry)
    except SkillError as exc:
        print(_safe_cli_field(str(exc), max_length=240), file=sys.stderr)
        return 2
    if args.mcp_config is None:
        try:
            tools = resolve_skill_tool_references(skill, registry)
        except SkillError as exc:
            print(_safe_cli_field(str(exc), max_length=240), file=sys.stderr)
            return 2
        return await _run_chat(
            message,
            registry,
            model_config=model_config,
            system_prompt=skill.instructions,
            tools=tools,
            limits=limits,
            workspace=workspace,
            state_db=args.state_db, record_content=args.record_content, run_kind=RunKind.SKILL, skill_name=skill.name,
        )
    try:
        required_server_ids = skill_mcp_server_ids(skill)
        configs = load_mcp_server_configs(args.mcp_config, required_server_ids)
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        return 2
    except SkillError as exc:
        print(_safe_cli_field(str(exc), max_length=240), file=sys.stderr)
        return 2
    except Exception:
        print("MCP error: MCP skill tool setup failed", file=sys.stderr)
        return 1
    return await _run_skill_with_mcp(message, skill, registry, configs=configs, model_config=model_config, limits=limits, state_db=args.state_db, record_content=args.record_content)


async def _run_skill_with_mcp(
    message: str,
    skill,
    registry: ToolRegistry,
    *,
    configs,
    model_config: ModelConfig,
    limits: AgentLoopLimits,
    state_db: str | None = None, record_content: bool = False,
) -> int:
    store = None; run_id = None; text_sink: list[str] = []; return_code = 1
    try:
        if state_db:
            store = SQLiteRunStore(state_db)
            run_id = await store.start_run(RunStartContext(RunKind.SKILL, skill_name=skill.name, model_name=getattr(model_config, "model", None), input_text=message, record_content=record_content))
    except StateStoreError:
        if store:
            try: store.close()
            except Exception: pass
        print("State error: Run store could not be initialized", file=sys.stderr); return 2
    group = MCPClientGroup(configs)
    try:
        await group.__aenter__()
    except asyncio.CancelledError:
        await _best_effort_cancel(store, run_id)
        raise
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_start_failed", error_message="MCP server failed to start")
        _best_effort_close(store)
        return 1
    except Exception:
        print("MCP error: MCP skill tool setup failed", file=sys.stderr)
        await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_start_failed", error_message="MCP server setup failed")
        _best_effort_close(store)
        return 1
    try:
        try:
            await group.register_tools(registry)
            tools = resolve_skill_tool_references(skill, registry)
        except SkillError as exc:
            print(_safe_cli_field(str(exc), max_length=240), file=sys.stderr)
            await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_register_failed", error_message="Skill tool setup failed")
            return_code = 2
        except MCPError as exc:
            print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
            await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_register_failed", error_message="MCP tool registration failed")
            return_code = 1
        except Exception:
            print("MCP error: MCP chat tool setup failed", file=sys.stderr)
            await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_register_failed", error_message="MCP tool registration failed")
            return_code = 1
        else:
            return_code = await _run_chat(
                message,
                registry,
                model_config=model_config,
                system_prompt=skill.instructions,
                tools=tools,
                limits=limits,
                workspace=_registry_workspace(registry),
                state_db=state_db, record_content=record_content, run_kind=RunKind.SKILL, skill_name=skill.name,
                external_store=store, external_run_id=run_id, final_text_sink=text_sink,
            )
    except BaseException as exc:
        try:
            await group.__aexit__(type(exc), exc, exc.__traceback__)
        except BaseException:
            pass
        if isinstance(exc, asyncio.CancelledError) and store and run_id:
            await _best_effort_cancel(store, run_id)
        elif store and run_id:
            await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="runtime_error", error_message="Skill execution failed")
        _best_effort_close(store)
        raise
    try:
        await group.__aexit__(None, None, None)
    except asyncio.CancelledError:
        await _best_effort_cancel(store, run_id)
        raise
    except MCPError as exc:
        print(f"MCP error: {_safe_cli_field(str(exc), max_length=240)}", file=sys.stderr)
        await _best_effort_finish(store, run_id, RunStatus.FAILED, error_code="mcp_cleanup_failed", error_message="MCP server cleanup failed")
        _best_effort_close(store)
        return 1
    if store and run_id:
        try: await store.finish_run(run_id, RunStatus.FAILED if return_code else RunStatus.COMPLETED, final_text=''.join(text_sink) if record_content else None)
        except StateStoreError:
            print("State error: Run final status could not be persisted", file=sys.stderr); return_code = 1
        _best_effort_close(store)
    return return_code


def _find_skill(skills_dir: str, name: str):
    for skill in discover_skills(skills_dir):
        if skill.name == name:
            return skill
    raise SkillError(f"Skill error: skill not found: {name}")


def _build_workspace(path: str | None) -> Workspace | None:
    if path is None:
        return None
    return Workspace(Path(path))


def _build_command_config(path: str | None, workspace: Workspace | None) -> CommandConfig:
    if path is None:
        raise CommandConfigError("--workspace-exec requires --command-config")
    if workspace is None:
        raise CommandConfigError("--workspace-exec requires --workspace")
    return load_command_config(path, workspace)


def _registry_workspace(registry: ToolRegistry) -> Workspace | None:
    for name in ("write_file", "replace_text", "read_file", "list_files", "search_text"):
        if registry.contains(name):
            workspace = getattr(registry.get(name), "_workspace", None)
            if isinstance(workspace, Workspace):
                return workspace
    return None


def _registry_command_config(registry: ToolRegistry) -> CommandConfig | None:
    if registry.contains("run_command"):
        config = getattr(registry.get("run_command"), "_config", None)
        if isinstance(config, CommandConfig):
            return config
    return None


def _approval_display_path(workspace: Workspace | None, value, *, create: bool) -> str:
    if type(value) is not str:
        return "<invalid>"
    if workspace is not None:
        try:
            if create:
                relative = resolve_workspace_create_target(workspace, value).relative_path
            else:
                path = resolve_workspace_path(workspace, value, expected_type="file")
                relative = workspace_relative_path(workspace, path)
            return _abbreviate_path_tail(relative, max_length=80)
        except WorkspaceError:
            pass
    return _abbreviate_path_tail(_normalize_display_path_text(value), max_length=80)


def _normalize_display_path_text(value: str) -> str:
    parts = []
    for part in value.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        parts.append(part)
    return "/".join(parts) or "."


def _abbreviate_path_tail(value: str, *, max_length: int) -> str:
    sanitized = _safe_cli_field(value, max_length=max(len(value), max_length))
    if len(sanitized) <= max_length:
        return sanitized
    return "..." + sanitized[-(max_length - 3) :]


def _path_relative_display(path, root) -> str:
    try:
        return str(path.relative_to(Path(root).resolve()))
    except Exception:
        return path.name


def _split_skill_run_separator_message(argv: list[str] | None) -> tuple[list[str] | None, list[str]]:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "skill" and argv[1] == "run" and "--" in argv:
        separator_index = argv.index("--")
        return argv[:separator_index], argv[separator_index + 1 :]
    return argv, []


def _skill_run_extra_args_are_message(extra_args: list[str]) -> bool:
    return all(not item.startswith("-") for item in extra_args)


def _strip_argument_separator(args: list[str]) -> list[str]:
    return [arg for arg in args if arg != "--"]


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

