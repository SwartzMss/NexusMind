"""KnowledgeBase-only command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import logging
import sys
from typing import Any

from nexusmind.answer_provider import OpenAICompatibleAnswerProvider
from nexusmind.config import ConfigError, load_model_config_from_env
from nexusmind.knowledge_answer import KnowledgeAnswerError
from nexusmind.knowledge_base import KnowledgeBase
from nexusmind.knowledge_base_manifest import KnowledgeBaseError, LocalDirectorySourceConfig, LocalFileSourceConfig
from nexusmind.knowledge_query import knowledge_query_result_dict
from nexusmind.runtime_support import runtime_operation


RUNTIME_LOGGER = logging.getLogger("nexusmind.runtime")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexusmind", description="Local KnowledgeBase management, retrieval, and cited answers.")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create a knowledge base")
    create.add_argument("path"); create.add_argument("--id", required=True, dest="knowledge_base_id"); create.add_argument("--name", dest="display_name")

    source = commands.add_parser("source", help="manage registered sources")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    source_add = source_commands.add_parser("add", help="register a local source")
    source_add.add_argument("--knowledge-base", default="."); source_add.add_argument("--id", required=True, dest="source_id"); source_add.add_argument("--path", required=True, dest="source_path"); source_add.add_argument("--type", choices=("file", "directory"), required=True)
    source_list = source_commands.add_parser("list", help="list registered sources")
    source_list.add_argument("--knowledge-base", default="."); source_list.add_argument("--json", action="store_true")
    source_remove = source_commands.add_parser("remove", help="remove a source and its documents")
    source_remove.add_argument("--knowledge-base", default="."); source_remove.add_argument("--id", required=True, dest="source_id")

    sync = commands.add_parser("sync", help="synchronize registered sources")
    sync.add_argument("--knowledge-base", default="."); sync.add_argument("--source", dest="source_id"); sync.add_argument("--json", action="store_true")
    search = commands.add_parser("search", help="search canonical knowledge")
    search.add_argument("query"); search.add_argument("--knowledge-base", default="."); search.add_argument("--limit", type=int, default=10); search.add_argument("--json", action="store_true")
    query = commands.add_parser("query", help="generate a cited answer")
    query.add_argument("question"); query.add_argument("--knowledge-base", default="."); query.add_argument("--debug", action="store_true"); query.add_argument("--json", action="store_true")
    inspect = commands.add_parser("inspect", help="inspect canonical knowledge")
    inspect.add_argument("--knowledge-base", default="."); inspect.add_argument("--document"); inspect.add_argument("--preview-chars", type=int, default=160); inspect.add_argument("--json", action="store_true")
    diagnose = commands.add_parser("diagnose", help="diagnose retrieval stages")
    diagnose.add_argument("query"); diagnose.add_argument("--knowledge-base", default="."); diagnose.add_argument("--limit", type=int, default=10); diagnose.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return {"create": _create, "source": _source, "sync": _sync, "search": _search, "query": _query, "inspect": _inspect, "diagnose": _diagnose}[args.command](args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr); return 2
    except (KnowledgeBaseError, KnowledgeAnswerError, TypeError, ValueError):
        print("KnowledgeBase operation failed.", file=sys.stderr); return 1


def _create(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.create(args.path, knowledge_base_id=args.knowledge_base_id, display_name=args.display_name)
    kb.close(); print(f"Created KnowledgeBase: {args.path}"); return 0


def _source(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.open(args.knowledge_base)
    try:
        if args.source_command == "add":
            kind = LocalFileSourceConfig if args.type == "file" else LocalDirectorySourceConfig
            kb.add_source(kind(source_id=args.source_id, path=args.source_path)); print(f"Registered source: {args.source_id}")
        elif args.source_command == "remove":
            kb.remove_source(args.source_id); print(f"Removed source: {args.source_id}")
        else:
            sources = kb.list_sources()
            if args.json: _print_json(sources)
            else:
                for item in sources: print(f"{item.source_id}\t{item.type}\t{item.path}")
    finally: kb.close()
    return 0


def _sync(args: argparse.Namespace) -> int:
    fields = {"source_id": args.source_id} if args.source_id else {}
    with runtime_operation(RUNTIME_LOGGER, "sync", **fields) as operation:
        kb = KnowledgeBase.open(args.knowledge_base)
        try:
            result: Any = kb.sync_source(args.source_id) if args.source_id else kb.sync()
            sync_results = result if isinstance(result, tuple) else (result,)
            operation["document_count"] = sum(
                item.documents_added + item.documents_updated + item.documents_unchanged
                for item in sync_results
            )
            if args.json: _print_json(result)
            elif args.source_id: print(f"Synchronized source: {args.source_id}")
            else: print(f"Synchronized {len(result)} source(s)")
        finally: kb.close()
    return 0


def _search(args: argparse.Namespace) -> int:
    with runtime_operation(RUNTIME_LOGGER, "search") as operation:
        kb = KnowledgeBase.open(args.knowledge_base)
        try:
            results = kb.search(args.query, limit=args.limit)
            operation["result_count"] = len(results)
            if args.json: _print_json(results)
            else:
                for index, result in enumerate(results, start=1):
                    print(f"{index}. {result.document.logical_path} ({result.hit.score:.6f})"); print(result.hit.chunk.content)
        finally: kb.close()
    return 0


def _query(args: argparse.Namespace) -> int:
    with runtime_operation(RUNTIME_LOGGER, "query") as operation:
        config = load_model_config_from_env(); provider = OpenAICompatibleAnswerProvider(config)
        kb = KnowledgeBase.open(args.knowledge_base, answer_generator=provider)
        try: result = kb.query(args.question)
        finally: kb.close()
        operation["citation_count"] = len(result.citations)
        if args.json:
            print(json.dumps(knowledge_query_result_dict(result, include_debug=args.debug), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)); return 0
        print("Answer:"); print(result.answer.text); print("\nSources:")
        for citation in result.citations: print(f"[{citation.citation_id}] {citation.logical_path}")
        if args.debug:
            print(f"\nRetrieval backend: {result.trace.retrieval_backend}"); print(f"Context: {result.trace.context_character_count} chars"); print(f"Trace: {result.trace_id}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.open(args.knowledge_base)
    try:
        result = kb.inspect_document(args.document, preview_chars=args.preview_chars) if args.document else kb.inspect()
        if args.json: _print_json(result)
        elif args.document:
            print(f"Document: {result.document.logical_path}")
            for chunk in result.chunks: print(f"{chunk.ordinal}\t{chunk.start_offset}:{chunk.end_offset}\t{chunk.preview}")
        else:
            print(f"KnowledgeBase: {result.status.knowledge_base_id}"); print(f"Sources: {len(result.sources)}"); print(f"Documents: {len(result.documents)}")
    finally: kb.close()
    return 0


def _diagnose(args: argparse.Namespace) -> int:
    kb = KnowledgeBase.open(args.knowledge_base)
    try:
        result = kb.diagnose_search(args.query, limit=args.limit)
        if args.json: _print_json(result)
        else:
            for item in result.candidates:
                row = item.diagnostic; print(f"{row.stage.value}\t{row.rank}\t{row.score:.6f}\t{item.document.logical_path}")
    finally: kb.close()
    return 0


def _jsonable(value: Any) -> Any:
    if is_dataclass(value): return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum): return value.value
    if isinstance(value, (tuple, list)): return [_jsonable(item) for item in value]
    if isinstance(value, dict): return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _print_json(value: Any) -> None:
    print(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__": raise SystemExit(main())
