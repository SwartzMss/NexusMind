from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nexusmind.tools import ToolCall, ToolErrorCode, ToolExecutor, ToolRegistry, ToolResultBudget, ToolRiskLevel
from nexusmind.tools.builtin import workspace as workspace_tools
from nexusmind.tools.builtin import ListFilesTool, ReadFileTool, SearchTextTool
from nexusmind.workspace import Workspace


def _registry(workspace: Workspace) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many([ListFilesTool(workspace), ReadFileTool(workspace), SearchTextTool(workspace)])
    return registry


def test_workspace_tools_register_as_read_only(tmp_path: Path) -> None:
    registry = _registry(Workspace(tmp_path))

    assert registry.definition("list_files").risk_level is ToolRiskLevel.READ_ONLY
    assert registry.definition("read_file").risk_level is ToolRiskLevel.READ_ONLY
    assert registry.definition("search_text").risk_level is ToolRiskLevel.READ_ONLY


def test_list_files_is_stable_relative_and_does_not_recurse_symlinks(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "nested.txt").write_text("nested", encoding="utf-8")
    try:
        (tmp_path / "link").symlink_to(tmp_path / "a", target_is_directory=True)
    except OSError:
        pass
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="list_files", arguments={"path": ".", "recursive": True})))

    assert result.error is None
    assert [entry["path"] for entry in result.output["entries"]] == sorted(entry["path"] for entry in result.output["entries"])
    assert all(not Path(entry["path"]).is_absolute() for entry in result.output["entries"])
    if any(entry["path"] == "link" for entry in result.output["entries"]):
        assert not any(entry["path"] == "link/nested.txt" for entry in result.output["entries"])


def test_read_file_supports_utf8_lines_and_rejects_binary(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_bytes(b"one\ntwo\nthree\n")
    (tmp_path / "binary.bin").write_bytes(b"a\x00b")
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(ToolCall(id="1", name="read_file", arguments={"path": "note.txt", "start_line": 2, "max_lines": 1}))
    )
    assert result.error is None
    assert result.output["path"] == "note.txt"
    assert result.output["content"] == "two\n"
    assert result.output["truncated"] is True

    failed = asyncio.run(executor.execute(ToolCall(id="2", name="read_file", arguments={"path": "binary.bin"})))
    assert failed.error is not None
    assert failed.error.code is ToolErrorCode.EXECUTION_FAILED


def test_read_file_compacts_content_to_runtime_result_budget(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("x" * 2000, encoding="utf-8")
    executor = ToolExecutor(_registry(Workspace(tmp_path)))
    call = ToolCall(id="1", name="read_file", arguments={"path": "note.txt"})
    requirements = executor.result_requirements(call)

    result = asyncio.run(
        executor.execute_with_result_budget(
            call,
            result_budget=ToolResultBudget(
                max_bytes=requirements.min_bytes + 100,
                max_nodes=requirements.min_nodes,
                max_depth=requirements.min_depth,
            ),
        )
    )

    assert result.error is None
    assert result.output["truncated"] is True
    assert len(result.output["content"]) < 2000


def test_list_files_compacts_entries_for_node_and_depth_budgets(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    executor = ToolExecutor(_registry(Workspace(tmp_path)))
    call = ToolCall(id="1", name="list_files", arguments={"path": "."})
    requirements = executor.result_requirements(call)

    for max_depth in (2, 3):
        result = asyncio.run(
            executor.execute_with_result_budget(
                call,
                result_budget=ToolResultBudget(
                    max_bytes=4096,
                    max_nodes=requirements.min_nodes,
                    max_depth=max_depth,
                ),
            )
        )
        assert result.error is None
        assert result.output["entries"] == []
        assert result.output["truncated"] is True


def test_search_text_skips_binary_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("class Runner:\npass\n", encoding="utf-8")
    (tmp_path / "bad.bin").write_bytes(b"Runner\x00")
    try:
        (tmp_path / "linked.py").symlink_to(tmp_path / "src" / "main.py")
    except OSError:
        pass
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(ToolCall(id="1", name="search_text", arguments={"query": "runner", "case_sensitive": False}))
    )

    assert result.error is None
    assert result.output["matches"] == [{"path": "src/main.py", "line": 1, "text": "class Runner:"}]
    assert result.output["files_scanned"] == 1


def test_search_text_compacts_matches_for_node_and_depth_budgets(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")
    executor = ToolExecutor(_registry(Workspace(tmp_path)))
    call = ToolCall(id="1", name="search_text", arguments={"query": "needle"})
    requirements = executor.result_requirements(call)

    for max_depth in (2, 3):
        result = asyncio.run(
            executor.execute_with_result_budget(
                call,
                result_budget=ToolResultBudget(
                    max_bytes=4096,
                    max_nodes=requirements.min_nodes,
                    max_depth=max_depth,
                ),
            )
        )
        assert result.error is None
        assert result.output["matches"] == []
        assert result.output["truncated"] is True


def test_search_text_consumes_budget_for_skipped_files(tmp_path: Path, monkeypatch) -> None:
    for index in range(3):
        (tmp_path / f"binary-{index}.bin").write_bytes(b"Runner\x00")
    monkeypatch.setattr(workspace_tools, "MAX_SEARCH_FILES_CONSIDERED", 2)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="search_text", arguments={"query": "Runner"})))

    assert result.error is None
    assert result.output["files_scanned"] == 0
    assert result.output["truncated"] is True


def test_search_text_does_not_exceed_total_byte_budget(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.txt").write_bytes(b"aaaa")
    (tmp_path / "b.txt").write_bytes(b"bbbb")
    monkeypatch.setattr(workspace_tools, "MAX_SEARCH_TOTAL_BYTES", 5)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="search_text", arguments={"query": "z"})))

    assert result.error is None
    assert result.output["bytes_scanned"] <= 5
    assert result.output["truncated"] is True


def test_search_text_binary_files_consume_total_byte_budget(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.bin").write_bytes(b"aaaa\x00")
    (tmp_path / "b.bin").write_bytes(b"bbbb\x00")
    monkeypatch.setattr(workspace_tools, "MAX_SEARCH_TOTAL_BYTES", 6)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="search_text", arguments={"query": "missing"})))

    assert result.error is None
    assert result.output["bytes_scanned"] == 5
    assert result.output["truncated"] is True


def test_search_text_rolls_back_matches_over_json_output_limit(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.txt").write_text("needle " + ("x" * 200), encoding="utf-8")
    monkeypatch.setattr(workspace_tools, "MAX_SEARCH_OUTPUT_BYTES", 120)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="search_text", arguments={"query": "needle"})))

    assert result.error is None
    assert result.output["truncated"] is True
    encoded = json.dumps(result.output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 120


def test_workspace_tools_reject_overlong_path_arguments(tmp_path: Path) -> None:
    executor = ToolExecutor(_registry(Workspace(tmp_path)))
    long_path = "./" * (workspace_tools.MAX_WORKSPACE_PATH_CHARS + 1)

    listed = asyncio.run(executor.execute(ToolCall(id="1", name="list_files", arguments={"path": long_path})))
    searched = asyncio.run(executor.execute(ToolCall(id="2", name="search_text", arguments={"query": "x", "path": long_path})))

    assert listed.error is not None
    assert listed.error.code is ToolErrorCode.INVALID_ARGUMENTS
    assert searched.error is not None
    assert searched.error.code is ToolErrorCode.INVALID_ARGUMENTS


def test_workspace_tools_return_normalized_result_path(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("needle\n", encoding="utf-8")
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    listed = asyncio.run(executor.execute(ToolCall(id="1", name="list_files", arguments={"path": "./src/."})))
    searched = asyncio.run(executor.execute(ToolCall(id="2", name="search_text", arguments={"query": "needle", "path": "./src/."})))

    assert listed.error is None
    assert listed.output["path"] == "src"
    assert searched.error is None
    assert searched.output["path"] == "src"


def test_search_text_file_read_failure_does_not_skip_siblings(tmp_path: Path, monkeypatch) -> None:
    bad = tmp_path / "a_bad.txt"
    good = tmp_path / "b_good.txt"
    bad.write_text("needle in bad\n", encoding="utf-8")
    good.write_text("needle in good\n", encoding="utf-8")
    original_read_limited = workspace_tools._read_limited

    def flaky_read(path: Path, max_bytes: int) -> bytes:
        if path == bad:
            raise OSError("cannot read")
        return original_read_limited(path, max_bytes)

    monkeypatch.setattr(workspace_tools, "_read_limited", flaky_read)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="search_text", arguments={"query": "needle"})))

    assert result.error is None
    assert result.output["matches"] == [{"path": "b_good.txt", "line": 1, "text": "needle in good"}]
    assert result.output["truncated"] is True


def test_search_text_file_read_failure_does_not_skip_pending_directories(tmp_path: Path, monkeypatch) -> None:
    bad_dir = tmp_path / "a_bad"
    good_dir = tmp_path / "b_good"
    bad_dir.mkdir()
    good_dir.mkdir()
    bad = bad_dir / "bad.txt"
    good = good_dir / "good.txt"
    bad.write_text("needle in bad\n", encoding="utf-8")
    good.write_text("needle in good\n", encoding="utf-8")
    original_read_limited = workspace_tools._read_limited

    def flaky_read(path: Path, max_bytes: int) -> bytes:
        if path == bad:
            raise OSError("cannot read")
        return original_read_limited(path, max_bytes)

    monkeypatch.setattr(workspace_tools, "_read_limited", flaky_read)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(executor.execute(ToolCall(id="1", name="search_text", arguments={"query": "needle"})))

    assert result.error is None
    assert result.output["matches"] == [{"path": "b_good/good.txt", "line": 1, "text": "needle in good"}]
    assert result.output["truncated"] is True
