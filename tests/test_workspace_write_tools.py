from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

from nexusmind.tools import ToolCall, ToolErrorCode, ToolExecutor, ToolRegistry, ToolRiskLevel
from nexusmind.tools.builtin import ReadFileTool, ReplaceTextTool, WriteFileTool
from nexusmind.tools.builtin.workspace import WorkspaceWriteBudget, WorkspaceWriteLimits
from nexusmind.workspace import Workspace


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _registry(workspace: Workspace, budget: WorkspaceWriteBudget | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    budget = budget or WorkspaceWriteBudget()
    registry.register_many([ReadFileTool(workspace), WriteFileTool(workspace, budget), ReplaceTextTool(workspace, budget)])
    return registry


def test_write_tools_are_local_write(tmp_path: Path) -> None:
    registry = _registry(Workspace(tmp_path))

    assert registry.definition("write_file").risk_level is ToolRiskLevel.LOCAL_WRITE
    assert registry.definition("replace_text").risk_level is ToolRiskLevel.LOCAL_WRITE


def test_read_file_returns_full_file_sha_and_size(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_bytes(b"one\ntwo\nthree\n")
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(ToolCall(id="1", name="read_file", arguments={"path": "note.txt", "start_line": 2, "max_lines": 1}))
    )

    assert result.error is None
    assert result.output["sha256"] == _sha(b"one\ntwo\nthree\n")
    assert result.output["size"] == len(b"one\ntwo\nthree\n")
    assert result.output["content"] == "two\n"


def test_write_file_create_and_replace(tmp_path: Path) -> None:
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    created = asyncio.run(
        executor.execute(ToolCall(id="1", name="write_file", arguments={"path": "new.txt", "mode": "create", "content": "a\n"}))
    )
    assert created.error is None
    assert created.output["previous_sha256"] is None
    assert created.output["sha256"] == _sha(b"a\n")
    assert (tmp_path / "new.txt").read_bytes() == b"a\n"

    replaced = asyncio.run(
        executor.execute(
            ToolCall(
                id="2",
                name="write_file",
                arguments={
                    "path": "new.txt",
                    "mode": "replace",
                    "content": "b\n",
                    "expected_sha256": created.output["sha256"],
                },
            )
        )
    )
    assert replaced.error is None
    assert replaced.output["previous_sha256"] == created.output["sha256"]
    assert replaced.output["sha256"] == _sha(b"b\n")
    assert (tmp_path / "new.txt").read_bytes() == b"b\n"
    assert "content" not in replaced.output


def test_write_file_conflicts_do_not_modify(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("original\n", encoding="utf-8")
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    missing_hash = asyncio.run(
        executor.execute(ToolCall(id="1", name="write_file", arguments={"path": "note.txt", "mode": "replace", "content": "new\n"}))
    )
    bad_hash = asyncio.run(
        executor.execute(
            ToolCall(
                id="2",
                name="write_file",
                arguments={"path": "note.txt", "mode": "replace", "content": "new\n", "expected_sha256": "0" * 64},
            )
        )
    )
    create_existing = asyncio.run(
        executor.execute(ToolCall(id="3", name="write_file", arguments={"path": "note.txt", "mode": "create", "content": "new\n"}))
    )

    assert missing_hash.error.code is ToolErrorCode.EXECUTION_FAILED
    assert bad_hash.error.code is ToolErrorCode.EXECUTION_FAILED
    assert create_existing.error.code is ToolErrorCode.EXECUTION_FAILED
    assert path.read_text(encoding="utf-8") == "original\n"


def test_replace_text_exact_count_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("alpha beta alpha\n", encoding="utf-8")
    previous = _sha(path.read_bytes())
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    mismatch = asyncio.run(
        executor.execute(
            ToolCall(
                id="1",
                name="replace_text",
                arguments={
                    "path": "note.txt",
                    "expected_sha256": previous,
                    "old_text": "alpha",
                    "new_text": "omega",
                    "expected_occurrences": 1,
                },
            )
        )
    )
    assert mismatch.error.code is ToolErrorCode.EXECUTION_FAILED
    assert path.read_text(encoding="utf-8") == "alpha beta alpha\n"

    replaced = asyncio.run(
        executor.execute(
            ToolCall(
                id="2",
                name="replace_text",
                arguments={
                    "path": "note.txt",
                    "expected_sha256": previous,
                    "old_text": "alpha",
                    "new_text": "omega",
                    "expected_occurrences": 2,
                },
            )
        )
    )
    assert replaced.error is None
    assert replaced.output["replacements"] == 2
    assert path.read_text(encoding="utf-8") == "omega beta omega\n"


def test_write_budget_is_shared_and_failures_do_not_consume_mutation_budget(tmp_path: Path) -> None:
    budget = WorkspaceWriteBudget(WorkspaceWriteLimits(max_successful_mutations=1, max_total_bytes_written=10))
    executor = ToolExecutor(_registry(Workspace(tmp_path), budget))

    failed = asyncio.run(
        executor.execute(ToolCall(id="1", name="write_file", arguments={"path": "one.txt", "mode": "create", "content": "\x00"}))
    )
    first = asyncio.run(
        executor.execute(ToolCall(id="2", name="write_file", arguments={"path": "one.txt", "mode": "create", "content": "one"}))
    )
    second = asyncio.run(
        executor.execute(ToolCall(id="3", name="write_file", arguments={"path": "two.txt", "mode": "create", "content": "two"}))
    )

    assert failed.error is not None
    assert first.error is None
    assert second.error is not None
    assert not (tmp_path / "two.txt").exists()


def test_replace_preserves_permissions_and_cleans_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "script.sh"
    path.write_text("echo old\n", encoding="utf-8")
    os.chmod(path, 0o700)
    previous_mode = path.stat().st_mode & 0o777
    previous_sha = _sha(path.read_bytes())
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": "script.sh", "mode": "replace", "content": "echo new\n", "expected_sha256": previous_sha},
            )
        )
    )

    assert result.error is None
    assert (path.stat().st_mode & 0o777) == previous_mode
    assert not list(tmp_path.glob(".nexusmind-*.tmp"))


def test_create_failure_after_temp_creation_leaves_no_target_or_temp(tmp_path: Path, monkeypatch) -> None:
    budget = WorkspaceWriteBudget()

    def failing_link(src, dst):
        raise OSError("publish failed")

    import nexusmind.tools.builtin.workspace as workspace_module

    monkeypatch.setattr(workspace_module.os, "link", failing_link)
    executor = ToolExecutor(_registry(Workspace(tmp_path), budget))

    result = asyncio.run(
        executor.execute(ToolCall(id="1", name="write_file", arguments={"path": "new.txt", "mode": "create", "content": "data"}))
    )

    assert result.error is not None
    assert not (tmp_path / "new.txt").exists()
    assert not list(tmp_path.glob(".nexusmind-*.tmp"))
    assert budget.successful_mutations == 0


def test_write_file_replace_rechecks_sha_at_commit(tmp_path: Path, monkeypatch) -> None:
    import nexusmind.tools.builtin.workspace as workspace_module

    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    previous_sha = _sha(path.read_bytes())
    original_mkstemp = workspace_module.tempfile.mkstemp
    changed = {"done": False}

    def racing_mkstemp(*args, **kwargs):
        result = original_mkstemp(*args, **kwargs)
        if not changed["done"]:
            path.write_text("user change\n", encoding="utf-8")
            changed["done"] = True
        return result

    monkeypatch.setattr(workspace_module.tempfile, "mkstemp", racing_mkstemp)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": "note.txt", "mode": "replace", "content": "new\n", "expected_sha256": previous_sha},
            )
        )
    )

    assert result.error is not None
    assert "changed since it was read" in result.error.message
    assert path.read_text(encoding="utf-8") == "user change\n"
    assert not list(tmp_path.glob(".nexusmind-*.tmp"))


def test_replace_text_rechecks_sha_at_commit(tmp_path: Path, monkeypatch) -> None:
    import nexusmind.tools.builtin.workspace as workspace_module

    path = tmp_path / "note.txt"
    path.write_text("old value\n", encoding="utf-8")
    previous_sha = _sha(path.read_bytes())
    original_mkstemp = workspace_module.tempfile.mkstemp
    changed = {"done": False}

    def racing_mkstemp(*args, **kwargs):
        result = original_mkstemp(*args, **kwargs)
        if not changed["done"]:
            path.write_text("user value\n", encoding="utf-8")
            changed["done"] = True
        return result

    monkeypatch.setattr(workspace_module.tempfile, "mkstemp", racing_mkstemp)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="1",
                name="replace_text",
                arguments={
                    "path": "note.txt",
                    "expected_sha256": previous_sha,
                    "old_text": "old",
                    "new_text": "new",
                },
            )
        )
    )

    assert result.error is not None
    assert "changed since it was read" in result.error.message
    assert path.read_text(encoding="utf-8") == "user value\n"
    assert not list(tmp_path.glob(".nexusmind-*.tmp"))


def test_create_success_with_temp_cleanup_failure_reports_committed_warning(tmp_path: Path, monkeypatch) -> None:
    import nexusmind.tools.builtin.workspace as workspace_module

    budget = WorkspaceWriteBudget()
    original_unlink = workspace_module.os.unlink

    def failing_unlink(path):
        if str(path).endswith(".tmp"):
            raise OSError("cleanup failed")
        return original_unlink(path)

    monkeypatch.setattr(workspace_module.os, "unlink", failing_unlink)
    executor = ToolExecutor(_registry(Workspace(tmp_path), budget))

    result = asyncio.run(
        executor.execute(ToolCall(id="1", name="write_file", arguments={"path": "new.txt", "mode": "create", "content": "data"}))
    )

    assert result.error is None
    assert result.output["committed"] is True
    assert result.output["cleanup_warning"] is True
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "data"
    assert list(tmp_path.glob(".nexusmind-*.tmp"))
    assert budget.successful_mutations == 1
    assert budget.total_bytes_written == 4


def test_create_cleanup_failure_still_consumes_write_budget(tmp_path: Path, monkeypatch) -> None:
    import nexusmind.tools.builtin.workspace as workspace_module

    budget = WorkspaceWriteBudget(WorkspaceWriteLimits(max_successful_mutations=1, max_total_bytes_written=4))
    original_unlink = workspace_module.os.unlink

    def failing_unlink(path):
        if str(path).endswith(".tmp"):
            raise OSError("cleanup failed")
        return original_unlink(path)

    monkeypatch.setattr(workspace_module.os, "unlink", failing_unlink)
    executor = ToolExecutor(_registry(Workspace(tmp_path), budget))

    first = asyncio.run(
        executor.execute(ToolCall(id="1", name="write_file", arguments={"path": "one.txt", "mode": "create", "content": "data"}))
    )
    second = asyncio.run(
        executor.execute(ToolCall(id="2", name="write_file", arguments={"path": "two.txt", "mode": "create", "content": "x"}))
    )

    assert first.error is None
    assert first.output["cleanup_warning"] is True
    assert second.error is not None
    assert "budget exceeded" in second.error.message
    assert not (tmp_path / "two.txt").exists()


def test_replace_sha_conflict_with_temp_cleanup_failure_reports_cleanup(tmp_path: Path, monkeypatch) -> None:
    import nexusmind.tools.builtin.workspace as workspace_module

    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    previous_sha = _sha(path.read_bytes())
    original_mkstemp = workspace_module.tempfile.mkstemp
    original_unlink = workspace_module.os.unlink

    def racing_mkstemp(*args, **kwargs):
        result = original_mkstemp(*args, **kwargs)
        path.write_text("changed\n", encoding="utf-8")
        return result

    def failing_unlink(target):
        if str(target).endswith(".tmp"):
            raise OSError("cleanup failed")
        return original_unlink(target)

    monkeypatch.setattr(workspace_module.tempfile, "mkstemp", racing_mkstemp)
    monkeypatch.setattr(workspace_module.os, "unlink", failing_unlink)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": "note.txt", "mode": "replace", "content": "new\n", "expected_sha256": previous_sha},
            )
        )
    )

    assert result.error is not None
    assert "temporary file could not be cleaned up" in result.error.message
    assert path.read_text(encoding="utf-8") == "changed\n"
    assert list(tmp_path.glob(".nexusmind-*.tmp"))


def test_replace_failure_with_temp_cleanup_failure_reports_cleanup(tmp_path: Path, monkeypatch) -> None:
    import nexusmind.tools.builtin.workspace as workspace_module

    path = tmp_path / "note.txt"
    path.write_text("old\n", encoding="utf-8")
    previous_sha = _sha(path.read_bytes())
    original_unlink = workspace_module.os.unlink

    def failing_replace(src, dst):
        raise OSError("replace failed")

    def failing_unlink(target):
        if str(target).endswith(".tmp"):
            raise OSError("cleanup failed")
        return original_unlink(target)

    monkeypatch.setattr(workspace_module.os, "replace", failing_replace)
    monkeypatch.setattr(workspace_module.os, "unlink", failing_unlink)
    executor = ToolExecutor(_registry(Workspace(tmp_path)))

    result = asyncio.run(
        executor.execute(
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": "note.txt", "mode": "replace", "content": "new\n", "expected_sha256": previous_sha},
            )
        )
    )

    assert result.error is not None
    assert "temporary file could not be cleaned up" in result.error.message
    assert path.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".nexusmind-*.tmp"))
