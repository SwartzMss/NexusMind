from __future__ import annotations

import asyncio
import sqlite3

import pytest

from nexusmind import cli
from nexusmind.config import ModelConfig
from nexusmind.models.base import ChatModel
from nexusmind.runtime.chat import ChatRuntime, ToolExecutionCancelled
from nexusmind.runtime.events import RuntimeEvent, RuntimeEventType
from nexusmind.runtime.harness import (
    CheckpointBoundary,
    HarnessCheckpoint,
    HarnessPhase,
    HarnessRequest,
    HarnessRunner,
    HarnessState,
    InMemoryCheckpointStore,
)
from nexusmind.runtime.harness.checkpointing import CheckpointCoordinator
from nexusmind.runtime.harness.resume import HarnessResumeRequest
from nexusmind.runtime.harness.state import HarnessStatus
from nexusmind.runtime.harness.stop import StopReason
from nexusmind.runtime.messages import Message, MessageRole
from nexusmind.tools.contracts import ToolCall, ToolDefinition, ToolResult, ToolResultRequirements, ToolRiskLevel


class ScriptedModel(ChatModel):
    def __init__(self, turns: list[list[RuntimeEvent]]) -> None:
        self._turns = list(turns)
        self.calls = 0

    async def stream(self, messages, tools=None):
        index = self.calls
        self.calls += 1
        for event in self._turns[min(index, len(self._turns) - 1)]:
            yield event


class RecordingExecutor:
    def __init__(self, actions: list[str]) -> None:
        self.actions = actions
        self.calls: list[str] = []
        self.definition_value = ToolDefinition(name="echo", risk_level=ToolRiskLevel.READ_ONLY)

    def definition(self, name: str):
        return self.definition_value if name == "echo" else None

    def result_requirements(self, call: ToolCall) -> ToolResultRequirements:
        return ToolResultRequirements(min_bytes=16, min_nodes=3, min_depth=1)

    async def execute(self, call: ToolCall) -> ToolResult:
        return await self.execute_with_result_budget(call, result_budget=None)

    async def execute_with_result_budget(self, call: ToolCall, *, result_budget) -> ToolResult:
        self.actions.append(f"execute:{call.id}")
        self.calls.append(call.id)
        return ToolResult(call_id=call.id, name=call.name, output={"ok": True})


class RecordingStore(InMemoryCheckpointStore):
    def __init__(self, actions: list[str], fail_on_save: int | None = None) -> None:
        super().__init__()
        self.actions = actions
        self.fail_on_save = fail_on_save
        self.attempts = 0

    async def save(self, checkpoint: HarnessCheckpoint) -> None:
        self.attempts += 1
        self.actions.append(f"save:{checkpoint.boundary.value}:{checkpoint.sequence}")
        if self.fail_on_save == self.attempts:
            raise RuntimeError("store unavailable")
        await super().save(checkpoint)


def _tool_model(*calls: ToolCall) -> list[RuntimeEvent]:
    return [
        RuntimeEvent(RuntimeEventType.MODEL_STARTED),
        *[RuntimeEvent(RuntimeEventType.TOOL_CALL_COMPLETED, tool_call=call) for call in calls],
        RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="tool_calls"),
    ]


def _text_model(text: str = "done") -> list[RuntimeEvent]:
    return [
        RuntimeEvent(RuntimeEventType.MODEL_STARTED),
        RuntimeEvent(RuntimeEventType.TEXT_DELTA, text=text),
        RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop"),
    ]


def test_automatic_checkpoint_order_and_boundaries() -> None:
    async def run() -> None:
        actions: list[str] = []
        call = ToolCall(id="call-1", name="echo", arguments={})
        model = ScriptedModel([_tool_model(call), _text_model()])
        executor = RecordingExecutor(actions)
        store = RecordingStore(actions)
        events = [
            event
            async for event in ChatRuntime(
                model,
                tool_executor=executor,
                checkpoint_store=store,
                checkpoint_run_id="run-1",
            ).stream_user_message("hello", tools=[executor.definition_value])
        ]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        assert actions.index("save:after_model:0") < actions.index("execute:call-1")
        assert actions.index("save:after_tool:1") < actions.index("save:after_model:2")
        assert actions[-1] == "save:run_terminal:3"
        checkpoints = await store.list("run-1")
        assert [item.boundary for item in checkpoints] == [
            CheckpointBoundary.AFTER_MODEL,
            CheckpointBoundary.AFTER_TOOL,
            CheckpointBoundary.AFTER_MODEL,
            CheckpointBoundary.RUN_TERMINAL,
        ]

    asyncio.run(run())


def test_chat_cli_opt_in_persists_sqlite_checkpoints(tmp_path, monkeypatch, capsys) -> None:
    class Model(ChatModel):
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="done")
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: Model())
    state_path = tmp_path / "runs.db"
    code = asyncio.run(
        cli._run_chat(
            "hello",
            cli.ToolRegistry(),
            model_config=ModelConfig("https://example.test", "test-key", "fake"),
            state_db=str(state_path),
            checkpoint_db=str(tmp_path / "chat-checkpoints.db"),
        )
    )
    assert code == 0
    assert "done" in capsys.readouterr().out
    store = cli.SQLiteCheckpointStore(tmp_path / "chat-checkpoints.db")

    # The CLI-generated run ID is intentionally opaque; inspect the database
    # through its public store API using the SQLite rows only for verification.
    import sqlite3

    connection = sqlite3.connect(tmp_path / "chat-checkpoints.db")
    run_id = connection.execute("SELECT run_id FROM harness_checkpoints LIMIT 1").fetchone()[0]
    connection.close()
    connection = sqlite3.connect(state_path)
    history_run_id = connection.execute("SELECT id FROM runs LIMIT 1").fetchone()[0]
    connection.close()
    assert run_id == history_run_id

    async def load_for_run() -> tuple:
        await store.initialize()
        items = await store.list(run_id)
        await store.close()
        return items

    checkpoints = asyncio.run(load_for_run())
    assert [item.boundary for item in checkpoints] == [
        CheckpointBoundary.AFTER_MODEL,
        CheckpointBoundary.RUN_TERMINAL,
    ]


def test_skill_cli_opt_in_persists_sqlite_checkpoints(tmp_path, monkeypatch, capsys) -> None:
    class Model(ChatModel):
        async def stream(self, messages, tools=None):
            yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
            yield RuntimeEvent(RuntimeEventType.TEXT_DELTA, text="skill-done")
            yield RuntimeEvent(RuntimeEventType.MODEL_TURN_COMPLETED, finish_reason="stop")

    skill_dir = tmp_path / "review"
    skill_dir.mkdir()
    (skill_dir / "skill.toml").write_text(
        'schema_version = 1\nname = "review"\ndescription = "Review"\ninstructions_file = "instructions.md"\nallowed_tools = []\n',
        encoding="utf-8",
    )
    (skill_dir / "instructions.md").write_text("Review this", encoding="utf-8")
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: ModelConfig("https://example.test", "test-key", "fake"))
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: Model())
    checkpoint_path = tmp_path / "skill-checkpoints.db"

    assert cli.main([
        "skill", "run", "review", "--skills-dir", str(tmp_path),
        "--checkpoint-db", str(checkpoint_path), "hello",
    ]) == 0
    assert "skill-done" in capsys.readouterr().out
    import sqlite3

    connection = sqlite3.connect(checkpoint_path)
    run_id = connection.execute("SELECT run_id FROM harness_checkpoints LIMIT 1").fetchone()[0]
    connection.close()
    store = cli.SQLiteCheckpointStore(checkpoint_path)

    async def load_for_run() -> tuple:
        await store.initialize()
        items = await store.list(run_id)
        await store.close()
        return items

    checkpoints = asyncio.run(load_for_run())
    assert checkpoints[-1].boundary is CheckpointBoundary.RUN_TERMINAL


def test_after_model_save_failure_blocks_first_tool() -> None:
    async def run() -> None:
        actions: list[str] = []
        call = ToolCall(id="call-1", name="echo", arguments={})
        executor = RecordingExecutor(actions)
        store = RecordingStore(actions, fail_on_save=1)
        events = [
            event
            async for event in ChatRuntime(
                ScriptedModel([_tool_model(call)]),
                tool_executor=executor,
                checkpoint_store=store,
                checkpoint_run_id="run-failed",
            ).stream_user_message("hello", tools=[executor.definition_value])
        ]

        assert executor.calls == []
        assert events[-1].type is RuntimeEventType.RUN_FAILED
        assert events[-1].metadata["checkpoint_persistence_failed"] is True
        assert "after_model" in events[-1].error
        assert await store.load_latest("run-failed") is None

    asyncio.run(run())


def test_after_tool_save_failure_blocks_sibling_tool_and_next_model() -> None:
    async def run() -> None:
        actions: list[str] = []
        calls = (
            ToolCall(id="call-1", name="echo", arguments={}),
            ToolCall(id="call-2", name="echo", arguments={}),
        )
        executor = RecordingExecutor(actions)
        store = RecordingStore(actions, fail_on_save=2)
        events = [
            event
            async for event in ChatRuntime(
                ScriptedModel([_tool_model(*calls), _text_model()]),
                tool_executor=executor,
                checkpoint_store=store,
                checkpoint_run_id="run-tool-failed",
            ).stream_user_message("hello", tools=[executor.definition_value])
        ]

        assert executor.calls == ["call-1"]
        assert not any(event.type is RuntimeEventType.RUN_COMPLETED for event in events)
        assert events[-1].type is RuntimeEventType.RUN_FAILED
        assert events[-1].metadata["checkpoint_boundary"] == CheckpointBoundary.AFTER_TOOL.value
        assert [item.sequence for item in await store.list("run-tool-failed")] == [0]

    asyncio.run(run())


def test_tool_failure_after_start_preserves_tool_failed_without_fake_persistence_failure() -> None:
    async def run() -> None:
        class FailingExecutor(RecordingExecutor):
            async def execute_with_result_budget(self, call: ToolCall, *, result_budget) -> ToolResult:
                self.actions.append(f"execute:{call.id}")
                self.calls.append(call.id)
                raise RuntimeError("result unavailable")

        actions: list[str] = []
        call = ToolCall(id="call-failed", name="echo", arguments={})
        executor = FailingExecutor(actions)
        store = RecordingStore(actions)
        runtime = ChatRuntime(
            ScriptedModel([_tool_model(call)]),
            tool_executor=executor,
            checkpoint_store=store,
            checkpoint_run_id="run-tool-failure",
        )
        events = [event async for event in runtime.stream_user_message("hello", tools=[executor.definition_value])]

        assert executor.calls == ["call-failed"]
        assert events[-1].type is RuntimeEventType.RUN_FAILED
        assert events[-1].metadata["tool_execution_started"] is True
        assert "checkpoint_persistence_failed" not in events[-1].metadata
        assert runtime._harness.stop_reason is StopReason.TOOL_FAILED
        checkpoints = await store.list("run-tool-failure")
        assert [item.boundary for item in checkpoints] == [CheckpointBoundary.AFTER_MODEL]

    asyncio.run(run())


def test_tool_cancellation_does_not_create_unsafe_terminal_checkpoint() -> None:
    async def run() -> None:
        class CancellingExecutor(RecordingExecutor):
            async def execute_with_result_budget(self, call: ToolCall, *, result_budget) -> ToolResult:
                self.actions.append(f"execute:{call.id}")
                self.calls.append(call.id)
                raise asyncio.CancelledError()

        actions: list[str] = []
        call = ToolCall(id="call-cancelled", name="echo", arguments={})
        executor = CancellingExecutor(actions)
        store = RecordingStore(actions)
        runtime = ChatRuntime(
            ScriptedModel([_tool_model(call)]),
            tool_executor=executor,
            checkpoint_store=store,
            checkpoint_run_id="run-tool-cancelled",
        )
        with pytest.raises(ToolExecutionCancelled):
            [event async for event in runtime.stream_user_message("hello", tools=[executor.definition_value])]

        assert executor.calls == ["call-cancelled"]
        assert runtime._harness.stop_reason is StopReason.CANCELLED
        checkpoints = await store.list("run-tool-cancelled")
        assert [item.boundary for item in checkpoints] == [CheckpointBoundary.AFTER_MODEL]

    asyncio.run(run())


def test_after_tool_durability_failure_marks_sqlite_run_history_incomplete(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class FailingCheckpointStore(RecordingStore):
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

    actions: list[str] = []
    call = ToolCall(id="call-1", name="echo", arguments={})
    executor = RecordingExecutor(actions)
    checkpoint_store = FailingCheckpointStore(actions, fail_on_save=2)
    monkeypatch.setattr(
        cli,
        "OpenAICompatibleChatModel",
        lambda config: ScriptedModel([_tool_model(call), _text_model()]),
    )
    monkeypatch.setattr(cli, "ToolExecutor", lambda registry, timeout: executor)
    monkeypatch.setattr(cli, "SQLiteCheckpointStore", lambda path: checkpoint_store)
    state_path = tmp_path / "runs.db"

    code = asyncio.run(
        cli._run_chat(
            "hello",
            cli.ToolRegistry(),
            model_config=ModelConfig("https://example.test", "test-key", "fake"),
            tools=[executor.definition_value],
            state_db=str(state_path),
            checkpoint_db=str(tmp_path / "checkpoints.db"),
        )
    )
    capsys.readouterr()

    connection = sqlite3.connect(state_path)
    run_id, status, trace_complete, error_code = connection.execute(
        "SELECT id, status, trace_complete, error_code FROM runs LIMIT 1"
    ).fetchone()
    connection.close()
    latest = asyncio.run(checkpoint_store.load_latest(run_id))
    assert code == 1
    assert status == "failed"
    assert trace_complete == 0
    assert error_code == "checkpoint_persistence_failed_after_tool"
    assert executor.calls == ["call-1"]
    assert latest is not None
    assert latest.boundary is CheckpointBoundary.AFTER_MODEL


def test_save_failure_does_not_advance_execution_sequence() -> None:
    async def run() -> None:
        actions: list[str] = []
        execution = HarnessRunner(ScriptedModel([_text_model()])).create_execution(
            HarnessRequest(messages=(Message(MessageRole.USER, "hello"),))
        )
        store = RecordingStore(actions, fail_on_save=1)
        events = [event async for event in CheckpointCoordinator(execution, store, run_id="run-sequence").stream()]
        assert events[-1].type is RuntimeEventType.RUN_FAILED
        assert execution.last_checkpoint_sequence is None
        assert await store.load_latest("run-sequence") is None

    asyncio.run(run())


def test_terminal_save_failure_converts_completion_to_runtime_failure() -> None:
    async def run() -> None:
        actions: list[str] = []
        store = RecordingStore(actions, fail_on_save=2)
        events = [
            event
            async for event in ChatRuntime(
                ScriptedModel([_text_model()]),
                checkpoint_store=store,
                checkpoint_run_id="run-terminal-failed",
            ).stream_user_message("hello")
        ]
        assert events[-1].type is RuntimeEventType.RUN_FAILED
        assert events[-1].metadata["checkpoint_boundary"] == CheckpointBoundary.RUN_TERMINAL.value
        assert [item.sequence for item in await store.list("run-terminal-failed")] == [0]

    asyncio.run(run())


def test_resumed_execution_starts_automatic_sequence_after_source() -> None:
    async def run() -> None:
        call = ToolCall(id="call-resume", name="echo", arguments={})
        source = HarnessCheckpoint.create(
            HarnessState(
                messages=(Message(MessageRole.ASSISTANT, content=None, tool_calls=(call,)),),
                model_turns=1,
                phase=HarnessPhase.AFTER_MODEL,
            ),
            "run-resume",
            7,
        )
        executor = RecordingExecutor([])
        execution = HarnessRunner(ScriptedModel([_text_model()]), tool_executor=executor).resume_execution(
            HarnessResumeRequest(source, tools=(executor.definition_value,))
        )
        store = InMemoryCheckpointStore()
        events = [
            event
            async for event in CheckpointCoordinator(execution, store, run_id="run-resume").stream()
        ]

        assert events[-1].type is RuntimeEventType.RUN_COMPLETED
        checkpoints = await store.list("run-resume")
        assert checkpoints[0].sequence == 8
        assert checkpoints[-1].sequence == 10
        assert executor.calls == ["call-resume"]

    asyncio.run(run())


def test_cancellation_saves_terminal_checkpoint_when_safe() -> None:
    async def run() -> None:
        class CancelModel(ChatModel):
            async def stream(self, messages, tools=None):
                yield RuntimeEvent(RuntimeEventType.MODEL_STARTED)
                raise asyncio.CancelledError()

        store = InMemoryCheckpointStore()
        runtime = ChatRuntime(
            CancelModel(),
            checkpoint_store=store,
            checkpoint_run_id="run-cancel",
        )
        with pytest.raises(asyncio.CancelledError):
            [event async for event in runtime.stream_user_message("hello")]
        checkpoint = await store.load_latest("run-cancel")
        assert checkpoint is not None
        assert checkpoint.boundary is CheckpointBoundary.RUN_TERMINAL
        assert checkpoint.state.status is HarnessStatus.CANCELLED
        assert checkpoint.state.stop_reason is StopReason.CANCELLED

    asyncio.run(run())
