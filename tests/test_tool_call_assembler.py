import pytest

from nexusmind.models.tool_calls import ToolCallAssembler, ToolCallAssemblyError, ToolCallDelta


def test_assembler_builds_fragmented_tool_call() -> None:
    assembler = ToolCallAssembler()
    assembler.apply(ToolCallDelta(index=0, call_id_fragment="call_", type_fragment="func", name_fragment="ec"))
    assembler.apply(ToolCallDelta(index=0, call_id_fragment="1", type_fragment="tion", name_fragment="ho"))
    assembler.apply(ToolCallDelta(index=0, arguments_fragment='{"te'))
    assembler.apply(ToolCallDelta(index=0, arguments_fragment='xt":"hello"}'))

    calls = assembler.finalize()

    assert calls[0].id == "call_1"
    assert calls[0].name == "echo"
    assert calls[0].arguments == {"text": "hello"}


def test_assembler_normalizes_blank_arguments_to_object() -> None:
    assembler = ToolCallAssembler()
    assembler.apply(ToolCallDelta(index=0, call_id_fragment="call_1", type_fragment="function", name_fragment="echo"))

    assert assembler.finalize()[0].arguments == {}


def test_assembler_requires_function_type() -> None:
    assembler = ToolCallAssembler()
    assembler.apply(ToolCallDelta(index=0, call_id_fragment="call_1", name_fragment="echo", arguments_fragment="{}"))

    with pytest.raises(ToolCallAssemblyError):
        assembler.finalize()


def test_assembler_sorts_parallel_calls_by_index() -> None:
    assembler = ToolCallAssembler()
    assembler.apply(ToolCallDelta(index=1, call_id_fragment="call_b", type_fragment="function", name_fragment="b", arguments_fragment="{}"))
    assembler.apply(ToolCallDelta(index=0, call_id_fragment="call_a", type_fragment="function", name_fragment="a", arguments_fragment="{}"))

    assert [call.id for call in assembler.finalize()] == ["call_a", "call_b"]


@pytest.mark.parametrize(
    "delta",
    [
        ToolCallDelta(index=-1),
        ToolCallDelta(index=32),
        ToolCallDelta(index=0, type_fragment="custom"),
        ToolCallDelta(index=0, arguments_fragment="x" * (1024 * 1024 + 1)),
    ],
)
def test_assembler_rejects_invalid_or_oversized_delta(delta: ToolCallDelta) -> None:
    with pytest.raises(ToolCallAssemblyError):
        ToolCallAssembler().apply(delta)


@pytest.mark.parametrize(
    "arguments",
    ['{"text"', "[]", '{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}'],
)
def test_assembler_rejects_invalid_final_arguments(arguments: str) -> None:
    assembler = ToolCallAssembler()
    assembler.apply(
        ToolCallDelta(index=0, call_id_fragment="call_1", type_fragment="function", name_fragment="echo", arguments_fragment=arguments)
    )

    with pytest.raises(ToolCallAssemblyError):
        assembler.finalize()


def test_assembler_converts_recursion_error_to_controlled_error(monkeypatch) -> None:
    assembler = ToolCallAssembler()
    assembler.apply(
        ToolCallDelta(index=0, call_id_fragment="call_1", type_fragment="function", name_fragment="echo", arguments_fragment="{}")
    )

    def raise_recursion(*args, **kwargs):
        raise RecursionError()

    monkeypatch.setattr("nexusmind.models.tool_calls.json.loads", raise_recursion)

    with pytest.raises(ToolCallAssemblyError):
        assembler.finalize()


def test_tool_call_repr_does_not_include_arguments() -> None:
    assembler = ToolCallAssembler()
    assembler.apply(
        ToolCallDelta(
            index=0,
            call_id_fragment="call_1",
            type_fragment="function",
            name_fragment="echo",
            arguments_fragment='{"token":"sk-live-secret"}',
        )
    )

    call = assembler.finalize()[0]

    assert "sk-live-secret" not in repr(call)


def test_assembler_rejects_missing_id_or_name_and_duplicate_ids() -> None:
    missing = ToolCallAssembler()
    missing.apply(ToolCallDelta(index=0, type_fragment="function", arguments_fragment="{}"))
    with pytest.raises(ToolCallAssemblyError):
        missing.finalize()

    duplicate = ToolCallAssembler()
    duplicate.apply(ToolCallDelta(index=0, call_id_fragment="call_1", type_fragment="function", name_fragment="a", arguments_fragment="{}"))
    duplicate.apply(ToolCallDelta(index=1, call_id_fragment="call_1", type_fragment="function", name_fragment="b", arguments_fragment="{}"))
    with pytest.raises(ToolCallAssemblyError):
        duplicate.finalize()
