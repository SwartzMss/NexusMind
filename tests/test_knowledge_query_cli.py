from __future__ import annotations

import json
from types import SimpleNamespace

from nexusmind import cli


class FakeKnowledgeBase:
    closed = False

    def query(self, question):
        assert question == "Binder UID?"
        citation = SimpleNamespace(
            citation_id="K1",
            source_id="docs",
            document_id="doc-1",
            logical_path="binder.md",
            document_content_hash="a" * 64,
            chunk_id="chunk-1",
            start_offset=0,
            end_offset=6,
            content_hash="b" * 64,
        )
        passage = SimpleNamespace(
            citation_id="K1", logical_path="binder.md", chunk_id="chunk-1"
        )
        return SimpleNamespace(
            answer=SimpleNamespace(text="Kernel credentials [K1]"),
            citations=(citation,),
            trace_id="trace-1",
            trace=SimpleNamespace(
                retrieval_backend="HybridChunkIndex",
                passages=(passage,),
                candidate_count=2,
                context_character_count=3200,
                context_estimated_token_count=800,
            ),
        )

    def close(self):
        self.closed = True


def _patch_query_dependencies(monkeypatch):
    knowledge = FakeKnowledgeBase()
    monkeypatch.setattr(
        cli, "load_model_config_from_env", lambda: SimpleNamespace(model="fake")
    )
    monkeypatch.setattr(cli, "OpenAICompatibleChatModel", lambda config: object())
    monkeypatch.setattr(cli, "ChatModelAnswerGenerator", lambda model, config_identity: object())
    monkeypatch.setattr(
        cli.KnowledgeBase,
        "open",
        lambda path, answer_generator: knowledge,
    )
    return knowledge


def test_query_cli_outputs_answer_sources_and_debug_trace(monkeypatch, capsys) -> None:
    knowledge = _patch_query_dependencies(monkeypatch)

    assert cli.main(["query", "Binder UID?", "--debug"]) == 0

    captured = capsys.readouterr()
    assert "Answer:\nKernel credentials [K1]" in captured.out
    assert "[K1] binder.md" in captured.out
    assert "backend: HybridChunkIndex" in captured.out
    assert "3200 chars" in captured.out
    assert "trace-1" in captured.out
    assert captured.err == ""
    assert knowledge.closed is True


def test_query_cli_json_is_stable_and_machine_readable(monkeypatch, capsys) -> None:
    _patch_query_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli,
        "knowledge_query_result_dict",
        lambda result, include_debug: {
            "answer": result.answer.text,
            "citations": [{"citation_id": "K1", "logical_path": "binder.md"}],
            "debug": {"retrieval_backend": result.trace.retrieval_backend}
            if include_debug
            else None,
            "trace_id": result.trace_id,
        },
    )

    assert cli.main(["query", "Binder UID?", "--json", "--debug"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "answer": "Kernel credentials [K1]",
        "citations": [{"citation_id": "K1", "logical_path": "binder.md"}],
        "debug": {"retrieval_backend": "HybridChunkIndex"},
        "trace_id": "trace-1",
    }
