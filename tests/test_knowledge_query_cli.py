from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from nexusmind import cli


class FakeKnowledgeBase:
    closed = False

    def query(self, question):
        citation = SimpleNamespace(citation_id="K1", logical_path="binder.md")
        return SimpleNamespace(
            answer=SimpleNamespace(text="Kernel credentials [K1]"),
            citations=(citation,),
            trace_id="trace-1",
            trace=SimpleNamespace(retrieval_backend="BM25", context_character_count=3200),
        )

    def close(self): self.closed = True


def _patch(monkeypatch):
    knowledge = FakeKnowledgeBase()
    monkeypatch.setattr(cli, "load_model_config_from_env", lambda: object())
    monkeypatch.setattr(cli, "OpenAICompatibleAnswerProvider", lambda config: object())
    monkeypatch.setattr(cli.KnowledgeBase, "open", lambda path, answer_generator: knowledge)
    return knowledge


def test_query_cli_outputs_answer_sources_and_debug(monkeypatch, capsys, caplog) -> None:
    knowledge = _patch(monkeypatch)
    logger = logging.getLogger("nexusmind.runtime")
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=logger.name)
    assert cli.main(["query", "Binder UID?", "--debug"]) == 0
    output = capsys.readouterr().out
    assert "Kernel credentials [K1]" in output
    assert "[K1] binder.md" in output
    assert "Retrieval backend: BM25" in output
    assert "3200 chars" in output
    assert knowledge.closed
    assert [record.event for record in caplog.records] == ["query_started", "query_completed"]
    assert caplog.records[-1].citation_count == 1
    assert all("Binder UID?" not in record.getMessage() for record in caplog.records)


def test_query_cli_json(monkeypatch, capsys) -> None:
    _patch(monkeypatch)
    monkeypatch.setattr(cli, "knowledge_query_result_dict", lambda result, include_debug: {"answer": result.answer.text, "debug": include_debug})
    assert cli.main(["query", "Binder UID?", "--json", "--debug"]) == 0
    assert json.loads(capsys.readouterr().out) == {"answer": "Kernel credentials [K1]", "debug": True}
