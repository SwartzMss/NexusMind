# End-to-End RAG Answer Quality Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic offline evaluator that runs real `KnowledgeBase.query()` cases, scores final answer quality and citation support, compares context configurations, and writes a stable checked-in report.

**Architecture:** Keep evaluation in a new `answer_quality_evaluation` module. A strict authored-case loader feeds a temporary local KnowledgeBase and an injected deterministic AnswerGenerator; a pure evaluator consumes the resulting `KnowledgeQueryResult` and produces per-case and aggregate metrics. Production retrieval, answer generation, citation validation, and query defaults remain unchanged.

**Tech Stack:** Python 3.11+, dataclasses, enums, JSON fixtures, pytest, existing local ingestion and KnowledgeBase contracts.

---

## File map

- Create `src/nexusmind/answer_quality_evaluation.py`: strict authored schema, deterministic fixture generator/runner, pure scoring, aggregate report, Markdown renderer, and CLI entry point.
- Modify `src/nexusmind/__init__.py`: export the evaluator's public types and functions.
- Create `tests/test_answer_quality_evaluation.py`: unit and integration coverage for loading, scoring, A/B execution, diagnostics, and stable rendering.
- Create `evals/knowledge/answer_quality/corpus/android-binder.md`: multi-section Binder/SELinux dogfooding document.
- Create `evals/knowledge/answer_quality/corpus/qnx-resource-manager.md`: multi-section QNX document with a distractor section.
- Create `evals/knowledge/answer_quality/corpus/security.md`: TrustZone/PKI document with a nearby unsupported claim.
- Create `evals/knowledge/answer_quality/corpus/distractor.md`: unrelated technical content that should not support cases.
- Create `evals/knowledge/answer_quality/cases.json`: versioned strict cases covering complete, partial, insufficient-evidence, unsupported, and context A/B behavior.
- Create `evals/knowledge/answer_quality/baseline.md`: generated deterministic report.

## Task 1: Define and validate authored evaluation cases

**Files:**
- Modify: `src/nexusmind/answer_quality_evaluation.py`
- Create: `tests/test_answer_quality_evaluation.py`

- [ ] **Step 1: Write failing schema tests**

Add tests that express the public loader contract:

```python
def test_load_cases_requires_version_and_unique_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")
    with pytest.raises(AnswerQualityEvaluationDatasetError, match="at least one case"):
        load_answer_quality_cases(path)


def test_case_result_targets_are_canonical_and_fact_ids_unique(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "cases": [{
            "case_id": "case",
            "question": "question",
            "required_facts": [{
                "fact_id": "fact",
                "answer": "answer",
                "match_phrases": ["answer"],
                "evidence_match_phrases": ["evidence marker"],
                "required_evidence": [{"source_id": "docs", "logical_path": "doc.md"}],
            }],
            "forbidden_claims": [],
            "required_evidence": [{"source_id": "docs", "logical_path": "doc.md"}],
            "allow_insufficient_evidence": False,
            "answer_profile": "grounded",
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    cases = load_answer_quality_cases(path)
    assert cases[0].required_facts[0].fact_id == "fact"
    assert cases[0].required_facts[0].required_evidence[0].logical_path == "doc.md"
```

Also test rejection of unknown root/case fields, blank text, duplicate case or
fact IDs, empty fact/evidence arrays, invalid evidence target fields, invalid
profiles, and non-boolean `allow_insufficient_evidence`.

- [ ] **Step 2: Run the schema tests and verify the expected RED failure**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_answer_quality_evaluation.py -q
```

Expected: collection fails because `AnswerQualityEvaluationDatasetError` and
`load_answer_quality_cases` do not exist yet.

- [ ] **Step 3: Implement strict immutable schema and loader**

Add these public values and strict parsing behavior:

```python
class AnswerQualityEvaluationError(Exception):
    """Base class for answer-quality evaluation failures."""


class AnswerQualityEvaluationDatasetError(AnswerQualityEvaluationError):
    """Authored answer-quality cases are unreadable or invalid."""


class AnswerQualityStatus(str, Enum):
    FULLY_CORRECT = "fully_correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class AnswerQualityEvidenceTarget:
    source_id: str
    logical_path: str
    chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class RequiredAnswerFact:
    fact_id: str
    answer: str
    match_phrases: tuple[str, ...]
    evidence_match_phrases: tuple[str, ...]
    required_evidence: tuple[AnswerQualityEvidenceTarget, ...]


@dataclass(frozen=True, slots=True)
class AnswerQualityCase:
    case_id: str
    question: str
    required_facts: tuple[RequiredAnswerFact, ...]
    forbidden_claims: tuple[str, ...]
    required_evidence: tuple[AnswerQualityEvidenceTarget, ...]
    allow_insufficient_evidence: bool
    answer_profile: str = "grounded"
```

Validate all plain text, tuple, uniqueness, and profile constraints in
`__post_init__`. `load_answer_quality_cases(path)` must read UTF-8 JSON, require
exact root fields `version` and `cases`, require version `1`, convert every
case into the dataclasses, reject malformed input as
`AnswerQualityEvaluationDatasetError`, and return cases sorted by `case_id`.

- [ ] **Step 4: Run schema tests and commit**

Run the focused test command again; expected: all schema tests pass. Commit:

```bash
git add src/nexusmind/answer_quality_evaluation.py tests/test_answer_quality_evaluation.py
git commit -m "feat: add answer quality evaluation case schema"
```

## Task 2: Add deterministic full-pipeline fixture execution

**Files:**
- Modify: `src/nexusmind/answer_quality_evaluation.py`
- Modify: `tests/test_answer_quality_evaluation.py`

- [ ] **Step 1: Write failing runner tests**

Create a temporary corpus fixture and assert that the runner invokes the real
query pipeline twice with only `expand_context` changing:

```python
def test_fixture_runner_executes_both_context_configurations(tmp_path: Path) -> None:
    cases_path = _write_case_dataset(tmp_path, profile="grounded")
    report = run_answer_quality_benchmark(cases_path, corpus_dir=tmp_path / "corpus")

    assert {(item.case_id, item.configuration) for item in report.case_results} == {
        ("case", "expand_context=false"),
        ("case", "expand_context=true"),
    }
    assert all(item.query_trace.retrieval_queries == ("question",) for item in report.case_results)
    assert all(item.answer for item in report.case_results)
```

Use a fake corpus file with a heading and two paragraphs so the true variant
can expose an adjacent fact while the false variant cannot. Assert that both
results retain `KnowledgeQueryResult.trace`, validated citations, final passage
IDs, and backend/generator identities.

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_answer_quality_evaluation.py::test_fixture_runner_executes_both_context_configurations -q
```

Expected: FAIL because the benchmark runner and deterministic generator do not
exist.

- [ ] **Step 3: Implement deterministic generator and runner**

Implement a private `FixtureAnswerGenerator` that satisfies the existing
`AnswerGenerator` protocol. It receives one `AnswerQualityCase`, scans each
`ModelContextRecord` passage in stable order, and for every required fact whose
required evidence target is represented by a passage and whose answer marker
is present in passage content, emits the authored `answer` string and cites
that passage's validated `[K#]` handle. `answer_profile` controls only fixture
behavior: `grounded` emits all available facts, `partial` omits the last
available fact, `unsupported` appends the authored forbidden claim and cites
the available passage, and `insufficient` emits a deterministic uncertainty
sentence with the available citations. It must never manufacture citation IDs.

Implement `run_answer_quality_benchmark(cases_path, *, corpus_dir, configurations=DEFAULT_ANSWER_QUALITY_CONFIGURATIONS)`:

```python
@dataclass(frozen=True, slots=True)
class AnswerQualityRunConfiguration:
    name: str
    expand_context: bool


DEFAULT_ANSWER_QUALITY_CONFIGURATIONS = (
    AnswerQualityRunConfiguration("expand_context=false", False),
    AnswerQualityRunConfiguration("expand_context=true", True),
)
```

For each configuration, create a temporary KnowledgeBase, register the corpus
directory with `LocalDirectorySourceConfig`, sync it, inject the deterministic
generator, and call the query with
`options=KnowledgeQueryOptions(generator=generator)` and
`expand_context=configuration.expand_context`.
Use the same case order and retrieval limit for every configuration. Convert
expected controlled query failures into a recorded evaluation result rather
than leaking provider details; unexpected evaluator input errors still raise.

- [ ] **Step 4: Run runner tests and commit**

Run the focused runner test and the complete schema/runner file; expected: pass.
Commit:

```bash
git add src/nexusmind/answer_quality_evaluation.py tests/test_answer_quality_evaluation.py
git commit -m "feat: run deterministic answer quality queries"
```

## Task 3: Implement pure answer, citation, and evidence scoring

**Files:**
- Modify: `src/nexusmind/answer_quality_evaluation.py`
- Modify: `tests/test_answer_quality_evaluation.py`

- [ ] **Step 1: Write failing scoring tests**

Build small real `KnowledgeQueryResult` fixtures through the existing answer
contracts and test each status and metric:

```python
def test_evaluator_distinguishes_complete_partial_incorrect_and_unsupported():
    case = _case_with_two_facts()
    complete = evaluate_answer_quality_case(case, _result("fact one fact two", ("K1", "K2")))
    partial = evaluate_answer_quality_case(case, _result("fact one", ("K1",)))
    incorrect = evaluate_answer_quality_case(case, _result("unrelated", ("K1",)))
    unsupported = evaluate_answer_quality_case(
        case, _result("fact one fact two forbidden claim", ("K1", "K2"))
    )
    assert complete.status is AnswerQualityStatus.FULLY_CORRECT
    assert complete.required_fact_coverage == 1.0
    assert partial.status is AnswerQualityStatus.PARTIALLY_CORRECT
    assert partial.required_fact_coverage == 0.5
    assert incorrect.status is AnswerQualityStatus.INCORRECT
    assert unsupported.status is AnswerQualityStatus.UNSUPPORTED
```

Add tests proving citation support requires the citation's canonical
`source_id`, `logical_path`, and optional `chunk_id` to match a fact's required
evidence; a valid citation to an unrelated allowed passage lowers support
precision. Test allowed citation validity, required-fact citation coverage,
forbidden claims, and insufficient-evidence success/failure.

- [ ] **Step 2: Run scoring tests and verify RED**

Run the named scoring tests; expected: FAIL because result dataclasses and
`evaluate_answer_quality_case` do not exist.

- [ ] **Step 3: Implement immutable per-case result and pure evaluator**

Add `AnswerQualityCaseResult` with fields for configuration, status, answer,
validated citations, required fact IDs satisfied/missed, coverage metrics,
forbidden claims, unsupported fact IDs, insufficient-evidence outcome, final
passage IDs, retrieval queries, fused provenance, expansion trace fields,
backend identity, and generator identity.

Implement:

```python
def evaluate_answer_quality_case(
    case: AnswerQualityCase,
    result: KnowledgeQueryResult | None,
    *,
    configuration: str = "direct",
    error: str | None = None,
) -> AnswerQualityCaseResult:
    """Score one validated query result against one authored case."""
```

Use Unicode casefold substring matching for authored phrases. Count fact
coverage as satisfied facts divided by total facts. Citation validity is true
only when every result citation exactly matches a model-context passage and
belongs to the case's top-level `required_evidence`. Citation coverage is the
fraction of satisfied facts with at least one citation whose target matches
that fact's required evidence and whose cited passage content contains one of
the fact's `evidence_match_phrases`; citation support precision is supported
cited fact claims divided by cited fact claims, with zero-safe denominators.
Detect forbidden
claims by exact casefold phrase presence. For insufficient cases, success means
the answer contains one of the deterministic uncertainty phrases and no
forbidden claim; for normal cases, an uncertainty-only answer is incorrect.

Classify in this order: unsupported if forbidden/unsupported claims or invalid
support are present; fully correct if all facts, citations, and sufficiency
expectations pass; partially correct if at least one but not all facts pass;
otherwise incorrect. Preserve the original `KnowledgeQueryResult` diagnostics
in bounded immutable result fields.

- [ ] **Step 4: Run scoring tests and commit**

Run all tests in `tests/test_answer_quality_evaluation.py`; expected: pass for
schema, runner, and scoring coverage. Commit:

```bash
git add src/nexusmind/answer_quality_evaluation.py tests/test_answer_quality_evaluation.py
git commit -m "feat: score answer correctness and citation support"
```

## Task 4: Add aggregate reports, authored corpus, and CLI

**Files:**
- Modify: `src/nexusmind/answer_quality_evaluation.py`
- Modify: `tests/test_answer_quality_evaluation.py`
- Modify: `src/nexusmind/__init__.py`
- Create: `evals/knowledge/answer_quality/corpus/android-binder.md`
- Create: `evals/knowledge/answer_quality/corpus/qnx-resource-manager.md`
- Create: `evals/knowledge/answer_quality/corpus/security.md`
- Create: `evals/knowledge/answer_quality/corpus/distractor.md`
- Create: `evals/knowledge/answer_quality/cases.json`
- Create: `evals/knowledge/answer_quality/baseline.md`

- [ ] **Step 1: Write failing aggregate/report tests**

Add assertions for stable aggregation and rendering:

```python
def test_report_aggregates_metrics_and_renders_byte_stably(tmp_path: Path) -> None:
    report = run_answer_quality_benchmark(
        _write_real_cases(tmp_path), corpus_dir=Path("evals/knowledge/answer_quality/corpus")
    )
    rendered_once = render_answer_quality_report(report)
    rendered_twice = render_answer_quality_report(report)
    assert rendered_once == rendered_twice
    assert 0.0 <= report.aggregate.answer_pass_rate <= 1.0
    assert "citation support precision" in rendered_once
    assert "expand_context=true" in rendered_once
```

Also test that calling `main(("--cases", str(cases_path), "--corpus",
str(corpus_dir), "--write", str(output_path)))` writes exactly the renderer
output and that exported public names import from `nexusmind`.

- [ ] **Step 2: Run aggregate tests and verify RED**

Run the named aggregate tests; expected: FAIL because aggregate report types,
renderer, CLI, and exports do not exist.

- [ ] **Step 3: Implement aggregate report and stable Markdown renderer**

Add `AnswerQualityAggregate`, `AnswerQualityBenchmarkReport`,
`render_answer_quality_report`, and `main`. Aggregate arithmetic must average
case metrics over the complete ordered result set and expose at least:

```python
answer_pass_rate
required_fact_coverage
citation_coverage
citation_support_precision
groundedness
unsupported_claim_rate
insufficient_evidence_success_rate
```

Render a Markdown table grouped by configuration, followed by aggregate rows
and a per-case diagnostics table. Sort configurations and cases explicitly;
format every float with six decimal places; include the reproduction command.
The CLI must require `--cases`, `--corpus`, and `--write`, use no network, and
write UTF-8 bytes from the renderer.

Export the evaluator's public schema, status, configuration, report, loader,
runner, evaluator, renderer, and `main` in `__init__.py` and `__all__`.

- [ ] **Step 4: Author representative corpus and cases**

Write real multi-section notes with stable headings and enough adjacent context
for the context A/B difference:

- Binder/SELinux: caller identity and permission enforcement, plus a nearby
  caveat explaining why a zero PID does not prove an anonymous caller;
- QNX: Resource Manager pathname dispatch and a distractor about unrelated
  scheduling;
- TrustZone/PKI: secure-world key isolation and certificate-chain validation;
- unrelated distractor: terms that overlap only superficially.

Author at least six cases: complete Binder, partial Binder, QNX complete,
TrustZone complete, intentionally insufficient evidence, and explicit
unsupported/forbidden claim. Include required evidence targets and fact phrase
markers in every case. Keep all values stable and human-readable.

- [ ] **Step 5: Generate the checked-in baseline and run aggregate tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m nexusmind.answer_quality_evaluation \
  --cases evals/knowledge/answer_quality/cases.json \
  --corpus evals/knowledge/answer_quality/corpus \
  --write evals/knowledge/answer_quality/baseline.md
```

Run the complete evaluation test file; expected: pass and byte-identical
baseline output on a second generation.

- [ ] **Step 6: Commit the public API, corpus, and report**

```bash
git add src/nexusmind/answer_quality_evaluation.py src/nexusmind/__init__.py \
  tests/test_answer_quality_evaluation.py evals/knowledge/answer_quality
git commit -m "feat: add deterministic answer quality benchmark"
```

## Task 5: Full verification and handoff

**Files:**
- Modify: none unless verification exposes an issue.

- [ ] **Step 1: Run focused and complete non-release tests**

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_answer_quality_evaluation.py tests/test_knowledge_answer.py tests/test_knowledge_query.py tests/test_context_assembly.py -q
PYTHONPATH=src .venv/bin/python -m pytest --ignore=tests/test_release_workflow.py -q
```

Expected: both commands exit 0.

- [ ] **Step 2: Run compile, diff, and determinism checks**

```bash
PYTHONPATH=src .venv/bin/python -m compileall -q src tests
git diff --check origin/main...HEAD
PYTHONPATH=src .venv/bin/python -m nexusmind.answer_quality_evaluation --cases evals/knowledge/answer_quality/cases.json --corpus evals/knowledge/answer_quality/corpus --write /tmp/answer-quality-baseline.md
cmp evals/knowledge/answer_quality/baseline.md /tmp/answer-quality-baseline.md
git status --short --branch
```

Expected: all commands exit 0, `cmp` reports no differences, and the worktree
is clean.

- [ ] **Step 3: Run the full suite and record environment-only failures**

```bash
PYTHONPATH=src .venv/bin/python -m pytest --tb=short -q
```

If the existing Python 3.10 environment reproduces the known
`tests/test_release_workflow.py`/`tomllib` failures, report those exact failures
without changing unrelated release code. All new and non-release tests must
remain green.

- [ ] **Step 4: Request review, then push and create the PR**

Request an independent review of `origin/main...HEAD`, address all critical or
important findings, rerun the verification commands, then push:

```bash
git push -u origin agent/issue-130-answer-quality
gh pr create --base main --head agent/issue-130-answer-quality \
  --title "feat: add end-to-end RAG answer quality evaluation" \
  --body $'## Summary\n\n- Add deterministic end-to-end answer and citation quality evaluation.\n- Add representative offline corpus and expand_context A/B benchmark.\n\nCloses #130\n\n## Test plan\n\n- Focused evaluator tests passed.\n- Non-release suite passed.\n- Full-suite environment-only failures recorded if present.'
```

The PR body must link Issue #130 with `Closes #130`, summarize the evaluator,
deterministic corpus, and A/B configurations, and list focused/non-release/full
test outcomes plus any environment-only failure.
