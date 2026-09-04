# Section-Aware Context Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add deterministic same-section neighboring context to \`KnowledgeBase.query()\` by default, with \`expand_context=False\` preserving the old context path and all retrieval contracts unchanged.

**Architecture:** Keep \`assemble_context()\` as the final budget/dedupe/provenance gate. Add a standalone expansion module that receives ranked \`KnowledgeSearchResult\` anchors and the collection’s atomically maintained derived chunk catalog, then emits anchors followed by bounded same-section neighbors. Add the query switch and bounded expansion diagnostics at the \`KnowledgeBase\`/query-trace layer; do not persist derived chunks or modify search/ranking/diagnostic APIs.

**Tech Stack:** Python 3.11–3.13, dataclasses, existing \`Chunk\`/\`KnowledgeSearchResult\`/\`ContextPackage\` contracts, pytest, JSON-backed offline benchmark fixtures.

---

### Task 1: Add the standalone expansion contract and deterministic neighbor selection

**Files:**
- Create: \`src/nexusmind/context_expansion.py\`
- Create: \`tests/test_context_expansion.py\`
- Modify: \`src/nexusmind/__init__.py\`

- [ ] **Step 1: Write failing tests**

Create exact \`Document\`/\`Chunk\` slices for \`definition\`, \`anchor\`, \`caveat\`, and \`sibling\`. Use \`KnowledgeSearchResult\` anchors and assert:

    def test_expansion_emits_ranked_anchors_before_same_section_neighbors() -> None:
        expanded = expand_context_candidates(
            (_anchor(fixture, "anchor", 5.0),),
            chunk_catalog={fixture.document.document_id: fixture.chunks},
        )
        assert [item.hit.chunk.chunk_id for item in expanded.candidates] == [
            "anchor", "definition", "caveat"
        ]
        assert expanded.anchor_chunk_ids == ("anchor",)
        assert expanded.expanded_chunk_ids == ("definition", "caveat")
        assert expanded.candidates[1].hit.score == 0.0
        assert expanded.candidates[1].hit.matched_terms == ()

    def test_expansion_skips_adjacent_sibling_section() -> None:
        expanded = expand_context_candidates(
            (_anchor(fixture, "caveat", 5.0),),
            chunk_catalog={fixture.document.document_id: fixture.chunks},
        )
        assert [item.hit.chunk.chunk_id for item in expanded.candidates] == ["caveat"]
        assert expanded.section_boundary_skips == 1

    def test_expansion_rejects_missing_catalog_entries() -> None:
        with pytest.raises(ValueError, match="chunk catalog"):
            expand_context_candidates((_anchor(fixture, "anchor", 5.0),), chunk_catalog={})

Also test shared-neighbor deduplication, exact canonical offsets/content, and that the expansion result preserves the original anchor order.

- [ ] **Step 2: Verify RED**

Run: \`PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_context_expansion.py -q\`

Expected: import/API failure because \`expand_context_candidates\` does not exist.

- [ ] **Step 3: Implement minimal expansion**

Create \`src/nexusmind/context_expansion.py\` with \`ContextExpansionResult\`, \`MAX_CONTEXT_EXPANSION_NEIGHBORS = 1\`, \`MAX_CONTEXT_EXPANSION_CANDIDATES = 100\`, and:

    def expand_context_candidates(
        anchors: tuple[KnowledgeSearchResult, ...],
        *,
        chunk_catalog: Mapping[str, tuple[Chunk, ...]],
    ) -> ContextExpansionResult:
        # Validate anchors, select bounded same-section neighbors, and return
        # ContextExpansionResult(candidates, anchor_chunk_ids,
        # expanded_chunk_ids, expanded_document_ids, section_boundary_skips).

Validate tuple/mapping types. For each anchor, locate its exact chunk in the catalog for the same document, inspect only the previous and next entries, accept only exact \`heading_path\` matches, count incompatible neighbors as \`section_boundary_skips\`, and emit all anchors before expansion candidates. Deduplicate chunk IDs and cap expansion-only candidates at 100. Build expansion hits with the canonical neighbor chunk, the anchor’s canonical source/document, score \`0.0\`, and empty matched terms. Raise sanitized \`ValueError\` for missing or incoherent catalog data. Export the module API from \`src/nexusmind/__init__.py\`.

- [ ] **Step 4: Verify GREEN**

Run: \`PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_context_expansion.py -q\`

Expected: all expansion tests pass.

- [ ] **Step 5: Commit**

    git add src/nexusmind/context_expansion.py src/nexusmind/__init__.py tests/test_context_expansion.py
    git commit -m "feat: add deterministic context expansion"

### Task 2: Maintain the canonical chunk catalog with collection state

**Files:**
- Modify: \`src/nexusmind/knowledge_collection.py:223-334, 944-980\`
- Modify: \`tests/test_knowledge_collection.py\`
- Modify: \`tests/test_knowledge_base_atomicity.py\`

- [ ] **Step 1: Write failing tests**

Using a custom multi-chunk fixture chunker, assert:

    catalog = collection.context_chunk_catalog((document.document_id,))
    assert catalog[document.document_id] == expected_chunks

    collection.sync(updated_adapter)
    assert collection.context_chunk_catalog((document.document_id,))[document.document_id] == updated_chunks

    collection.remove_source("docs")
    with pytest.raises(KnowledgeInspectionError, match="unknown document"):
        collection.context_chunk_catalog((document.document_id,))

Add a clone/replace failure case proving failed sync leaves both search state and catalog unchanged, and a restore case proving the catalog is rebuilt while \`KnowledgeSnapshot\` contains no derived chunk field.

- [ ] **Step 2: Verify RED**

Run: \`PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_knowledge_collection.py tests/test_knowledge_base_atomicity.py -q\`

Expected: failures because \`context_chunk_catalog()\` and catalog state do not exist.

- [ ] **Step 3: Implement atomic derived state**

Initialize \`self._chunks_by_document: dict[str, tuple[Chunk, ...]] = {}\`. In \`sync()\`, copy the catalog, remove deleted document IDs, add prepared chunks for added/changed documents, and assign it only with the staged index/sources/documents/versions. Update \`remove_source()\` and \`restore()\` at the same commit points.

Add:

    def context_chunk_catalog(
        self, document_ids: tuple[str, ...]
    ) -> dict[str, tuple[Chunk, ...]]:
        if type(document_ids) is not tuple:
            raise TypeError("document_ids must be a tuple")
        result: dict[str, tuple[Chunk, ...]] = {}
        for document_id in document_ids:
            if type(document_id) is not str or not document_id.strip():
                raise ValueError("document_ids must contain non-empty strings")
            chunks = self._chunks_by_document.get(document_id)
            if chunks is None:
                raise KnowledgeInspectionError(
                    "chunk catalog references an unknown document"
                )
            result[document_id] = tuple(chunks)
        return result

Keep the catalog out of \`KnowledgeSnapshot\` and the SQLite schema.

- [ ] **Step 4: Verify GREEN**

Run: \`PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_knowledge_collection.py tests/test_knowledge_base_atomicity.py -q\`

Expected: focused collection and atomicity tests pass.

- [ ] **Step 5: Commit**

    git add src/nexusmind/knowledge_collection.py tests/test_knowledge_collection.py tests/test_knowledge_base_atomicity.py
    git commit -m "feat: retain canonical chunk catalog for context expansion"

### Task 3: Wire query opt-in/opt-out and bounded diagnostics

**Files:**
- Modify: \`src/nexusmind/knowledge_base.py:763-861\`
- Modify: \`src/nexusmind/knowledge_query.py:37-170\`
- Modify: \`tests/test_knowledge_query.py\`
- Modify: \`tests/test_knowledge_answer.py\`
- Modify: \`tests/test_context_assembly.py\`

- [ ] **Step 1: Write failing tests**

Add tests with a recording generator and a multi-chunk fixture:

    def test_query_expands_same_section_by_default(tmp_path: Path) -> None:
        kb, generator = _multi_chunk_knowledge_base(tmp_path)
        result = kb.query(
            "anchor",
            options=KnowledgeQueryOptions(generator=generator),
        )
        assert [item.chunk_id for item in generator.calls[0][2].passages] == [
            "anchor", "definition", "caveat"
        ]
        assert result.trace.context_expansion_enabled is True
        assert result.trace.anchor_passage_count == 1
        assert result.trace.expanded_passage_count == 2

    def test_query_expand_context_false_reproduces_old_context(tmp_path: Path) -> None:
        kb, generator = _multi_chunk_knowledge_base(tmp_path)
        result = kb.query(
            "anchor",
            expand_context=False,
            options=KnowledgeQueryOptions(generator=generator),
        )
        assert [item.chunk_id for item in generator.calls[0][2].passages] == ["anchor"]
        assert result.trace.context_expansion_enabled is False
        assert result.trace.expanded_passage_count == 0

    def test_query_rejects_non_boolean_expand_context(tmp_path: Path) -> None:
        kb, _ = _multi_chunk_knowledge_base(tmp_path)
        with pytest.raises(TypeError, match="expand_context"):
            kb.query("anchor", expand_context=1)  # type: ignore[arg-type]

Also assert sibling sections are absent, tight \`max_passages\` retains higher-ranked anchors before expansions, debug JSON includes expansion metadata, and \`diagnose_search()\` output is unchanged.

- [ ] **Step 2: Verify RED**

Run: \`PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_knowledge_query.py tests/test_knowledge_answer.py tests/test_context_assembly.py -q\`

Expected: failures for the missing query parameter, missing trace fields, and missing neighboring passages.

- [ ] **Step 3: Implement query orchestration**

Add keyword-only \`expand_context: bool = True\` to \`KnowledgeBase.query()\`; reject non-plain booleans before retrieval. Leave search/fusion code unchanged. After \`fused\` is produced, use:

    if expand_context:
        catalog = self._collection.context_chunk_catalog(
            tuple(item.document.document_id for item in fused)
        )
        expansion = expand_context_candidates(fused, chunk_catalog=catalog)
        context_candidates = expansion.candidates
        context_max_candidates = max(
            active_options.retrieval_limit, len(context_candidates)
        )
    else:
        expansion = None
        context_candidates = fused
        context_max_candidates = active_options.retrieval_limit

    context = assemble_context(
        question,
        context_candidates,
        max_passages=active_limits.max_passages,
        max_candidates=context_max_candidates,
        max_chars=active_limits.max_context_chars,
        max_tokens=active_limits.max_context_tokens,
    )

Add defaulted \`KnowledgeQueryTrace\` fields: \`context_expansion_enabled\`, \`anchor_passage_count\`, \`expanded_passage_count\`, \`expanded_document_count\`, and \`section_boundary_skips\`. Validate types/counts and derive selected counts from final assembled passage chunk IDs. Add only to the \`include_debug=True\` JSON object. Do not add the switch to \`KnowledgeQueryOptions\`; do not change search, search_backend, ranking, scores, or diagnose_search.

- [ ] **Step 4: Verify**

Run: \`PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_knowledge_query.py tests/test_knowledge_answer.py tests/test_context_assembly.py tests/test_knowledge_diagnostics.py -q\`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

    git add src/nexusmind/knowledge_base.py src/nexusmind/knowledge_query.py tests/test_knowledge_query.py tests/test_knowledge_answer.py tests/test_context_assembly.py
    git commit -m "feat: enable section-aware query context expansion"

### Task 4: Add deterministic offline evaluation

**Files:**
- Create: \`src/nexusmind/context_expansion_evaluation.py\`
- Create: \`tests/test_context_expansion_evaluation.py\`
- Create: \`evals/knowledge/context_expansion/cases.json\`
- Create: \`evals/knowledge/context_expansion/baseline.md\`

- [ ] **Step 1: Write failing benchmark tests and fixture cases**

Add cases for previous definition, next caveat, sibling exclusion, and multiple-document budget competition. Every case declares \`case_id\`, \`query\`, ranked \`anchor_ids\`, \`relevant_ids\`, and \`forbidden_ids\`. Tests require anchor retention, relevant coverage, expansion precision, irrelevant expansion rate, boundary skips, and equality of two consecutive reports/rendered bytes.

- [ ] **Step 2: Verify RED**

Run: \`PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_context_expansion_evaluation.py -q\`

Expected: import failure because the benchmark module and fixtures do not exist.

- [ ] **Step 3: Implement evaluator and renderer**

Construct exact \`Document\`/\`Chunk\`/\`KnowledgeSearchResult\` fixtures, treat \`anchor_ids\` as already-ranked retrieval output, invoke \`expand_context_candidates()\`, then \`assemble_context()\` with explicit budgets. Compute:

    anchor_retention = retained_anchor_count / len(anchor_ids)
    relevant_coverage = len(retained_ids & relevant_ids) / len(relevant_ids)
    expansion_precision = len(expanded_ids & relevant_ids) / max(1, len(expanded_ids))
    irrelevant_expansion_rate = len(expanded_ids & forbidden_ids) / max(1, len(expanded_ids))

Sort case IDs, use fixed six-decimal rendering, and write \`baseline.md\` with reproduction command and per-case/overall tables. Do not use randomness, wall-clock values, model calls, or live retrieval.

- [ ] **Step 4: Verify and regenerate**

Run:

    PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m pytest tests/test_context_expansion_evaluation.py -q
    PYTHONPATH=src /home/swartz/WorkSpace/NexusMind/.venv/bin/python -m nexusmind.context_expansion_evaluation --write evals/knowledge/context_expansion/baseline.md
    git diff --check

Expected: benchmark tests pass and report bytes are stable.

- [ ] **Step 5: Commit**

    git add src/nexusmind/context_expansion_evaluation.py tests/test_context_expansion_evaluation.py evals/knowledge/context_expansion
    git commit -m "test: add context expansion benchmark"

### Task 5: Review, verify, and create the PR

**Files:**
- Modify: only files from Tasks 1–4 if review identifies a concrete issue.

- [ ] **Step 1: Run final supported-version tests**

Run \`PYTHONPATH=src python -m pytest -q\` on Python 3.11+ when available. In the current Python 3.10 environment, run the complete suite with the existing venv and record the pre-existing release-workflow \`tomllib\` failures separately; run all feature-focused tests successfully.

- [ ] **Step 2: Verify diff and requirements**

Run:

    git diff --check origin/main...HEAD
    git diff --stat origin/main...HEAD
    git status --short

Confirm branch base is the fetched \`origin/main\`; search/ranking/score/diagnostic code is unchanged; \`expand_context=False\` follows the old path; expansion content is canonical; budgets remain enforced by \`assemble_context()\`; benchmark output is deterministic.

- [ ] **Step 3: Request code review**

Use \`superpowers:requesting-code-review\` with base \`origin/main\`, final \`HEAD\`, issue #128 requirements, and test evidence. Fix Critical/Important findings, rerun tests, and re-review after code changes.

- [ ] **Step 4: Final verification**

Run the full available test command, focused expansion/query/diagnostic tests, \`git diff --check\`, and \`git status --short\` at one fresh checkpoint. Record exit codes and failure counts before claiming readiness.

- [ ] **Step 5: Push and create the PR**

    git push -u origin agent/issue-128-context-expansion
    gh pr create --base main --head agent/issue-128-context-expansion --title "feat: add section-aware context expansion" --body "$(cat <<'EOF'
    ## Summary
    - Add deterministic same-document, same-section context expansion around ranked retrieval anchors.
    - Keep retrieval ranking, scores, search diagnostics, and assemble_context defaults unchanged.
    - Add query-level opt-out, bounded expansion diagnostics, and an offline deterministic benchmark.

    Closes #128

    ## Test plan
    - Focused context-expansion, query, context-assembly, and diagnostics tests.
    - Full pytest suite on supported Python when available.
    - Deterministic benchmark regeneration and diff check.
    EOF
    )"

Keep the worktree after PR creation so review feedback can be applied safely.
