# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **MVP readiness plan executed.** Legal KG wired into UI as default. 2,200+ triplets across 10 predicates, 496 subjects. 3,325 golden answers. 50 hand-crafted benchmark cases.
- **Ralph Wiggum converged at 97.4%** on 3,325 test cases — above 90% threshold. No further dictionary additions needed.
- **UI upgraded.** KG mode selector (Legal/Remulak), confidence indicators on answers, grounding transparency (shows source KG facts), KG Explorer tab for browsing entities.
- **Next milestone: Demo benchmark (A/B comparison).** Three-arm comparison: naked LLM vs. LLM + real opinion text vs. Crystal. Requires downloading real opinion documents from CourtListener (not synthetic), auditing which predicates are document-answerable, adding obscure cases. See TODO.md "Next Up" section.
- **After that:** Phase 1c (judge bios), Phase 2 (opinion text extraction). Citation re-ingestion running in background.
- **Test count:** 729 passing, 5 skipped.

---

## 2026-04-11 — MVP Readiness Plan Execution

### What changed
- **Phase A: Legal KG wired into UI** — `src/crystal/tools/kg/legal.py`: added `load_legal_kg()` convenience function. `src/crystal/ui/app.py`: KG mode selector dropdown (Legal SQLite / Remulak demo), loads SQLite KG at startup when `data/legal.sqlite` exists, legal KG is the default experience.
- **Phase B: Expanded COLD Cases extraction** — 4 new predicates: `opinion_author` (208 records), `per_curiam` (10), `attorneys` (182), `precedential_status` (496). Ontology updated with aliases. Question templates added. SQLite KG rebuilt: 2,206 triplets (up from 1,615). 1,956 auto-accepted questions generated.
- **Phase C: Cancelled** — CourtListener `/dockets/` endpoint returns empty `nature_of_suit` and `disposition` for SCOTUS cases. These fields are only populated for district court PACER data.
- **Phase D: Ralph Wiggum loop** — Ran on 3,325 accepted cases against expanded SQLite KG. 97.4% accuracy (3,237/3,325 correct). Converged immediately above 90% threshold. LLM proposals yielded no further improvements.
- **Phase E: UI polish** — Confidence indicators (HIGH/MEDIUM/LOW based on route), grounding transparency (shows source KG facts for grounded answers), KG Explorer tab (entity search, predicate summary, entity list). Request verb and subject scan improvements.
- **Phase F: Benchmarks expanded to 50** — `benchmarks/ground_truth/legal.py`: 40 positive + 10 negative cases covering all predicates, WH-word variation, citation-format entities, request verb variation, subject scan queries. `tests/fixtures/scotus_sample.py`: added `precedential_status`, `attorneys`, `opinions` fields to key records. All 94 integration tests pass.
- **Test fixture updates** — SCOTUS sample records enriched with new fields for integration test coverage.

### Decisions
- CourtListener `/dockets/` skipped: SCOTUS cases lack `nature_of_suit`/`disposition` data. These fields are PACER-specific.
- 97.4% Ralph Wiggum score accepted as sufficient — remaining 88 failures are entity resolution edge cases, not predicate mapping issues.
- `_format_kg_facts()` reused for KG Explorer; `_search_kg_entity()` uses `_resolve_entity()` 3-tier cascade for search.

---

## 2026-04-09 — Phase 1a+1b Bulk Ingestion, Tiered Data Strategy, Review UI Fixes

### What changed
- **Tiered data strategy** — Established phased ingestion roadmap (TODO.md rewritten):
  - Phase 1 (structured API): auto-accept, no human review needed. Golden answers are verbatim API field values.
  - Phase 2 (unstructured text): Crystal proposes, human verifies. NER/LLM extraction with confidence-tiered review.
- **Phase 1a bulk ingestion** — `scripts/bulk_ingest.py`: streams SCOTUS cases from COLD Cases (HuggingFace), filters by court name, builds SQLite KG, generates questions, auto-accepts everything with `confidence_tier: 0`.
  - 500 SCOTUS cases → 1,305 triplets → 1,321 auto-accepted questions (Tier 1 factual + 20 negatives)
- **Phase 1b citation graph** — `scripts/bulk_citations.py`: searches CourtListener by case name → cluster → opinion → citations. Resolves cited opinion IDs to case names.
  - 200 cases searched → 57 with citations → 316 citation triplets → 48 Tier 2 relational questions
  - Rate-limited at 0.6s/request, ~15 min for 200 cases
- **Auto-accepted original 100 questions** — `review/pending_questions.json` status changed from `pending_review` to `accepted`.
- **Final totals:** 1,469 accepted golden answers, 1,615 triplets (1,301 metadata + 314 citations), 496 subjects in SQLite KG.
- **Review UI fixes** (Gradio 6 compatibility):
  - `list_batches()` now discovers all `*.json` files with `cases` lists, not just `batch_*.json`
  - `load_batch_questions()`, `load_batch_context()`, `save_review_decisions()` use `_resolve_batch_path()` helper
  - `load_batch_context()` falls back to extracting source triplets from each case's `source_triplet` field
  - All Dataframe returns converted to `pandas.DataFrame` (Gradio 6 `type="pandas"` default)
  - Questions table pre-populated on page load (`.change` event doesn't fire on initial render)
  - Refresh button now updates all Review tab components (dashboard, dropdown, table, context, gaps)
  - "Textbox" label renamed to "Save Status"

### Decisions
- Structured API data needs no human approval: the golden answer is the API field value, and the odds of incorrect data from a structured source are negligible
- Batching is unnecessary for auto-accepted data — bulk ingest replaces the cron → review → accept workflow for Tier 0
- 500 cases / ~2,500 questions is the sufficiency threshold for Ralph Wiggum convergence (~50 per predicate)
- Citation resolution requires 2 API calls per cited opinion (opinion → cluster → case_name) — capped at 10 citations/case and 200 cases for first run
- Phase 1c (CourtListener /people/ judge bios) deferred — requires new adapter + entity type

---

## 2026-04-04 — Cron Ingestion, Interactive Review UI, Ralph Wiggum Phase 6b

### What changed
- **Ingestion cron CLI** — `src/crystal/ingest/cron.py`
  - `run_ingestion_batch()` runs the full pipeline: stream COLD Cases → build/update SQLite KG → generate questions → export to `review/batch_YYYYMMDD_HHMMSS.json` with batch metadata (id, source, record count, timestamp, source triplets)
  - CLI: `PYTHONPATH=src:. python src/crystal/ingest/cron.py --source cold-cases --limit 100`
  - Tracks accepted case count across batches, prompts when ready for Ralph Wiggum
  - 8 tests in `tests/unit/ingest/test_cron.py`
- **Batch-aware review API** — `src/crystal/review.py` extended
  - `list_batches()`, `load_batch_questions()`, `load_batch_context()`, `save_review_decisions()`, `collect_accepted_cases()`
  - `format_review_dashboard()` now includes batch summary and Ralph Wiggum readiness indicator
  - 13 tests in `tests/unit/test_review_batches.py`
- **Interactive review UI** — `src/crystal/ui/app.py` Review tab redesigned
  - Batch selector dropdown with per-batch metadata display
  - Interactive questions table: edit Status column → Save Decisions button
  - Batch context panel: shows source triplets grouped by entity
- **Ralph Wiggum Phase 6b (autonomous mutation)** — `benchmarks/ralph_wiggum.py`
  - `_propose_changes()`: LLM analyzes failures, proposes dict additions to QUESTION_PREDICATE_MAP and LEGAL_PREDICATE_ALIASES
  - `_validate_proposal()`: string→string only, additions only, allowed sections only
  - `_apply_proposal()` / `_insert_dict_entries()`: regex-based insertion before closing brace of target dict literals
  - `_revert_proposal()` / `_remove_dict_entries()`: non-git revert path for undoing failed proposals
  - Git integration: `--use-git` flag creates `ralph/*` branch, commits changes, `git reset --hard` on regression
  - `_parse_llm_proposal()`: extracts JSON from code fences or bare text
  - CLI: `PYTHONPATH=src:. python benchmarks/ralph_wiggum.py --threshold 0.90 [--use-git] [--dry-run]`
  - `ralph_results.tsv` output log (iteration, score, commit, keep/discard)
  - 14 tests in `tests/unit/test_ralph_wiggum_mutations.py`
  - All 12 original tests still pass
- **Test total:** 659 passing, 5 skipped (+35 new tests)

### Decisions
- Adopted karpathy/autoresearch pattern for Ralph Wiggum: one mutable file (predicate map), immutable evaluation, git as undo, no human approval
- Two mutable files rather than one: QUESTION_PREDICATE_MAP (maps extracted phrases → predicates) and LEGAL_PREDICATE_ALIASES (maps surface forms → canonical predicates) serve complementary roles
- Non-git revert path available (`_revert_proposal`) for use without `--use-git` — safer for development
- Consecutive discard limit (3) prevents infinite loops when LLM proposals aren't useful
- `collect_accepted_cases()` aggregates across all batch files — Ralph Wiggum draws from entire reviewed corpus
- `format_review_dashboard()` now takes optional `review_dir` parameter for testability

---

## 2026-03-29 — Legal Data Ingestion Pipeline (Phases 0-6)

### What changed
- **Phase 0: Legal ontology** — `src/crystal/data/legal_ontology.py`
  - 7 canonical predicates (cites, cited_by_count, court, date_filed, judges, disposition, nature_of_suit)
  - 40+ predicate aliases, COURT_TYPE_MAP (12 codes)
  - `normalize_case_name()`, `parse_citation()`, `generate_case_aliases()` with slug-derived aliases
  - 31 tests in `tests/unit/test_legal_ontology.py`
- **Phase 1: COLD Cases adapter** — `src/crystal/ingest/sources/cold_cases.py`
  - `_record_to_triplets()`, `ingest_cold_cases()` (streaming iterator with filters, injectable records)
  - `scotus_filter()` for SCOTUS-specific detection
  - 22 tests in `tests/unit/ingest/test_cold_cases.py`
- **Phase 3: SQLite persistence** — `src/crystal/tools/kg/store.py`
  - `SqliteKnowledgeGraph` with same lookup()/traverse() API as in-memory KG
  - `bulk_insert()` with dedup, WAL mode, 3-tier resolution cascade via SQL
  - `src/crystal/tools/kg/legal.py` — convenience builders (memory + SQLite)
  - 26 tests in `tests/unit/tools/test_sqlite_kg.py`
- **Phase 2: CourtListener client** — `src/crystal/ingest/sources/courtlistener.py`
  - REST client with rate limiting, retry, cursor pagination
  - `citations_to_triplets()` for mapping citation results to KG format
  - 12 tests with mock HTTP transport
- **Phase 4: Question generator** — `src/crystal/ingest/question_gen.py`
  - Tier 1 (factual lookup) + Tier 2 (relational traversal) + negative case generation
  - Predicate-specific natural language templates, `QuestionCase` dataclass
  - 22 tests
- **Phase 5: Fitness function** — `benchmarks/fitness.py`
  - `binary_correct()`, `fitness_score()`, `evaluate_cases()` — single metric for Ralph Wiggum loop
  - 17 tests
- **Phase 6: Ralph Wiggum loop** — `benchmarks/ralph_wiggum.py`
  - `RalphWiggumLoop` with `run_iteration()`, `run()`, failure analysis
  - Converges on easy cases, early-stops on score plateau
  - 12 tests
- **Benchmark cases** — `benchmarks/legal_ground_truth.py`
  - 11 positive + 3 negative legal questions, `LEGAL_KNOWN_GAPS` for future detector work
  - 29 integration tests in `tests/integration/test_legal_pipeline.py`
- **Detector updates**: QUESTION_PREDICATE_MAP extended (filed, decided, ruled), LEGAL_PREDICATE_ALIASES expanded

### Decisions
- SQLite before CourtListener (Phase 3 before Phase 2): persistence needed before handling large citation volumes
- Citation-format entity spans deferred: spaCy splits "384 U.S. 436" into separate tokens, needs regex pre-pass in detector
- "decided" mapped to date_filed (not court/judges): "who" context lost after noise word stripping — Ralph Wiggum optimization target
- In-memory KG collapses multi-valued predicates (cites): SQLite KG used for question gen tests with citations
- No neo4j needed: SQLite handles the scale (8M records at sub-second query time via indexed lookups)

---

## 2026-03-25 — D2 Phase 2: LLM-assisted relationship extraction

### What changed
- `src/crystal/ingest/schema.py`: New `ReviewableTriplet` dataclass (subject/predicate/object + source_sentence, confidence, status) with `to_dict()`/`from_dict()` serialization. New `LLMExtractionResult` dataclass with `accepted_triplets()`, `pending_triplets()`, `to_review_dict()`/`from_review_dict()` for the human review workflow, and `to_ingest_result()` for converting accepted triplets back into the pipeline.
- `src/crystal/ingest/ner.py`: New `find_unresolved_sentences()` — identifies sentences where spaCy found ≥2 noun chunks but dep-tree patterns produced zero triplets. New `_get_noun_chunks()` helper.
- `src/crystal/ingest/llm_extract.py`: New module. Structured LLM prompt for relationship extraction from gap sentences. `_format_sentences()` builds numbered sentence+entity list. `_parse_llm_response()` robustly extracts JSON array from LLM output (handles code fences, surrounding text, malformed JSON). `extract_triplets_llm()` orchestrates batched extraction with injectable `call_llm_fn` for testability.
- `src/crystal/ingest/loader.py`: New `load_review()` — reads a human-reviewed JSON file and imports only triplets with `status: "accepted"`.
- `src/crystal/ingest/__init__.py`: New `ingest_with_llm()` two-pass pipeline (NER first, LLM for gaps). Re-exports all new public symbols.
- `src/crystal/ingest/__main__.py`: New CLI flags `--llm-assist` (triggers two-pass extraction), `--review-output` (custom path for review JSON), `--load-review` (import accepted triplets from reviewed file). Refactored into `_print_table()` and `_format_result_json()` helpers.
- 50 new tests: `test_schema.py` (+11), `test_ner.py` (+6), `test_llm_extract.py` (26), `test_loader.py` (+6), `test_integration.py` (+5)
- 421/421 passing, 5 skipped

### Decisions
- LLM function is injectable (`call_llm_fn` parameter) — tests use mocks, no real API calls in the test suite
- Gap detection requires ≥2 noun chunks (not just 1) — a sentence with a single entity and no predicate is unlikely to contain an extractable relationship
- LLM-extracted triplets are always `pending_review` — never auto-accepted into the KG. The human-in-the-loop design is intentional: LLM extraction is a suggestion engine, not a truth source
- Review JSON format designed for hand-editing: flat list of dicts with a `status` field the reviewer changes to "accepted" or "rejected"
- Batching (default 20 sentences/call) keeps LLM context manageable while reducing round-trips

---

## Session Template

```
## YYYY-MM-DD — [Title]

### What changed
-

### Decisions
-

### Next steps
-
```
