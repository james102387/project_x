# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Goal: ship a defensible demo showing Crystal+KG beats LLM+docs on
hallucination + citation precision for opinion-text questions.

**Demo success criteria (measurable):**
1. `opinion_golden` corpus populated with ≥20 hand-authored Q/A (currently **0 — blocker**).
2. `three_arm_comparison --corpus opinion_golden` baseline captured: expect Crystal > LLM+docs on accuracy by ≥10pp and on hallucination rate by ≥15pp.
3. One full Ralph Wiggum cycle (all loops incl. `ExtractionLoop`) runs to convergence on the golden set and the delta is logged.
4. `--corpus opinion_holdout` confirms the mutations generalize (no overfitting).

**Open blockers / known issues:**
- `opinion_golden.py` is empty — user-authored, cannot be automated.
- `opinion_holdout.py` is empty — needed for generalization check.
- Scaffold is still mostly `api_metadata` (3,263 rows, 0 `opinion_doc` after the old NER rows were purged). Real opinion-doc ingestion needs to happen before the golden set is meaningful.
- `ExtractionLoop` is opt-in (`run_extraction_loop=True`) — decide whether it should be default-on for the cycle above.

**Recent programmatic cleanup (this session):**
- Fixed silent `bulk_insert` counter inflation + phantom empty batch rows.
- `ingest_document` now dedups across NER + LLM + validation-rejected paths.
- Deleted 8 backward-compat shim modules under `benchmarks/`, repointed callers.
- Added `debug_confidence_trace()` in `planner.py` for grounded-confidence debugging.
- `QuestionGenLoop` no longer reports vacuous 1.0 when 0 questions are generated; it returns 0.0 with a `no_questions_generated` diagnosis so the orchestrator will mutate the prompt.
- `three_arm_comparison --corpus` default switched to `opinion_golden`; `all` now prints per-corpus sections instead of a merged dilute report.
- Merged the two question-gen template dicts (`compare._PRED_TEMPLATES` + `question_gen._PREDICATE_QUESTION_FORMS`) into a single source of truth with shared helpers.
- Split `src/crystal/ui/app.py` (1,970 lines) into `ui/state.py`, `ui/formatting.py`, and per-tab modules under `ui/tabs/`. `app.py` is now a 93-line composition root; all tests still pass.

---

## 2026-04-16 — Merge question-gen template codepaths

### What changed
- **Single source of truth for predicate → question templates.** `ingest/question_gen.py` now owns `PREDICATE_QUESTION_FORMS` (the richer multi-variant dict). New helpers `canonical_template(pred)` and `object_length_limit(pred)` replace the scattered logic. `_PREDICATE_QUESTION_FORMS`/`_LONG_PREDICATES`/`_SKIP_PREDICATES` kept as private aliases for back-compat.
- **`compare.py` simplified.** Deleted `_PRED_TEMPLATES` (13-predicate single-template dict) and the local `_JUNK_SUBJECTS`/`_JUNK_PREFIXES` shadows. Both `generate_questions_from_triplets` and the `generate_questions_llm` fallback now resolve templates via `canonical_template(pred)`. Junk-subject checks in `_is_plausible_case_name` delegate to `validation._JUNK_SUBJECTS` (strictly larger superset — catches `"the state"`, `"government"`, etc. that the compare.py shadow missed).
- **`cites` predicate added to `PREDICATE_QUESTION_FORMS`** to match what `compare._PRED_TEMPLATES` had; previously tier1 in `question_gen.py` silently fell through to the generic `"What is the {predicate} of {subject}?"` template for `cites`.
- **Tests updated.** `test_question_gen_loop.py::TestDoctrinalTemplates` now asserts via the public API (`canonical_template`, `PREDICATE_QUESTION_FORMS`) instead of the two-dict invariant. `QUESTION_GEN_PROMPT` stays in `compare.py` (the QuestionGenLoop mutates it by file path).
- **1,272 tests pass, 8 skipped.**

### Decisions
- Kept the question-gen LLM prompt (`QUESTION_GEN_PROMPT`) and the LLM generator in `compare.py` — moving them would require rewiring `QuestionGenLoop._read_current_prompt` / `_write_prompt` to a new file path, and the loop is the only mutator, so the cost/benefit didn't justify it this pass.
- Added `cites` to the shared dict rather than suppressing it in `generate_tier1`. Tier2 already handles multi-valued `cites` (subjects with ≥2 citations); tier1 will now also emit single-`cites` questions for subjects with exactly one citation. Acceptable — same behavior as the old compare.py path.

---

## 2026-04-15 — Provenance Tracking + Adversarial Golden Benchmark

### What changed
- **Schema migration** — Added `origin` and `source_document` columns to `triplets` table with indexes, migrations for existing DBs, and backfill logic in `backfill_provenance()`.
- **Pipeline wiring** — `ScoredTriplet` now has `source_document` field and `origin` property (maps extraction_source → storage-level origin). `ingest_document()` passes per-row origins via 5-tuples. `review_pipeline.py` propagates origin/source_document into proposed rows. `review.py` persists them in batch JSON with filter support in `collect_accepted_cases()`.
- **UI provenance** — KG Facts table shows Origin and Source Document columns with filter dropdowns. Review overview table shows origin and source doc per question. Entity search shows origin tags. KG stats banner breaks down by origin type. KG subgraph viewer in Review tab shows all facts for the source entity with provenance.
- **KG audit** — Backfilled 3,277 triplets: 2,940 api_metadata, 337 ner_extraction. Deleted 225 garbage NER subjects (pronouns, fragments, common nouns). 3,052 clean triplets remain.
- **Adversarial golden benchmark** — `benchmarks/ground_truth/opinion_golden.py`: 35 hand-authored Q&A targeting multi-hop citation chains, negative-existence, citation verification, holding vs. dicta, cross-document accuracy. Demo cluster: Gideon, Loving, Marbury, Miranda, Brown, Roe, Mapp, Powell, Betts.
- **Benchmark infrastructure** — `three_arm_comparison.py` accepts `--corpus` flag. `package_results.py` runs all corpora and produces unified demo report. `opinion_holdout.py` placeholder for validation batch.
- **ExtractionLoop integration** — Wired into orchestrator as opt-in loop with separate init path (different __init__ signature from other loops).
- **138 new/updated tests** — Provenance schema, bulk_insert round-trip, backfill, origin property, review batch fields, filter helpers.

### Decisions
- Backfill precision over recall: ambiguous rows stay `origin='unknown'` for manual triage.
- ExtractionLoop kept as opt-in (not in default LOOP_CLASSES) because it has a different init contract.
- Garbage NER subjects deleted outright rather than quarantined — clear false positives (pronouns, fragments).

---

## 2026-04-15 — LLM-Based Question Generation + QuestionGenLoop

### What changed
- **Doctrinal templates** — Added `holding`, `doctrine`, `reasoning` to `_PRED_TEMPLATES` (compare.py) and `_PREDICATE_QUESTION_FORMS` (question_gen.py). Raised object length cap from 200→2000 for these predicates.
- **`generate_questions_llm()`** — LLM-based question generator in `compare.py` with `QUESTION_GEN_PROMPT` constant. Groups triplets by subject, sends to LLM, parses JSON. Falls back to templates on failure.
- **Pipeline wiring** — `review_pipeline.py` uses LLM generation when available, template fallback otherwise. `question_gen.py` `generate_all()` accepts optional `call_llm_fn`.
- **QuestionGenLoop** — New Ralph Wiggum loop targeting `QUESTION_GEN_PROMPT`. Self-consistency fitness metric: can Crystal answer the questions it generates?
- **31 new tests** — Covers templates, golden doctrinal facts, LLM parse/generate/fallback, loop proposal/apply/revert, orchestrator registration.

### Decisions
- QuestionGenLoop returns score=1.0 when 0 questions generated (vacuously true) to not break orchestrator overall score for non-legal KGs.
- `QUESTION_GEN_PROMPT` stored as module-level constant (not inline) so the loop can read, rewrite, and revert it.

---

## 2026-04-16 — UI Split (per-tab modules)

### What changed
- `src/crystal/ui/app.py` went from 1,970 lines to 93. Split into a composition-root layout:
  - `ui/state.py` — compiled graph, `KG_MODES`, default KG.
  - `ui/formatting.py` — pure formatters (stats, facts DF, banners, route labels). No Gradio imports.
  - `ui/tabs/ask.py` (142) — `ask_question` + `build_ask_tab`.
  - `ui/tabs/ingest.py` (619) — ingestion + proposed-answers + comparison actions + `build_ingest_tab`.
  - `ui/tabs/kg.py` (405) — explorer + KG switch/import actions + `build_kg_tab` (owns cross-tab wiring that used to touch both Ingest and KG widgets).
  - `ui/tabs/review.py` (852) — batch review + benchmark/RW actions + `build_review_tab`.
- `ui/app.py::build_ui()` is now a thin composition root: creates shared `kg_state` + `kg_banner`, then delegates to each `build_<tab>` factory, passing `IngestTab` into `build_kg_tab` so cross-tab outputs stay explicit.
- Each `build_<tab>` returns a dataclass of its components so future cross-tab wiring stays typed rather than free-form.
- `crystal.ui.app` still re-exports `_kg_info`, `_format_kg_stats`, `_format_kg_facts`, `_default_kg_info`, `import_structured_data`, `KG_MODES`, `build_ui`, `main` for back-compat with `tests/unit/ui/test_ui_helpers.py`.
- Smoke-tested: `build_ui()` constructs cleanly; 1,272 tests pass, 8 skipped (unchanged from baseline).

### Decisions
- `switch_kg_mode` and `import_structured_data` live in `tabs/kg.py` (not a separate `actions/` module) because they are wired from the KG tab; ownership follows the event source rather than the touched components.
- Kept dataclasses per tab (`AskTab`, `IngestTab`, `KgTab`, `ReviewTab`) instead of plain dicts so component references are discoverable and renames surface at type-check time.

---

## 2026-04-16 — Project Review + Programmatic Cleanup

### What changed
- **`bulk_insert` silent counter bug fixed.** `INSERT OR IGNORE` never raises `IntegrityError`, so the old `try/except + inserted += 1` counted attempts, not actual rows. Switched to `cursor.rowcount` and skipped writing the `ingestion_batches` row when 0 rows landed (removes phantom-batch rows). Cleaned 4 orphaned batch rows from `data/legal.sqlite`.
- **`ingest_document` dedup leak fixed.** The NER and LLM extraction paths could hand the same `(s,p,o)` tuple to validation twice; one copy went to `auto_accepted`/`pending_review`, the duplicate went to `rejected`. `test_no_duplicate_triplets_in_result` now passes for terry-v-ohio. Dedup now happens pre-validation using a single `extraction_seen` set.
- **Deleted 8 `benchmarks/*` shim modules** (`run_benchmark.py`, `run_augmented_benchmark.py`, `run_reasoning_benchmark.py`, `ground_truth.py`, `legal_ground_truth.py`, `fitness.py`, `rubric.py`, `scoring.py`). The last four were already dead (package `benchmarks/ground_truth/` and `benchmarks/scoring/` take resolution priority). Updated tests, runner docstrings, and README.
- **`debug_confidence_trace()` helper** in `nodes/planner.py` — returns a per-factor breakdown of the grounding-confidence score (entity tier/score, predicate modifier, ambiguity penalty, final, tier). Useful for Ralph Wiggum diagnosis and the review UI.
- **`QuestionGenLoop` no longer reports vacuous success.** When 0 questions are generated it returns `score=0.0` with a `no_questions_generated` failure diagnosis so the orchestrator will actually trigger prompt mutation.
- **`three_arm_comparison`**: default corpus is now `opinion_golden` (the headline demo), and `--corpus all` prints per-corpus sections instead of a merged dilute report.

### Decisions
- Not re-ingesting opinion text to resurrect `origin='ner_extraction'` rows. Those were purified on purpose; the right move is real opinion-doc ingestion, not backfilling.

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
