# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **Extraction baselines established.** NER-only: 59.1% accuracy (13/22). NER+LLM: 63.6% (14/22). 2,575 vs 3,241 KG facts from 15 SCOTUS opinions.
- **Review pipeline ready.** Document → extract → generate questions → Crystal proposes → human verifies → review batch → Ralph Wiggum. First batch: `review/batch_doc_20260412_103346.json` (7 questions from 5 landmark cases).
- **Test count:** 769 unit tests passing.
- **Product direction:** Crystal (engine) + pre-built legal KG (scaffold) + customer document uploads.
- **Path to demo:** (1) Get extraction to 80%+ via Ralph Wiggum iterations, (2) expand scaffold 543→5K subjects, (3) pick practice area + curate demo batch, (4) three-arm comparison → marketing materials.
- **Next:** Human review of proposed answers → Ralph Wiggum → re-benchmark → iterate.

---

## 2026-04-12 — Extraction Baselines + Review Pipeline

### What changed
- **NER-only extraction baseline** — 59.1% accuracy (13/22) on 15 SCOTUS opinions, 2,575 KG facts. NER handles court, judges, attorneys for well-known cases. Fails on citation counts (metadata-only), opinion authors for obscure cases, and older opinion formats.
- **NER+LLM extraction** — 63.6% accuracy (14/22), 3,241 facts (+26%). LLM adds 666 triplets. Fixed Pennsylvania Railroad court question. Remaining 8 failures are extraction quality issues (opinion author, court for obscure cases) and one routing bug (Coventry returned date_filed instead of court on kg_answerable route).
- **Review pipeline** — `src/crystal/ingest/review_pipeline.py`: CLI tool ingests documents, generates questions from extracted facts, has Crystal propose answers, saves as review batch. `review.py`: new `save_proposed_as_batch()` + `_derive_match_strings()`. UI: editable "Golden Answer" column + "Save to Review" button in Ingest Documents tab.
- **Question generator hardened** — `src/crystal/compare.py`: only generates questions for known predicate templates (court, date_filed, judges, etc.). `_is_plausible_case_name()` filters NER noise subjects (pronouns, common nouns, "this case", "justice X"). Prevents garbage questions like "What is the take of I?".
- **Switched to Anthropic Haiku** for benchmarks — Gemini 2.5 Flash had severe rate limiting (5 retries/question, ~3 min each). Haiku completes 22 questions in ~100s.
- **10 new review tests, 6 updated compare tests** — 769 unit tests passing.

### Decisions
- Question generation rejects unknown predicates entirely instead of generating "What is the {pred} of {subj}?" — the NER dep-tree predicates ("deliver of", "yield to of") produce meaningless questions.
- Review batch format extends existing batch_*.json schema with `crystal_proposed`, `crystal_route`, `crystal_confidence` fields so the user can see what Crystal thought vs their correction.
- Path to demo crystallized: extraction quality iterations → scaffold density → demo batch → marketing. Structured metadata is table stakes; citation network and hallucination prevention are the selling points.

---

## 2026-04-12 — Extraction Quality + Metrics Overhaul

### What changed
- **Simplified benchmark metrics** — `benchmarks/scoring/rubric.py`: Removed `specificity_score()` and `grounding_score()` from the active rubric. `RubricResult` now has `accuracy` + `abstention` (was accuracy + specificity + no_hallucination). All runners (comparison, baseline, augmented) updated. Report prints accuracy, hallucination rate, and token cost — the three metrics customers care about.
- **Extraction quality benchmark** — `benchmarks/extraction_quality.py`: Runs `ingest_document()` on 15 cached SCOTUS opinions (NER or NER+LLM), builds an ephemeral `SqliteKnowledgeGraph` from only extracted facts, then runs benchmark questions through the Crystal pipeline against this document-only KG. Directly measures "can Crystal learn from documents and answer questions?"
- **ExtractionLoop** — `benchmarks/ralph_wiggum/extraction_loop.py`: New Ralph Wiggum loop targeting ingestion quality. Evaluates by comparing extracted triplets against CourtListener ground truth. Four diagnosis categories: subject_mismatch, predicate_mismatch, missing_fact, hallucinated_fact. Mutation targets: `LEGAL_EXTRACTION_PROMPT` hints, `LEGAL_PREDICATE_ALIASES` additions. Wired into Orchestrator.
- **Crystal-proposed answers in UI** — `src/crystal/ui/app.py`: New "Crystal's Proposed Answers" section in Ingest Documents tab. After ingestion, generates questions from extracted facts, runs each through Crystal with the updated KG, shows results in a table (Question, Crystal Answer, Route, Confidence, Expected). Verifies the system is actually learning post-ingestion.
- **25 new tests** — 11 extraction quality, 14 extraction loop.

### Decisions
- Dropped specificity/no_hallucination rather than fixing them — they measured KG triple regurgitation, not answer quality. The asymmetry (vacuously 1.0 when `kg_results` empty) made them misleading for naked LLM and LLM+doc arms.
- Extraction benchmark uses `SqliteKnowledgeGraph(":memory:")` so each run starts fresh with no scaffold contamination.
- ExtractionLoop doesn't inherit from BaseLoop in the usual way (it has a different iteration model — extract + compare vs pipeline eval) but follows the same interface contract.

---

## 2026-04-11 — Document Ingestion MVP (Phase 2a)

### What changed
- **Ingestion confidence scorer** — `src/crystal/ingest/confidence.py`: `score_ingestion_confidence()` scores 0.0–1.0 based on extraction source (NER=0.85, LLM high=0.80, medium=0.55, low=0.30), entity known bonus (+0.10), predicate alignment bonus (+0.10). `ScoredTriplet` dataclass with conversion helpers. `INGEST_AUTO_ACCEPT = 0.70`.
- **Legal-tuned extraction prompt** — `src/crystal/ingest/llm_extract.py`: `LEGAL_EXTRACTION_PROMPT` lists preferred predicates (court, date_filed, judges, opinion_author, cites, etc.), instructs "Party v. Party" format, extraction of citation relationships. `normalize_predicate()` fuzzy-maps extracted predicates to canonical ontology forms. `extract_triplets_llm()` accepts `domain="legal"` parameter.
- **Document ingestion orchestrator** — `src/crystal/ingest/__init__.py`: `ingest_document()` runs NER + optional LLM extraction, scores all triplets, auto-accepts above threshold, deduplicates against existing KG, inserts into SQLite via `bulk_insert()`. `DocumentIngestionResult` with `accept_pending()`, `reject_pending()`, `accept_all_pending()` methods.
- **UI Ingest Documents tab** — `src/crystal/ui/app.py`: new tab between Ask and KG with multi-file upload, paste text, configurable auto-accept threshold slider, extraction results panel (stats, auto-accepted table, pending review table), bulk accept/reject buttons.
- **Before/after comparison** — `src/crystal/compare.py`: `before_after_comparison()` runs questions through Crystal + KG, LLM + docs, and naked LLM. `generate_questions_from_triplets()` auto-generates questions from extracted facts. Wired into UI as "Test Your Ingestion" section.
- **Real document validation** — `tests/integration/test_ingest_validation.py`: 8 tests validate extraction on 5 real SCOTUS opinions. 543 triplets extracted, all auto-accepted (NER-only path). Validation report with per-case stats and predicate coverage.
- **21 confidence tests, 15 extraction tests, 14 orchestrator tests, 11 comparison tests** — comprehensive unit coverage for all new components.

### Decisions
- NER base confidence (0.85) means NER extractions with aligned predicates (0.95) or known entities (0.95) always auto-accept. LLM "medium" (0.55) stays below threshold unless both bonuses apply.
- Path vs text detection in `ingest_document()` uses length + newline heuristic (not `Path.exists()` on arbitrary strings) to avoid OS errors on long text inputs.
- Before/after comparison uses Crystal's full pipeline (graph.invoke), not a simplified path — ensures the demo shows real production behavior.
- LLM extraction is opt-in via `call_llm_fn` parameter; NER-only mode is the fast default for development and testing.

---

## 2026-04-11 — Ralph Wiggum v3: Multi-Loop Architecture

### What changed
- **Decomposed monolith** — `benchmarks/ralph_wiggum.py` (single 1,048-line file) → `benchmarks/ralph_wiggum/` package with 6 modules:
  - `base.py`: BaseLoop ABC with shared evaluation, diagnosis, scoring, reporting, git ops, pipeline runner
  - `predicate_loop.py`: PredicateLoop — owns `predicate_mismatch` → mutates `QUESTION_PREDICATE_MAP` + `LEGAL_PREDICATE_ALIASES`
  - `entity_loop.py`: EntityLoop — owns `entity_mismatch` → mutates entity alias tables
  - `threshold_loop.py`: ThresholdLoop — owns `routing_error` → mutates `CONFIDENCE_LOW` (bounded [0.5, 0.85])
  - `orchestrator.py`: Orchestrator runs all loops in sequence, produces unified report
  - `__main__.py`: CLI with `--loop` flag to run individual or all loops
- **Each loop is tightly contained**: owns exactly one `FailureCategory` set, exactly one set of `TARGET_FILES`, its own `_validate_proposal()`, `_apply_proposal()`, `_revert_proposal()`, and `_build_proposal_prompt()`.
- **Backward-compatible `__init__.py`**: old `RalphWiggumLoop`, `_validate_proposal`, `_apply_proposal`, `_parse_llm_proposal`, `_update_threshold`, `_build_change_report` all still importable from `benchmarks.ralph_wiggum`.
- **19 new tests** covering orchestrator, per-loop metadata, per-loop validation, `_my_failures` filtering. All existing tests preserved and passing.

### Decisions
- Each loop filters failures with `_my_failures()` so it only proposes changes for its own domain — no accidental cross-contamination.
- Orchestrator runs loops in sequence (PredicateLoop → EntityLoop → ThresholdLoop) because earlier loops may fix issues that change the failure distribution for later ones.
- EntityLoop `_apply_proposal` currently only logs proposed aliases (requires manual KG addition) — this keeps the entity alias mutation safe and auditable.
- Backward compat shim in `__init__.py` provides a unified `_validate_proposal()` that accepts all loop formats, so existing tests pass unchanged.

---

## 2026-04-11 — Pipeline Safety Guarantees + Ralph Wiggum v2

### What changed
- **Unified confidence scoring** — `src/crystal/nodes/planner.py`: replaced binary `_kg_detection_is_confident()` with `score_grounding_confidence()` returning a 0.0–1.0 float. Factors: entity match tier (piecewise bands for fuzzy), predicate specificity (targeted vs subject_scan), entity ambiguity penalty. Three bands: HIGH (≥0.9), MEDIUM (0.7–0.9), LOW (<0.7 → LLM fallback). `grounding_confidence` added to `CrystalState`.
- **Qualified prompt framing** — `src/crystal/nodes/compiler/kg.py`: HIGH confidence → "verified from the knowledge graph", MEDIUM → "possibly relevant, use your own judgment". Medium forces `kg_augmented` route (never `kg_answerable`) so LLM can cross-check. `_build_kg_augmented_prompt` accepts `grounding_confidence` kwarg.
- **Graceful error degradation** — `src/crystal/graph.py`: `_safe_node()` wrapper on all pre-LLM nodes. Any exception → `fallback_to_llm=True` with `prompt_type="no_math"` and `compiled_prompt=raw_prompt`, so downstream routing degrades cleanly to LLM.
- **Contract test suite** — `tests/integration/test_never_worse.py`: 52 tests covering KG answerable, KG augmented, alias resolution, negatives, adversarial negatives, wrong-entity protection, and graceful degradation (broken KG detection, broken compiler).
- **Ralph Wiggum v2** — `benchmarks/ralph_wiggum.py` rewritten:
  - Full-pipeline evaluation via `graph.invoke()` (v1 `detect_kg_query`-only mode available with `--legacy`)
  - Component-level failure diagnosis: `diagnose_failure()` classifies into entity_mismatch, predicate_mismatch, routing_error, framing_error, math_false_positive, no_detection
  - Expanded mutation targets: predicate maps, predicate aliases, entity aliases (additions only), confidence threshold (bounded [0.5, 0.85])
  - Autonomous loop produces `ralph_report.md` change report for human review
  - `_build_change_report()` summarizes iterations, diagnoses, proposed changes, remaining failures

### Decisions
- Entity confidence dominates the scorer via piecewise bands (not linear mapping): <90% fuzzy = 0.4, 90-95% = 0.8, ≥95% = 0.95. This preserves the original 90% threshold behavior while allowing nuanced routing.
- `_safe_node` sets all fields needed for clean downstream routing — not just `fallback_to_llm` but also `prompt_type`, `compiled_prompt`, `final_response`. This handles crashes in post-planner nodes where the graph can't re-route to `llm_fallback_node`.
- Confidence threshold mutation bounded at [0.5, 0.85] — low enough to be useful for experimentation but can't accidentally disable the safety gate entirely.

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
