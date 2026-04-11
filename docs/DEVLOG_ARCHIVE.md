# Crystal Development Log — Archive

Older entries moved from `DEVLOG.md` to keep the active log short.
Only the most recent ~5 entries stay in `DEVLOG.md`.

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
- **Review UI fixes** (Gradio 6 compatibility)

### Decisions
- Structured API data needs no human approval: the golden answer is the API field value, and the odds of incorrect data from a structured source are negligible
- Batching is unnecessary for auto-accepted data — bulk ingest replaces the cron → review → accept workflow for Tier 0
- 500 cases / ~2,500 questions is the sufficiency threshold for Ralph Wiggum convergence (~50 per predicate)
- Citation resolution requires 2 API calls per cited opinion (opinion → cluster → case_name) — capped at 10 citations/case and 200 cases for first run
- Phase 1c (CourtListener /people/ judge bios) deferred — requires new adapter + entity type

---

## 2026-04-11 — Per-question Caching, Batch API, First Benchmark Run

### What changed
- **Per-question result caching** — `benchmarks/cache.py`: SHA256(question+arm+model) → JSON on disk. Runners check cache before LLM calls, write back after. Rate limits just slow you down, never lose progress. `--clear-cache` flag to reset.
- **Case-name sorting** — Benchmark cases sorted by extracted case name before running. Groups same-case questions together so Gemini's implicit context caching kicks in (same document prefix → cached tokens at ~90% discount).
- **Gemini Batch API** — `src/crystal/llm.py`: `submit_batch()`, `poll_batch()`, `call_llm_batch()`. Separate rate limits from interactive calls, 50% cost discount. `--batch` flag in comparison runner. Falls back to sequential for non-Gemini providers.
- **Multi-provider LLM** — `src/crystal/llm.py`: `LLM_PROVIDER=anthropic` support alongside Gemini. Haiku 4.5 as default Anthropic model ($1/M in, $5/M out).
- **Rubric scoring fix** — `_flatten_kg_results()` normalizes Crystal's nested operation dicts into flat subject/object triples before specificity and grounding scoring.
- **Case name extraction fix** — Strips leading question words ("of", "for", "is") from extracted case names so "of Miranda v. Arizona" correctly becomes "Miranda v. Arizona".

### First benchmark results (Haiku 4.5, 81 cases)
Fair A/B (64 doc-answerable): Naked LLM 46.9% → LLM+Doc 67.2% → Crystal 95.3%.
Crystal-Only (7 KG metadata): 28.6%. Negatives (10): 100% abstention across all arms.

### Decisions
- Haiku 4.5 chosen over Sonnet for benchmarks: ~25x cheaper, sufficient for factual Q&A accuracy evaluation.
- Per-question cache keyed on model tag so switching providers doesn't pollute results.
- 4s sleep between sequential calls (down from 6s) — Anthropic handles higher throughput than Gemini free tier.

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
- **Phase 1: COLD Cases adapter** — `src/crystal/ingest/sources/cold_cases.py`
- **Phase 3: SQLite persistence** — `src/crystal/tools/kg/store.py`
- **Phase 2: CourtListener client** — `src/crystal/ingest/sources/courtlistener.py`
- **Phase 4: Question generator** — `src/crystal/ingest/question_gen.py`
- **Phase 5: Fitness function** — `benchmarks/fitness.py`
- **Phase 6: Ralph Wiggum loop** — `benchmarks/ralph_wiggum.py`
- **Benchmark cases** — 11 positive + 3 negative legal questions

### Decisions
- SQLite before CourtListener (Phase 3 before Phase 2)
- "decided" mapped to date_filed
- No neo4j needed: SQLite handles the scale

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

## 2026-03-16 — D2 Phase 1: KG ingestion pipeline (NER + hand-curated)

### What changed
- New `src/crystal/ingest/` package:
  - `schema.py`: `Triplet` dataclass (subject/predicate/object), `IngestResult` dataclass (triplets + alias maps + source), `merge()` for combining results
  - `ner.py`: spaCy dep-tree-based triplet extraction. Five sentence patterns: copular (`X is Y`, including `of`-flip), possessive (`X has Y`), active transitive (`X verbs Y`), passive (`X was Vd by Y`), prepositional (`X verb PREP Y`). Handles hyphenated entities, determiner stripping, passive predicate text (not lemma).
  - `loader.py`: CSV loader (auto-detects header), JSON loader (object-with-aliases format + flat array format), `load_file()` auto-detect
  - `__init__.py`: top-level `ingest(path)` auto-detects format, `build_kg(result)` converts to `KnowledgeGraph`
  - `__main__.py`: CLI entry point `python -m crystal.ingest <document>` with `--output` and `--format` flags
- Test fixtures: `sample_triplets.csv`, `sample_triplets_no_header.csv`, `sample_triplets.json`, `sample_triplets_flat.json`, `sample_text.txt`
- 53 new tests: `test_schema.py` (7), `test_ner.py` (19), `test_loader.py` (18), `test_integration.py` (9)
- 331/331 passing, 5 skipped

### Decisions
- NER uses dep tree patterns rather than NER entity labels because `en_core_web_sm` misses most fiction entities — noun chunks + dependency structure are more reliable
- Passive predicates use `token.text.lower()` not `token.lemma_` to avoid unintuitive lemma forms ("born" → "bear")
- Copular `of`-flip checks both subject and attr for `prep(of)` to handle both "The capital of X is Y" and "Y is the capital of X"
- Determiner stripping in `_span_text` only (not in `_get_subject_span`) — subjects keep their structure, objects get cleaned
- `_join_tokens` collapses spaces around hyphens to preserve "Dark-ore" from spaCy's "Dark", "-", "ore" tokenization

---

## 2026-03-16 — D6: Reasoning cost benchmark (K-reduction)

### What changed
- `src/crystal/llm.py`: Extracted `_extract_usage()` helper. Now captures `thoughts_token_count` (reasoning tokens) from Gemini usage metadata alongside prompt and output tokens. Computes `total_tokens` sum.
- `src/crystal/metrics.py`: `TokenMetrics` extended with `actual_reasoning_tokens` and `actual_total_tokens`. New `ReasoningComparison` dataclass for per-query grounded-vs-ungrounded comparison with computed properties (`total_token_delta`, `reasoning_token_delta`, savings percentages). New `summarize_reasoning_comparisons()` aggregates per-query data into summary statistics.
- `src/crystal/nodes/llm_nodes.py`: Extracted `_update_metrics_from_usage()` helper. Both augmented and fallback nodes propagate all four token fields (prompt, output, reasoning, total).
- `benchmarks/run_reasoning_benchmark.py`: New benchmark runner. Runs identical queries through both naked LLM and Crystal-grounded pipeline using a thinking-capable model (default: gemini-2.5-flash). Records per-query: accuracy, prompt/output/reasoning/total tokens. Reports accuracy delta and token savings. Handles kg_answerable (LLM bypass = 0 tokens) and kg_augmented paths. CLI with `--model` and `--cases` flags.
- 23 new unit tests: `_extract_usage` (4), `_update_metrics_from_usage` (3), `ReasoningComparison` (9), `summarize_reasoning_comparisons` (5), `TokenMetrics` reasoning fields (2)
- 278/278 passing, 5 skipped

### Decisions
- Reasoning benchmark uses existing `BENCHMARK_CASES` from ground_truth.py rather than waiting for D4 — Remulak cases exercise both kg_answerable and kg_augmented paths, sufficient for initial K-reduction measurement
- kg_answerable cases counted as 0 grounded tokens (LLM fully bypassed) — this is the strongest form of K-reduction
- `_extract_usage()` made a module-level function (not method) so the reasoning benchmark can call it directly without going through `call_llm()`

---

## 2026-03-16 — D5: Entity aliases + fuzzy matching + multi-hop traversal

### What changed
- New `src/crystal/tools/kg/fuzzy.py`: `fuzzy_match()` using `rapidfuzz.fuzz.token_sort_ratio`
- `graph.py`: added `entity_aliases` param, `_resolve_entity()` (exact → alias → fuzzy cascade), `_resolve_predicate_fuzzy()` (same cascade for predicates), `subjects` property, `traverse()` (BFS depth-limited multi-hop, default depth=2)
- `data/remulak.py`: `ENTITY_ALIASES` dict (~20 entries) mapping short forms to canonical entities
- `detectors/kg.py`: `find_entity_spans()` now uses 3-tier cascade with spaCy noun phrase extraction as fallback, length-ratio guard against derived forms (e.g., "Remulakian" ≠ "Remulak"), `detect_kg_query()` accepts `multi_hop` and `max_depth` params, detection results include `match_tier`, `match_score`, `original_text`, `predicate_match_tier`
- `QUESTION_PREDICATE_MAP`: added `"long last" → "duration"`, `"long" → "duration"`
- `requirements.txt`: added `rapidfuzz>=3.0.0`
- Golden test cases: 3 alias cases + 2 fuzzy string cases
- Unit tests: `test_fuzzy.py` (8 tests), expanded `test_kg.py` (entity aliases, fuzzy entity, fuzzy predicate, multi-hop traversal, subjects property — 20+ new tests), expanded `test_kg.py` detector tests (match_tier, alias detection, fuzzy detection, multi-hop detection)
- 255/255 passing, 5 skipped

### Decisions
- Length-ratio guard (0.7–1.3) on fuzzy entity matches from noun phrase extraction prevents derived adjectives ("Remulakian") from fuzzy-matching their base entity ("Remulak")
- Entity aliases include article variants ("the fracture war" alongside "fracture war") because spaCy noun phrase extraction includes determiners
- Multi-hop is opt-in (`multi_hop=False` default) — caller decides when to expand; keeps subject_scan as the default fallback for backward compatibility
- Multi-hop uses BFS with visited set for cycle prevention, collects only facts where objects are also KG subjects (avoids expanding leaf values like "Zelphos" that have no forward facts)

---

## 2026-03-20 — D4: Augmented output quality benchmark (thin)

### What changed
- `benchmarks/ground_truth.py`: New `AUGMENTED_BENCHMARK_CASES` — 5 KG augmented + 3 math augmented cases with `match_strings` that prove grounding worked (fictional KG values a naked LLM can't produce).
- `benchmarks/run_augmented_benchmark.py`: Benchmark runner comparing naked LLM vs. full Crystal pipeline (real LLM calls) on augmented paths. Scores both with D1 rubric (accuracy, specificity, no-hallucination). Side-by-side report + JSON output. CLI: `python -m benchmarks.run_augmented_benchmark`.
- `tests/golden/test_cases.py`: `KG_AUGMENTED_CASES` now have expected result strings (previously `None`).
- 20 new tests in `test_augmented_benchmark.py`: case format validation (4), scoring with synthetic responses (5), summary aggregation (3), routing verification for KG cases (5) and math reasoning signals (3).
- 371/371 passing, 5 skipped.

### Decisions
- Treatment arm uses the full `build_crystal_graph()` pipeline (not manual detection) — exercises the real code path including LLM call for augmented routes.
- Baseline is scored against treatment's KG results for rubric specificity/grounding — otherwise there's no rubric baseline for comparison since naked LLM has no KG context.
- Match strings are intentionally minimal (one KG value per case). The rubric dimensions handle deeper quality measurement; binary accuracy just gates "did grounding reach the output at all."

---

## 2026-03-17 — D3: Web UI (Gradio)

### What changed
- **KG injection into pipeline**: Added `kg` field to `CrystalState`, `make_initial_state(prompt, *, kg=None)` accepts optional KG override, `kg_detection_node` reads from `state["kg"]` or falls back to `remulak_kg`. Fully backward compatible.
- **Gradio UI** at `src/crystal/ui/`:
  - `app.py`: Two-tab layout. "Ask" tab: question input → side-by-side Crystal vs. naked LLM with route/token metadata. "Knowledge Graph" tab: file upload (CSV/JSON/TXT), paste text for NER extraction, reset to Remulak, scrollable facts table.
  - `__main__.py`: `python -m crystal.ui` entry point
  - Pre-loaded Remulak KG works out of the box, `gr.State` manages active KG across tabs
- `requirements.txt`: added `gradio>=5.0.0`
- 20 new tests: `test_kg_injection.py` (5), `test_ui_helpers.py` (12), plus 3 structural
- 351/351 passing, 5 skipped

### Decisions
- KG injection via state field (not factory/closure) — minimal invasive change, one line in detection node, full backward compat
- Gradio Blocks over Streamlit — event-driven model maps directly to "user clicks Ask → both pipelines run → two boxes fill in", avoids rerun model side effects
- UI is pure wiring — zero business logic in `app.py`, all calls go through existing `build_crystal_graph()`, `ingest()`, `build_kg()`, `call_llm()`

---

## 2026-03-14 — Subject-scan fallback → kg_augmented

### What changed
- KG detector: detection result now includes `lookup_type` field (`"targeted"` or `"subject_scan"`)
  - `targeted`: extracted predicate matched a KG predicate → specific fact(s)
  - `subject_scan`: predicate extraction failed or didn't match → all entity facts
- Compiler: `subject_scan` lookups route as `kg_augmented` (inject entity facts as context for LLM reasoning) instead of `kg_answerable` (dump facts, skip LLM)
- `lookup_type` propagated through planner → preprocessor → execution node → tool_results
- Benchmark runner: `kg_augmented` cases produce augmented prompts (not raw fact dumps)
- Golden test cases: adversarial negatives updated from `kg_answerable` to `kg_augmented`
- New unit tests: `test_targeted_lookup_type`, `test_subject_scan_lookup_type`, `test_alias_is_targeted`, `test_kg_augmented_subject_scan`, `test_kg_augmented_reasoning_signals_override`, `test_kg_answerable_no_lookup_type_backward_compat`
- 210/210 passing, 5 skipped

### Decisions
- Predecessor case ("Who was the leader before Grand Vizier Korth?") — KG has `predecessor: Vizier Aamra Sel` but predicate extraction yields "leader before" which doesn't match. Stays as subject_scan → `kg_augmented`. Fuzzy matching (D5) would fix predicate resolution.
- Missing `lookup_type` in tool_results (backward compat) defaults to `kg_answerable`, not `kg_augmented` — safer for existing code paths.
- `kg_augmented` from subject_scan vs. reasoning signals is the same classification — both inject facts for LLM. The distinction matters for understanding *why* it was routed that way.

---

## 2026-03-14 — D1: Minimal Quality Rubric

### What changed
- New `benchmarks/rubric.py`: `accuracy_score()`, `specificity_score()`, `grounding_score()`, `calibration_score()`, `score_rubric()`, `RubricResult` dataclass
- Extended `benchmarks/ground_truth.py`: added `is_negative` field (4th tuple element) to all 20 existing cases + 5 adversarial negative cases (GDP, predecessor, second city, oceans, Zelphos population)
- Updated `benchmarks/scoring.py`: added `score_batch_rubric()` that returns per-dimension averages alongside binary correct/incorrect
- Updated `benchmarks/run_benchmark.py`: uses `score_batch_rubric()`, prints per-dimension breakdown and comparison
- New `tests/unit/test_rubric.py`: 25 unit tests covering all scorers + batch integration + backward compatibility
- Extended `tests/golden/test_cases.py`: added `KG_ADVERSARIAL_NEGATIVES` (5 cases)
- Extended `tests/integration/test_pipeline.py`: parametrized tests for adversarial negatives
- 204/204 passing, 5 skipped

### Decisions
- Adversarial negatives where entity exists but predicate doesn't still classify as `kg_answerable` (entity found → subject scan returns all facts). Rubric calibration dimension evaluates quality separately.
- Added "no kg match" to `ABSTENTION_PHRASES` since the treatment pipeline emits `[NO KG MATCH]` for negative cases
- `specificity_score` returns 1.0 for empty `kg_results` (vacuously satisfied) — avoids penalizing non-KG paths

---

## 2026-03-14 — Roadmap Restructure: Demo Phase

### What changed
- Restructured `TODO.md` from MVP/2.0 into three tiers: MVP (complete), Demo, Future
- Demo phase: 5 tasks (D1–D5) focused on making the value proposition experienceable — quality rubric, KG ingestion pipeline, web UI, augmented benchmarks, fuzzy matching
- Bifurcated `PLAN_GRADING_RUBRIC.md` into Phase 1 (3 dimensions: accuracy, specificity, no-hallucination) and Phase 2 (full 6-metric weighted composite)
- Bifurcated `PLAN_FUZZY_MATCHING.md` into Phase 1 (entity aliases + rapidfuzz, no model downloads) and Phase 2 (embedding similarity via sentence-transformers)
- Promoted fuzzy matching (D5) into Demo tier — exact-match-only entity resolution is a dealbreaker for user-supplied documents

### Decisions
- KG construction (offline batch/ETL) is architecturally separate from query-time pipeline — using an LLM for extraction does not undermine Crystal's deterministic query-time value
- Quality rubric ships with demo but doesn't block it; fuzzy matching is a prerequisite
- Suggested implementation order: D5 (fuzzy) → D2 (ingestion) → D3 (UI) → D1/D4 (rubric + benchmarks)

---

## 2026-03-14 — MVP Complete: KG Integration + Benchmarks

### What changed
- `CrystalState` schema: added `kg_detections`, `kg_results`, `kg_entities_found` fields
- KG detection node: removed math-priority gate — both math and KG detectors now fire independently
- KG execution node: populates new `kg_results` state field
- Compiler: added `kg_augmented` path (grounded KG facts + reasoning signals → LLM)
- KG detector: added `QUESTION_PREDICATE_MAP` entries (`born → birthplace`, `known → known for`), `why` to question words, subject-entity preference in entity selection
- Metrics: `kg_augmented` treated same as `math_augmented` for savings calculation
- Golden tests: expanded from 8 → 20 KG answerable cases, added 3 KG augmented + 3 KG negative cases
- Benchmark infrastructure: `benchmarks/` with ground truth (20 questions), auto-scoring, baseline and treatment runners
- Treatment benchmark: **100% accuracy** (20/20)
- 167/167 passing, 5 skipped

### Decisions
- `kg_augmented` uses same reasoning signals as `math_augmented` — keeps logic consistent
- Benchmark scoring: substring match (case-insensitive, all match strings must appear)
- Entity selection prefers KG subjects over objects to avoid picking object-only entities as primary

---

## 2026-03-14 — Module Restructuring

### What changed
- Moved `detectors/calculator.py` + `detectors/semantic.py` into `detectors/math/` subfolder
  - `calculator.py` → `explicit.py`, `semantic.py` unchanged
  - `__init__.py` re-exports all public symbols
- Moved `tools/kg.py` + `tools/remulak_kg.py` into `tools/kg/` subfolder
  - `kg.py` → `graph.py`, `remulak_kg.py` → `remulak.py`
- Renamed `nodes/detector.py` → `nodes/math_detection.py`
- Renamed `nodes/kg_detector.py` → `nodes/kg_detection.py`
- Function names: `math_detection_node`, `kg_detection_node`
- Graph node labels: `"math_detection"`, `"kg_detection"`
- 149/149 passing, 5 skipped

### Decisions
- `semantic.py` grouped with calculator (same detection pipeline)
- Node files use `_detection` (noun) not `_detector` (agent noun) to avoid
  confusion with the actual detector modules in `detectors/`

---

## 2026-03-08 — Three-View Savings Metrics

### What changed
- Three savings views: `token_savings_pct` (billing), `marginal_savings_pct` (contextual cost), `savings_pct` (legacy)
- New `marginal_cost()` function, `estimate_metrics()` accepts `base_context` param
- 8 new tests; 95/95 passing

### Decisions
- Isolated N+N² overstated penalties ~5x; kept for backwards compat, clearly labelled

---

## 2026-03-08 — Token Metrics & math_answerable Bypass

### What changed
- Split `math_in_context` → `math_answerable` (LLM bypass) + `math_augmented` (LLM required)
- `REASONING_SIGNALS` set gates augmented path; default is bypass
- tiktoken-based metrics, Gemini `usage_metadata` integration
- 3 new golden cases, 13 new tests; 93/93 passing

### Decisions
- Conservative bypass: only advisory/explanatory/comparative/predictive words trigger LLM

---

## 2026-02-15 — Project Scaffolding & Verb-Semantic Detection

### What happened
- Migrated from single-file Colab prototype to proper Python package structure
- Decomposed monolithic `crystal_router.py` into modular `src/crystal/` package
- Implemented verb-semantic detection for implied math word problems
- Added three-way prompt classification (pure_math / math_in_context / no_match)
- Created comprehensive test suite with golden test cases
- Added `.cursorrules` following minimal-requirements approach (per ETH Zurich paper on context files)

### What works
- All 4 explicit addition patterns (verb, conjunction, noun, symbol): 14/14 passing
- Semantic verb detection with acquire/lose/state classification
- Zero false positives on all 14 negative test cases
- Pure math queries skip LLM entirely
- Prompt compiler injects results into simplified prompts for LLM

### Known bugs
- **Semantic verb conj-skip needs verification:** The fix to skip conjoined verb
  children is in `detectors/semantic.py` but was never confirmed working (Colab
  module cache issue). First priority: run pytest and check these cases:
  - "Adam has 10 chairs, sells 6, and then makes 7 more" → expected: 11
  - "I had 100 dollars, earned 50, and spent 30" → expected: 120

### Known limitations
- Calculator only supports addition (explicit) and add/subtract (semantic verbs)
- Semantic verb list intentionally small (~15 verbs)
- LLM integration untested (Gemini rate limits on free tier)

---

## 2026-03-08 — LLM Test Infrastructure

### What happened
- Added `--run-llm` pytest flag: LLM integration tests are skipped by default,
  opt-in with `pytest --run-llm`
- Added `cached_llm` fixture: caches real Gemini responses to
  `tests/fixtures/llm_cache.json` so only the first run hits the API
- Created `tests/test_llm_integration.py` with tests for all three graph paths
  (augmented, fallback, direct return)
- Made Gemini client lazy-init in `src/crystal/llm.py` so imports work without
  `GOOGLE_API_KEY` set (was crashing test collection)

### Decisions made
- LLM tests gated behind a flag rather than always-on — prevents accidental
  API spending and rate limiting during normal development
- Cache keyed by SHA-256 of prompt text; delete the JSON file to force refresh

### Known issues
- `tests/fixtures/llm_cache.json` does not exist yet; first `pytest --run-llm`
  invocation will create it (requires `GOOGLE_API_KEY` env var)
