# Crystal Development Log — Archive

Older entries moved from `DEVLOG.md` to keep the active log short.
Only the most recent ~5 entries stay in `DEVLOG.md`.

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
