# Crystal Development Log — Archive

Older entries moved from `DEVLOG.md` to keep the active log short.
Only the most recent ~5 entries stay in `DEVLOG.md`.

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
