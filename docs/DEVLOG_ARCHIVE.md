# Crystal Development Log — Archive

Older entries moved from `DEVLOG.md` to keep the active log short.
Only the most recent ~5 entries stay in `DEVLOG.md`.

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
