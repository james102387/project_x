# Crystal Development Log

Reverse-chronological record of changes, decisions, and known issues.
Point the LLM at this file at the start of each session for full context.

---

## 2026-03-08 — Token Metrics & math_answerable Bypass

### What happened
- Split `math_in_context` into `math_answerable` (LLM bypass) and `math_augmented`
  (LLM required). All 11 former math_in_context golden cases are now math_answerable.
- Added `REASONING_SIGNALS` set in compiler: advisory/explanatory/comparative/predictive
  words trigger math_augmented; everything else bypasses the LLM.
- Added `src/crystal/metrics.py`: tiktoken-based token counting, N + N^2 compute proxy,
  `TokenMetrics` dataclass, `estimate_metrics()` function.
- `call_llm` now returns `(text, usage_dict)` tuple with Gemini `usage_metadata`.
- LLM nodes merge actual API token counts into `state["token_metrics"]`.
- `scripts/run.py` displays token savings per prompt and summary by type.
- Fixed `llm_fallback_node` not setting `prompt_type` (was empty string).
- Fixed `llm_nodes.py` import binding so `cached_llm` monkeypatch works.
- Seeded `tests/fixtures/llm_cache.json` so LLM tests pass without API key.
- 3 new `math_augmented` golden cases, 13 new tests. 93/93 passing.

### Decisions made
- `math_answerable` returns bare numeric result (same as `pure_math`)
- `REASONING_SIGNALS` is conservative: default is bypass, only explicit reasoning
  demands go to LLM
- tiktoken `cl100k_base` for token counting — approximate but sufficient for
  relative comparisons

### Known issues
- `math_augmented` compiled prompts are longer than raw prompts (negative savings)
  — this is expected; the value is accuracy, not token reduction
- LLM cache contains canned responses, not real Gemini output

### Next steps
1. Run with real API key to populate cache with actual Gemini responses
2. Measure accuracy delta: math_answerable direct return vs LLM-narrated response
3. Begin KG tool implementation

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

### Next steps
1. Run `pytest --run-llm` once with a valid API key to populate the cache
2. Decide whether to commit the cache file to git

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

### What's next
1. Run pytest, fix any failing golden tests (especially conj-skip cases)
2. Get LLM API working (add credits or switch provider)
3. Run accuracy comparison: tool-augmented vs pure LLM
4. Begin KG tool implementation

---

## Session Template

```
## YYYY-MM-DD — [Title]

### What happened
-

### Decisions made
-

### Known issues
-

### Next steps
-
```
