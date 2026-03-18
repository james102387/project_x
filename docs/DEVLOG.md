# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **D1, D5, D6 complete.** Reasoning cost benchmark infrastructure in place.
- **Next milestone:** D2 (KG ingestion pipeline), D3 (Web UI), D4 (augmented benchmark cases).
- **Key architectural additions:** `call_llm()` captures reasoning tokens. `ReasoningComparison` + `summarize_reasoning_comparisons()` for K-reduction analysis. Benchmark runner at `benchmarks/run_reasoning_benchmark.py`.
- **Test count:** 278 passing, 5 skipped.

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
