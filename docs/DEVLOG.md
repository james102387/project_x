# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **D1 complete:** Minimal quality rubric + subject-scan → `kg_augmented` routing.
- **Next milestone:** D2 (KG ingestion pipeline), D3 (Web UI), D5 (fuzzy matching).
- **Key architectural insight:** Entity-found/predicate-not-found queries now route as `kg_augmented` (definitional grounding for LLM) instead of `kg_answerable` (raw fact dump). The `lookup_type` field in the detection result drives this.
- **Test count:** 210 passing, 5 skipped.

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
