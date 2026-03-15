# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **MVP complete:** All 8 MVP tasks done. KG integration, benchmarks, golden tests all passing.
- **Next milestone:** Demo phase — make the value proposition experienceable. Minimal quality rubric (D1), KG ingestion pipeline (D2), web UI (D3).
- **Key insight:** KG construction (offline batch/ETL) is a separate concern from query-time pipeline. LLM-based extraction for KG building does not undermine Crystal's value — the KG is a verified artifact at query time.
- **math_augmented cost:** compiled prompts are longer than raw — by design, but worth revisiting.

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
