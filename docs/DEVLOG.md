# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **Document Ingestion MVP (Phase 2a) implemented.** Full pipeline: NER + LLM extraction → confidence scoring → auto-accept/review → SQLite KG persistence. Progressive trust model with configurable threshold.
- **New UI "Ingest Documents" tab.** Multi-file upload, extraction with progress, auto-accepted/pending review tables, bulk accept/reject, before/after comparison.
- **Legal-tuned extraction prompt.** `LEGAL_EXTRACTION_PROMPT` steers LLM toward ontology predicates. `normalize_predicate()` maps extracted predicates to canonical forms.
- **Validated on real SCOTUS opinions.** 543 triplets extracted from 5 opinions (Brown, Gideon, Loving, Terry, Hibbs). All NER-based extractions auto-accepted at 0.85+ confidence.
- **Test count:** 1038 passing, 5 skipped.
- **Scaffold hit rate analysis:** 20.7% of case citations in 323 cached opinions match KG entities. 39.9% of documents have ≥1 KG link. Top-connected: Berisha (28 matches), Smith v. Arizona (19), Citizens United (13).
- **Product direction:** Crystal (engine) + pre-built legal KG (scaffold) + customer document uploads. Scaffold provides day-one value; segmented by jurisdiction/practice area with delta updates via webhook.
- **Next:** Expand scaffold from 543 to ~5K subjects (target 70%+ hit rate). Run LLM extraction on real opinions. Phase 1c (judge bios). Segmented KG distribution.

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

## 2026-04-11 — Planner Confidence Gate ("Never Worse Than LLM")

### What changed
- **Confidence gate in `plan_builder_node`** — `src/crystal/nodes/planner.py`: `_kg_detection_is_confident()` checks entity match quality before using KG results. Exact and alias matches are always trusted. Fuzzy matches require `match_score >= 90.0` (threshold: `KG_FUZZY_CONFIDENCE_THRESHOLD`). Below that, the KG detection is filtered out and Crystal falls back to the raw LLM.
- **17 unit tests** — `tests/unit/nodes/test_planner.py`: covers exact/alias/fuzzy confident, fuzzy below threshold, mixed detections, math-survives-filtered-KG, all-low-confidence-falls-back.
- **4 integration tests** — `tests/integration/test_confidence_gate.py`: end-to-end LangGraph pipeline with custom KG and mocked LLM. Verifies exact match → KG facts, missing entity → LLM fallback, close-wrong-entity → safe fallback, no cross-contamination.

### Decisions
- Gate is in the planner (not in `detect_kg_query`) so detection remains sensitive while the trust decision is centralized. The detection still captures fuzzy matches for diagnostics; the planner decides whether to act on them.
- Threshold at 90 (not 85): the "never worse than LLM" contract means false negatives (rejecting a valid fuzzy match → LLM answers) are always safer than false positives (accepting a wrong entity → confidently wrong answer). A rejected fuzzy match just means the LLM answers instead — by definition not worse.
- The 28.6% Crystal-Only failure in the benchmark was actually the LLM fallback path working correctly — entities weren't in the KG, detection returned None, LLM answered but couldn't provide exact citation counts. The confidence gate adds protection for the edge case where a fuzzy match DOES fire (80-89% score) but resolves to the wrong entity.

---

## 2026-04-11 — B1-B6 Demo Benchmark Implementation

### What changed
- **B1: Opinion text downloader** — `scripts/download_opinions.py`: searches CourtListener by case name → cluster → lead opinion, fetches `html_with_citations`, strips HTML to plain text, caches to `benchmarks/documents/{slug}.json`. `benchmarks/documents.py`: loader utilities (`load_opinion`, `load_all_opinions`, `opinion_token_estimate`, `list_cached_opinions`). Rate-limited at 0.6s/request. 16 tests.
- **B2: Document-answerability audit** — `benchmarks/answerability.py`: classifies predicates as document-answerable (court, date_filed, judges, opinion_author, cites, attorneys) vs KG-only (cited_by_count, precedential_status, per_curiam). `partition_cases()` splits benchmark cases into 4 buckets. Regex-based predicate inference from question text. 18 tests.
- **B3: Document-context baseline runner** — `benchmarks/runners/document.py`: Arm 2 (LLM + real opinion text). Extracts case name from question (handles "v." names + citation formats), looks up cached opinion, builds realistic prompt. Injectable `call_llm_fn` for testing. 13 tests.
- **B4: Obscure long-tail cases** — `scripts/select_obscure_cases.py`: queries SQLite KG for low-citation cases not in SCOTUS_SAMPLE. 15 obscure cases added (Coventry Health Care v. Nevils, Flanders v. Seelye, Gray v. Coan, Richards v. Mackall, etc.). 32 new benchmark questions in `benchmarks/ground_truth/legal.py`. 15 fixture records in `tests/fixtures/scotus_sample.py`.
- **B5: Stratified sampler** — `benchmarks/sampling.py`: `sample_from_review_cases()` groups by predicate from `source_triplet`, proportional sampling with minimum per stratum. `sample_benchmark_cases()` for tuple-format cases. Export/load for Tier 2 caching. 15 tests.
- **B6: Three-arm comparison report** — `benchmarks/runners/comparison.py`: orchestrates all 3 arms across partitioned cases (fair A/B on document-answerable, Crystal-only on KG metadata, abstention on negatives). Results cached per-arm to JSON. `print_report()` with table formatting. CLI with `--from-cache`, `--tier2` flags. 8 tests.

### Decisions
- Predicate answerability is structural (predicate-level), not per-question text search — `cited_by_count` never appears in any opinion text regardless of the case.
- Case name extraction from questions uses " v. " as anchor and expands outward by capitalization, rather than a monolithic regex.
- Subject-scan questions ("Tell me about X") included in the fair A/B comparison since they test entity detection, which works identically across all arms.
- 15 obscure cases selected: zero citation count in CourtListener index, most from 1850s-1920s. Mix of modern obscure (Coventry 2017, Zubik 2015) and historical deep cuts (Gray v. Coan 1871, Sturgis v. Clough 1863).

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
