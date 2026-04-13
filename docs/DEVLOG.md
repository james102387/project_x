# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **Demo pipeline operational.** Full crystallization cycle: ingest → extract → purify → generate questions → human review → benchmark → RW improvement. CLI (`scripts/crystallize.py`) and UI buttons wired.
- **Test count:** 1,199 tests passing.
- **KG provenance complete.** All 2,197 scaffold facts backfilled with source_sentence. UI shows sentences in auto-accepted table. Review batch JSON includes source_sentence.
- **Three-arm comparison ready.** `benchmarks/three_arm_comparison.py` + UI section. Compares Crystal+KG vs LLM+Docs vs Naked LLM on accepted golden answers with accuracy scoring.
- **Golden-answer feedback loop wired.** Benchmark + RW buttons in Review tab. Questions → human golden answers → benchmark scoring → RW improvement loop.
- **Next:** Run 5 real crystallization cycles on curated SCOTUS opinions. Build golden answer set. Run three-arm comparison for marketing demo.

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

## 2026-04-12 — Demo Pipeline Build

### What changed
- **Provenance visibility** — Added "Sentence" column to auto-accepted table in UI. Review batch JSON exports now include `source_sentence`. Backfilled 2,197 scaffold facts (118 matched against opinion text, 2,079 got `[COLD Cases metadata]` provenance).
- **Golden-answer feedback loop** — "Run Benchmark on Accepted" and "Improve with Ralph Wiggum" buttons in Review tab. Runs Crystal pipeline on accepted golden answers, shows accuracy scores and per-question results.
- **Crystallization CLI** — `scripts/crystallize.py` with three subcommands: `ingest` (documents → extract → questions → review batch), `benchmark` (score accepted golden answers, optional `--rw`), `purify` (run KG audit).
- **Three-arm comparison** — `benchmarks/three_arm_comparison.py`: runs Crystal+KG, LLM+Docs, Naked LLM on accepted golden answers. Produces markdown report with accuracy table and per-question results. Also wired into UI as "Three-Arm Comparison on Golden Answers" section in Ingest tab.

---

## 2026-04-12 — KG Quality Gates, Rollback, and Audit

### What changed
- **Triplet validation gates** — `src/crystal/ingest/validation.py`: Three fast, deterministic gates run before KG insertion. Subject gate rejects pronouns/common nouns/junk prefixes. Predicate gate rejects non-canonical predicates. Object type gate validates format per predicate (date_filed must be date-like, cited_by_count must be numeric, etc.). Hard vs soft severity classification.
- **Fixed `normalize_predicate()` substring match** — `src/crystal/ingest/llm_extract.py`: Removed dangerous `canon in low or low in canon` fallback that could silently reclassify predicates. Now exact match + alias lookup only. Same fix applied in `confidence.py` scorer.
- **Source sentence persistence** — `Triplet` dataclass now has `source_sentence` field. NER extraction stamps `sent.text` on each triplet. `SqliteKnowledgeGraph.bulk_insert()` accepts 4-tuples `(s, p, o, source_sentence)`. New `source_sentence` column in `triplets` table with migration for existing DBs.
- **Batch provenance + rollback** — `ingestion_batches` table tracks every `bulk_insert()` call with `batch_id`, source, count, status. `delete_batch(batch_id)` rolls back all triplets from a batch. `list_batches()`, `batch_stats()`, `delete_by_ids()` for visibility and cleanup.
- **KG audit tool** — `src/crystal/tools/kg/audit.py`: Fast, deterministic health check (zero LLM cost). Runs validation gates on all KG facts. CLI: `python -m crystal.tools.kg.audit --db data/legal.sqlite [--fix]`.
- **KG proofreader** — `src/crystal/tools/kg/proofreader.py`: Two-pass deep clean. Pass 1: fast gates (free). Pass 2: LLM proofreading (semantic verification against source sentences). Tiered trust: hard failures auto-delete, soft+LLM → auto-delete, LLM-only → human review queue. CLI: `python -m crystal.tools.kg.proofread --db data/legal.sqlite [--fix] [--fast]`.
- **LLM proofreading** — `proofread_triplets()` in validation.py. Groups by source sentence, sends verification prompt per group. Falls back to plausibility check for legacy data without source sentences.
- **Wired validation into ingestion pipeline** — `ingest_document()` now runs `validate_triplet()` before confidence scoring. Rejected triplets never reach the KG.
- **KG cleaned** — Proofreader fast-mode deleted 764 garbage facts (pronoun subjects, verb predicates, bad date_filed/cites objects). Backfilled 306 empty `source` fields. Health score: 0.75 → 0.93.
- **59 new tests** — validation gates, LLM proofreading, batch provenance, rollback, audit, proofreader, ingestion integration. 1,122 tests total.

### Decisions
- Tiered trust policy for KG proofreader: fast gates are deterministic and trusted for auto-delete. LLM proofreading is probabilistic — rejections go to human review queue, never auto-delete on LLM judgment alone. Exception: soft gate failure + LLM rejection = auto-delete (two independent signals agree).
- `normalize_predicate()` substring match removed entirely rather than tightened. Better to reject an unmapped predicate and let the predicate gate handle it than to silently misclassify.
- `source_sentence` column added via migration for backward compatibility with existing DBs. Existing facts get empty string, new facts get the actual sentence.

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
