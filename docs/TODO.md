# Crystal — Prioritized Task List

## MVP (complete) — Prove the core pipeline works

- [x] **1. KG detection node** — Scan raw text against `kg.entities` (hash lookup, no spaCy). Detect entity mentions, match predicates via aliases.
- [x] **2. KG execution node** — Takes preprocessed KG lookups, runs `kg.lookup()`, returns results.
- [x] **3. Compiler: `kg_answerable` / `kg_augmented`** — `kg_answerable` returns the fact directly; `kg_augmented` injects grounded facts for LLM reasoning.
- [x] **4. Golden test cases for KG** — 20 KG answerable + 3 KG augmented + 3 KG negative Remulak cases.
- [x] **5. Baseline benchmark** — Naked LLM on Remulak questions. Result: 0% accuracy.
- [x] **6. Treatment benchmark** — Crystal + KG on same questions. Result: 100% accuracy.
- [x] **7. CrystalState schema update** — KG-related fields added.
- [x] **8. Planner update** — Multiple detection types in the same plan.

## Demo — Make the value proposition experienceable

The core pipeline works. Now someone needs to be able to *use* it.

- [x] **D1. Minimal quality rubric** — Extend benchmark scoring beyond binary substring match. Three dimensions: factual accuracy, specificity, no-hallucination. Score all three paths (answerable, augmented, fallback) so the safety invariant is measurable. See `docs/PLAN_GRADING_RUBRIC.md` Phase 1.
- [x] **D2. KG ingestion pipeline (offline) — Phase 1** — `src/crystal/ingest/` package: NER-based triplet extraction (5 dep-tree patterns), CSV/JSON hand-curated loaders, `Triplet`/`IngestResult` schema, `ingest()` → `build_kg()` pipeline, CLI at `python -m crystal.ingest <document>`.
  - [x] **Phase 2 (LLM-assisted):** Use LLM for relationship extraction on sentences where NER finds entities but can't resolve predicates. Offline batch job, human-reviewable output. `llm_extract.py`, `find_unresolved_sentences()`, `ingest_with_llm()`, `load_review()`, CLI `--llm-assist`.
  - [ ] **Phase 3 (community detection):** Leiden or similar for discovering entity clusters and implicit relationships in larger corpora. See L3 for legal-specific application (doctrinal families via citation graph clustering, GraphRAG-inspired).
- [x] **D3. Web UI** — Gradio-based demo at `python -m crystal.ui`. Three tabs: Ask, Knowledge Graph, Review. Pre-loaded Remulak KG works out of the box.
- [x] **D4. Augmented benchmark cases** — `AUGMENTED_BENCHMARK_CASES` in `benchmarks/ground_truth.py` (5 KG + 3 math). Benchmark runner at `benchmarks/run_augmented_benchmark.py`.
- [x] **D5. Entity aliases + fuzzy string matching + multi-hop** — 3-tier resolution cascade (exact → alias → rapidfuzz) for entities and predicates.
- [x] **D6. Reasoning cost benchmark (K-reduction)** — Token-level comparison with thinking models. Benchmark runner at `benchmarks/run_reasoning_benchmark.py`.

## Ingestion Roadmap — Phased data strategy

Golden answers from structured API sources are deterministic and require no human review. Human review is reserved for LLM-extracted content where confidence is lower.

### Phase 1: Structured API data (auto-accept)

All structured-source questions bypass human review. The golden answer is the API field value verbatim.

| Step | Source | Predicates | Target |
|------|--------|-----------|--------|
| **1a** | COLD Cases (SCOTUS) | court, date_filed, judges, disposition, cited_by_count, nature_of_suit | ~500 cases, ~2,500 questions |
| **1b** | CourtListener citations | cites (forward + reverse) | Citation graph for Tier 2 relational questions |
| **1c** | CourtListener /people/ | appointing_president, law_school, birth_year, active_service | Judge biographical entity type |

- [x] **Phase 1a** — Bulk SCOTUS ingest from COLD Cases via HuggingFace streaming. Auto-accept all questions. 500 cases, 1,321 questions.
- [x] **Phase 1b** — Citation graph from CourtListener `/opinions-cited/`. 200 cases searched, 316 citation triplets, 48 Tier 2 questions.
- [ ] **Phase 1c** — Judge biographical data from CourtListener `/people/`. New entity type (judges as subjects).

### Phase 2: Unstructured text (Crystal proposes, human verifies)

This is where the review UI earns its keep. Crystal proposes question + answer from NER/LLM extraction of opinion text. Human verifies or corrects. Still faster than authoring from scratch.

- [ ] **Phase 2a** — Opinion text extraction (holdings, doctrines, reasoning chains). NER + LLM pipeline with `ReviewableTriplet` workflow.
- [ ] **Phase 2b** — Oral argument transcripts. New source adapter.

### Sufficiency threshold

~500 cases / ~2,500 questions across all 6 predicates is sufficient for:
- Ralph Wiggum to converge reliably (~50 cases per predicate)
- Statistically significant benchmark comparisons vs. naked LLMs
- Regression detection when adding new extraction methods

### Operational pipeline (complete)

- [x] **L-OPS-1. Ingestion cron CLI** — `src/crystal/ingest/cron.py`
- [x] **L-OPS-2. Batch-aware review API** — `src/crystal/review.py`
- [x] **L-OPS-3. Interactive review UI** — Gradio Review tab
- [x] **L-OPS-4. Ralph Wiggum Phase 6b (autonomous mutation)** — Autoresearch pattern

### Structured data enrichment (deterministic golden answers)

- [x] **L-ENRICH-5. First real data run** — Bulk SCOTUS from COLD Cases, auto-accepted.
- [ ] **L-ENRICH-1. COLD Cases field expansion** (Tier 2) — `opinions[].author_str`, `opinions[].type`, `per_curiam`, `attorneys`, `precedential_status`. ~4-5 new predicates.
- [ ] **L-ENRICH-2. CourtListener /people/ endpoint** (Tier 2) — Judge biographical data.
- [ ] **L-ENRICH-3. CourtListener /clusters/ + /dockets/** (Tier 2) — Procedural posture, cause of action, lower court info.
- [ ] **L-ENRICH-4. Supreme Court Database (SCDB)** (Tier 3) — ~200 coded variables per case.

## Future — After the demo proves adoption potential

- [ ] **F1. Dependent detection / multi-hop** — Planner resolves dependencies between tools.
- [ ] **F2. Mixed prompts (independent)** — Both KG and calculator fire, compiler merges results.
- [ ] **F3. KG-grounded reasoning chain** — KG confirms facts → calculator computes → LLM reasons.
- [ ] **F4. Embedding-based KG matching** — Tier 4 resolution via sentence-transformers.
- [ ] **F5. Full grading rubric** — Six weighted metrics with composite scoring.
- [ ] **F6. Separate semantic eval from detector** — Move `evaluate_semantic_steps` out of detector.

## Infrastructure extensions

- [ ] **L1. Knowledge fusion layer** — Entity deduplication, citation normalization, conflict resolution.
- [ ] **L2. Incremental sync CLI** — `python -m crystal.ingest sync --source courtlistener --since <date>`.
- [ ] **L3. Community detection over citation graph** — Leiden clustering for doctrinal families.

## LLM-extracted content (requires human-verified golden answers)

- [ ] **L4. Tier 4 LLM extraction** — Extract-Define-Canonicalize pipeline for holdings/doctrines from opinion text.
- [ ] **L5. LLM-assisted ontology expansion** — CQ-based approach for predicate structure.
- [ ] **L6. Temporal validity for legal facts** — Track `overruled_by`, `modified_by` with validity windows.
- [ ] **L7. Corpus reasoning detector** — Routes to community cluster summaries for corpus-level questions.
- [ ] **L8. LLM-generated test case pipeline** — Automated Ralph Wiggum: LLM generates test cases, failing tests reveal gaps, LLM proposes fixes.

Research references:
[GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/),
[LLM-empowered KG Construction (Bian et al.)](https://arxiv.org/html/2510.20345v1),
[KGGen (Mo et al.)](https://arxiv.org/html/2502.09956v1)
