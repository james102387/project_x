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
- [x] **D3. Web UI** — Gradio-based demo at `python -m crystal.ui`. Two tabs: "Ask" (side-by-side Crystal vs. naked LLM with route/token metadata) and "Knowledge Graph" (upload CSV/JSON/TXT, paste text for NER, browse facts). KG injectable into pipeline via `make_initial_state(prompt, kg=custom_kg)`. Pre-loaded Remulak KG works out of the box.
- [x] **D4. Augmented benchmark cases** — `AUGMENTED_BENCHMARK_CASES` in `benchmarks/ground_truth.py` (5 KG + 3 math). Benchmark runner at `benchmarks/run_augmented_benchmark.py`: naked LLM vs. full Crystal pipeline with real LLM calls, scored with D1 rubric (accuracy, specificity, no-hallucination). Side-by-side report. 20 new tests, 371 passing.
- [x] **D5. Entity aliases + fuzzy string matching + multi-hop** — 3-tier resolution cascade (exact → alias → rapidfuzz) for entities and predicates. Entity alias tables per dataset, `rapidfuzz` for typos and word reordering. Depth-limited recursive multi-hop traversal (BFS, default depth=2). Match tier metadata in detection results.
- [x] **D6. Reasoning cost benchmark (K-reduction)** — `call_llm()` captures `thoughts_token_count` (reasoning tokens). `TokenMetrics` extended with `actual_reasoning_tokens` and `actual_total_tokens`. `ReasoningComparison` dataclass + `summarize_reasoning_comparisons()` for per-query grounded-vs-ungrounded token analysis. Benchmark runner at `benchmarks/run_reasoning_benchmark.py` (uses thinking-capable model, default gemini-2.5-flash). LLM nodes propagate all four token fields. Can use Remulak cases now; full D4 augmented cases expand coverage later.

## Future — After the demo proves adoption potential

- [ ] **F1. Dependent detection / multi-hop** — Planner resolves dependencies between tools (e.g., "add the populations of Draveth and Sulari" → KG lookup first, then calculator). Requires plan interpreter with step threading.
- [ ] **F2. Mixed prompts (independent)** — "What is the capital of Remulak, and what is 5 + 3?" — both KG and calculator fire, compiler merges both results.
- [ ] **F3. KG-grounded reasoning chain** — KG confirms facts → calculator computes → LLM reasons over the grounded, pre-computed result. Three tools in sequence.
- [ ] **F4. Embedding-based KG matching** — Tier 4 resolution via sentence-transformers for genuine semantic paraphrases that fuzzy string matching can't catch ("ruler" → "leader"). See `docs/PLAN_FUZZY_MATCHING.md` Phase 2. Also needed for L4's predicate canonicalization (comparing definitions by embedding similarity).
- [ ] **F5. Full grading rubric** — Expand minimal rubric to six weighted metrics with composite scoring (completeness, efficiency, refined calibration). See `docs/PLAN_GRADING_RUBRIC.md` Phase 2.
- [ ] **F6. Separate semantic eval from detector** — Move `evaluate_semantic_steps` out of the detector into the calculator node. Detector detects, calculator calculates.

## Legal KG — Build on the legal data ingestion pipeline

See `PLAN: Legal Data Ingestion Architecture` for the core pipeline (Phases 0–3). Items below extend it.

Research references:
[GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/),
[LLM-empowered KG Construction (Bian et al.)](https://arxiv.org/html/2510.20345v1),
[KGGen (Mo et al.)](https://arxiv.org/html/2502.09956v1)

- [ ] **L1. Knowledge fusion layer** — Dedicated `ingest/fusion.py` for advanced entity deduplication beyond inline normalization. LLM-guided entity clustering for ambiguous cases where string matching fails (adapt KGGen's iterative LLM-as-judge clustering pattern). Citation normalization across partial/variant formats. Conflict resolution when COLD Cases and CourtListener disagree. Runs as batch post-processing, not inline. (Bian et al. Section 5; KGGen Section 3.3)
- [ ] **L2. Incremental sync CLI** — `python -m crystal.ingest sync --source courtlistener --since <date>`. Reads `sync_state` table, fetches delta from CourtListener API, upserts into SQLite. Crystal stays a CLI tool — wrap in system cron or GitHub Actions for automation.
- [ ] **L3. Community detection over citation graph** — Extends D2 Phase 3 with legal context. Leiden clustering over the citation graph produces doctrinal families. Per-cluster LLM summarization enables whole-dataset reasoning ("What are the major themes in Fourth Amendment jurisprudence?"). Hierarchical: tight clusters (specific case groups) → broad groupings (doctrinal areas). Dependencies: `leidenalg`, `igraph`. (GraphRAG)
- [ ] **L4. Tier 3 LLM extraction (EDC + KGGen techniques)** — Extract-Define-Canonicalize pipeline for holdings, doctrines, statutes from opinion text. Adapt KGGen's two-step extraction (entities first via separate LLM call, then relations given explicit entity list) and iterative edge clustering (canonicalize extracted predicates against existing vocabulary via LLM-as-judge). Reimplement in Crystal's style with injectable `call_llm_fn` — do not use `kg-gen` package directly. Prioritize landmark cases (high citation count from Tier 2). Same `ReviewableTriplet` human-review workflow. Progressive crystallization: patterns discovered by Tier 3 LLM extraction should be codified as Tier 1/2 deterministic rules (new dep-tree patterns, new predicate maps) so future occurrences bypass the LLM. L8's test pipeline drives this progression. (Bian et al. Section 4; KGGen Sections 3.1, 3.3; EDC: Zhang & Soh 2024)
- [ ] **L5. LLM-assisted ontology expansion** — When adding Phase 2 metadata predicates (authorship, jurisdiction, disposition), use LLM-assisted CQ-based approach: sample records, have LLM propose predicate structure, validate with competency questions ("Can I answer 'Who wrote the majority opinion in X?'"). Resolves ambiguous predicate boundaries (e.g., should `opinion_author` and `judges` be the same?). (Bian et al. Section 3.1.1)
- [ ] **L6. Temporal validity for legal facts** — Holdings get overruled/modified. Track `overruled_by`, `modified_by` metadata from COLD Cases `history` and `cross_reference` fields. Facts have validity windows. (Bian et al. Section 6.2; Zep/A-MEM temporal KG concept)
- [ ] **L7. Corpus reasoning detector** — New Crystal route type alongside KG and math. Routes to community cluster summaries (L3) instead of individual triplet lookups. Enables questions no single-case lookup can answer ("What are the trends in X?"). Requires new detector, new compiler path. (GraphRAG)
- [ ] **L8. LLM-generated test case pipeline** — For each CourtListener ingestion batch: LLM generates test cases (question + expected route + expected answer) from the new data. Failing tests reveal where detection/routing heuristics break on real legal text. LLM proposes new patterns (predicate map entries, entity detection rules, dep-tree patterns) to handle failures. Human reviews the diff. Test suite grows, coverage compounds. This is the mechanism that progressively crystallizes `llm_fallback` paths into `kg_answerable`/`kg_augmented` paths — a manual RL loop with transparent policy updates. Pairs with L2 (incremental sync) and L4 (Tier 3 extraction).
