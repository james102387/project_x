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
- [ ] **D2. KG ingestion pipeline (offline)** — Phased approach, all in-repo at `src/crystal/ingest/`:
  - **Phase 1 (NER):** spaCy NER → extract entities + verb-phrase relationships → normalized `(subject, predicate, object)` triplets. Basic but reliable. Also support hand-curated triplets (CSV/JSON) for bootstrapping new datasets without any extraction.
  - **Phase 2 (LLM-assisted):** Use LLM for relationship extraction on sentences where NER finds entities but can't resolve predicates. Offline batch job, human-reviewable output.
  - **Phase 3 (community detection):** Leiden or similar for discovering entity clusters and implicit relationships in larger corpora.
  - Interface contract: produce `list[Triplet]` + optional `dict[str, str]` alias maps → feeds directly into `KnowledgeGraph`. CLI entry point: `python -m crystal.ingest <document>`.
- [ ] **D3. Web UI** — Simple interface (Streamlit/Gradio to start): upload documents or paste a URL, Crystal ingests → builds KG, then ask questions. Side-by-side comparison (Crystal vs. naked LLM) to make the value visible. Include a pre-loaded sample KG so the demo works out of the box.
- [ ] **D4. Augmented benchmark cases** — Extend benchmark ground truth to measure LLM *output quality* on augmented paths (`kg_augmented`, `math_augmented`), not just routing classification. These are the paths where Crystal injects context for the LLM — need to verify the augmentation actually helps rather than misleads. Requires real LLM calls in the benchmark and ground-truth expected answers for reasoning questions. Start with Remulak augmented cases, expand to ingested datasets once D2 Phase 1 lands.
- [x] **D5. Entity aliases + fuzzy string matching + multi-hop** — 3-tier resolution cascade (exact → alias → rapidfuzz) for entities and predicates. Entity alias tables per dataset, `rapidfuzz` for typos and word reordering. Depth-limited recursive multi-hop traversal (BFS, default depth=2). Match tier metadata in detection results.
- [x] **D6. Reasoning cost benchmark (K-reduction)** — `call_llm()` captures `thoughts_token_count` (reasoning tokens). `TokenMetrics` extended with `actual_reasoning_tokens` and `actual_total_tokens`. `ReasoningComparison` dataclass + `summarize_reasoning_comparisons()` for per-query grounded-vs-ungrounded token analysis. Benchmark runner at `benchmarks/run_reasoning_benchmark.py` (uses thinking-capable model, default gemini-2.5-flash). LLM nodes propagate all four token fields. Can use Remulak cases now; full D4 augmented cases expand coverage later.

## Future — After the demo proves adoption potential

- [ ] **F1. Dependent detection / multi-hop** — Planner resolves dependencies between tools (e.g., "add the populations of Draveth and Sulari" → KG lookup first, then calculator). Requires plan interpreter with step threading.
- [ ] **F2. Mixed prompts (independent)** — "What is the capital of Remulak, and what is 5 + 3?" — both KG and calculator fire, compiler merges both results.
- [ ] **F3. KG-grounded reasoning chain** — KG confirms facts → calculator computes → LLM reasons over the grounded, pre-computed result. Three tools in sequence.
- [ ] **F4. Embedding-based KG matching** — Tier 4 resolution via sentence-transformers for genuine semantic paraphrases that fuzzy string matching can't catch ("ruler" → "leader"). See `docs/PLAN_FUZZY_MATCHING.md` Phase 2.
- [ ] **F5. Full grading rubric** — Expand minimal rubric to six weighted metrics with composite scoring (completeness, efficiency, refined calibration). See `docs/PLAN_GRADING_RUBRIC.md` Phase 2.
- [ ] **F6. Separate semantic eval from detector** — Move `evaluate_semantic_steps` out of the detector into the calculator node. Detector detects, calculator calculates.
