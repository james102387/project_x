# Crystal — Task List

## Completed

<details>
<summary>MVP — Core pipeline (8/8)</summary>

- [x] KG detection node (hash lookup, predicate matching via aliases)
- [x] KG execution node (`kg.lookup()`)
- [x] Compiler: `kg_answerable` / `kg_augmented` routing
- [x] Golden test cases (20 answerable + 3 augmented + 3 negative)
- [x] Baseline benchmark (naked LLM → 0% accuracy)
- [x] Treatment benchmark (Crystal + KG → 100% accuracy)
- [x] CrystalState schema update
- [x] Planner update (multiple detection types)

</details>

<details>
<summary>Demo tooling (6/6)</summary>

- [x] D1. Quality rubric — accuracy + abstention scoring
- [x] D2. KG ingestion pipeline — NER (5 dep-tree patterns), CSV/JSON loaders, LLM-assisted extraction
- [x] D3. Web UI — Gradio (Ask, KG Explorer, Ingest Documents, Review tabs)
- [x] D4. Augmented benchmark cases (5 KG + 3 math)
- [x] D5. Entity aliases + fuzzy matching + multi-hop (3-tier resolution cascade)
- [x] D6. Reasoning cost benchmark (K-reduction, token-level comparison)

</details>

<details>
<summary>Ingestion infrastructure (complete)</summary>

- [x] Phase 1a — Bulk SCOTUS from COLD Cases (500 cases, 1,956 questions, auto-accepted)
- [x] Phase 1b — Citation graph from CourtListener (316 triplets, 48 Tier 2 questions)
- [x] L-ENRICH-1 — COLD Cases field expansion (opinion_author, per_curiam, attorneys, precedential_status)
- [x] Ingestion cron CLI (`src/crystal/ingest/cron.py`)
- [x] Batch-aware review API (`src/crystal/review.py`)
- [x] Interactive review UI (Gradio Review tab)
- [x] Ralph Wiggum Phase 6b (autonomous mutation, autoresearch pattern)
- [~~L-ENRICH-3~~] Cancelled — SCOTUS dockets lack nature_of_suit/disposition on CourtListener

</details>

<details>
<summary>Extraction Quality + Metrics (complete)</summary>

- [x] Simplified benchmark metrics — accuracy + abstention (dropped specificity/no_hallucination)
- [x] Extraction quality benchmark (`benchmarks/extraction_quality.py`) — NER-only: 59.1%, NER+LLM: 63.6%
- [x] ExtractionLoop for Ralph Wiggum — targets ingestion quality
- [x] Crystal-proposed answers in UI — generates questions from extracted facts, shows Crystal's answers
- [x] Review pipeline (`src/crystal/ingest/review_pipeline.py`) — document → KG → questions → Crystal proposes → review batch
- [x] Ground truth workflow — editable golden answers, save to review, flows into Ralph Wiggum
- [x] Question generator filters — predicate templates + subject plausibility (no NER noise)

</details>

---

## Path to Demo

The demo proves: Crystal + scaffold KG eliminates hallucinations, provides grounded answers, and costs 10x less than document-in-context. The selling points are citation network verification and zero hallucination, not yet holdings/doctrines extraction.

### Phase 1: Extraction Quality (current — iterate 3-5 times)

Target: 80%+ accuracy on document-derived questions.

1. [ ] **EQ-1. Human ground truth** — Review `review/batch_doc_*.json`, correct golden answers, accept/reject. Need ~50 accepted cases for Ralph Wiggum.
2. [ ] **EQ-2. Run Ralph Wiggum loops** — ExtractionLoop improves extraction prompts/aliases. PredicateLoop + EntityLoop + ThresholdLoop improve routing/matching. `python -m benchmarks.ralph_wiggum`
3. [ ] **EQ-3. Re-run extraction benchmark** — Measure improvement: `LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-4-5 python -m benchmarks.extraction_quality`
4. [ ] **EQ-4. Iterate** — Repeat EQ-1 → EQ-3 until accuracy ≥ 80% and hallucination ≤ 10%.

### Phase 2: Scaffold Density

Target: 70%+ citation hit rate (currently 20.7% with 543 subjects).

5. [ ] **S1. Expand scaffold** — CourtListener: 543 → ~5,000 subjects. Top-cited cases first (highest connectivity value per case).
6. [ ] **S2. Priority ingestion** — Rank cases by cited_by_count, ingest top-N first.
7. [ ] **Phase 1c** — Judge bios from CourtListener `/people/` (appointing_president, law_school, birth_year). Links judge entities across cases.

### Phase 3: Demo Batch

Target: A compelling three-arm comparison on a realistic practice area.

8. [ ] **DEMO-1. Pick practice area** — Choose a specific area with good scaffold coverage (e.g., 4th Amendment, equal protection, commerce clause).
9. [ ] **DEMO-2. Curate demo documents** — 5-10 real briefs/opinions. Must include citations to cases in the scaffold.
10. [ ] **DEMO-3. Run three-arm comparison** — Naked LLM vs LLM+document vs Crystal. On the demo documents.
11. [ ] **DEMO-4. Package results** — "LLM fabricated 40% of citations, Crystal 0%. 95% accuracy, 10x cheaper."
12. [ ] **DEMO-5. Marketing materials** — Demo video, one-pager, pitch deck from comparison results.

---

## Demo Benchmark (A/B Comparison)

Three arms, fair comparison on real opinion text.

### Tasks

- [ ] **B1. Opinion text downloader** — Fetch real opinion text from CourtListener for benchmark cases. Cache to `benchmarks/documents/`.
- [ ] **B2. Document-answerability audit** — Tag benchmark cases with `document_answerable`.
- [ ] **B3. Document-context baseline runner** — LLM + real opinion text arm.
- [ ] **B4. Add obscure cases** — 10-15 long-tail cases no LLM would know.
- [ ] **B5. Stratified sampler** — `sample_benchmark_cases(all_cases, n, by='predicate')` for Tier 2.
- [ ] **B6. Three-arm comparison report** — Side-by-side output.

---

## Ingestion — Open Items

### Structured API data

- [ ] **Phase 1c** — Judge bios from CourtListener `/people/` (appointing_president, law_school, birth_year, active_service). New entity type.
- [ ] **L-ENRICH-2** — CourtListener `/people/` endpoint for judge biographical data.
- [ ] **L-ENRICH-4** — Supreme Court Database (SCDB): ~200 coded variables per case.

### Unstructured text (Crystal proposes, human verifies)

- [x] **Phase 2a** — Document ingestion MVP. Full pipeline validated on real SCOTUS opinions.
- [x] **Phase 2a+** — LLM extraction on real opinions. NER-only: 59.1%, NER+LLM: 63.6% on 22 questions.
- [ ] **Phase 2b** — Oral argument transcripts. New source adapter.
- [ ] **D2 Phase 3** — Community detection (Leiden) for entity clusters.

---

## KG Scaffold Expansion & Distribution

**Business model:** Crystal (engine) + pre-built legal KG (scaffold) + customer document uploads. The scaffold provides day-one value and citation network connectivity. Current scaffold: 543 subjects, 20.7% citation hit rate against uploaded documents.

### Scaffold density

- [ ] **S1.** Expand CourtListener scaffold from 543 to ~5,000 subjects. Target 70%+ citation hit rate.
- [ ] **S2.** Priority ingestion: top-cited cases first (highest connectivity value per case added).
- [ ] **Phase 1c** — Judge bios from CourtListener `/people/` (links judge entities across cases).

### Segmented distribution

- [ ] **S3.** KG segmentation by jurisdiction (federal appellate, SCOTUS, state courts) and practice area.
- [ ] **S4.** Delta update mechanism — incremental triplet bundles via webhook/pull. Uses existing `sync_state` table + `bulk_insert()` idempotency.
- [ ] **S5.** Segment subscription model — customer selects relevant segments, receives targeted updates.

---

## RAG as Ingestion Faucet

RAG is a **document discovery layer** that feeds the ingestion pipeline, not a competing answer delivery mechanism. Retrieval quality affects ingestion throughput, not answer quality. The review step is the quality firewall.

**Feedback loop:** Ralph Wiggum finds gaps → gaps become retrieval queries → RAG retrieves documents → ingestion pipeline extracts triplets → human review → KG grows targeted at its own gaps → Ralph Wiggum re-evaluates.

- [ ] **R1.** Gap-to-query translator (Ralph Wiggum failure analysis → retrieval queries)
- [ ] **R2.** Document retrieval layer (embeddings over CourtListener opinion corpus)
- [ ] **R3.** Retrieved-document ingestion (wire into `ingest_with_llm()`)
- [ ] **R4.** Autonomous growth loop (full cycle)

---

## Future

### Pipeline extensions

- [ ] **F1.** Dependent detection / multi-hop planner
- [ ] **F2.** Mixed prompts (KG + calculator fire independently, compiler merges)
- [ ] **F3.** KG-grounded reasoning chain (KG confirms → calculator computes → LLM reasons)
- [ ] **F4.** Embedding-based KG matching (Tier 4 resolution via sentence-transformers)
- [ ] **F5.** Full grading rubric (6 weighted metrics, composite score)
- [ ] **F6.** Separate semantic eval from detector

### Infrastructure

- [ ] **L1.** Knowledge fusion layer (entity dedup, citation normalization, conflict resolution)
- [ ] **L2.** Incremental sync CLI (`crystal.ingest sync --source courtlistener --since <date>`)
- [ ] **L3.** Community detection over citation graph (Leiden clustering for doctrinal families)

### LLM-extracted content (requires human-verified golden answers)

- [ ] **L4.** Tier 4 LLM extraction (Extract-Define-Canonicalize for holdings/doctrines)
- [ ] **L5.** LLM-assisted ontology expansion (CQ-based predicate structure)
- [ ] **L6.** Temporal validity for legal facts (`overruled_by`, `modified_by` with validity windows)
- [ ] **L7.** Corpus reasoning detector (routes to community cluster summaries)
- [ ] **L8.** LLM-generated test case pipeline (automated Ralph Wiggum)

---

**References:**
[GraphRAG](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) ·
[LLM-empowered KG Construction (Bian et al.)](https://arxiv.org/html/2510.20345v1) ·
[KGGen (Mo et al.)](https://arxiv.org/html/2502.09956v1)
