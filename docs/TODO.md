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

- [x] D1. Quality rubric — 3-dimension scoring (accuracy, specificity, no-hallucination)
- [x] D2. KG ingestion pipeline — NER (5 dep-tree patterns), CSV/JSON loaders, LLM-assisted extraction
- [x] D3. Web UI — Gradio (Ask, KG, Review tabs)
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

---

## Next Up — Demo Benchmark (A/B comparison)

The naked LLM baseline is a strawman. Lawyers paste real documents into ChatGPT. The demo needs a fair, realistic comparison.

### Design

**Three arms:**

| Arm | What it tests | Simulates |
|-----|---------------|-----------|
| Naked LLM | `call_llm(question)` | Lawyer trusts training data |
| LLM + document | `call_llm(real_opinion_text + question)` | Lawyer pastes case opinion into ChatGPT |
| Crystal | Full pipeline | Crystal |

**Three metrics:**

| Metric | Why |
|--------|-----|
| Hallucination rate | Headline: "LLM fabricated 40%, Crystal 0%" |
| Rubric quality (accuracy, specificity, no-hallucination) | Crystal is precise, not just non-hallucinatory |
| Token cost | Crystal's `kg_answerable` path: 0 tokens vs. 30k for document-in-context |

**Documents must be real opinion text** (5k-50k tokens from CourtListener), not synthetic docs from KG metadata. Synthetic docs are reformatted metadata — a critic would dismiss the comparison.

**Not all predicates are document-answerable.** Fair A/B only on questions answerable from both KG and opinion text:

| Category | Predicates | Fair A/B? |
|----------|-----------|-----------|
| Both | court, date_filed, judges, opinion_author, cites, attorneys | Yes |
| KG only | cited_by_count, precedential_status, per_curiam | Show separately ("Crystal-only") |
| Document only | holdings, doctrines, reasoning | Crystal can't answer yet (Phase 2a) |

**Scaling:** Tier 1 (~50 hand-crafted cases, run live, cached). Tier 2 (~100-200 stratified sample from bulk corpus, run once, cached). Scoring works against cached result dicts.

### Tasks

- [ ] **B1. Opinion text downloader** — Fetch real opinion text from CourtListener for benchmark cases. Cache to `benchmarks/documents/`. Uses existing `courtlistener.py` client.
- [ ] **B2. Document-answerability audit** — Read downloaded opinions, verify which predicates appear. Tag benchmark cases with `document_answerable`.
- [ ] **B3. Document-context baseline runner** — New arm in `benchmarks/runners/`. LLM + real opinion text. Same scoring interface. Only runs on document-answerable questions.
- [ ] **B4. Add obscure cases** — 10-15 cases from the long tail of the 500-case corpus. Cases no LLM would know from training data.
- [ ] **B5. Stratified sampler** — `sample_benchmark_cases(all_cases, n, by='predicate')` for Tier 2.
- [ ] **B6. Three-arm comparison report** — Side-by-side output with separate sections for fair A/B and Crystal-only questions.

---

## Ingestion — Open items

### Structured API data

- [ ] **Phase 1c** — Judge bios from CourtListener `/people/` (appointing_president, law_school, birth_year, active_service). New entity type.
- [ ] **L-ENRICH-2** — CourtListener `/people/` endpoint for judge biographical data.
- [ ] **L-ENRICH-4** — Supreme Court Database (SCDB): ~200 coded variables per case.

### Unstructured text (Crystal proposes, human verifies)

- [ ] **Phase 2a** — Opinion text extraction (holdings, doctrines, reasoning chains). NER + LLM pipeline with `ReviewableTriplet` workflow.
- [ ] **Phase 2b** — Oral argument transcripts. New source adapter.
- [ ] **D2 Phase 3** — Community detection (Leiden) for entity clusters. See also L3.

### Sufficiency threshold

~500 cases / ~2,500 questions across 6+ predicates — sufficient for Ralph Wiggum convergence (~50 per predicate), statistically significant benchmarks, and regression detection.

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
