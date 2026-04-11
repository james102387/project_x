# Session — 2026-04-10

## Task: Demo benchmark strategy (A/B comparison design)

Designed the three-arm benchmark comparison for the MVP demo.

### Three arms

1. **Naked LLM** — `call_llm(question)`, no context. Useful as a floor but is a strawman — nobody uses LLMs this way for obscure cases. Fails 100% on cases outside training data.
2. **LLM + real opinion text** — Lawyer pastes actual case opinion into ChatGPT. This is the realistic comparison. Must use real opinion documents from CourtListener, not synthetic docs constructed from KG metadata.
3. **Crystal** — Full pipeline (`kg_answerable` / `kg_augmented` / `fallback`).

### Key concerns and decisions

**Why not synthetic documents?** A critic would say "you gave the LLM junk to work with." Reformatting KG metadata as prose tests the LLM's ability to parse our synthetic format, not its ability to extract facts from real legal text. Must download actual opinion text (5k-50k tokens).

**Metadata vs. document content mismatch.** The KG was trained on structured API metadata. The opinion documents contain different information. Not all KG predicates appear in opinion text:
- **In both:** court, date_filed, judges, opinion_author, cites, attorneys
- **KG only:** cited_by_count, precedential_status, per_curiam (these are CourtListener index fields, not in the opinion)
- **Document only:** holdings, doctrines, reasoning (Phase 2a, Crystal can't answer yet)

Fair A/B comparison can only run on "both" category questions. KG-only questions shown separately as a second value proposition.

**Requires manual audit (B2):** Must download a few opinions and verify which predicates actually appear in the text before building the benchmark.

**Document grouping.** One opinion per case, multiple questions per document. Questions reference cases by name — entity detection (already built) resolves to canonical case name = document key. Alias and citation-format questions resolve through existing 3-tier cascade.

**Scaling.** ~1,500 questions through an LLM is expensive. Two-tier approach:
- Tier 1: ~50 hand-crafted cases, run live, cache results to JSON
- Tier 2: ~100-200 stratified sample from bulk corpus, run once, cache. `score_batch_rubric()` works on cached dicts.

**Must add obscure cases (B4).** Current `LEGAL_BENCHMARK_CASES` are all famous (Miranda, Brown, Roe). LLM might know these from training data. Need 10-15 long-tail cases.

**Three metrics:** hallucination rate (headline), rubric quality (depth), token cost (economics).

**RAG as ingestion faucet.** RAG is not a competitor to Crystal — it's a future document discovery layer. RAG retrieves documents targeted at KG gaps (identified by Ralph Wiggum failure analysis). Documents go through the existing ingestion pipeline. The review step is the quality firewall.

### Changes made

- Restructured `TODO.md` — collapsed completed work into `<details>` blocks, "Next Up" is the benchmark section
- Added B1-B6 implementation tasks for demo benchmark
- Added R1-R4 tasks for RAG-as-faucet (post-demo)
