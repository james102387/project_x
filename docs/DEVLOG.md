# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **D1, D2 Phase 1, D3, D4, D5, D6 complete.** Web UI live at `python -m crystal.ui`.
- **Next milestone:** D2 Phase 2 (LLM-assisted extraction). D2 Phase 3 (community detection) and F-series items after that.
- **Key architectural additions:** KG injectable into pipeline via `make_initial_state(prompt, kg=custom_kg)`. Gradio UI at `src/crystal/ui/`. Augmented quality benchmark at `python -m benchmarks.run_augmented_benchmark`.
- **Test count:** 371 passing, 5 skipped.

---

## 2026-03-16 — D2 Phase 1: KG ingestion pipeline (NER + hand-curated)

### What changed
- New `src/crystal/ingest/` package:
  - `schema.py`: `Triplet` dataclass (subject/predicate/object), `IngestResult` dataclass (triplets + alias maps + source), `merge()` for combining results
  - `ner.py`: spaCy dep-tree-based triplet extraction. Five sentence patterns: copular (`X is Y`, including `of`-flip), possessive (`X has Y`), active transitive (`X verbs Y`), passive (`X was Vd by Y`), prepositional (`X verb PREP Y`). Handles hyphenated entities, determiner stripping, passive predicate text (not lemma).
  - `loader.py`: CSV loader (auto-detects header), JSON loader (object-with-aliases format + flat array format), `load_file()` auto-detect
  - `__init__.py`: top-level `ingest(path)` auto-detects format, `build_kg(result)` converts to `KnowledgeGraph`
  - `__main__.py`: CLI entry point `python -m crystal.ingest <document>` with `--output` and `--format` flags
- Test fixtures: `sample_triplets.csv`, `sample_triplets_no_header.csv`, `sample_triplets.json`, `sample_triplets_flat.json`, `sample_text.txt`
- 53 new tests: `test_schema.py` (7), `test_ner.py` (19), `test_loader.py` (18), `test_integration.py` (9)
- 331/331 passing, 5 skipped

### Decisions
- NER uses dep tree patterns rather than NER entity labels because `en_core_web_sm` misses most fiction entities — noun chunks + dependency structure are more reliable
- Passive predicates use `token.text.lower()` not `token.lemma_` to avoid unintuitive lemma forms ("born" → "bear")
- Copular `of`-flip checks both subject and attr for `prep(of)` to handle both "The capital of X is Y" and "Y is the capital of X"
- Determiner stripping in `_span_text` only (not in `_get_subject_span`) — subjects keep their structure, objects get cleaned
- `_join_tokens` collapses spaces around hyphens to preserve "Dark-ore" from spaCy's "Dark", "-", "ore" tokenization

---

## 2026-03-16 — D6: Reasoning cost benchmark (K-reduction)

### What changed
- `src/crystal/llm.py`: Extracted `_extract_usage()` helper. Now captures `thoughts_token_count` (reasoning tokens) from Gemini usage metadata alongside prompt and output tokens. Computes `total_tokens` sum.
- `src/crystal/metrics.py`: `TokenMetrics` extended with `actual_reasoning_tokens` and `actual_total_tokens`. New `ReasoningComparison` dataclass for per-query grounded-vs-ungrounded comparison with computed properties (`total_token_delta`, `reasoning_token_delta`, savings percentages). New `summarize_reasoning_comparisons()` aggregates per-query data into summary statistics.
- `src/crystal/nodes/llm_nodes.py`: Extracted `_update_metrics_from_usage()` helper. Both augmented and fallback nodes propagate all four token fields (prompt, output, reasoning, total).
- `benchmarks/run_reasoning_benchmark.py`: New benchmark runner. Runs identical queries through both naked LLM and Crystal-grounded pipeline using a thinking-capable model (default: gemini-2.5-flash). Records per-query: accuracy, prompt/output/reasoning/total tokens. Reports accuracy delta and token savings. Handles kg_answerable (LLM bypass = 0 tokens) and kg_augmented paths. CLI with `--model` and `--cases` flags.
- 23 new unit tests: `_extract_usage` (4), `_update_metrics_from_usage` (3), `ReasoningComparison` (9), `summarize_reasoning_comparisons` (5), `TokenMetrics` reasoning fields (2)
- 278/278 passing, 5 skipped

### Decisions
- Reasoning benchmark uses existing `BENCHMARK_CASES` from ground_truth.py rather than waiting for D4 — Remulak cases exercise both kg_answerable and kg_augmented paths, sufficient for initial K-reduction measurement
- kg_answerable cases counted as 0 grounded tokens (LLM fully bypassed) — this is the strongest form of K-reduction
- `_extract_usage()` made a module-level function (not method) so the reasoning benchmark can call it directly without going through `call_llm()`

---

## 2026-03-16 — D5: Entity aliases + fuzzy matching + multi-hop traversal

### What changed
- New `src/crystal/tools/kg/fuzzy.py`: `fuzzy_match()` using `rapidfuzz.fuzz.token_sort_ratio`
- `graph.py`: added `entity_aliases` param, `_resolve_entity()` (exact → alias → fuzzy cascade), `_resolve_predicate_fuzzy()` (same cascade for predicates), `subjects` property, `traverse()` (BFS depth-limited multi-hop, default depth=2)
- `data/remulak.py`: `ENTITY_ALIASES` dict (~20 entries) mapping short forms to canonical entities
- `detectors/kg.py`: `find_entity_spans()` now uses 3-tier cascade with spaCy noun phrase extraction as fallback, length-ratio guard against derived forms (e.g., "Remulakian" ≠ "Remulak"), `detect_kg_query()` accepts `multi_hop` and `max_depth` params, detection results include `match_tier`, `match_score`, `original_text`, `predicate_match_tier`
- `QUESTION_PREDICATE_MAP`: added `"long last" → "duration"`, `"long" → "duration"`
- `requirements.txt`: added `rapidfuzz>=3.0.0`
- Golden test cases: 3 alias cases + 2 fuzzy string cases
- Unit tests: `test_fuzzy.py` (8 tests), expanded `test_kg.py` (entity aliases, fuzzy entity, fuzzy predicate, multi-hop traversal, subjects property — 20+ new tests), expanded `test_kg.py` detector tests (match_tier, alias detection, fuzzy detection, multi-hop detection)
- 255/255 passing, 5 skipped

### Decisions
- Length-ratio guard (0.7–1.3) on fuzzy entity matches from noun phrase extraction prevents derived adjectives ("Remulakian") from fuzzy-matching their base entity ("Remulak")
- Entity aliases include article variants ("the fracture war" alongside "fracture war") because spaCy noun phrase extraction includes determiners
- Multi-hop is opt-in (`multi_hop=False` default) — caller decides when to expand; keeps subject_scan as the default fallback for backward compatibility
- Multi-hop uses BFS with visited set for cycle prevention, collects only facts where objects are also KG subjects (avoids expanding leaf values like "Zelphos" that have no forward facts)

---

## 2026-03-20 — D4: Augmented output quality benchmark (thin)

### What changed
- `benchmarks/ground_truth.py`: New `AUGMENTED_BENCHMARK_CASES` — 5 KG augmented + 3 math augmented cases with `match_strings` that prove grounding worked (fictional KG values a naked LLM can't produce).
- `benchmarks/run_augmented_benchmark.py`: Benchmark runner comparing naked LLM vs. full Crystal pipeline (real LLM calls) on augmented paths. Scores both with D1 rubric (accuracy, specificity, no-hallucination). Side-by-side report + JSON output. CLI: `python -m benchmarks.run_augmented_benchmark`.
- `tests/golden/test_cases.py`: `KG_AUGMENTED_CASES` now have expected result strings (previously `None`).
- 20 new tests in `test_augmented_benchmark.py`: case format validation (4), scoring with synthetic responses (5), summary aggregation (3), routing verification for KG cases (5) and math reasoning signals (3).
- 371/371 passing, 5 skipped.

### Decisions
- Treatment arm uses the full `build_crystal_graph()` pipeline (not manual detection) — exercises the real code path including LLM call for augmented routes.
- Baseline is scored against treatment's KG results for rubric specificity/grounding — otherwise there's no rubric baseline for comparison since naked LLM has no KG context.
- Match strings are intentionally minimal (one KG value per case). The rubric dimensions handle deeper quality measurement; binary accuracy just gates "did grounding reach the output at all."

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
