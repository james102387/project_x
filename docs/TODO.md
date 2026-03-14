# Crystal — Prioritized Task List

## MVP — Prove the value proposition with numbers

- [ ] **7. CrystalState schema update** — Add KG-related fields (`kg_detections`, `kg_results`, etc.) to the TypedDict so new nodes can communicate through state.
- [ ] **1. KG detection node** — Scan raw text against `kg.entities` (hash lookup, no spaCy). Detect entity mentions, match predicates via aliases. Wire into `graph.py` alongside `math_detection`.
- [ ] **2. KG execution node** — Takes preprocessed KG lookups, runs `kg.lookup()`, returns results. Parallel to `calculator_node` in the graph.
- [ ] **8. Planner update** — Handle multiple detection types (calculator + KG) in the same plan. Flat/parallel only — no dependency resolution yet.
- [ ] **3. Compiler: `kg_answerable` / `kg_augmented`** — `kg_answerable` returns the fact directly; `kg_augmented` injects grounded facts into a simplified prompt for the LLM.
- [ ] **4. Golden test cases for KG** — ~20 Remulak questions with known answers (forward lookup, reverse lookup, alias resolution, negatives). Add to `tests/golden/test_cases.py`.
- [ ] **5. Baseline benchmark** — Send the same Remulak questions to a naked LLM (no KG). Record answers. Auto-score against ground truth. This is the "before" measurement.
- [ ] **6. Treatment benchmark** — Run the same questions through Crystal with KG. Auto-score. Compare accuracy/hallucination rate against baseline. This produces the pitch-deck number.

## 2.0 — After the MVP proves the concept

- [ ] **9. Dependent detection / multi-hop** — Planner resolves dependencies between tools (e.g., "add the populations of Draveth and Sulari" → KG lookup first, then calculator). Requires plan interpreter with step threading.
- [ ] **10. Grading rubric** — Specificity scoring (did the response include exact entities/values?), completeness scoring, provenance citation, confidence calibration.
- [ ] **11. Real-domain dataset** — Replace Remulak with a large, automatically verifiable ground-truth corpus (law/CourtListener or other). Build ingest pipeline.
- [ ] **12. Separate semantic eval from detector** — Move `evaluate_semantic_steps` out of the detector into the calculator node. Detector detects, calculator calculates.
- [ ] **13. Mixed prompts (independent)** — "What is the capital of Remulak, and what is 5 + 3?" — both KG and calculator fire, compiler merges both results.
- [ ] **14. KG-grounded reasoning chain** — KG confirms facts → calculator computes → LLM reasons over the grounded, pre-computed result. Three tools in sequence.
