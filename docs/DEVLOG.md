# Crystal Development Log

Reverse-chronological record of changes and decisions.
Only the most recent ~5 entries live here. Older entries are in `DEVLOG_ARCHIVE.md`.

---

## Active Focus

Update this section each session with current priorities.

- **KG integration:** detector, execution node, and compiler wired into graph. Needs golden test benchmarks.
- **Benchmarking:** baseline (naked LLM) vs treatment (Crystal + tools) accuracy comparison not yet run.
- **math_augmented cost:** compiled prompts are longer than raw — by design, but worth revisiting.

---

## 2026-03-14 — Module Restructuring

### What changed
- Moved `detectors/calculator.py` + `detectors/semantic.py` into `detectors/math/` subfolder
  - `calculator.py` → `explicit.py`, `semantic.py` unchanged
  - `__init__.py` re-exports all public symbols
- Moved `tools/kg.py` + `tools/remulak_kg.py` into `tools/kg/` subfolder
  - `kg.py` → `graph.py`, `remulak_kg.py` → `remulak.py`
- Renamed `nodes/detector.py` → `nodes/math_detection.py`
- Renamed `nodes/kg_detector.py` → `nodes/kg_detection.py`
- Function names: `math_detection_node`, `kg_detection_node`
- Graph node labels: `"math_detection"`, `"kg_detection"`
- 149/149 passing, 5 skipped

### Decisions
- `semantic.py` grouped with calculator (same detection pipeline)
- Node files use `_detection` (noun) not `_detector` (agent noun) to avoid
  confusion with the actual detector modules in `detectors/`

---

## 2026-03-08 — Three-View Savings Metrics

### What changed
- Three savings views: `token_savings_pct` (billing), `marginal_savings_pct` (contextual cost), `savings_pct` (legacy)
- New `marginal_cost()` function, `estimate_metrics()` accepts `base_context` param
- 8 new tests; 95/95 passing

### Decisions
- Isolated N+N² overstated penalties ~5x; kept for backwards compat, clearly labelled

---

## 2026-03-08 — Token Metrics & math_answerable Bypass

### What changed
- Split `math_in_context` → `math_answerable` (LLM bypass) + `math_augmented` (LLM required)
- `REASONING_SIGNALS` set gates augmented path; default is bypass
- tiktoken-based metrics, Gemini `usage_metadata` integration
- 3 new golden cases, 13 new tests; 93/93 passing

### Decisions
- Conservative bypass: only advisory/explanatory/comparative/predictive words trigger LLM

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
