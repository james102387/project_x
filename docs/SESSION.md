# Session — 2026-03-17

## Goal
Implement D3: Web UI (Gradio-based demo interface).

## Completed
- Made KG injectable into pipeline: added `kg` field to `CrystalState`, updated `make_initial_state(prompt, *, kg=None)`, updated `kg_detection_node` to use `state["kg"]` or fall back to `remulak_kg`
- Built Gradio UI at `src/crystal/ui/app.py`:
  - "Ask" tab: question input, side-by-side Crystal vs. naked LLM comparison with route/token metadata
  - "Knowledge Graph" tab: file upload (CSV/JSON/TXT), paste text for NER extraction, reset to Remulak, facts table
  - Pre-loaded Remulak KG works out of the box
  - `gr.State` manages active KG across tabs
  - Runnable via `python -m crystal.ui`
- Added `gradio>=5.0.0` to requirements.txt (installed 6.9.0)
- 20 new tests: `test_kg_injection.py` (5 tests for KG injection through pipeline), `test_ui_helpers.py` (12 tests for UI helper functions)
- 351/351 passing, 5 skipped
