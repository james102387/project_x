# Session Journal

Ephemeral scratchpad for the current working session.
Archive useful findings into `DEVLOG.md` before clearing.

---

## How to use this file

**At session start:** The LLM reads this + `DEVLOG.md` for context.

**During session:** Write observations and reasoning here as you work.

**At session end:** Move anything worth keeping into `DEVLOG.md`.
Then clear the "Current Session" section for next time.

## Current Session

### Module restructuring
- Moved `detectors/calculator.py` + `detectors/semantic.py` → `detectors/math/{explicit,semantic}.py`
- Moved `tools/kg.py` + `tools/remulak_kg.py` → `tools/kg/{graph,remulak}.py`
- Renamed `nodes/detector.py` → `nodes/math_detection.py` (fn: `math_detection_node`)
- Renamed `nodes/kg_detector.py` → `nodes/kg_detection.py` (fn: `kg_detection_node`)
- Graph labels updated to `"math_detection"`, `"kg_detection"`
- All imports updated, 149/149 passing

### Previous session notes (KG implementation)
- Starting KG tool implementation (was listed as next step in previous sessions)
- Created `src/crystal/data/` package for pure synthetic datasets
- Moved Remulak triplets from `tools/remulak_kg.py` to `data/remulak.py` (pure data, no logic)
- Created `tools/kg.py`: generic `KnowledgeGraph` class — dataset-agnostic, forward+reverse hash indexes
- Refactored `tools/remulak_kg.py` into a thin convenience module that wires data → engine
- Design: data separate from tools. Adding a new KG corpus = write a data module + instantiate `KnowledgeGraph`
- Updated README, ARCHITECTURE docs to reflect new structure
- Added `PREDICATE_ALIASES` dict to `data/remulak.py` — ~80 alias→canonical mappings covering all predicate categories
- `KnowledgeGraph` now accepts optional `predicate_aliases` param, resolves aliases in `_resolve_predicate()` before lookup
- Added `_entity_index` set (all subjects+objects lowercased) with `has_entity()` and `entities` property
- `remulak_kg.py` wiring now passes both triplets and aliases
- Added `tests/test_kg.py`: 27 tests covering forward/reverse/scan, alias resolution, entity index, construction edge cases

### Decoupled detector architecture (Option A)
- Addressed #13: detector.py was too coupled to calculator. Implemented Option A (separate detector nodes)
- Created `detectors/kg.py`: two-stage detection (entity recognition + question structure) with predicate extraction
- Created `nodes/kg_detector.py`: separate KG detector node, runs after calculator detector
- Created `nodes/kg.py`: KG execution node, runs after calculator node
- Updated `nodes/planner.py`: handles KG detections alongside calculator
- Updated `nodes/preprocessor.py`: validates KG plan items
- Updated `nodes/compiler.py`: new `kg_answerable` prompt type, `_format_kg_results()`, classification updated
- Updated `graph.py`: wired KG detector + KG execution into the graph as separate nodes
- Updated `state.py`: documented `kg_answerable` prompt type
- Updated `metrics.py`: `kg_answerable` gets 100% savings (LLM bypassed)
- Predicate extraction: strips question noise, preserves internal prepositions ("head of state"), strips trailing connectors ("of Remulak")
- `QUESTION_PREDICATE_MAP` for natural phrasing ("old" → "age", "big" → "diameter")
- 8 KG golden test cases: exact predicates, alias predicates, multi-word entities, "tell me" style
- 17 KG detector unit tests: entity spans, question structure, end-to-end detection
- 147/147 passing, 5 skipped (LLM integration), 50 golden cases total

### Test reorganization
- Reorganized flat `tests/` into structured subfolders:
  - `tests/unit/detectors/` — calculator, semantic, KG detector tests
  - `tests/unit/tools/` — KG tool tests (lookup, aliases, entity index)
  - `tests/unit/nodes/` — compiler, metrics tests
  - `tests/integration/` — full pipeline (local) + LLM integration
  - `tests/golden/` — golden test cases (unchanged)
  - `tests/fixtures/` — LLM cache (unchanged)
- `conftest.py` stays at `tests/` root — fixtures available to all subfolders
- Deleted old flat test files, all imports updated
- Added `kg_answerable` metrics test and compiler classification test
- 149/149 passing, 5 skipped (LLM integration)
