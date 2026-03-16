# Session — 2026-03-16

## Goal
Implement D5: Entity aliases + fuzzy string matching + multi-hop recursive processor.

## Completed
- `src/crystal/tools/kg/fuzzy.py` — `fuzzy_match()` with rapidfuzz
- `src/crystal/data/remulak.py` — `ENTITY_ALIASES` dict (~20 entries)
- `src/crystal/tools/kg/graph.py` — entity_aliases, `_resolve_entity()`, `_resolve_predicate_fuzzy()`, `subjects` property, `traverse()` (BFS multi-hop)
- `src/crystal/tools/kg/remulak.py` — wired entity_aliases
- `src/crystal/detectors/kg.py` — 3-tier cascade in `find_entity_spans()`, match_tier metadata, `multi_hop` param, length-ratio guard
- `requirements.txt` — added rapidfuzz>=3.0.0
- Golden test cases: 3 alias + 2 fuzzy string
- Unit tests: 8 fuzzy, 20+ new KG tool tests, 8+ new detector tests
- 255 passing, 5 skipped, 0 failed

## Key decisions
- Length-ratio guard (0.7–1.3) prevents derived adjectives from fuzzy-matching base entities
- Entity aliases include article variants for spaCy compatibility
- Multi-hop is opt-in (default off) for backward compat
