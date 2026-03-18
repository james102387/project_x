# Session — 2026-03-16 (cont.)

## Goal
Implement D2 Phase 1: KG ingestion pipeline (NER-based + hand-curated loaders).

## Completed
- `src/crystal/ingest/schema.py`: Triplet, IngestResult dataclasses
- `src/crystal/ingest/ner.py`: 5 dep-tree patterns (copular, possessive, active, passive, prepositional)
- `src/crystal/ingest/loader.py`: CSV (auto-header), JSON (object + flat array)
- `src/crystal/ingest/__init__.py`: ingest() auto-detect, build_kg()
- `src/crystal/ingest/__main__.py`: CLI entry point
- Test fixtures: CSV, JSON, text samples
- 53 new tests, 331 total passing
- Also: cached_content_token_count added to _extract_usage()
