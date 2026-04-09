# Session — 2026-04-09

## Phase 1a Bulk Ingestion + Review UI Fixes

### Done
- Updated TODO.md with phased ingestion roadmap (Phase 1a/1b/1c structured, Phase 2 unstructured)
- Auto-accepted 100 existing structured-API questions in `review/pending_questions.json`
- Created `scripts/bulk_ingest.py` for Phase 1a bulk ingestion
- Executed Phase 1a: 500 SCOTUS cases → 1,305 triplets → 1,321 auto-accepted questions
- SQLite KG: 1,301 triplets, 496 subjects at `data/legal.sqlite`
- Total golden answers: 1,421 (100 original + 1,321 Phase 1a)
- Fixed Review UI: batch discovery, Gradio 6 DataFrame compatibility, pre-populated table on load
- All 659 tests passing, 5 skipped

### Phase 1b/1c deferred
- CourtListener citation graph + judge bios require `COURTLISTENER_API_TOKEN`
- Set token in `.env` and run corresponding scripts when ready
