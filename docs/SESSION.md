# Session Notes — 2026-04-16

## UI split — per-tab modules

### Completed
- Extracted `src/crystal/ui/app.py` (1,970 lines) into a composition root + per-tab modules:
  - `ui/state.py` (26) — compiled graph, `KG_MODES`, `_default_kg`, `_DEFAULT_KG_MODE`.
  - `ui/formatting.py` (154) — `_kg_info`, `_format_kg_stats`, `_format_kg_facts`, `_kg_facts_df`, `_format_kg_banner`, `_format_ingest_target`, `_label_for_kg`, origin/route label tables. Zero Gradio imports.
  - `ui/tabs/ask.py` (142) — `ask_question` + `build_ask_tab`.
  - `ui/tabs/ingest.py` (619) — ingestion, proposed answers, comparison actions + `build_ingest_tab`.
  - `ui/tabs/kg.py` (405) — explorer helpers + `switch_kg_mode`/`import_structured_data` + `build_kg_tab`. Takes `IngestTab` so cross-tab outputs (ingest_kg_stats, ingest_target_label) stay explicit.
  - `ui/tabs/review.py` (852) — batch review, accept/reject, benchmark, RW loop + `build_review_tab`.
  - `ui/app.py` (93) — `build_ui()` composes tabs around a shared `kg_state` + `kg_banner`. Re-exports symbols the tests expect (`_kg_info`, `_format_kg_stats`, `_format_kg_facts`, `_default_kg_info`, `import_structured_data`, `KG_MODES`, `build_ui`, `main`).
- Every `build_<tab>` returns a dataclass (`AskTab`, `IngestTab`, `KgTab`, `ReviewTab`) of component handles.
- Verified: `python -c "from crystal.ui.app import build_ui; build_ui()"` builds cleanly. `pytest tests/` → 1,272 passed, 8 skipped (matches pre-refactor baseline).
- DEVLOG + DEVLOG_ARCHIVE hygiene: added a "UI Split" entry; moved the oldest 2026-04-12 entry ("Extraction Baselines + Review Pipeline") to `DEVLOG_ARCHIVE.md`; dropped the stale "Not doing a UI split" decision from the previous 2026-04-16 entry; updated the Active Focus cleanup bullet.

### Key files touched
- Added: `src/crystal/ui/state.py`, `src/crystal/ui/formatting.py`, `src/crystal/ui/tabs/__init__.py`, `src/crystal/ui/tabs/ask.py`, `src/crystal/ui/tabs/ingest.py`, `src/crystal/ui/tabs/kg.py`, `src/crystal/ui/tabs/review.py`.
- Rewrote: `src/crystal/ui/app.py` (1,970 → 93 lines; `build_ui()` + re-exports).
- Unchanged by design: `src/crystal/ui/__init__.py`, `src/crystal/ui/__main__.py`, `src/crystal/ui/dev.py`.
- Docs: `docs/DEVLOG.md`, `docs/DEVLOG_ARCHIVE.md`, `docs/SESSION.md`.

### Notes for future work
- Two `SqliteKnowledgeGraph` import sites remain (intentional — tab modules need the isinstance check locally). Could consolidate via a typing protocol if we end up with more polymorphic branching.
- `save_pending_decisions` in `tabs/ingest.py` is still a "accept all remaining" stopgap; the row-level accept/reject UI from the original file was never wired up.
- `dev.py` path is unchanged (`gradio src/crystal/ui/dev.py --watch-dirs src/crystal`) — it imports `build_ui` from `crystal.ui.app` same as before.
