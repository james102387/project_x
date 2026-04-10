# Session — 2026-04-09 (continued)

## Detector Known Gaps — Resolved

### Citation-format entity spans
- **Problem:** spaCy splits "384 U.S. 436" into three tokens (384, U.S., 436), so `find_entity_spans` never matched the full citation string against the KG.
- **Fix:** Added a regex pre-scan (Tier 0) in `find_entity_spans` that runs `_CITATION_PATTERN` over raw text before the spaCy-based cascade. Reuses the same citation regex from `legal_ontology.py`. Matched citations are resolved via `kg._resolve_entity()` and injected into matched_ranges to prevent overlap with later tiers.

### WH-word-aware predicate resolution
- **Problem:** "decided" always mapped to `date_filed` via `QUESTION_PREDICATE_MAP`, but "Who decided X?" expects judges.
- **Fix:** Added `_WH_PREDICATE_OVERRIDES` dict keyed on `(predicate_phrase, wh_word)` tuples. New `_leading_wh_word(doc)` helper extracts the first non-punctuation token if it's a question word. Override is checked before the default map in `detect_kg_query`.
- Override pairs: `("decided", "who") → "judges"`, `("ruled", "who") → "judges"`.

### Tests added
- `TestCitationSpanDetection`: 2 tests (entity found + targeted lookup)
- `TestWhWordPredicateOverride`: 3 tests (who→judges, when→date, wh-word extraction)
- Full suite: 664 passed, 5 skipped

### Files changed
- `src/crystal/detectors/kg.py` — regex pre-scan, WH-word override, `_leading_wh_word()`
- `tests/unit/detectors/test_kg.py` — 5 new tests with legal KG fixture
- `benchmarks/ground_truth/legal.py` — `LEGAL_KNOWN_GAPS` emptied, resolution comments added
- `docs/DEVLOG.md` — Active Focus updated (gaps resolved, test count)
