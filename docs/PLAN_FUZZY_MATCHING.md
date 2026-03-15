# Plan: Fuzzy KG Matching

Two-phase plan. Phase 1 (Demo) solves the immediate brittleness problem with entity aliases and lightweight string fuzzing. Phase 2 (Future) adds embedding-based semantic matching for genuine paraphrases.

---

## Phase 1: Entity Aliases + Fuzzy String Match (Demo — D5)

### Problem

Entity matching in `detectors/kg.py` is exact substring only. Predicate matching has an alias table but no fallback. This means:

- "How old is Korth?" → no match (entity index has `"grand vizier korth"`, not `"korth"`)
- "What is the capital of Remulack?" → no match (typo)
- "Tell me about the Veldran Guard" → no match (entity is `"the veldran guard"`, casing aside the article position varies)
- "What is Remulak's city capital?" → no match ("city capital" not in predicate aliases)

These are all cases where a human would understand the intent. On a controlled dataset like Remulak this is tolerable; on user-supplied documents it's a dealbreaker. If the demo can't handle minor phrasing variations, it looks broken regardless of how good the pipeline is.

### Scope

Two additions:

1. **Entity alias tables** — per-dataset mappings of common short forms / alternate names to canonical entity strings. O(1) dict lookup, no new dependencies.
2. **Fuzzy string matching** — `rapidfuzz` for near-miss typos, pluralization, and word reordering. Sub-millisecond on thousands of candidates. One new lightweight dependency.

No embeddings, no model downloads, no GPU. This stays fast and deterministic.

### Architecture: 3-Tier Resolution Cascade

Resolution fires top-down; first match wins. Applied to both entities and predicates.

```
Query term
  │
  ├─ Tier 1: Exact match (O(1) hash lookup)         ← current behavior
  │    └─ hit → return
  │
  ├─ Tier 2: Alias table (O(1) dict lookup)          ← new for entities, exists for predicates
  │    └─ hit → return
  │
  └─ Tier 3: Fuzzy string match (rapidfuzz)           ← new
       └─ score >= threshold → return
       └─ miss → no match
```

### Entity Aliases

Add `entity_aliases: dict[str, str]` param to `KnowledgeGraph.__init__`. Maps alternate surface forms to canonical entity names (lowercased).

Example aliases for Remulak (in `data/remulak.py`):

```python
ENTITY_ALIASES: dict[str, str] = {
    "korth": "grand vizier korth",
    "vizier korth": "grand vizier korth",
    "korth vellan": "grand vizier korth",
    "quorum": "the quorum of twelve",
    "quorum of twelve": "the quorum of twelve",
    "veldran guard": "the veldran guard",
    "draya kess": "marshal draya kess",
    "marshal kess": "marshal draya kess",
    "orath yenn": "physicist orath yenn",
    "aamra sel": "vizier aamra sel",
    "festival of vohn": "the festival of vohn",
    "pellux cup": "the pellux cup",
    "sulari fracture war": "the sulari fracture war",
    "fracture war": "the sulari fracture war",
    "crucible mines": "the crucible mines",
    "vreth": "the vreth",
}
```

### Fuzzy String Match

- Library: `rapidfuzz` (pure C extension, no model downloads)
- Scorer: `fuzz.token_sort_ratio` — handles word reordering and partial matches
- Default threshold: 80 (configurable per-KG instance)
- Candidate sets kept narrow:
  - Entity fuzzy: match against `kg.entities` (typically < 100 strings)
  - Predicate fuzzy: match against predicates for the *resolved subject only* (typically < 15)

```python
# src/crystal/tools/kg/fuzzy.py

from rapidfuzz import fuzz

def fuzzy_match(
    query: str,
    candidates: Iterable[str],
    threshold: float = 80.0,
) -> tuple[str, float] | None:
    """Best rapidfuzz match above threshold. Returns (match, score) or None."""
    best_match = None
    best_score = 0.0
    for candidate in candidates:
        score = fuzz.token_sort_ratio(query.lower(), candidate.lower())
        if score > best_score:
            best_score = score
            best_match = candidate
    if best_score >= threshold:
        return (best_match, best_score)
    return None
```

### Detection Metadata

When alias or fuzzy match fires, enrich the detection dict so downstream components know the match quality:

```python
{
    "tool": "kg",
    "entity": "grand vizier korth",    # canonical
    "match_tier": "alias",             # "exact", "alias", or "fuzzy"
    "match_score": 1.0,                # 1.0 for exact/alias, 0.0-1.0 for fuzzy
    "original_text": "Korth",          # what the user actually wrote
    ...
}
```

This metadata flows to the compiler and rubric for confidence-aware decisions.

### Files to Change

| File | Change |
|------|--------|
| `src/crystal/tools/kg/fuzzy.py` (new) | `fuzzy_match()` function, standalone and testable |
| `src/crystal/tools/kg/graph.py` | Add `entity_aliases` param to `__init__`. Add `_resolve_entity()` method (exact → alias → fuzzy cascade). Extend `_resolve_predicate()` to fall through to fuzzy. Add `fuzzy_threshold` instance attribute. |
| `src/crystal/detectors/kg.py` | `find_entity_spans`: after exact substring scan yields nothing, extract noun phrases from spaCy doc and try `kg._resolve_entity()` on each. Return fuzzy matches with metadata. |
| `src/crystal/data/remulak.py` | Add `ENTITY_ALIASES` dict. |
| `src/crystal/tools/kg/remulak.py` | Pass `entity_aliases=ENTITY_ALIASES` to constructor. |
| `requirements.txt` | Add `rapidfuzz>=3.0.0` |

### Test Plan

**Golden test cases** (add to `tests/golden/test_cases.py`):
- Alias: `"How old is Korth?"` → 142 (tier 2, entity alias)
- Alias: `"Tell me about the Veldran Guard"` → entity alias resolves (tier 2)
- Typo: `"What is the capital of Remulack?"` → Zelphos (tier 3, fuzzy)
- Word reorder: `"What is Remulak's city capital?"` → Zelphos (tier 3, predicate fuzzy)
- True negative: `"What is the capital of Zorgon?"` → no match at any tier

**Unit tests** (`tests/unit/test_fuzzy.py`):
- `fuzzy_match()` returns correct match above threshold
- `fuzzy_match()` returns None below threshold
- `_resolve_entity()` cascade: exact beats alias beats fuzzy
- `_resolve_predicate()` cascade preserved
- Entity aliases wired correctly in Remulak instance

### Implementation Order

1. `src/crystal/tools/kg/fuzzy.py` — standalone, testable in isolation
2. `ENTITY_ALIASES` in `data/remulak.py`
3. `graph.py` — `_resolve_entity()`, extend `_resolve_predicate()`, wire `entity_aliases`
4. `tools/kg/remulak.py` — pass aliases
5. `detectors/kg.py` — fuzzy fallback in `find_entity_spans`
6. Golden tests + unit tests

---

## Phase 2: Embedding Similarity (Future — F4)

### Problem

Fuzzy string matching handles typos and word reordering but cannot bridge genuine semantic gaps: "ruler" → "leader", "birthplace" → "where born", "What star is Remulak near?" → "star system". These require understanding *meaning*, not just character overlap.

### Scope

Add a 4th tier to the resolution cascade:

```
  Tier 3: Fuzzy string match
       └─ miss ↓
  Tier 4: Embedding similarity (sentence-transformers, cosine)
       └─ score >= threshold → return
       └─ miss → no match
```

### Embedding Index

- Library: `sentence-transformers` with `all-MiniLM-L6-v2` (~80MB model)
- Pre-compute normalized embeddings for all entities and all predicates at `_build()` time
- At query time: encode the query term, compute cosine similarity against the pre-built index
- Default threshold: 0.7 (configurable)
- **Optional** — controlled by `enable_embeddings: bool = False` on the KG constructor
- Lazy-loaded: model only downloaded/loaded if embeddings are enabled

```python
# Extension to src/crystal/tools/kg/fuzzy.py

class EmbeddingIndex:
    """Pre-computed embedding index for cosine nearest-neighbor lookup."""

    def __init__(self, terms: list[str], model_name: str = "all-MiniLM-L6-v2"):
        ...

    def nearest(self, query: str, threshold: float = 0.7) -> tuple[str, float] | None:
        ...
```

### Semantic paraphrase cases this unlocks

- "Who rules Remulak?" — "rules" → embedding matches "leader" (predicate alias already covers "ruler", but not all verb forms)
- "What star is Remulak near?" — "star near" → embedding matches "star system"
- "Where is Korth from?" — "from" → embedding matches "birthplace"
- "What do Remulaki look like?" — "look like" → embedding matches "distinguishing feature"

### Files to Change

- `src/crystal/tools/kg/fuzzy.py` — add `EmbeddingIndex` class
- `src/crystal/tools/kg/graph.py` — add `enable_embeddings` param, wire tier 4 into cascade
- `requirements.txt` / `pyproject.toml` — add `sentence-transformers` as optional (`crystal[embeddings]`)

### Prerequisites

- Phase 1 complete (cascade infrastructure, `_resolve_entity()`, `_resolve_predicate()` already support multiple tiers)
- Decision needed: is the ~80MB model download acceptable for the demo, or is this strictly a self-hosted / enterprise feature?

### Test Plan

- `EmbeddingIndex.nearest()` finds semantic matches above threshold
- `EmbeddingIndex.nearest()` returns None for unrelated terms
- Cascade order: exact → alias → fuzzy → embedding (earlier tiers take priority)
- Golden cases for semantic paraphrases that fuzzy cannot catch
- Benchmark: report match tier distribution to measure how often embedding fires vs. earlier tiers
