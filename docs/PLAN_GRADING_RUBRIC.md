# Plan: Grading Rubric

Two-phase plan. Phase 1 (Demo) proves the safety invariant is measurable across all paths. Phase 2 (Future) expands to a full multi-dimensional scoring system for real-domain evaluation.

---

## Phase 1: Minimal Quality Rubric (Demo — D1)

### Problem

Current scoring is binary: a response either contains all `match_strings` or it doesn't. This tells us nothing about the augmented and fallback paths — the exact paths where Crystal could theoretically make things *worse* than a naked LLM. To credibly claim "Crystal never degrades response quality," we need to measure quality, not just factual hit/miss.

### Scope

Three dimensions, scored per-response as floats in [0.0, 1.0]:

| Metric | What it measures |
|--------|------------------|
| Factual Accuracy | Are the core factual claims correct? (Extends current binary match to a ratio.) |
| Specificity | Does the response include exact KG values, not vague paraphrases? |
| No-Hallucination | Does the response avoid fabricating facts not in the KG? On KG misses, does the system correctly abstain? |

No weighting, no composite score. Each dimension reported independently — simpler to explain, easier to trust.

### Factual Accuracy

Extends `score_response()` from bool to a ratio:

```python
def accuracy_score(response: str, match_strings: list[str]) -> float:
    hits = sum(1 for m in match_strings if m.lower() in response.lower())
    return hits / len(match_strings) if match_strings else 0.0
```

- 1.0 = all match strings present (current "correct")
- 0.5 = half present (partial credit)
- 0.0 = none present

### Specificity

Did the response include the exact KG object values?

```python
def specificity_score(response: str, kg_results: list[dict]) -> float:
    if not kg_results:
        return 0.0
    hits = sum(1 for r in kg_results if r["object"].lower() in response.lower())
    return hits / len(kg_results)
```

Distinguishes "The capital of Remulak is Zelphos" (1.0) from "Remulak has a capital city" (0.0).

### No-Hallucination

Two sub-checks depending on path:

**Positive cases (KG hit):** Extract factual tokens from the response and check what fraction trace to `kg_results`. Ungrounded factual claims reduce the score.

```python
def grounding_score(response: str, kg_results: list[dict]) -> float:
    if not kg_results:
        return 0.0
    grounded_values = {r["object"].lower() for r in kg_results}
    grounded_values |= {r["subject"].lower() for r in kg_results}
    # Count response tokens traceable to KG vs. factual-looking but ungrounded
    ...
```

**Negative cases (KG miss):** The system should abstain rather than fabricate.

```python
ABSTENTION_PHRASES = {
    "i don't have", "no information", "not in the knowledge",
    "i'm not sure", "cannot determine", "no match",
}

def calibration_score(response: str, is_negative_case: bool) -> float:
    if not is_negative_case:
        return 1.0
    response_lower = response.lower()
    return 1.0 if any(p in response_lower for p in ABSTENTION_PHRASES) else 0.0
```

### Ground Truth Extension

Current format:

```python
(question, ground_truth_answer, match_strings)
```

Extended format (backward compatible — scoring checks tuple length):

```python
(question, ground_truth_answer, match_strings, is_negative)
```

Where `is_negative: bool` indicates the system should abstain. Defaults to `False`.

Add 5 adversarial negative cases:
- "What is the GDP of Remulak?" (no GDP triplet → subject scan → `kg_augmented`)
- "Who was the leader before Grand Vizier Korth?" (predecessor triplet exists but predicate extraction yields "leader before" which doesn't match → subject scan → `kg_augmented`)
- "What is the second largest city on Remulak?" (no such triplet → subject scan → `kg_augmented`)
- "How many oceans does Remulak have?" (no ocean data → subject scan → `kg_augmented`)
- "What is the population of Zelphos?" (Zelphos is object-only, not a subject → `no_match`)

**Implementation note:** Entity-found/predicate-not-found queries route as `kg_augmented` (not `kg_answerable`), injecting all known entity facts as definitional grounding for LLM reasoning. The `lookup_type` field (`"targeted"` vs `"subject_scan"`) in the detection result drives this classification.

### Files to Change

| File | Change |
|------|--------|
| `benchmarks/rubric.py` (new) | `accuracy_score()`, `specificity_score()`, `grounding_score()`, `calibration_score()`. Simple dataclass for results. |
| `benchmarks/scoring.py` | Keep existing `score_response()` / `score_batch()`. Add `score_batch_rubric()` that returns per-dimension scores alongside binary. |
| `benchmarks/ground_truth.py` | Add `is_negative` field to existing cases. Add 5 adversarial negatives. |
| `benchmarks/run_benchmark.py` | Pass `kg_results` to scoring. Add per-dimension breakdown to report. |

### Test Plan

- Unit tests for each scorer in isolation (`tests/unit/test_rubric.py`)
- Run rubric on existing treatment results (100% binary) — accuracy and specificity should be high
- Run rubric on baseline results (0% binary) — accuracy should be low, specificity 0.0
- Verify backward compat: `score_batch()` still works with old 3-tuple ground truth

### Implementation Order

1. `benchmarks/rubric.py` — scorers + dataclass
2. Unit tests
3. Extend `benchmarks/ground_truth.py` with `is_negative` + adversarial cases
4. `benchmarks/scoring.py` — `score_batch_rubric()` wrapper
5. `benchmarks/run_benchmark.py` — wire rubric into reporting
6. Run benchmarks, verify

---

## Phase 2: Full Multi-Dimensional Rubric (Future — F5)

### Problem

The minimal rubric proves the safety invariant. The full rubric provides the evaluation infrastructure needed for real-domain datasets where questions are harder, answers are longer, and partial credit matters.

### Additional Dimensions

Extends Phase 1 from 3 → 6 scored metrics:

| New Metric | Weight | What it measures |
|------------|--------|------------------|
| Completeness | 0.10 | For multi-fact queries, what fraction of expected facts appeared? |
| Token Efficiency | 0.10 | Ratio of useful information to total response length |
| Confidence Calibration | 0.05 | Refined version of Phase 1 abstention check with graduated scoring |

Phase 1 metrics are also weighted:

| Phase 1 Metric | Weight |
|----------------|--------|
| Factual Accuracy | 0.35 |
| Factual Grounding (No-Hallucination) | 0.25 |
| Specificity | 0.15 |

Composite score = weighted sum. Weights sum to 1.0.

### Completeness

For queries returning multiple KG facts (e.g. "Tell me about Draveth" → 5 triplets):

```python
def completeness_score(response: str, expected_triplets: list[tuple]) -> float:
    if not expected_triplets:
        return 1.0  # single-fact query, trivially satisfied
    hits = sum(
        1 for s, p, o in expected_triplets
        if o.lower() in response.lower()
    )
    return hits / len(expected_triplets)
```

### Token Efficiency

```python
def efficiency_score(response: str, kg_results: list[dict]) -> float:
    if not kg_results:
        return 0.0
    fact_text = " ".join(f"{r['subject']} {r['predicate']} {r['object']}" for r in kg_results)
    fact_tokens = count_tokens(fact_text)
    response_tokens = count_tokens(response)
    if response_tokens == 0:
        return 0.0
    return min(1.0, fact_tokens / response_tokens)
```

### Refined Confidence Calibration

Graduated scoring (not just binary abstain/fabricate):

- 1.0 = explicit abstention ("I don't have information about...")
- 0.5 = hedged response ("I'm not certain, but...")
- 0.0 = confident fabrication

### Ground Truth Extension

Extends Phase 1 format further:

```python
(question, ground_truth_answer, match_strings, expected_triplets, is_negative)
```

Where `expected_triplets: list[tuple[str, str, str]]` lists the KG triplets the response should be grounded in. `None` for cases where any matching triplets are acceptable.

### Composite Scoring

```python
@dataclass
class DimensionScore:
    name: str
    score: float
    weight: float

@dataclass
class RubricScore:
    accuracy: DimensionScore
    grounding: DimensionScore
    specificity: DimensionScore
    completeness: DimensionScore
    efficiency: DimensionScore
    calibration: DimensionScore
    composite: float  # weighted sum

DIMENSION_WEIGHTS = {
    "accuracy": 0.35,
    "grounding": 0.25,
    "specificity": 0.15,
    "completeness": 0.10,
    "efficiency": 0.10,
    "calibration": 0.05,
}
```

### Files to Change

- `benchmarks/rubric.py` — add `completeness_score()`, `efficiency_score()`, refined `calibration_score()`, `DIMENSION_WEIGHTS`, `RubricScore` with composite calculation
- `benchmarks/ground_truth.py` — add `expected_triplets` field to cases
- `benchmarks/run_benchmark.py` — add composite score and per-dimension averages to report JSON

### Prerequisites

- Phase 1 rubric complete and passing
- Augmented benchmark cases (D4) in place so there's data worth scoring with the full rubric
