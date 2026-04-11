"""
B5: Stratified sampler for Tier 2 benchmark questions.

Draws a reproducible stratified sample from the full accepted question corpus,
ensuring proportional representation of each predicate type.

Used for the larger (100-200 question) benchmark tier that runs once and
is cached — scoring works against the cached results.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _extract_predicate_from_case(case: dict | tuple) -> str:
    """Extract the predicate from a review case dict or benchmark tuple.

    For review dicts with source_triplet, uses the predicate field.
    For tuples without metadata, infers from the question text.
    """
    if isinstance(case, dict):
        triplet = case.get("source_triplet")
        if triplet and len(triplet) >= 2:
            return triplet[1]

        if case.get("is_negative"):
            return "_negative"
        return "_unknown"

    if isinstance(case, tuple) and len(case) >= 4 and case[3]:
        return "_negative"
    return "_unknown"


def sample_from_review_cases(
    cases: list[dict],
    n: int = 200,
    seed: int = 42,
    min_per_stratum: int = 5,
    negative_fraction: float = 0.05,
) -> list[dict]:
    """Stratified sample from review case dicts (with source_triplet metadata).

    Groups by predicate, samples proportionally, ensures minimum per stratum.
    """
    rng = random.Random(seed)

    strata: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        pred = _extract_predicate_from_case(case)
        strata[pred].append(case)

    negatives = strata.pop("_negative", [])
    unknowns = strata.pop("_unknown", [])

    n_negatives = max(min_per_stratum, int(n * negative_fraction))
    n_positives = n - n_negatives

    total_positive = sum(len(v) for v in strata.values())
    if total_positive == 0:
        return []

    sampled: list[dict] = []

    allocation: dict[str, int] = {}
    for pred, pool in strata.items():
        share = max(min_per_stratum, int(n_positives * len(pool) / total_positive))
        allocation[pred] = min(share, len(pool))

    allocated = sum(allocation.values())
    if allocated > n_positives:
        scale = n_positives / allocated
        allocation = {p: max(min_per_stratum, int(v * scale)) for p, v in allocation.items()}

    for pred, count in allocation.items():
        pool = strata[pred]
        k = min(count, len(pool))
        sampled.extend(rng.sample(pool, k))

    if negatives:
        k = min(n_negatives, len(negatives))
        sampled.extend(rng.sample(negatives, k))

    if unknowns and len(sampled) < n:
        remaining = n - len(sampled)
        sampled.extend(rng.sample(unknowns, min(remaining, len(unknowns))))

    rng.shuffle(sampled)
    return sampled[:n]


def sample_benchmark_cases(
    all_cases: list[tuple[str, str, list[str], bool]],
    n: int = 200,
    seed: int = 42,
) -> list[tuple[str, str, list[str], bool]]:
    """Stratified sample from benchmark tuples (without source_triplet).

    Uses the answerability module to infer predicates for stratification.
    """
    from benchmarks.answerability import infer_predicate

    rng = random.Random(seed)

    strata: dict[str, list[tuple]] = defaultdict(list)
    for case in all_cases:
        question, _golden, _match, is_negative = case
        if is_negative:
            strata["_negative"].append(case)
        else:
            pred = infer_predicate(question) or "_other"
            strata[pred].append(case)

    negatives = strata.pop("_negative", [])

    n_negatives = max(5, int(n * 0.05))
    n_positives = n - n_negatives
    total_positive = sum(len(v) for v in strata.values())

    if total_positive == 0:
        return []

    sampled: list[tuple] = []
    for pred, pool in strata.items():
        share = max(5, int(n_positives * len(pool) / total_positive))
        k = min(share, len(pool))
        sampled.extend(rng.sample(pool, k))

    if negatives:
        k = min(n_negatives, len(negatives))
        sampled.extend(rng.sample(negatives, k))

    rng.shuffle(sampled)
    return sampled[:n]


def export_sample(
    cases: list[dict],
    output_path: Path | None = None,
) -> Path:
    """Export sampled cases to a JSON file for caching."""
    output_path = output_path or RESULTS_DIR / "tier2_sample.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total": len(cases),
        "cases": cases,
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_sample(path: Path | None = None) -> list[dict]:
    """Load a cached sample from JSON."""
    path = path or RESULTS_DIR / "tier2_sample.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("cases", [])


def sample_summary(cases: list[dict]) -> dict:
    """Return a summary of predicate distribution in a sample."""
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        pred = _extract_predicate_from_case(case)
        counts[pred] += 1
    return {
        "total": len(cases),
        "by_predicate": dict(sorted(counts.items())),
    }
