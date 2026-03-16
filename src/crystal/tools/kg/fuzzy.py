"""Fuzzy string matching for KG entity and predicate resolution.

Uses rapidfuzz for near-miss typos, pluralization, and word reordering.
Sub-millisecond on typical KG candidate sets (< 100 entities, < 15 predicates).
"""

from __future__ import annotations

from collections.abc import Iterable

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
    if best_match is not None and best_score >= threshold:
        return (best_match, best_score)
    return None
