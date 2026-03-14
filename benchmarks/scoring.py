"""
Auto-scoring for benchmark responses.

Scoring rule: a response is "correct" if ALL required match_strings appear
in the response (case-insensitive). This catches both exact answers and
answers embedded in longer text.
"""

from __future__ import annotations


def score_response(response: str, match_strings: list[str]) -> bool:
    """Return True if the response contains ALL required match strings."""
    response_lower = response.lower()
    return all(m.lower() in response_lower for m in match_strings)


def score_batch(
    results: list[dict],
) -> dict:
    """Score a batch of benchmark results.

    Each result dict must have: question, response, match_strings, ground_truth.
    Returns summary stats + per-question details.
    """
    scored = []
    correct = 0

    for r in results:
        is_correct = score_response(r["response"], r["match_strings"])
        if is_correct:
            correct += 1
        scored.append({
            **r,
            "correct": is_correct,
        })

    total = len(results)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "hallucination_rate": 1.0 - (correct / total) if total > 0 else 1.0,
        "details": scored,
    }
