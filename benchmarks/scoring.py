"""
Auto-scoring for benchmark responses.

Scoring rule: a response is "correct" if ALL required match_strings appear
in the response (case-insensitive). This catches both exact answers and
answers embedded in longer text.

The rubric scorer extends this with three-dimensional quality scoring.
"""

from __future__ import annotations

from dataclasses import asdict

from benchmarks.rubric import score_rubric


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


def score_batch_rubric(
    results: list[dict],
) -> dict:
    """Score a batch with both binary and rubric dimensions.

    Each result dict must have: question, response, match_strings, ground_truth.
    Optional keys: kg_results (list[dict]), is_negative (bool).

    Returns summary stats + per-dimension averages + per-question details.
    """
    scored = []
    correct = 0
    totals = {"accuracy": 0.0, "specificity": 0.0, "no_hallucination": 0.0}
    positive_count = 0
    negative_count = 0

    for r in results:
        is_correct = score_response(r["response"], r["match_strings"])
        if is_correct:
            correct += 1

        is_negative = r.get("is_negative", False)
        kg_results = r.get("kg_results", None)

        rubric = score_rubric(
            response=r["response"],
            match_strings=r["match_strings"],
            kg_results=kg_results,
            is_negative=is_negative,
        )

        if is_negative:
            negative_count += 1
        else:
            positive_count += 1

        totals["accuracy"] += rubric.accuracy
        totals["specificity"] += rubric.specificity
        totals["no_hallucination"] += rubric.no_hallucination

        scored.append({
            **r,
            "correct": is_correct,
            "rubric": asdict(rubric),
        })

    total = len(results)
    avg = {k: v / total if total > 0 else 0.0 for k, v in totals.items()}

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "hallucination_rate": 1.0 - (correct / total) if total > 0 else 1.0,
        "positive_cases": positive_count,
        "negative_cases": negative_count,
        "rubric_averages": avg,
        "details": scored,
    }
