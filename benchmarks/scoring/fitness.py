"""
Ralph Wiggum fitness function — single metric for the self-testing loop.

Binary correctness per question, aggregated as a percentage.
  - Positive cases: correct if ALL match_strings present in response
  - Negative cases: correct if response contains an abstention phrase

One number. This drives the loop.
Existing rubric.py is retained for diagnostic breakdowns.
"""

from __future__ import annotations

from benchmarks.scoring.rubric import is_abstention


def binary_correct(
    response: str,
    match_strings: list[str],
    is_negative: bool = False,
) -> bool:
    """Score a single response as binary correct/incorrect.

    Positive: all match_strings present (case-insensitive).
    Negative: response is an abstention.
    """
    if is_negative:
        return is_abstention(response)

    if not match_strings:
        return False

    response_lower = response.lower()
    return all(m.lower() in response_lower for m in match_strings)


def fitness_score(
    results: list[dict],
) -> float:
    """Compute the Ralph Wiggum fitness score for a batch of results.

    Each dict must have: response, match_strings.
    Optional: is_negative (bool, default False).

    Returns a float in [0.0, 1.0] — percentage of questions answered correctly.
    """
    if not results:
        return 0.0

    correct = sum(
        1 for r in results
        if binary_correct(
            r["response"],
            r["match_strings"],
            r.get("is_negative", False),
        )
    )
    return correct / len(results)


def evaluate_cases(
    cases: list[tuple[str, str, list[str], bool]],
    run_fn,
) -> tuple[float, list[dict]]:
    """Run a batch of benchmark cases through a function and score them.

    cases: list of (question, golden_answer, match_strings, is_negative)
    run_fn: callable that takes a question string and returns a response string

    Returns (fitness_score, detailed_results).
    """
    results: list[dict] = []

    for question, golden_answer, match_strings, is_negative in cases:
        response = run_fn(question)
        correct = binary_correct(response, match_strings, is_negative)
        results.append({
            "question": question,
            "golden_answer": golden_answer,
            "response": response,
            "match_strings": match_strings,
            "is_negative": is_negative,
            "correct": correct,
        })

    score = fitness_score(results)
    return score, results
