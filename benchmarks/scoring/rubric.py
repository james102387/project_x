"""
Quality rubric scorers for benchmark evaluation.

Three dimensions scored per-response as floats in [0.0, 1.0]:
  - Factual Accuracy: ratio of match_strings present in the response
  - Specificity: ratio of exact KG object values present in the response
  - No-Hallucination: grounding check (positive cases) or abstention check (negative cases)

No weighting, no composite score. Each dimension reported independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


ABSTENTION_PHRASES = (
    # Crystal-generated refusals
    "no match",
    "no kg match",
    # Explicit "I don't know" family
    "i don't have",
    "i don't know",
    "i do not have",
    "i do not know",
    "don't have information",
    "don't have enough information",
    "do not have information",
    "do not have enough information",
    # Hedging / uncertainty
    "i'm not sure",
    "i am not sure",
    "i'm unable to",
    "i am unable to",
    "i cannot",
    "i can't",
    # Absence signals
    "no information",
    "no data",
    "no record",
    "no relevant",
    "not in the knowledge",
    "not available",
    "not found",
    "not contain",
    "not present",
    "not included",
    # Inability verbs
    "cannot determine",
    "cannot answer",
    "cannot find",
    "cannot confirm",
    "unable to find",
    "unable to determine",
    "unable to answer",
    "unable to confirm",
    # Absence of data phrasing
    "doesn't contain",
    "does not contain",
    "doesn't include",
    "does not include",
    "there is no",
    "there are no",
    "lack the data",
    "lack the information",
    "outside the scope",
    "beyond what is available",
)

_ABSTENTION_PATTERN = re.compile(
    r"(?:i\s+)?(?:can(?:'t|not)|don(?:'t|\ not)|do\s+not|am\s+(?:not\s+able|unable))\s+"
    r"(?:to\s+)?(?:find|answer|determine|confirm|provide|verify|locate|access)",
    re.IGNORECASE,
)


@dataclass
class RubricResult:
    """Per-response rubric scores."""
    accuracy: float
    specificity: float
    no_hallucination: float


def accuracy_score(response: str, match_strings: list[str]) -> float:
    """Fraction of required match_strings found in the response.

    1.0 = all present, 0.0 = none present.
    Extends the binary score_response() to a ratio.
    """
    if not match_strings:
        return 0.0
    hits = sum(1 for m in match_strings if m.lower() in response.lower())
    return hits / len(match_strings)


def specificity_score(response: str, kg_results: list[dict]) -> float:
    """Fraction of KG object values that appear verbatim in the response.

    Distinguishes "The capital of Remulak is Zelphos" (1.0)
    from "Remulak has a capital city" (0.0).

    Returns 1.0 when kg_results is empty (no KG facts to check against —
    specificity is vacuously satisfied).
    """
    if not kg_results:
        return 1.0
    hits = sum(1 for r in kg_results if r["object"].lower() in response.lower())
    return hits / len(kg_results)


def grounding_score(response: str, kg_results: list[dict]) -> float:
    """For positive (KG-hit) cases: fraction of response factual tokens
    traceable to KG subject/object values.

    A simple heuristic: check that each KG value (subject + object) appears
    in the response. Ungrounded factual-looking content reduces the score.

    Returns 1.0 when kg_results is empty (nothing to ground against).
    """
    if not kg_results:
        return 1.0
    grounded_values = set()
    for r in kg_results:
        grounded_values.add(r["object"].lower())
        grounded_values.add(r["subject"].lower())
    hits = sum(1 for v in grounded_values if v in response.lower())
    return hits / len(grounded_values)


def is_abstention(response: str) -> bool:
    """Check whether a response is an abstention (refusal to answer).

    Uses a two-tier approach:
      1. Fast phrase scan against ABSTENTION_PHRASES
      2. Regex fallback for "I can't/cannot/don't [verb]" patterns
    """
    response_lower = response.lower()
    if any(p in response_lower for p in ABSTENTION_PHRASES):
        return True
    if _ABSTENTION_PATTERN.search(response_lower):
        return True
    return False


def calibration_score(response: str, is_negative: bool) -> float:
    """For negative (KG-miss) cases: did the system abstain rather than fabricate?

    Returns 1.0 if:
      - The case is positive (not a negative case — calibration is vacuously satisfied)
      - The case is negative AND the response is an abstention

    Returns 0.0 if the case is negative and the response does not abstain.
    """
    if not is_negative:
        return 1.0
    return 1.0 if is_abstention(response) else 0.0


def score_rubric(
    response: str,
    match_strings: list[str],
    kg_results: list[dict] | None = None,
    is_negative: bool = False,
) -> RubricResult:
    """Score a single response across all three rubric dimensions."""
    return RubricResult(
        accuracy=accuracy_score(response, match_strings),
        specificity=specificity_score(response, kg_results or []),
        no_hallucination=_no_hallucination_score(
            response, kg_results or [], is_negative
        ),
    )


def _no_hallucination_score(
    response: str,
    kg_results: list[dict],
    is_negative: bool,
) -> float:
    """Combined no-hallucination score: grounding for positive, calibration for negative."""
    if is_negative:
        return calibration_score(response, is_negative=True)
    return grounding_score(response, kg_results)
