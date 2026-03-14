"""Math detection node — runs explicit and semantic pattern matchers."""

from crystal.detectors.math import EXPLICIT_PATTERNS
from crystal.detectors.math import match_semantic_verb_pattern, evaluate_semantic_steps


def math_detection_node(state: dict) -> dict:
    """
    Run all detection patterns against the spaCy doc.

    Priority:
    1. Explicit math patterns (add, plus, sum, +) — high confidence
    2. Verb-semantic patterns (buy, sell, have) — medium confidence,
       only fires if no explicit pattern matched
    """
    doc = state["spacy_doc"]
    detections = list(state.get("tool_detections", []))

    # Try explicit patterns first
    for pattern_name, matcher in EXPLICIT_PATTERNS:
        args = matcher(doc)
        if args is not None:
            detections.append({
                "tool": "calculator",
                "operation": "add",
                "raw_args": args,
                "matched_pattern": pattern_name,
            })
            return {"tool_detections": detections}

    # Try verb-semantic pattern
    semantic_steps = match_semantic_verb_pattern(doc)
    if semantic_steps is not None:
        evaluation = evaluate_semantic_steps(semantic_steps)
        if evaluation is not None:
            detections.append({
                "tool": "calculator",
                "operation": "semantic_math",
                "raw_args": evaluation["args"],
                "steps": evaluation["steps"],
                "result": evaluation["result"],
                "matched_pattern": "semantic_verb",
            })

    return {"tool_detections": detections}
