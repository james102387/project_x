"""Plan builder node — converts tool detections into an execution plan.

Includes a unified confidence scorer for KG detections that considers
entity match quality, predicate match quality, entity ambiguity, and
lookup specificity. The confidence score drives three behaviors:

  HIGH  (>= 0.9): proceed normally (kg_answerable or kg_augmented)
  MEDIUM (0.7–0.9): force kg_augmented with softened prompt framing
  LOW   (< 0.7): fall back to the naked LLM entirely

This enforces the "never worse than LLM" contract — confidently
returning wrong-entity facts is worse than letting the LLM answer
from its training data.
"""

from __future__ import annotations

CONFIDENCE_HIGH = 0.9
CONFIDENCE_LOW = 0.7


def score_grounding_confidence(detection: dict) -> float:
    """Score a KG detection's trustworthiness on a 0.0–1.0 scale.

    Entity match quality dominates: if the entity is wrong, nothing else
    matters. Predicate specificity and entity ambiguity are secondary.

    Fuzzy entity matches use piecewise bands that preserve the original
    90% threshold behavior: below 90% → LOW, 90-95 → MEDIUM, 95+ → HIGH.
    """
    entity_tier = detection.get("match_tier", "exact")
    entity_raw_score = detection.get("match_score", 1.0)

    if entity_tier in ("exact", "alias"):
        entity_confidence = 1.0
    elif entity_raw_score >= 95:
        entity_confidence = 0.95
    elif entity_raw_score >= 90:
        entity_confidence = 0.8
    else:
        entity_confidence = 0.4

    lookup_type = detection.get("lookup_type", "subject_scan")
    if lookup_type == "targeted":
        predicate_modifier = 1.0
    elif lookup_type == "multi_hop":
        predicate_modifier = 0.9
    else:
        predicate_modifier = 0.85

    n_entity_spans = len(detection.get("entity_spans", []))
    if n_entity_spans > 2:
        ambiguity_penalty = 0.1
    elif n_entity_spans > 1:
        ambiguity_penalty = 0.05
    else:
        ambiguity_penalty = 0.0

    confidence = entity_confidence * predicate_modifier - ambiguity_penalty
    return max(0.0, min(1.0, confidence))


def plan_builder_node(state: dict) -> dict:
    """Build a compiler plan from tool detections.

    KG detections with grounding confidence below CONFIDENCE_LOW are
    filtered out entirely (LLM fallback).  Medium-confidence detections
    are kept but the confidence score propagates to the compiler for
    softened prompt framing.
    """
    detections = state.get("tool_detections", [])

    if not detections:
        return {"plan": [], "fallback_to_llm": True, "grounding_confidence": 0.0}

    plan = []
    best_kg_confidence = 0.0

    for detection in detections:
        if detection["tool"] == "kg":
            confidence = score_grounding_confidence(detection)
            if confidence < CONFIDENCE_LOW:
                continue
            best_kg_confidence = max(best_kg_confidence, confidence)
            plan.append({
                "tool": "kg",
                "operation": "lookup",
                "entity": detection["entity"],
                "results": detection["results"],
                "lookup_type": detection.get("lookup_type", "subject_scan"),
                "grounding_confidence": confidence,
            })
        else:
            entry = {
                "tool": detection["tool"],
                "operation": detection["operation"],
                "args": detection["raw_args"],
            }
            if detection["operation"] == "semantic_math":
                entry["steps"] = detection.get("steps", [])
                entry["result"] = detection.get("result")
            plan.append(entry)

    if not plan:
        return {"plan": [], "fallback_to_llm": True, "grounding_confidence": 0.0}

    return {
        "plan": plan,
        "fallback_to_llm": False,
        "grounding_confidence": best_kg_confidence,
    }
