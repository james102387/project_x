"""Tests for the plan_builder_node unified confidence scorer.

The planner must enforce Crystal's "never worse than LLM" contract:
- LOW confidence (< 0.7): fall back to the raw LLM
- MEDIUM confidence (0.7–0.9): proceed but with softened framing
- HIGH confidence (>= 0.9): proceed normally
"""

import pytest

from crystal.nodes.planner import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    score_grounding_confidence,
    plan_builder_node,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_kg_detection(
    entity: str = "miranda v. arizona",
    match_tier: str = "exact",
    match_score: float = 1.0,
    results: list | None = None,
    lookup_type: str = "targeted",
    predicate_match_tier: str = "exact",
    entity_spans: list | None = None,
) -> dict:
    return {
        "tool": "kg",
        "operation": "lookup",
        "entity": entity,
        "results": results or [{"subject": entity, "predicate": "court", "object": "Supreme Court"}],
        "lookup_type": lookup_type,
        "match_tier": match_tier,
        "match_score": match_score,
        "original_text": entity,
        "predicate_match_tier": predicate_match_tier,
        "entity_spans": entity_spans or [{"entity": entity}],
    }


def _make_math_detection() -> dict:
    return {
        "tool": "calculator",
        "operation": "arithmetic",
        "raw_args": "2 + 2",
    }


# ── score_grounding_confidence ───────────────────────────────────────


class TestScoreGroundingConfidence:
    def test_exact_entity_targeted_predicate_is_high(self):
        d = _make_kg_detection(match_tier="exact", lookup_type="targeted")
        score = score_grounding_confidence(d)
        assert score >= CONFIDENCE_HIGH

    def test_alias_entity_targeted_predicate_is_high(self):
        d = _make_kg_detection(match_tier="alias", lookup_type="targeted")
        score = score_grounding_confidence(d)
        assert score >= CONFIDENCE_HIGH

    def test_exact_entity_subject_scan_is_medium(self):
        d = _make_kg_detection(
            match_tier="exact", lookup_type="subject_scan", predicate_match_tier="none",
        )
        score = score_grounding_confidence(d)
        assert CONFIDENCE_LOW <= score < CONFIDENCE_HIGH

    def test_fuzzy_95_targeted_is_high(self):
        d = _make_kg_detection(
            match_tier="fuzzy", match_score=95.0, lookup_type="targeted",
        )
        score = score_grounding_confidence(d)
        assert score >= CONFIDENCE_HIGH

    def test_fuzzy_83_targeted_is_low(self):
        """83% fuzzy → wrong entity territory → below LOW threshold."""
        d = _make_kg_detection(
            match_tier="fuzzy", match_score=83.0, lookup_type="targeted",
        )
        score = score_grounding_confidence(d)
        assert score < CONFIDENCE_LOW

    def test_fuzzy_50_is_very_low(self):
        d = _make_kg_detection(
            match_tier="fuzzy", match_score=50.0, lookup_type="targeted",
        )
        score = score_grounding_confidence(d)
        assert score < CONFIDENCE_LOW

    def test_ambiguity_penalty_many_spans(self):
        """3+ competing entity spans penalize confidence."""
        spans = [{"entity": "a"}, {"entity": "b"}, {"entity": "c"}]
        d = _make_kg_detection(
            match_tier="exact", lookup_type="targeted", entity_spans=spans,
        )
        score = score_grounding_confidence(d)
        d_single = _make_kg_detection(
            match_tier="exact", lookup_type="targeted",
            entity_spans=[{"entity": "a"}],
        )
        single_score = score_grounding_confidence(d_single)
        assert score < single_score

    def test_score_clamped_to_0_1(self):
        d = _make_kg_detection(match_tier="fuzzy", match_score=0.0, lookup_type="subject_scan")
        score = score_grounding_confidence(d)
        assert 0.0 <= score <= 1.0

    def test_multi_hop_is_moderate(self):
        d = _make_kg_detection(match_tier="exact", lookup_type="multi_hop")
        score = score_grounding_confidence(d)
        assert CONFIDENCE_LOW <= score <= 1.0


# ── plan_builder_node ────────────────────────────────────────────────


class TestPlanBuilderConfidenceGate:
    def test_no_detections_falls_back(self):
        state = {"tool_detections": []}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is True
        assert result["plan"] == []
        assert result["grounding_confidence"] == 0.0

    def test_exact_kg_match_builds_plan(self):
        det = _make_kg_detection(match_tier="exact", match_score=1.0)
        state = {"tool_detections": [det]}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is False
        assert len(result["plan"]) == 1
        assert result["plan"][0]["entity"] == det["entity"]
        assert result["grounding_confidence"] >= CONFIDENCE_HIGH

    def test_alias_kg_match_builds_plan(self):
        det = _make_kg_detection(match_tier="alias", match_score=1.0)
        state = {"tool_detections": [det]}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is False
        assert len(result["plan"]) == 1

    def test_high_confidence_fuzzy_builds_plan(self):
        det = _make_kg_detection(match_tier="fuzzy", match_score=95.0)
        state = {"tool_detections": [det]}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is False
        assert len(result["plan"]) == 1

    def test_low_confidence_fuzzy_falls_back_to_llm(self):
        """83% fuzzy match → wrong entity → fall back."""
        det = _make_kg_detection(
            entity="miranda v. united states",
            match_tier="fuzzy",
            match_score=83.0,
        )
        state = {"tool_detections": [det]}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is True
        assert result["plan"] == []

    def test_low_confidence_kg_with_math_still_uses_math(self):
        """Low-confidence KG is filtered but other detections survive."""
        kg_det = _make_kg_detection(match_tier="fuzzy", match_score=75.0)
        math_det = _make_math_detection()
        state = {"tool_detections": [kg_det, math_det]}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is False
        assert len(result["plan"]) == 1
        assert result["plan"][0]["tool"] == "calculator"

    def test_mixed_confident_and_unconfident_kg(self):
        good = _make_kg_detection(
            entity="roe v. wade", match_tier="exact", match_score=1.0,
        )
        bad = _make_kg_detection(
            entity="roe v. other", match_tier="fuzzy", match_score=50.0,
        )
        state = {"tool_detections": [good, bad]}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is False
        assert len(result["plan"]) == 1
        assert result["plan"][0]["entity"] == "roe v. wade"

    def test_all_kg_detections_low_confidence_falls_back(self):
        d1 = _make_kg_detection(match_tier="fuzzy", match_score=50.0)
        d2 = _make_kg_detection(match_tier="fuzzy", match_score=60.0)
        state = {"tool_detections": [d1, d2]}
        result = plan_builder_node(state)
        assert result["fallback_to_llm"] is True
        assert result["plan"] == []

    def test_grounding_confidence_propagated(self):
        """Confidence score flows through to state for compiler use."""
        det = _make_kg_detection(
            match_tier="exact", lookup_type="subject_scan",
            predicate_match_tier="none",
        )
        state = {"tool_detections": [det]}
        result = plan_builder_node(state)
        assert "grounding_confidence" in result
        assert 0.0 < result["grounding_confidence"] <= 1.0

    def test_plan_items_carry_confidence(self):
        det = _make_kg_detection(match_tier="exact")
        state = {"tool_detections": [det]}
        result = plan_builder_node(state)
        assert "grounding_confidence" in result["plan"][0]

    def test_threshold_values_are_sensible(self):
        assert 0.6 <= CONFIDENCE_LOW <= 0.8
        assert 0.85 <= CONFIDENCE_HIGH <= 0.95
