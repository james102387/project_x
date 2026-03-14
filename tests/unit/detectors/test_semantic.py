"""Unit tests for semantic verb pattern detection."""

import pytest
from crystal.detectors.math.semantic import match_semantic_verb_pattern, evaluate_semantic_steps


class TestSemanticVerbDetection:
    def test_acquire_basic(self, parse):
        steps = match_semantic_verb_pattern(parse("John has 10 apples and buys 5 more"))
        assert steps is not None
        assert len(steps) == 2

    def test_lose_basic(self, parse):
        steps = match_semantic_verb_pattern(parse("I have 20 dollars and spent 8"))
        assert steps is not None
        assert len(steps) == 2

    def test_mixed_operations(self, parse):
        steps = match_semantic_verb_pattern(
            parse("I had 100 dollars, earned 50, and spent 30")
        )
        assert steps is not None
        assert len(steps) == 3

    def test_single_verb_no_match(self, parse):
        result = match_semantic_verb_pattern(parse("I have 5 apples"))
        assert result is None

    def test_no_numbers_no_match(self, parse):
        assert match_semantic_verb_pattern(parse("I lost my keys")) is None

    def test_figurative_no_match(self, parse):
        assert match_semantic_verb_pattern(parse("She earned a reputation")) is None


class TestSemanticStepEvaluation:
    def test_state_plus_add(self):
        steps = [
            {"op": "state", "value": 10, "verb": "has"},
            {"op": "add", "value": 5, "verb": "buys"},
        ]
        result = evaluate_semantic_steps(steps)
        assert result["result"] == 15

    def test_state_plus_subtract(self):
        steps = [
            {"op": "state", "value": 20, "verb": "have"},
            {"op": "subtract", "value": 8, "verb": "spent"},
        ]
        result = evaluate_semantic_steps(steps)
        assert result["result"] == 12

    def test_mixed_three_step(self):
        steps = [
            {"op": "state", "value": 100, "verb": "had"},
            {"op": "add", "value": 50, "verb": "earned"},
            {"op": "subtract", "value": 30, "verb": "spent"},
        ]
        result = evaluate_semantic_steps(steps)
        assert result["result"] == 120

    def test_no_state_verb(self):
        steps = [
            {"op": "add", "value": 500, "verb": "earned"},
            {"op": "add", "value": 300, "verb": "earned"},
        ]
        result = evaluate_semantic_steps(steps)
        assert result["result"] == 800

    def test_empty_steps(self):
        assert evaluate_semantic_steps([]) is None
