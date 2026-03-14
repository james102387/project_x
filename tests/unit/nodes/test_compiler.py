"""Unit tests for prompt compiler — type classification and compilation."""

import pytest
from crystal.nodes.compiler import _classify_prompt_type, _build_simplified_prompt


class TestPromptClassification:
    def test_pure_math(self, parse):
        doc = parse("5 plus 3")
        results = [{"success": True, "operation": "add"}]
        assert _classify_prompt_type("5 plus 3", doc, results) == "pure_math"

    def test_pure_math_with_question(self, parse):
        doc = parse("what's 5 plus 3")
        results = [{"success": True, "operation": "add"}]
        assert _classify_prompt_type("what's 5 plus 3", doc, results) == "pure_math"

    def test_math_answerable(self, parse):
        prompt = "John has 10 apples and buys 5 more"
        doc = parse(prompt)
        results = [{"success": True, "operation": "semantic_math"}]
        assert _classify_prompt_type(prompt, doc, results) == "math_answerable"

    def test_math_augmented_advisory(self, parse):
        prompt = "She earned 500 and spent 300, is she managing her money wisely?"
        doc = parse(prompt)
        results = [{"success": True, "operation": "semantic_math"}]
        assert _classify_prompt_type(prompt, doc, results) == "math_augmented"

    def test_math_augmented_explanatory(self, parse):
        prompt = "I started with 50 and lost 30, explain what happened"
        doc = parse(prompt)
        results = [{"success": True, "operation": "semantic_math"}]
        assert _classify_prompt_type(prompt, doc, results) == "math_augmented"

    def test_math_augmented_predictive(self, parse):
        prompt = "He had 100 shares and sold 60, should he buy more?"
        doc = parse(prompt)
        results = [{"success": True, "operation": "semantic_math"}]
        assert _classify_prompt_type(prompt, doc, results) == "math_augmented"

    def test_no_results(self, parse):
        doc = parse("hello")
        assert _classify_prompt_type("hello", doc, []) == "no_math"

    def test_kg_answerable(self, parse):
        prompt = "What is the capital of Remulak?"
        doc = parse(prompt)
        results = [{"success": True, "tool": "kg", "results": [{"subject": "Remulak", "predicate": "capital", "object": "Zelphos"}]}]
        assert _classify_prompt_type(prompt, doc, results) == "kg_answerable"


class TestSimplifiedPromptBuild:
    def test_explicit_math(self):
        results = [{"success": True, "operation": "add", "args": [5, 3], "result": 8}]
        prompt = _build_simplified_prompt("what is 5 plus 3", results)
        assert "5 + 3 = 8" in prompt
        assert "Do not recalculate" in prompt

    def test_semantic_math(self):
        results = [{
            "success": True,
            "operation": "semantic_math",
            "args": [10, 5],
            "steps": [
                {"op": "state", "value": 10, "verb": "has"},
                {"op": "add", "value": 5, "verb": "buys"},
            ],
            "result": 15,
        }]
        prompt = _build_simplified_prompt("John has 10 and buys 5", results)
        assert "15" in prompt
