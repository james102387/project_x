"""
"Never worse than LLM" contract tests.

For every KG golden test case, runs the full Crystal pipeline and
verifies that Crystal's answer is at least as good as a naked LLM.
Since these tests don't make real LLM calls, the naked LLM baseline
is "no answer" — Crystal must either produce a correct KG answer or
gracefully fall back (not return wrong-entity facts).

These tests are the regression gate: if ANY change makes Crystal
confidently return wrong information that the LLM would not have
produced, the test fails.
"""

from unittest.mock import patch

import pytest

from crystal.graph import build_crystal_graph
from crystal.state import make_initial_state
from crystal.tools.kg import remulak_kg
from crystal.tools.kg.graph import KnowledgeGraph
from tests.golden.test_cases import (
    KG_ANSWERABLE_CASES,
    KG_AUGMENTED_CASES,
    KG_FUZZY_ENTITY_ALIAS_CASES,
    KG_FUZZY_STRING_CASES,
    KG_ADVERSARIAL_NEGATIVES,
    NEGATIVE_CASES,
)


LLM_FALLBACK = "[LLM_FALLBACK]"


def _mock_call_llm(prompt: str):
    return LLM_FALLBACK, {"prompt_tokens": 10, "output_tokens": 5}


def _run_pipeline(prompt: str, kg=None) -> dict:
    app = build_crystal_graph()
    state = make_initial_state(prompt, kg=kg or remulak_kg)
    with patch(
        "crystal.nodes.llm_nodes.crystal.llm.call_llm",
        side_effect=_mock_call_llm,
    ):
        return app.invoke(state)


# ── Contract: correct KG answers must still be correct ───────────────


@pytest.mark.parametrize(
    "prompt,expected_type,expected_result", KG_ANSWERABLE_CASES,
)
def test_kg_answerable_returns_correct_facts(prompt, expected_type, expected_result):
    """Crystal must return the correct answer for high-confidence KG lookups."""
    result = _run_pipeline(prompt)
    assert result["prompt_type"] in ("kg_answerable", "kg_augmented")
    if result["prompt_type"] == "kg_answerable":
        assert result["final_response"] == expected_result


@pytest.mark.parametrize(
    "prompt,expected_type,expected_result", KG_AUGMENTED_CASES,
)
def test_kg_augmented_calls_llm(prompt, expected_type, expected_result):
    """KG augmented should invoke LLM with grounding — not return raw KG."""
    result = _run_pipeline(prompt)
    assert result["prompt_type"] == "kg_augmented"
    assert expected_result in result.get("compiled_prompt", "")


@pytest.mark.parametrize(
    "prompt,expected_type,expected_result", KG_FUZZY_ENTITY_ALIAS_CASES,
)
def test_alias_cases_still_resolve(prompt, expected_type, expected_result):
    """Entity aliases (Korth → Grand Vizier Korth) must still work."""
    result = _run_pipeline(prompt)
    assert result["prompt_type"] in ("kg_answerable", "kg_augmented")
    if result["prompt_type"] == "kg_answerable":
        assert result["final_response"] == expected_result


# ── Contract: negatives must not produce hallucinated answers ────────


@pytest.mark.parametrize(
    "prompt,expected_type,expected_result", NEGATIVE_CASES,
)
def test_negatives_fall_back_to_llm(prompt, expected_type, expected_result):
    """Non-KG questions must fall back to LLM, not hallucinate KG output."""
    result = _run_pipeline(prompt)
    assert result["prompt_type"] in ("no_math", "no_match", "kg_augmented")


@pytest.mark.parametrize(
    "prompt,expected_type,expected_result", KG_ADVERSARIAL_NEGATIVES,
)
def test_adversarial_negatives_handled(prompt, expected_type, expected_result):
    """Entity found but predicate missing — must not return wrong predicate data."""
    result = _run_pipeline(prompt)
    assert result["prompt_type"] in ("kg_augmented", "no_math")


# ── Contract: wrong entity must never be returned confidently ────────


class TestWrongEntityProtection:
    """Crystal must never return facts from a different entity
    when the asked-about entity isn't in the KG."""

    def test_missing_entity_falls_back(self):
        """Ask about an entity NOT in the Remulak KG."""
        result = _run_pipeline("What is the capital of Atlantis?")
        assert result["prompt_type"] == "no_math"
        assert result["final_response"] == LLM_FALLBACK

    def test_similar_name_different_entity(self):
        """Custom KG with close-but-wrong entity names."""
        triplets = [
            ("miranda v. united states", "court", "Supreme Court"),
        ]
        kg = KnowledgeGraph(triplets)
        result = _run_pipeline(
            "What court decided Miranda v. Arizona?", kg=kg,
        )
        # Must NOT contain "Supreme Court" from miranda v. united states
        if result["prompt_type"] == "no_math":
            assert result["final_response"] == LLM_FALLBACK
        else:
            kg_facts = result.get("kg_results", [])
            for entry in kg_facts:
                for fact in entry.get("results", []):
                    assert fact["subject"] != "miranda v. united states"


# ── Contract: node crashes degrade to LLM, not error messages ────────


class TestGracefulDegradation:
    def test_broken_kg_detection_falls_back(self):
        """If KG detection throws, pipeline still returns an LLM answer."""
        with patch(
            "crystal.nodes.kg.detection.detect_kg_query",
            side_effect=RuntimeError("KG broke"),
        ):
            result = _run_pipeline("What is the capital of Remulak?")
        assert result["prompt_type"] == "no_math"
        assert result["final_response"] == LLM_FALLBACK

    def test_broken_compiler_falls_back(self):
        """If the compiler throws, pipeline still returns an LLM answer."""
        with patch(
            "crystal.nodes.compiler.core._classify_prompt_type",
            side_effect=RuntimeError("compiler broke"),
        ):
            result = _run_pipeline("What is the capital of Remulak?")
        assert result["prompt_type"] == "no_math"
        assert result["final_response"] == LLM_FALLBACK
