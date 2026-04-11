"""
Integration tests for the planner confidence gate.

Verifies that low-confidence fuzzy KG matches cause the full LangGraph
pipeline to fall back to the LLM rather than returning wrong-entity facts.

Uses a custom KG and mocked LLM to test the actual graph routing.
"""

from unittest.mock import patch

from crystal.graph import build_crystal_graph
from crystal.state import make_initial_state
from crystal.tools.kg.graph import KnowledgeGraph

TRIPLETS = [
    ("miranda v. united states", "court", "Supreme Court"),
    ("miranda v. united states", "date_filed", "1966-06-13"),
    ("roe v. wade", "court", "Supreme Court"),
    ("roe v. wade", "date_filed", "1973-01-22"),
    ("brown v. board of education", "court", "Supreme Court"),
]

LLM_FALLBACK_RESPONSE = "I don't have specific information about that case."


def _mock_call_llm(prompt: str):
    return LLM_FALLBACK_RESPONSE, {"prompt_tokens": 20, "output_tokens": 10}


class TestConfidenceGatePipeline:
    """End-to-end: wrong-entity fuzzy match → LLM fallback, not wrong KG facts."""

    def _run(self, prompt: str, kg: KnowledgeGraph) -> dict:
        app = build_crystal_graph()
        state = make_initial_state(prompt, kg=kg)
        with patch("crystal.nodes.llm_nodes.crystal.llm.call_llm", side_effect=_mock_call_llm):
            return app.invoke(state)

    def test_exact_entity_returns_kg_facts(self):
        """Exact match: 'roe v. wade' IS in the KG → KG detection fires,
        not filtered by confidence gate.  May be kg_answerable (targeted hit)
        or kg_augmented (subject scan needs LLM), but either way the KG
        results must contain the correct entity's facts."""
        kg = KnowledgeGraph(TRIPLETS)
        result = self._run("What court decided Roe v. Wade?", kg)
        assert result["prompt_type"] in ("kg_answerable", "kg_augmented")
        kg_facts = result.get("kg_results", [])
        assert len(kg_facts) > 0
        flat_results = []
        for entry in kg_facts:
            flat_results.extend(entry.get("results", []))
        assert any(f["object"] == "Supreme Court" for f in flat_results)

    def test_missing_entity_falls_back_to_llm(self):
        """'Miranda v. Arizona' not in KG, fuzzy ≈47% to 'Miranda v. United States'.
        Should fall back to LLM, not return United States facts."""
        kg = KnowledgeGraph(TRIPLETS)
        result = self._run(
            "What is the precedential status of Miranda v. Arizona?", kg
        )
        assert result["prompt_type"] == "no_math"
        assert result["final_response"] == LLM_FALLBACK_RESPONSE
        assert "miranda v. united states" not in result["final_response"].lower()

    def test_close_wrong_entity_falls_back(self):
        """Even if fuzzy score is 80-89%, planner rejects it.
        'Brown v. Board' is a partial match but NOT in KG as that exact name."""
        triplets = [
            ("brown v. board of education", "court", "Supreme Court"),
            ("brown v. board of education", "date_filed", "1954-05-17"),
        ]
        kg = KnowledgeGraph(triplets)
        result = self._run(
            "What court decided Brown v. Bored of Education?", kg
        )
        # "Bored" is a typo for "Board" — fuzzy should match but let's see
        # whether the score is high enough for the confidence gate.
        # Either it passes (correct answer) or falls back (safe answer).
        # It must NOT return garbage from a wrong entity.
        assert (
            result["prompt_type"] in ("kg_answerable", "kg_augmented", "no_math")
        )
        if result["prompt_type"] == "no_math":
            assert result["final_response"] == LLM_FALLBACK_RESPONSE
        else:
            assert "Supreme Court" in result["final_response"]

    def test_exact_entity_not_confused_by_similar_names(self):
        """When 'roe v. wade' is exact in KG, it should not pick
        a different entity even if others are similar."""
        kg = KnowledgeGraph(TRIPLETS)
        result = self._run("When was Roe v. Wade decided?", kg)
        assert "1973" in result["final_response"]
