"""Tests for KG injection — verifying custom KGs flow through the pipeline."""

import pytest
import spacy

from crystal.state import make_initial_state
from crystal.nodes.kg.detection import kg_detection_node
from crystal.tools.kg.graph import KnowledgeGraph


nlp = spacy.load("en_core_web_sm")

CUSTOM_TRIPLETS = [
    ("Thaloria", "capital", "Narveth"),
    ("Thaloria", "population", "3 million"),
    ("Narveth", "known for", "floating markets"),
]


@pytest.fixture
def custom_kg():
    return KnowledgeGraph(CUSTOM_TRIPLETS)


class TestMakeInitialState:
    def test_default_kg_is_none(self):
        state = make_initial_state("hello")
        assert state["kg"] is None

    def test_custom_kg_injected(self, custom_kg):
        state = make_initial_state("hello", kg=custom_kg)
        assert state["kg"] is custom_kg

    def test_all_fields_present(self):
        state = make_initial_state("test")
        required = [
            "raw_prompt", "spacy_doc", "tool_detections", "plan",
            "preprocessed", "tool_results", "compiled_prompt",
            "prompt_type", "llm_response", "final_response",
            "fallback_to_llm", "token_metrics", "kg",
            "kg_detections", "kg_results", "kg_entities_found",
        ]
        for field in required:
            assert field in state, f"Missing field: {field}"


class TestKgDetectionNodeInjection:
    def test_custom_kg_detects_custom_entity(self, custom_kg):
        state = make_initial_state("What is the capital of Thaloria?", kg=custom_kg)
        state["spacy_doc"] = nlp(state["raw_prompt"])
        result = kg_detection_node(state)
        assert len(result["kg_entities_found"]) > 0
        assert "thaloria" in result["kg_entities_found"]

    def test_custom_kg_does_not_detect_remulak(self, custom_kg):
        state = make_initial_state("What is the capital of Remulak?", kg=custom_kg)
        state["spacy_doc"] = nlp(state["raw_prompt"])
        result = kg_detection_node(state)
        assert "remulak" not in result.get("kg_entities_found", [])

    def test_none_kg_falls_back_to_remulak(self):
        state = make_initial_state("What is the capital of Remulak?")
        state["spacy_doc"] = nlp(state["raw_prompt"])
        result = kg_detection_node(state)
        assert len(result["kg_entities_found"]) > 0
        assert "remulak" in result["kg_entities_found"]

    def test_missing_kg_key_falls_back_to_remulak(self):
        """Backward compat: old states without 'kg' key still work."""
        state = {
            "raw_prompt": "What is the capital of Remulak?",
            "spacy_doc": nlp("What is the capital of Remulak?"),
            "tool_detections": [],
        }
        result = kg_detection_node(state)
        assert len(result["kg_entities_found"]) > 0
        assert "remulak" in result["kg_entities_found"]

    def test_custom_kg_lookup_results(self, custom_kg):
        state = make_initial_state("What is the capital of Thaloria?", kg=custom_kg)
        state["spacy_doc"] = nlp(state["raw_prompt"])
        result = kg_detection_node(state)
        detections = result["kg_detections"]
        assert len(detections) > 0
        facts = detections[0]["results"]
        assert any(f["object"] == "Narveth" for f in facts)
