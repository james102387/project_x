"""Unit tests for KG detector — entity matching and question structure."""

import pytest
import spacy

from crystal.detectors.kg import find_entity_spans, has_question_structure, detect_kg_query
from crystal.tools.kg import KnowledgeGraph


nlp = spacy.load("en_core_web_sm")

SAMPLE_TRIPLETS = [
    ("Remulak", "capital", "Zelphos"),
    ("Remulak", "leader", "Grand Vizier Korth"),
    ("Grand Vizier Korth", "real name", "Korth Vellan"),
    ("Draveth", "climate", "temperate with long dry seasons"),
]

SAMPLE_ALIASES = {
    "capital city": "capital",
    "head of state": "leader",
}


@pytest.fixture
def kg():
    return KnowledgeGraph(SAMPLE_TRIPLETS, predicate_aliases=SAMPLE_ALIASES)


class TestEntitySpans:
    def test_single_entity(self, kg):
        doc = nlp("What is the capital of Remulak?")
        spans = find_entity_spans(doc, kg)
        entities = {s["entity"] for s in spans}
        assert "remulak" in entities

    def test_multi_word_entity(self, kg):
        doc = nlp("Who is Grand Vizier Korth?")
        spans = find_entity_spans(doc, kg)
        entities = {s["entity"] for s in spans}
        assert "grand vizier korth" in entities

    def test_longest_match_first(self, kg):
        doc = nlp("Tell me about Grand Vizier Korth")
        spans = find_entity_spans(doc, kg)
        entities = [s["entity"] for s in spans]
        assert "grand vizier korth" in entities
        assert "korth vellan" not in entities

    def test_no_entity(self, kg):
        doc = nlp("What is the weather like today?")
        spans = find_entity_spans(doc, kg)
        assert spans == []

    def test_boundary_check(self, kg):
        doc = nlp("The Remulakian empire is vast")
        spans = find_entity_spans(doc, kg)
        entities = {s["entity"] for s in spans}
        assert "remulak" not in entities


class TestQuestionStructure:
    def test_what_question(self):
        assert has_question_structure(nlp("What is the capital?"))

    def test_who_question(self):
        assert has_question_structure(nlp("Who leads Remulak?"))

    def test_how_question(self):
        assert has_question_structure(nlp("How big is Remulak?"))

    def test_question_mark(self):
        assert has_question_structure(nlp("The capital of Remulak?"))

    def test_tell_me(self):
        assert has_question_structure(nlp("Tell me about Remulak"))

    def test_describe(self):
        assert has_question_structure(nlp("Describe the climate of Draveth"))

    def test_statement_no_question(self):
        assert not has_question_structure(nlp("Remulak is a planet"))

    def test_bare_entity_no_question(self):
        assert not has_question_structure(nlp("Remulak"))


class TestDetectKgQuery:
    def test_basic_question(self, kg):
        doc = nlp("What is the capital of Remulak?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["tool"] == "kg"
        assert result["entity"] == "remulak"
        assert len(result["results"]) > 0

    def test_no_entity(self, kg):
        doc = nlp("What is the meaning of life?")
        result = detect_kg_query(doc, kg)
        assert result is None

    def test_entity_but_no_question(self, kg):
        doc = nlp("Remulak is interesting")
        result = detect_kg_query(doc, kg)
        assert result is None

    def test_multi_word_entity_question(self, kg):
        doc = nlp("Who is Grand Vizier Korth?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["entity"] == "grand vizier korth"
