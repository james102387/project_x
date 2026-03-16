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
    ("Grand Vizier Korth", "age", "142 standard years"),
    ("Draveth", "climate", "temperate with long dry seasons"),
    ("Draveth", "capital", "Zelphos"),
    ("Sulari", "known for", "mining and heavy industry"),
]

SAMPLE_ALIASES = {
    "capital city": "capital",
    "head of state": "leader",
}

SAMPLE_ENTITY_ALIASES = {
    "korth": "grand vizier korth",
    "vizier korth": "grand vizier korth",
}


@pytest.fixture
def kg():
    return KnowledgeGraph(
        SAMPLE_TRIPLETS,
        predicate_aliases=SAMPLE_ALIASES,
        entity_aliases=SAMPLE_ENTITY_ALIASES,
    )


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

    def test_match_tier_exact(self, kg):
        doc = nlp("What is the capital of Remulak?")
        spans = find_entity_spans(doc, kg)
        remulak_span = next(s for s in spans if s["entity"] == "remulak")
        assert remulak_span["match_tier"] == "exact"
        assert remulak_span["match_score"] == 1.0

    def test_alias_entity_match(self, kg):
        doc = nlp("How old is Korth?")
        spans = find_entity_spans(doc, kg)
        assert len(spans) > 0
        korth_span = next(s for s in spans if s["entity"] == "grand vizier korth")
        assert korth_span["match_tier"] == "alias"

    def test_fuzzy_entity_match(self, kg):
        doc = nlp("What is the capital of Remulack?")
        spans = find_entity_spans(doc, kg)
        assert len(spans) > 0
        match = spans[0]
        assert match["match_tier"] == "fuzzy"
        assert match["entity"] == "remulak"


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

    def test_targeted_lookup_type(self, kg):
        doc = nlp("What is the capital of Remulak?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["lookup_type"] == "targeted"
        assert len(result["results"]) == 1
        assert result["results"][0]["predicate"] == "capital"

    def test_subject_scan_lookup_type(self, kg):
        doc = nlp("What is the GDP of Remulak?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["lookup_type"] == "subject_scan"
        assert len(result["results"]) > 1

    def test_alias_is_targeted(self, kg):
        doc = nlp("What is the capital city of Remulak?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["lookup_type"] == "targeted"
        assert result["results"][0]["predicate"] == "capital"

    def test_match_tier_in_result(self, kg):
        doc = nlp("What is the capital of Remulak?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["match_tier"] == "exact"
        assert result["match_score"] == 1.0

    def test_entity_alias_detection(self, kg):
        doc = nlp("How old is Korth?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["entity"] == "grand vizier korth"
        assert result["match_tier"] == "alias"

    def test_fuzzy_entity_detection(self, kg):
        doc = nlp("What is the capital of Remulack?")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["entity"] == "remulak"
        assert result["match_tier"] == "fuzzy"

    def test_multi_hop_collects_related_facts(self, kg):
        doc = nlp("Tell me about Remulak")
        result = detect_kg_query(doc, kg, multi_hop=True, max_depth=1)
        assert result is not None
        assert result["lookup_type"] == "multi_hop"
        subjects = {r["subject"] for r in result["results"]}
        assert "Remulak" in subjects
        assert "Grand Vizier Korth" in subjects

    def test_multi_hop_default_off(self, kg):
        doc = nlp("Tell me about Remulak")
        result = detect_kg_query(doc, kg)
        assert result is not None
        assert result["lookup_type"] == "subject_scan"
