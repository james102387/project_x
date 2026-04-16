"""Tests for ingestion confidence scoring."""

import pytest

from crystal.ingest.confidence import (
    INGEST_AUTO_ACCEPT,
    ScoredTriplet,
    classify_extraction_source,
    score_ingestion_confidence,
)
from crystal.tools.kg.graph import KnowledgeGraph


@pytest.fixture
def simple_kg():
    return KnowledgeGraph(
        [("remulak", "capital", "Zelphos")],
        predicate_aliases={"ruler": "leader"},
    )


LEGAL_PREDS = {"court", "date_filed", "judges", "cites", "opinion_author"}
LEGAL_ALIASES = {"decided by": "court", "filed on": "date_filed"}


class TestScoreIngestionConfidence:
    def test_ner_base_score(self):
        score = score_ingestion_confidence("x", "y", "ner")
        assert score == pytest.approx(0.85)

    def test_llm_high_base_score(self):
        score = score_ingestion_confidence("x", "y", "llm_high")
        assert score == pytest.approx(0.80)

    def test_llm_medium_base_score(self):
        score = score_ingestion_confidence("x", "y", "llm_medium")
        assert score == pytest.approx(0.55)

    def test_llm_low_base_score(self):
        score = score_ingestion_confidence("x", "y", "llm_low")
        assert score == pytest.approx(0.30)

    def test_structured_base_score(self):
        score = score_ingestion_confidence("x", "y", "structured")
        assert score == pytest.approx(1.0)

    def test_entity_known_bonus(self, simple_kg):
        score = score_ingestion_confidence(
            "remulak", "y", "llm_medium", kg=simple_kg,
        )
        assert score == pytest.approx(0.65)

    def test_predicate_aligned_bonus(self):
        score = score_ingestion_confidence(
            "x", "court", "llm_medium",
            ontology_predicates=LEGAL_PREDS,
        )
        assert score == pytest.approx(0.65)

    def test_predicate_alias_bonus(self):
        score = score_ingestion_confidence(
            "x", "decided by", "llm_medium",
            predicate_aliases=LEGAL_ALIASES,
        )
        assert score == pytest.approx(0.65)

    def test_both_bonuses_stack(self, simple_kg):
        score = score_ingestion_confidence(
            "remulak", "court", "llm_medium",
            kg=simple_kg, ontology_predicates=LEGAL_PREDS,
        )
        assert score == pytest.approx(0.75)

    def test_capped_at_1(self, simple_kg):
        score = score_ingestion_confidence(
            "remulak", "court", "structured",
            kg=simple_kg, ontology_predicates=LEGAL_PREDS,
        )
        assert score == 1.0

    def test_ner_with_known_entity_auto_accepts(self, simple_kg):
        score = score_ingestion_confidence(
            "remulak", "y", "ner", kg=simple_kg,
        )
        assert score >= INGEST_AUTO_ACCEPT

    def test_llm_medium_alone_below_threshold(self):
        score = score_ingestion_confidence("x", "y", "llm_medium")
        assert score < INGEST_AUTO_ACCEPT

    def test_llm_high_with_aligned_predicate_auto_accepts(self):
        score = score_ingestion_confidence(
            "x", "court", "llm_high",
            ontology_predicates=LEGAL_PREDS,
        )
        assert score >= INGEST_AUTO_ACCEPT

    def test_unknown_source_gets_default(self):
        score = score_ingestion_confidence("x", "y", "unknown_source")
        assert score == pytest.approx(0.50)


class TestClassifyExtractionSource:
    def test_high(self):
        assert classify_extraction_source("high") == "llm_high"

    def test_medium(self):
        assert classify_extraction_source("medium") == "llm_medium"

    def test_low(self):
        assert classify_extraction_source("low") == "llm_low"

    def test_unknown_defaults_medium(self):
        assert classify_extraction_source("garbage") == "llm_medium"


class TestScoredTriplet:
    def test_to_dict_roundtrip(self):
        st = ScoredTriplet(
            subject="Miranda v. Arizona", predicate="court",
            object="Supreme Court", source_sentence="test",
            extraction_source="llm_high", ingestion_confidence=0.9,
        )
        d = st.to_dict()
        restored = ScoredTriplet.from_dict(d)
        assert restored.subject == st.subject
        assert restored.ingestion_confidence == st.ingestion_confidence

    def test_as_tuple(self):
        st = ScoredTriplet(
            subject="a", predicate="b", object="c",
            source_sentence="", extraction_source="ner",
            ingestion_confidence=0.85,
        )
        assert st.as_tuple() == ("a", "b", "c")

    def test_to_triplet(self):
        st = ScoredTriplet(
            subject="a", predicate="b", object="c",
            source_sentence="", extraction_source="ner",
            ingestion_confidence=0.85,
        )
        t = st.to_triplet()
        assert t.subject == "a"
        assert t.predicate == "b"
        assert t.object == "c"

    def test_origin_defaults_to_opinion_doc(self):
        st = ScoredTriplet(
            subject="a", predicate="b", object="c",
            source_sentence="", extraction_source="ner",
            ingestion_confidence=0.85,
        )
        assert st.origin == "opinion_doc"

    def test_origin_explicit_api_metadata(self):
        st = ScoredTriplet(
            subject="a", predicate="b", object="c",
            source_sentence="", extraction_source="ner",
            ingestion_confidence=0.85,
            origin="api_metadata",
        )
        assert st.origin == "api_metadata"

    def test_source_document_field(self):
        st = ScoredTriplet(
            subject="a", predicate="b", object="c",
            source_sentence="", extraction_source="ner",
            ingestion_confidence=0.85,
            source_document="miranda.json",
        )
        assert st.source_document == "miranda.json"
        d = st.to_dict()
        assert d["source_document"] == "miranda.json"
        assert d["origin"] == "opinion_doc"

    def test_from_dict_with_source_document(self):
        d = {
            "subject": "a", "predicate": "b", "object": "c",
            "source_sentence": "", "extraction_source": "ner",
            "ingestion_confidence": 0.85,
            "source_document": "doc.json",
        }
        st = ScoredTriplet.from_dict(d)
        assert st.source_document == "doc.json"
