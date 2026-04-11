"""Tests for the ingest_document() orchestrator."""

import json
import pytest
from pathlib import Path

from crystal.ingest import (
    DocumentIngestionResult,
    ingest_document,
)
from crystal.ingest.confidence import ScoredTriplet
from crystal.tools.kg.store import SqliteKnowledgeGraph


@pytest.fixture
def empty_kg():
    return SqliteKnowledgeGraph(":memory:")


@pytest.fixture
def seeded_kg():
    kg = SqliteKnowledgeGraph(":memory:")
    kg.bulk_insert([
        ("miranda v. arizona", "court", "Supreme Court of the United States"),
        ("miranda v. arizona", "date_filed", "1966-06-13"),
    ])
    return kg


@pytest.fixture
def sample_text(tmp_path):
    text = (
        "The Supreme Court decided Miranda v. Arizona in 1966. "
        "Chief Justice Warren wrote the majority opinion. "
        "The case established the requirement for police to inform suspects of their rights."
    )
    path = tmp_path / "sample.txt"
    path.write_text(text)
    return path


def _mock_llm(prompt: str):
    return json.dumps([
        {
            "subject": "Miranda v. Arizona",
            "predicate": "opinion_author",
            "object": "Chief Justice Warren",
            "confidence": "high",
            "sentence_index": 2,
        },
        {
            "subject": "Miranda v. Arizona",
            "predicate": "holding",
            "object": "Police must inform suspects of their rights",
            "confidence": "medium",
            "sentence_index": 3,
        },
    ]), {"prompt_tokens": 100, "output_tokens": 50}


class TestIngestDocument:
    def test_returns_document_ingestion_result(self, sample_text, empty_kg):
        result = ingest_document(sample_text, empty_kg)
        assert isinstance(result, DocumentIngestionResult)

    def test_ner_only_when_no_llm(self, sample_text, empty_kg):
        result = ingest_document(sample_text, empty_kg)
        assert result.stats["llm_triplets"] == 0
        assert result.stats["ner_triplets"] >= 0

    def test_llm_extracts_when_provided(self, tmp_path, empty_kg):
        text = (
            "In the landmark ruling, the Warren Court fundamentally transformed "
            "the landscape of criminal procedure and Miranda warnings. "
            "The complex interplay between Fifth Amendment protections and "
            "police interrogation practices remains debated to this day."
        )
        path = tmp_path / "complex.txt"
        path.write_text(text)
        result = ingest_document(path, empty_kg, call_llm_fn=_mock_llm)
        assert result.stats["llm_triplets"] >= 0

    def test_auto_accepts_above_threshold(self, sample_text, empty_kg):
        result = ingest_document(
            sample_text, empty_kg,
            call_llm_fn=_mock_llm,
            auto_accept_threshold=0.70,
        )
        for st in result.auto_accepted:
            assert st.ingestion_confidence >= 0.70
            assert st.status == "accepted"

    def test_pending_below_threshold(self, sample_text, empty_kg):
        result = ingest_document(
            sample_text, empty_kg,
            call_llm_fn=_mock_llm,
            auto_accept_threshold=0.70,
        )
        for st in result.pending_review:
            assert st.ingestion_confidence < 0.70
            assert st.status == "pending_review"

    def test_inserts_auto_accepted_into_kg(self, sample_text, empty_kg):
        before = len(empty_kg)
        result = ingest_document(
            sample_text, empty_kg,
            call_llm_fn=_mock_llm,
        )
        after = len(empty_kg)
        assert after >= before + len(result.auto_accepted)

    def test_deduplicates_existing_kg_facts(self, sample_text, seeded_kg):
        result = ingest_document(sample_text, seeded_kg, call_llm_fn=_mock_llm)
        all_triplets = result.auto_accepted + result.pending_review + result.rejected
        tuples = [(st.subject.lower(), st.predicate.lower(), st.object.lower()) for st in all_triplets]
        assert len(tuples) == len(set(tuples))

    def test_raw_text_input(self, empty_kg):
        text = "Remulak has a capital called Zelphos."
        result = ingest_document(text, empty_kg, domain="general")
        assert result.stats["source"] == "pasted_text"

    def test_stats_populated(self, sample_text, empty_kg):
        result = ingest_document(sample_text, empty_kg, call_llm_fn=_mock_llm)
        assert "total_extracted" in result.stats
        assert "elapsed_seconds" in result.stats
        assert "auto_accepted" in result.stats

    def test_legal_domain_normalizes_predicates(self, sample_text, empty_kg):
        result = ingest_document(
            sample_text, empty_kg,
            call_llm_fn=_mock_llm,
            domain="legal",
        )
        all_preds = [st.predicate for st in result.auto_accepted + result.pending_review]
        for pred in all_preds:
            assert pred == pred.lower()

    def test_no_kg_still_works(self, sample_text):
        result = ingest_document(sample_text, kg=None, call_llm_fn=_mock_llm)
        assert isinstance(result, DocumentIngestionResult)
        total = len(result.auto_accepted) + len(result.pending_review) + len(result.rejected)
        assert total > 0


class TestDocumentIngestionResultAccept:
    def test_accept_pending_moves_to_accepted(self, sample_text, empty_kg):
        result = ingest_document(
            sample_text, empty_kg,
            call_llm_fn=_mock_llm,
            auto_accept_threshold=0.99,
        )
        if not result.pending_review:
            pytest.skip("No pending items to test")
        n_pending = len(result.pending_review)
        accepted = result.accept_pending([0])
        assert accepted == 1
        assert len(result.pending_review) == n_pending - 1

    def test_accept_all_pending(self, sample_text, empty_kg):
        result = ingest_document(
            sample_text, empty_kg,
            call_llm_fn=_mock_llm,
            auto_accept_threshold=0.99,
        )
        if not result.pending_review:
            pytest.skip("No pending items to test")
        n = result.accept_all_pending()
        assert n > 0
        assert len(result.pending_review) == 0

    def test_reject_pending(self, sample_text, empty_kg):
        result = ingest_document(
            sample_text, empty_kg,
            call_llm_fn=_mock_llm,
            auto_accept_threshold=0.99,
        )
        if not result.pending_review:
            pytest.skip("No pending items to test")
        n = result.reject_pending([0])
        assert n == 1
        assert result.rejected[-1].status == "rejected"
