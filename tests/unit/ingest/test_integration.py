"""Integration tests: ingest → build_kg → query."""

import json
import pytest
from pathlib import Path

from crystal.ingest import ingest, ingest_with_llm, build_kg
from crystal.ingest.loader import load_review
from crystal.ingest.schema import (
    IngestResult,
    LLMExtractionResult,
    ReviewableTriplet,
    Triplet,
)

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


class TestIngestAutoDetect:
    def test_csv(self):
        result = ingest(FIXTURES / "sample_triplets.csv")
        assert len(result.triplets) == 3

    def test_json(self):
        result = ingest(FIXTURES / "sample_triplets.json")
        assert len(result.triplets) == 2

    def test_text(self):
        result = ingest(FIXTURES / "sample_text.txt")
        assert len(result.triplets) >= 3


class TestBuildKg:
    def test_from_csv(self):
        result = ingest(FIXTURES / "sample_triplets.csv")
        kg = build_kg(result)
        assert len(kg) == 3
        facts = kg.lookup(subject="Remulak", predicate="capital")
        assert len(facts) == 1
        assert facts[0]["object"] == "Zelphos"

    def test_from_json_with_aliases(self):
        result = ingest(FIXTURES / "sample_triplets.json")
        kg = build_kg(result)
        assert len(kg) == 2
        # Predicate alias works
        facts = kg.lookup(subject="Remulak", predicate="capital city")
        assert len(facts) == 1
        assert facts[0]["object"] == "Zelphos"
        # Entity alias works
        assert kg.has_entity("korth")

    def test_from_text_ner(self):
        result = ingest(FIXTURES / "sample_text.txt")
        kg = build_kg(result)
        assert len(kg) >= 3
        assert len(kg.entities) >= 2

    def test_from_ingest_result_directly(self):
        result = IngestResult(
            triplets=[
                Triplet("Earth", "capital", "N/A"),
                Triplet("Earth", "population", "8 billion"),
            ],
            predicate_aliases={"main city": "capital"},
        )
        kg = build_kg(result)
        assert len(kg) == 2
        facts = kg.lookup(subject="Earth", predicate="main city")
        assert len(facts) == 1


class TestMergeAndBuild:
    def test_merge_csv_and_json(self):
        r1 = ingest(FIXTURES / "sample_triplets.csv")
        r2 = ingest(FIXTURES / "sample_triplets.json")
        merged = r1.merge(r2)
        kg = build_kg(merged)
        # CSV has 3 triplets, JSON has 2, with 2 overlapping
        assert len(kg) >= 3
        assert kg.has_entity("korth")  # from JSON aliases


class TestIngestWithLlm:
    """D2 Phase 2: two-pass ingestion (NER + LLM)."""

    def test_returns_both_results(self):
        def _mock_llm(prompt):
            return json.dumps([
                {"subject": "Mock", "predicate": "test", "object": "Value",
                 "confidence": "high", "sentence_index": 1},
            ]), None

        ner_result, llm_result = ingest_with_llm(
            FIXTURES / "sample_text.txt",
            call_llm_fn=_mock_llm,
        )
        assert isinstance(ner_result, IngestResult)
        assert isinstance(llm_result, LLMExtractionResult)
        assert len(ner_result.triplets) >= 3
        assert llm_result.source == str(FIXTURES / "sample_text.txt")

    def test_ner_result_matches_plain_ingest(self):
        def _noop_llm(prompt):
            return "[]", None

        ner_result, _ = ingest_with_llm(
            FIXTURES / "sample_text.txt",
            call_llm_fn=_noop_llm,
        )
        plain_result = ingest(FIXTURES / "sample_text.txt")
        assert ner_result.as_tuples() == plain_result.as_tuples()


class TestReviewRoundtrip:
    """D2 Phase 2: write review file → edit → load back."""

    def test_full_roundtrip(self, tmp_path):
        llm_result = LLMExtractionResult(
            reviewable=[
                ReviewableTriplet(
                    "Remulak", "borders", "Draveth",
                    "Remulak borders Draveth.", "high", "pending_review",
                ),
                ReviewableTriplet(
                    "Sulari", "exports", "minerals",
                    "Sulari exports minerals.", "medium", "pending_review",
                ),
            ],
            source="doc.txt",
        )
        review_path = tmp_path / "review.json"
        review_data = llm_result.to_review_dict()
        review_data["reviewable"][0]["status"] = "accepted"
        review_data["reviewable"][1]["status"] = "rejected"
        review_path.write_text(json.dumps(review_data))

        loaded = load_review(review_path)
        assert len(loaded.triplets) == 1
        assert loaded.triplets[0].subject == "Remulak"
        assert loaded.triplets[0].predicate == "borders"

    def test_merge_reviewed_with_ner(self, tmp_path):
        ner_result = IngestResult(
            triplets=[Triplet("Remulak", "capital", "Zelphos")],
            source="doc.txt",
        )
        review = {
            "reviewable": [
                {"subject": "Remulak", "predicate": "borders", "object": "Draveth",
                 "status": "accepted"},
            ],
        }
        review_path = tmp_path / "review.json"
        review_path.write_text(json.dumps(review))

        reviewed = load_review(review_path)
        merged = ner_result.merge(reviewed)
        kg = build_kg(merged)
        assert len(kg) == 2
        assert kg.lookup(subject="Remulak", predicate="capital")
        assert kg.lookup(subject="Remulak", predicate="borders")
