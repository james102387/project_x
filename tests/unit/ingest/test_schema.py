"""Unit tests for the ingestion schema types."""

import pytest
from crystal.ingest.schema import (
    Triplet,
    IngestResult,
    ReviewableTriplet,
    LLMExtractionResult,
)


class TestTriplet:
    def test_as_tuple(self):
        t = Triplet(subject="Remulak", predicate="capital", object="Zelphos")
        assert t.as_tuple() == ("Remulak", "capital", "Zelphos")

    def test_equality(self):
        a = Triplet(subject="A", predicate="B", object="C")
        b = Triplet(subject="A", predicate="B", object="C")
        assert a == b

    def test_fields(self):
        t = Triplet(subject="X", predicate="Y", object="Z")
        assert t.subject == "X"
        assert t.predicate == "Y"
        assert t.object == "Z"


class TestIngestResult:
    def test_as_tuples(self):
        result = IngestResult(triplets=[
            Triplet("A", "b", "C"),
            Triplet("D", "e", "F"),
        ])
        assert result.as_tuples() == [("A", "b", "C"), ("D", "e", "F")]

    def test_empty(self):
        result = IngestResult()
        assert result.as_tuples() == []
        assert result.entity_aliases == {}
        assert result.predicate_aliases == {}

    def test_merge_deduplicates(self):
        r1 = IngestResult(
            triplets=[Triplet("A", "b", "C"), Triplet("D", "e", "F")],
            entity_aliases={"x": "y"},
            source="file1",
        )
        r2 = IngestResult(
            triplets=[Triplet("A", "b", "C"), Triplet("G", "h", "I")],
            entity_aliases={"z": "w"},
            source="file2",
        )
        merged = r1.merge(r2)
        assert len(merged.triplets) == 3
        assert merged.entity_aliases == {"x": "y", "z": "w"}
        assert "file1" in merged.source
        assert "file2" in merged.source

    def test_merge_empty(self):
        r1 = IngestResult(triplets=[Triplet("A", "b", "C")])
        r2 = IngestResult()
        merged = r1.merge(r2)
        assert len(merged.triplets) == 1


class TestReviewableTriplet:
    def test_fields(self):
        rt = ReviewableTriplet(
            subject="Remulak", predicate="borders", object="Draveth",
            source_sentence="Remulak borders Draveth.",
            confidence="high",
        )
        assert rt.subject == "Remulak"
        assert rt.status == "pending_review"

    def test_to_triplet(self):
        rt = ReviewableTriplet(
            subject="A", predicate="b", object="C",
            source_sentence="A b C.", confidence="high",
        )
        t = rt.to_triplet()
        assert isinstance(t, Triplet)
        assert t.as_tuple() == ("A", "b", "C")

    def test_to_dict_roundtrip(self):
        rt = ReviewableTriplet(
            subject="A", predicate="b", object="C",
            source_sentence="A b C.", confidence="medium",
            status="accepted",
        )
        d = rt.to_dict()
        assert d["subject"] == "A"
        assert d["status"] == "accepted"
        rt2 = ReviewableTriplet.from_dict(d)
        assert rt2 == rt

    def test_from_dict_defaults(self):
        d = {"subject": "X", "predicate": "y", "object": "Z"}
        rt = ReviewableTriplet.from_dict(d)
        assert rt.confidence == "medium"
        assert rt.status == "pending_review"
        assert rt.source_sentence == ""


class TestLLMExtractionResult:
    def _make_result(self):
        return LLMExtractionResult(
            reviewable=[
                ReviewableTriplet("A", "b", "C", "A b C.", "high", "accepted"),
                ReviewableTriplet("D", "e", "F", "D e F.", "medium", "pending_review"),
                ReviewableTriplet("G", "h", "I", "G h I.", "low", "rejected"),
            ],
            skipped_sentences=["Some unclear sentence."],
            source="test.txt",
        )

    def test_accepted_triplets(self):
        result = self._make_result()
        accepted = result.accepted_triplets()
        assert len(accepted) == 1
        assert accepted[0].as_tuple() == ("A", "b", "C")

    def test_pending_triplets(self):
        result = self._make_result()
        pending = result.pending_triplets()
        assert len(pending) == 1
        assert pending[0].subject == "D"

    def test_to_ingest_result(self):
        result = self._make_result()
        ir = result.to_ingest_result()
        assert isinstance(ir, IngestResult)
        assert len(ir.triplets) == 1
        assert "llm-reviewed" in ir.source

    def test_to_review_dict_structure(self):
        result = self._make_result()
        d = result.to_review_dict()
        assert d["source"] == "test.txt"
        assert d["total_reviewable"] == 3
        assert d["total_skipped"] == 1
        assert len(d["reviewable"]) == 3
        assert d["skipped_sentences"] == ["Some unclear sentence."]

    def test_from_review_dict_roundtrip(self):
        result = self._make_result()
        d = result.to_review_dict()
        result2 = LLMExtractionResult.from_review_dict(d)
        assert len(result2.reviewable) == 3
        assert result2.reviewable[0].status == "accepted"
        assert result2.skipped_sentences == result.skipped_sentences

    def test_empty(self):
        result = LLMExtractionResult()
        assert result.accepted_triplets() == []
        assert result.pending_triplets() == []
        assert result.to_review_dict()["total_reviewable"] == 0

    def test_generated_at_populated(self):
        result = LLMExtractionResult()
        assert result.generated_at  # not empty
