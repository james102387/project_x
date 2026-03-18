"""Unit tests for the ingestion schema types."""

import pytest
from crystal.ingest.schema import Triplet, IngestResult


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
