"""Integration tests: ingest → build_kg → query."""

import pytest
from pathlib import Path

from crystal.ingest import ingest, build_kg
from crystal.ingest.schema import IngestResult, Triplet

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
