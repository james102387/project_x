"""Tests for SqliteKnowledgeGraph — mirrors test_kg.py's core tests."""

import pytest

from crystal.tools.kg.store import SqliteKnowledgeGraph


@pytest.fixture
def kg():
    """In-memory SQLite KG with Remulak-style sample data."""
    db = SqliteKnowledgeGraph(":memory:")
    triplets = [
        ("remulak", "capital", "Zelphos"),
        ("remulak", "population", "4.3 billion"),
        ("remulak", "leader", "Grand Vizier Korth"),
        ("remulak", "star system", "Veldra-7"),
        ("grand vizier korth", "age", "142 standard years"),
        ("grand vizier korth", "birthplace", "Zelphos"),
        ("draveth", "capital", "Zelphos"),
        ("draveth", "population", "1.8 billion"),
    ]
    db.bulk_insert(
        triplets,
        entity_aliases={"korth": "grand vizier korth", "gvk": "grand vizier korth"},
        predicate_aliases={"capital city": "capital", "ruler": "leader", "head of state": "leader"},
    )
    return db


# ── Forward lookup ───────────────────────────────────────────────────────


class TestForwardLookup:
    def test_exact_subject_predicate(self, kg):
        results = kg.lookup(subject="remulak", predicate="capital")
        assert len(results) == 1
        assert results[0]["object"] == "Zelphos"

    def test_subject_scan(self, kg):
        results = kg.lookup(subject="remulak")
        assert len(results) == 4

    def test_predicate_alias(self, kg):
        results = kg.lookup(subject="remulak", predicate="capital city")
        assert len(results) == 1
        assert results[0]["object"] == "Zelphos"

    def test_case_insensitive(self, kg):
        results = kg.lookup(subject="Remulak", predicate="Capital")
        assert len(results) == 1

    def test_no_match(self, kg):
        results = kg.lookup(subject="remulak", predicate="nonexistent")
        assert results == []


# ── Reverse lookup ───────────────────────────────────────────────────────


class TestReverseLookup:
    def test_reverse_by_predicate_and_object(self, kg):
        results = kg.lookup(predicate="capital", obj="Zelphos")
        assert len(results) >= 1
        subjects = {r["subject"] for r in results}
        assert "remulak" in subjects

    def test_reverse_no_match(self, kg):
        results = kg.lookup(predicate="capital", obj="Nonexistent")
        assert results == []


# ── Entity resolution ────────────────────────────────────────────────────


class TestEntityResolution:
    def test_exact_entity(self, kg):
        resolved, tier = kg._resolve_entity("remulak")
        assert resolved == "remulak"
        assert tier == "exact"

    def test_alias_entity(self, kg):
        resolved, tier = kg._resolve_entity("korth")
        assert resolved == "grand vizier korth"
        assert tier == "alias"

    def test_fuzzy_entity(self, kg):
        resolved, tier = kg._resolve_entity("remualk")
        assert resolved == "remulak"
        assert tier == "fuzzy"

    def test_no_match(self, kg):
        _, tier = kg._resolve_entity("zzzzzzz")
        assert tier == "none"


# ── Predicate resolution ────────────────────────────────────────────────


class TestPredicateResolution:
    def test_exact_predicate(self, kg):
        resolved = kg._resolve_predicate("capital")
        assert resolved == "capital"

    def test_alias_predicate(self, kg):
        resolved = kg._resolve_predicate("ruler")
        assert resolved == "leader"

    def test_fuzzy_predicate_resolution(self, kg):
        resolved, tier = kg._resolve_predicate_cascade("capitl", subject="remulak")
        assert resolved == "capital"
        assert tier == "fuzzy"


# ── Multi-hop traversal ─────────────────────────────────────────────────


class TestTraversal:
    def test_single_hop(self, kg):
        results = kg.traverse("remulak", max_depth=0)
        assert len(results) == 4

    def test_unknown_entity(self, kg):
        results = kg.traverse("nonexistent")
        assert results == []


# ── Bulk operations ──────────────────────────────────────────────────────


class TestBulkInsert:
    def test_insert_count(self):
        db = SqliteKnowledgeGraph(":memory:")
        count = db.bulk_insert([
            ("a", "rel", "b"),
            ("c", "rel", "d"),
        ])
        assert count == 2
        assert len(db) == 2

    def test_deduplication(self):
        db = SqliteKnowledgeGraph(":memory:")
        db.bulk_insert([("a", "rel", "b")])
        db.bulk_insert([("a", "rel", "b")])
        assert len(db) == 1

    def test_insert_count_reflects_actual_rowcount_on_dedup(self):
        """Counter must only count rows actually inserted, not attempts.

        Regression: INSERT OR IGNORE silently skips duplicates but used
        to still increment the counter, producing inflated triplet_count
        in ingestion_batches and phantom batches with no rows.
        """
        db = SqliteKnowledgeGraph(":memory:")
        db.bulk_insert([("a", "rel", "b")])
        count = db.bulk_insert([
            ("a", "rel", "b"),
            ("c", "rel", "d"),
        ])
        assert count == 1

    def test_no_batch_row_when_nothing_inserted(self):
        """Re-inserting only duplicates should not create an empty batch row."""
        db = SqliteKnowledgeGraph(":memory:")
        db.bulk_insert([("a", "rel", "b")])
        batches_before = len(db.list_batches())
        count = db.bulk_insert([("a", "rel", "b")])
        assert count == 0
        assert len(db.list_batches()) == batches_before

    def test_batch_id_stamped_on_inserted_rows(self):
        """Every inserted row must carry the batch_id for rollback to work."""
        db = SqliteKnowledgeGraph(":memory:")
        db.bulk_insert([("a", "rel", "b"), ("c", "rel", "d")], source="test")
        batches = db.list_batches()
        assert len(batches) == 1
        bid = batches[0]["batch_id"]
        rows = db._conn.execute(
            "SELECT batch_id FROM triplets"
        ).fetchall()
        assert all(r[0] == bid for r in rows)
        assert db.delete_batch(bid) == 2
        assert len(db) == 0

    def test_source_field(self):
        db = SqliteKnowledgeGraph(":memory:")
        db.bulk_insert([("a", "rel", "b")], source="test-source")
        row = db._conn.execute(
            "SELECT source FROM triplets WHERE subject='a'"
        ).fetchone()
        assert row[0] == "test-source"


# ── Properties ───────────────────────────────────────────────────────────


class TestProperties:
    def test_entities(self, kg):
        entities = kg.entities
        assert "remulak" in entities
        assert "grand vizier korth" in entities

    def test_subjects(self, kg):
        subjects = kg.subjects
        assert "remulak" in subjects
        assert "grand vizier korth" in subjects

    def test_len(self, kg):
        assert len(kg) == 8

    def test_repr(self, kg):
        assert "8 triplets" in repr(kg)

    def test_has_entity(self, kg):
        assert kg.has_entity("remulak")
        assert kg.has_entity("korth")
        assert not kg.has_entity("zzzzz")

    def test_triplets_property(self, kg):
        triplets = kg.triplets
        assert len(triplets) == 8
        assert all(len(t) == 3 for t in triplets)


# ── Integration with legal data ──────────────────────────────────────────


class TestLegalIntegration:
    def test_scotus_sample_in_sqlite(self):
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE
        from crystal.ingest.sources.cold_cases import ingest_cold_cases
        from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES

        db = SqliteKnowledgeGraph(":memory:")

        for batch in ingest_cold_cases(records=SCOTUS_SAMPLE, batch_size=100):
            db.bulk_insert(
                [t.as_tuple() for t in batch.triplets],
                entity_aliases=batch.entity_aliases,
                predicate_aliases=LEGAL_PREDICATE_ALIASES,
            )

        assert len(db) >= 250

        results = db.lookup(subject="miranda v. arizona", predicate="court")
        assert len(results) == 1
        assert "Supreme Court" in results[0]["object"]

        resolved, tier = db._resolve_entity("miranda")
        assert resolved == "miranda v. arizona"
        assert tier == "alias"

        results = db.lookup(subject="miranda v. arizona")
        assert len(results) >= 4
