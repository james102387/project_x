"""Unit tests for the Knowledge Graph tool."""

import pytest
from crystal.tools.kg import KnowledgeGraph



# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_TRIPLETS = [
    ("Remulak", "capital", "Zelphos"),
    ("Remulak", "leader", "Grand Vizier Korth"),
    ("Remulak", "population", "4.3 billion"),
    ("Draveth", "capital", "Zelphos"),
    ("Draveth", "climate", "temperate with long dry seasons"),
    ("Grand Vizier Korth", "real name", "Korth Vellan"),
    ("resonance crystals", "found in", "Sulari"),
]

SAMPLE_ALIASES = {
    "capital city": "capital",
    "main city": "capital",
    "head of state": "leader",
    "ruler": "leader",
    "who leads": "leader",
    "how many people": "population",
    "birth name": "real name",
    "located in": "found in",
    "where found": "found in",
}


@pytest.fixture
def kg():
    return KnowledgeGraph(SAMPLE_TRIPLETS, predicate_aliases=SAMPLE_ALIASES)


@pytest.fixture
def kg_no_aliases():
    return KnowledgeGraph(SAMPLE_TRIPLETS)


# ── Forward lookup (exact predicate) ──────────────────────────────────────


class TestForwardLookup:
    def test_exact_match(self, kg):
        results = kg.lookup(subject="Remulak", predicate="capital")
        assert len(results) == 1
        assert results[0]["object"] == "Zelphos"

    def test_case_insensitive(self, kg):
        results = kg.lookup(subject="remulak", predicate="Capital")
        assert len(results) == 1
        assert results[0]["object"] == "Zelphos"

    def test_miss(self, kg):
        results = kg.lookup(subject="Remulak", predicate="nonexistent")
        assert results == []


# ── Reverse lookup ────────────────────────────────────────────────────────


class TestReverseLookup:
    def test_exact(self, kg):
        results = kg.lookup(predicate="capital", obj="Zelphos")
        assert len(results) >= 1
        subjects = {r["subject"] for r in results}
        assert "Draveth" in subjects or "Remulak" in subjects

    def test_miss(self, kg):
        results = kg.lookup(predicate="capital", obj="Nonexistia")
        assert results == []


# ── Subject scan (all facts) ─────────────────────────────────────────────


class TestSubjectScan:
    def test_all_facts(self, kg):
        results = kg.lookup(subject="Remulak")
        assert len(results) == 3
        predicates = {r["predicate"] for r in results}
        assert predicates == {"capital", "leader", "population"}

    def test_single_fact_entity(self, kg):
        results = kg.lookup(subject="resonance crystals")
        assert len(results) == 1
        assert results[0]["predicate"] == "found in"

    def test_unknown_subject(self, kg):
        results = kg.lookup(subject="Zarquon")
        assert results == []


# ── Predicate alias resolution ────────────────────────────────────────────


class TestPredicateAliases:
    def test_alias_resolves_to_canonical(self, kg):
        results = kg.lookup(subject="Remulak", predicate="capital city")
        assert len(results) == 1
        assert results[0]["object"] == "Zelphos"

    def test_multiple_aliases_same_canonical(self, kg):
        r1 = kg.lookup(subject="Remulak", predicate="head of state")
        r2 = kg.lookup(subject="Remulak", predicate="ruler")
        r3 = kg.lookup(subject="Remulak", predicate="who leads")
        assert r1 == r2 == r3
        assert r1[0]["object"] == "Grand Vizier Korth"

    def test_alias_case_insensitive(self, kg):
        results = kg.lookup(subject="Remulak", predicate="Capital City")
        assert len(results) == 1
        assert results[0]["object"] == "Zelphos"

    def test_alias_with_reverse_lookup(self, kg):
        results = kg.lookup(predicate="located in", obj="Sulari")
        assert len(results) == 1
        assert results[0]["subject"] == "resonance crystals"

    def test_canonical_still_works(self, kg):
        results = kg.lookup(subject="Remulak", predicate="leader")
        assert len(results) == 1
        assert results[0]["object"] == "Grand Vizier Korth"

    def test_unknown_alias_passes_through(self, kg):
        results = kg.lookup(subject="Remulak", predicate="completely unknown")
        assert results == []

    def test_no_aliases_exact_only(self, kg_no_aliases):
        exact = kg_no_aliases.lookup(subject="Remulak", predicate="capital")
        assert len(exact) == 1
        alias = kg_no_aliases.lookup(subject="Remulak", predicate="capital city")
        assert alias == []

    def test_birth_name_alias(self, kg):
        results = kg.lookup(subject="Grand Vizier Korth", predicate="birth name")
        assert len(results) == 1
        assert results[0]["object"] == "Korth Vellan"

    def test_where_found_alias(self, kg):
        results = kg.lookup(subject="resonance crystals", predicate="where found")
        assert len(results) == 1
        assert results[0]["object"] == "Sulari"


# ── Entity index ──────────────────────────────────────────────────────────


class TestEntityIndex:
    def test_known_subject(self, kg):
        assert kg.has_entity("Remulak")

    def test_known_object(self, kg):
        assert kg.has_entity("Zelphos")

    def test_case_insensitive(self, kg):
        assert kg.has_entity("remulak")
        assert kg.has_entity("ZELPHOS")

    def test_multi_word_entity(self, kg):
        assert kg.has_entity("Grand Vizier Korth")
        assert kg.has_entity("resonance crystals")

    def test_unknown_entity(self, kg):
        assert not kg.has_entity("Zarquon")
        assert not kg.has_entity("")

    def test_entities_property_returns_set(self, kg):
        entities = kg.entities
        assert isinstance(entities, set)
        assert "remulak" in entities
        assert "zelphos" in entities
        assert "korth vellan" in entities


# ── Construction ──────────────────────────────────────────────────────────


class TestConstruction:
    def test_len(self, kg):
        assert len(kg) == len(SAMPLE_TRIPLETS)

    def test_repr(self, kg):
        assert "7 triplets" in repr(kg)

    def test_empty_kg(self):
        kg = KnowledgeGraph([])
        assert len(kg) == 0
        assert kg.lookup(subject="anything") == []
        assert kg.entities == set()

    def test_no_aliases_default(self):
        kg = KnowledgeGraph(SAMPLE_TRIPLETS)
        assert kg._resolve_predicate("capital") == "capital"
        assert kg._resolve_predicate("capital city") == "capital city"
