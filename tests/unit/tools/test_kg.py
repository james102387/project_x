"""Unit tests for the Knowledge Graph tool."""

import pytest
from crystal.tools.kg import KnowledgeGraph
from crystal.tools.kg.store import SqliteKnowledgeGraph


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_TRIPLETS = [
    ("Remulak", "capital", "Zelphos"),
    ("Remulak", "leader", "Grand Vizier Korth"),
    ("Remulak", "population", "4.3 billion"),
    ("Draveth", "capital", "Zelphos"),
    ("Draveth", "climate", "temperate with long dry seasons"),
    ("Grand Vizier Korth", "real name", "Korth Vellan"),
    ("Grand Vizier Korth", "age", "142 standard years"),
    ("resonance crystals", "found in", "Sulari"),
    ("Sulari", "known for", "mining and heavy industry"),
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

SAMPLE_ENTITY_ALIASES = {
    "korth": "grand vizier korth",
    "vizier korth": "grand vizier korth",
    "korth vellan": "grand vizier korth",
}


@pytest.fixture
def kg():
    return KnowledgeGraph(
        SAMPLE_TRIPLETS,
        predicate_aliases=SAMPLE_ALIASES,
        entity_aliases=SAMPLE_ENTITY_ALIASES,
    )


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


# ── Entity alias resolution ──────────────────────────────────────────────


class TestEntityAliases:
    def test_alias_resolves(self, kg):
        resolved, tier = kg._resolve_entity("Korth")
        assert resolved == "grand vizier korth"
        assert tier == "alias"

    def test_multi_word_alias(self, kg):
        resolved, tier = kg._resolve_entity("Vizier Korth")
        assert resolved == "grand vizier korth"
        assert tier == "alias"

    def test_exact_beats_alias(self, kg):
        resolved, tier = kg._resolve_entity("Remulak")
        assert resolved == "remulak"
        assert tier == "exact"

    def test_has_entity_with_alias(self, kg):
        assert kg.has_entity("Korth")

    def test_has_entity_exact(self, kg):
        assert kg.has_entity("Remulak")

    def test_has_entity_unknown(self, kg):
        assert not kg.has_entity("Zarquon")

    def test_no_entity_aliases(self, kg_no_aliases):
        resolved, tier = kg_no_aliases._resolve_entity("Korth")
        assert tier != "alias"


# ── Fuzzy entity resolution ──────────────────────────────────────────────


class TestFuzzyEntity:
    def test_typo_fuzzy_match(self, kg):
        resolved, tier = kg._resolve_entity("Remulack")
        assert tier == "fuzzy"
        assert resolved == "remulak"

    def test_exact_beats_fuzzy(self, kg):
        resolved, tier = kg._resolve_entity("Remulak")
        assert tier == "exact"

    def test_alias_beats_fuzzy(self, kg):
        resolved, tier = kg._resolve_entity("Korth")
        assert tier == "alias"

    def test_no_match(self, kg):
        resolved, tier = kg._resolve_entity("Zarquon the Destroyer")
        assert tier == "none"


# ── Fuzzy predicate resolution ────────────────────────────────────────────


class TestFuzzyPredicate:
    def test_exact_predicate(self, kg):
        resolved, tier = kg._resolve_predicate_cascade("capital", subject="remulak")
        assert resolved == "capital"
        assert tier == "exact"

    def test_alias_predicate(self, kg):
        resolved, tier = kg._resolve_predicate_cascade("capital city", subject="remulak")
        assert resolved == "capital"
        assert tier == "alias"

    def test_fuzzy_predicate(self, kg):
        resolved, tier = kg._resolve_predicate_cascade("capitl", subject="remulak")
        assert tier == "fuzzy"
        assert resolved == "capital"

    def test_no_match_predicate(self, kg):
        resolved, tier = kg._resolve_predicate_cascade("gdp", subject="remulak")
        assert tier == "none"


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

    def test_subjects_property(self, kg):
        subjects = kg.subjects
        assert isinstance(subjects, set)
        assert "remulak" in subjects
        assert "grand vizier korth" in subjects
        assert "zelphos" not in subjects


# ── Multi-hop traversal ──────────────────────────────────────────────────


class TestMultiHopTraversal:
    def test_depth_0_is_subject_scan(self, kg):
        results = kg.traverse("Remulak", max_depth=0)
        assert len(results) == 3
        predicates = {r["predicate"] for r in results}
        assert predicates == {"capital", "leader", "population"}

    def test_depth_1_follows_objects(self, kg):
        results = kg.traverse("Remulak", max_depth=1)
        subjects = {r["subject"] for r in results}
        assert "Remulak" in subjects
        assert "Grand Vizier Korth" in subjects
        assert "Draveth" not in subjects

    def test_depth_2_follows_two_hops(self, kg):
        results = kg.traverse("resonance crystals", max_depth=2)
        subjects = {r["subject"] for r in results}
        assert "resonance crystals" in subjects
        assert "Sulari" in subjects

    def test_default_depth_is_2(self, kg):
        results_default = kg.traverse("Remulak")
        results_explicit = kg.traverse("Remulak", max_depth=2)
        assert results_default == results_explicit

    def test_avoids_cycles(self):
        cyclic = KnowledgeGraph([
            ("A", "related", "B"),
            ("B", "related", "A"),
        ])
        results = cyclic.traverse("A", max_depth=5)
        assert len(results) == 2

    def test_unknown_entity_returns_empty(self, kg):
        results = kg.traverse("Zarquon")
        assert results == []

    def test_object_only_entity(self, kg):
        results = kg.traverse("Zelphos")
        assert results == []

    def test_facts_are_unique(self, kg):
        results = kg.traverse("Remulak", max_depth=2)
        unique = []
        for r in results:
            if r not in unique:
                unique.append(r)
        assert len(results) == len(unique)


# ── Construction ──────────────────────────────────────────────────────────


class TestConstruction:
    def test_len(self, kg):
        assert len(kg) == len(SAMPLE_TRIPLETS)

    def test_repr(self, kg):
        assert f"{len(SAMPLE_TRIPLETS)} triplets" in repr(kg)

    def test_empty_kg(self):
        kg = KnowledgeGraph([])
        assert len(kg) == 0
        assert kg.lookup(subject="anything") == []
        assert kg.entities == set()

    def test_no_aliases_default(self):
        kg = KnowledgeGraph(SAMPLE_TRIPLETS)
        assert kg._resolve_predicate("capital") == "capital"
        assert kg._resolve_predicate("capital city") == "capital city"


# ── Extend (mutation) ────────────────────────────────────────────────────


class TestExtend:
    def test_adds_new_triplets(self, kg):
        original_len = len(kg)
        added = kg.extend([("Zelphos", "type", "city"), ("Zelphos", "founded", "year 0")])
        assert added == 2
        assert len(kg) == original_len + 2
        results = kg.lookup(subject="Zelphos", predicate="type")
        assert results[0]["object"] == "city"

    def test_skips_duplicates(self, kg):
        original_len = len(kg)
        added = kg.extend([("Remulak", "capital", "Zelphos")])
        assert added == 0
        assert len(kg) == original_len

    def test_mixed_new_and_duplicate(self, kg):
        original_len = len(kg)
        added = kg.extend([
            ("Remulak", "capital", "Zelphos"),
            ("Remulak", "exports", "dilithium"),
        ])
        assert added == 1
        assert len(kg) == original_len + 1

    def test_updates_entity_index(self, kg):
        kg.extend([("NewPlanet", "type", "gas giant")])
        assert "newplanet" in kg.entities
        assert "gas giant" in kg.entities
        assert "newplanet" in kg.subjects

    def test_extend_empty(self, kg):
        original_len = len(kg)
        added = kg.extend([])
        assert added == 0
        assert len(kg) == original_len


# ── SqliteKnowledgeGraph Provenance ───────────────────────────────────────


class TestSqliteProvenance:
    """Tests for origin and source_document provenance tracking."""

    @pytest.fixture
    def skg(self):
        return SqliteKnowledgeGraph(":memory:")

    def test_schema_has_origin_and_source_document(self, skg):
        cols = {
            row[1]
            for row in skg._conn.execute("PRAGMA table_info(triplets)").fetchall()
        }
        assert "origin" in cols
        assert "source_document" in cols

    def test_bulk_insert_default_origin(self, skg):
        skg.bulk_insert(
            [("a", "b", "c")],
            source="test",
        )
        facts = skg.get_all_facts()
        assert len(facts) == 1
        assert facts[0]["origin"] == "unknown"
        assert facts[0]["source_document"] == ""

    def test_bulk_insert_with_origin(self, skg):
        skg.bulk_insert(
            [("miranda v. arizona", "court", "SCOTUS")],
            source="cold-cases-scotus",
            origin="api_metadata",
            source_document="cold-cases-scotus",
        )
        facts = skg.get_all_facts()
        assert facts[0]["origin"] == "api_metadata"
        assert facts[0]["source_document"] == "cold-cases-scotus"

    def test_bulk_insert_5tuple_per_row_origin(self, skg):
        skg.bulk_insert(
            [
                ("a", "p1", "o1", "sentence one", "opinion_doc"),
                ("b", "p2", "o2", "sentence two", "api_metadata"),
            ],
            source="test",
            origin="unknown",
            source_document="opinion.json",
        )
        facts = skg.get_all_facts()
        origins = {f["subject"]: f["origin"] for f in facts}
        assert origins["a"] == "opinion_doc"
        assert origins["b"] == "api_metadata"

    def test_lookup_returns_origin_and_source_document(self, skg):
        skg.bulk_insert(
            [("x", "y", "z")],
            source="test",
            origin="opinion_doc",
            source_document="loving-v-virginia.txt",
        )
        results = skg.lookup(subject="x", predicate="y")
        assert len(results) == 1
        assert results[0]["origin"] == "opinion_doc"
        assert results[0]["source_document"] == "loving-v-virginia.txt"

    def test_get_all_facts_includes_provenance(self, skg):
        skg.bulk_insert(
            [("a", "b", "c")],
            source="test",
            origin="opinion_doc",
            source_document="doc.txt",
        )
        facts = skg.get_all_facts()
        assert "origin" in facts[0]
        assert "source_document" in facts[0]
        assert facts[0]["origin"] == "opinion_doc"
        assert facts[0]["source_document"] == "doc.txt"

    def test_triplets_with_provenance(self, skg):
        skg.bulk_insert(
            [("a", "b", "c"), ("d", "e", "f")],
            source="src",
            origin="api_metadata",
            source_document="payload.json",
        )
        data = skg.triplets_with_provenance
        assert len(data) == 2
        assert len(data[0]) == 6
        s, p, o, source, origin, src_doc = data[0]
        assert origin == "api_metadata"
        assert src_doc == "payload.json"

    def test_provenance_counts(self, skg):
        skg.bulk_insert(
            [("a1", "p", "o1"), ("a2", "p", "o2")],
            source="s1",
            origin="api_metadata",
        )
        skg.bulk_insert(
            [("b1", "p", "o3")],
            source="s2",
            origin="opinion_doc",
        )
        counts = skg.provenance_counts()
        assert counts["api_metadata"] == 2
        assert counts["opinion_doc"] == 1

    def test_source_documents_list(self, skg):
        skg.bulk_insert(
            [("a", "p", "o1")],
            source="s",
            source_document="doc_a.json",
        )
        skg.bulk_insert(
            [("b", "p", "o2")],
            source="s",
            source_document="doc_b.json",
        )
        skg.bulk_insert(
            [("c", "p", "o3")],
            source="s",
            source_document="",
        )
        docs = skg.source_documents()
        assert "doc_a.json" in docs
        assert "doc_b.json" in docs
        assert "" not in docs

    def test_lookup_by_origin(self, skg):
        skg.bulk_insert(
            [("a", "p", "o1")],
            source="s",
            origin="api_metadata",
        )
        skg.bulk_insert(
            [("b", "p", "o2")],
            source="s",
            origin="opinion_doc",
        )
        api_facts = skg.lookup_by_origin("api_metadata")
        assert len(api_facts) == 1
        assert api_facts[0]["subject"] == "a"

        opinion_facts = skg.lookup_by_origin("opinion_doc")
        assert len(opinion_facts) == 1
        assert opinion_facts[0]["subject"] == "b"

    def test_lookup_by_document(self, skg):
        skg.bulk_insert(
            [("a", "p", "o1")],
            source="s",
            source_document="opinion_123.json",
        )
        skg.bulk_insert(
            [("b", "p", "o2")],
            source="s",
            source_document="opinion_456.json",
        )
        facts = skg.lookup_by_document("opinion_123.json")
        assert len(facts) == 1
        assert facts[0]["subject"] == "a"

    def test_backfill_provenance(self, skg):
        skg.bulk_insert(
            [("a", "p", "o1")],
            source="cold-cases-scotus",
        )
        skg.bulk_insert(
            [("b", "p", "o2")],
            source="pasted_text",
        )
        counts = skg.backfill_provenance()
        assert counts["api_metadata"] >= 1
        assert counts["opinion_doc"] >= 1

        facts = skg.get_all_facts()
        origins = {f["subject"]: f["origin"] for f in facts}
        assert origins["a"] == "api_metadata"
        assert origins["b"] == "opinion_doc"

    def test_migration_on_old_schema(self):
        """Verify migrations add origin/source_document to a DB created without them."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS triplets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source TEXT,
                source_sentence TEXT DEFAULT '',
                batch_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, predicate, object)
            );
            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias TEXT PRIMARY KEY, canonical TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predicate_aliases (
                alias TEXT PRIMARY KEY, canonical TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sync_state (
                source TEXT PRIMARY KEY, last_sync TEXT,
                last_id TEXT, item_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS ingestion_batches (
                batch_id TEXT PRIMARY KEY, source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                triplet_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            );
        """)
        conn.execute(
            "INSERT INTO triplets (subject, predicate, object, source) "
            "VALUES ('x', 'y', 'z', 'old_source')"
        )
        conn.commit()
        conn.close()

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert([("test", "pred", "obj")], source="new", origin="api_metadata")
        facts = kg.get_all_facts()
        assert any(f["origin"] == "api_metadata" for f in facts)
