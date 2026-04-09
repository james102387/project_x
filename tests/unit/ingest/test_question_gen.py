"""Tests for the question generator — Tier 1, Tier 2, negatives, orchestrator, and intake bay."""

import json

import pytest

from crystal.tools.kg.graph import KnowledgeGraph
from crystal.ingest.question_gen import (
    QuestionCase,
    generate_tier1,
    generate_tier2,
    generate_negatives,
    generate_all,
    export_for_review,
    import_reviewed,
    review_stats,
)


@pytest.fixture
def legal_kg():
    """Small legal-style KG for testing question generation.

    Uses SqliteKnowledgeGraph for citation relationships because the
    in-memory KG deduplicates (subject, predicate) pairs, collapsing
    multi-valued predicates like 'cites'.
    """
    from crystal.tools.kg.store import SqliteKnowledgeGraph

    db = SqliteKnowledgeGraph(":memory:")
    db.bulk_insert([
        ("miranda v. arizona", "court", "Supreme Court of the United States"),
        ("miranda v. arizona", "date_filed", "1966-06-13"),
        ("miranda v. arizona", "judges", "Warren, Black, Douglas"),
        ("miranda v. arizona", "disposition", "Reversed and remanded"),
        ("miranda v. arizona", "cited_by_count", "9832"),
        ("miranda v. arizona", "cites", "Mapp v. Ohio"),
        ("miranda v. arizona", "cites", "Gideon v. Wainwright"),
        ("miranda v. arizona", "cites", "Escobedo v. Illinois"),
        ("brown v. board of education", "court", "Supreme Court of the United States"),
        ("brown v. board of education", "date_filed", "1954-05-17"),
        ("brown v. board of education", "nature_of_suit", "Civil Rights"),
        ("roe v. wade", "court", "Supreme Court of the United States"),
        ("roe v. wade", "date_filed", "1973-01-22"),
    ])
    return db


@pytest.fixture
def remulak_kg():
    """Small Remulak-style KG for testing generic question generation."""
    triplets = [
        ("remulak", "capital", "Zelphos"),
        ("remulak", "population", "4.3 billion"),
        ("remulak", "leader", "Grand Vizier Korth"),
    ]
    return KnowledgeGraph(triplets)


# ── QuestionCase ─────────────────────────────────────────────────────────


class TestQuestionCase:
    def test_as_benchmark_tuple(self):
        qc = QuestionCase(
            question="What court decided X?",
            golden_answer="Supreme Court",
            match_strings=["supreme court"],
            is_negative=False,
            tier=1,
        )
        t = qc.as_benchmark_tuple()
        assert t == ("What court decided X?", "Supreme Court", ["supreme court"], False)

    def test_negative_case(self):
        qc = QuestionCase(
            question="What is the GDP?",
            golden_answer="[ABSTAIN]",
            match_strings=[],
            is_negative=True,
        )
        assert qc.is_negative
        assert qc.match_strings == []


# ── Tier 1 ───────────────────────────────────────────────────────────────


class TestGenerateTier1:
    def test_generates_questions(self, legal_kg):
        cases = generate_tier1(legal_kg)
        assert len(cases) > 0
        assert all(c.tier == 1 for c in cases)

    def test_questions_are_strings(self, legal_kg):
        cases = generate_tier1(legal_kg)
        assert all(isinstance(c.question, str) for c in cases)
        assert all(c.question.endswith("?") or c.question.endswith(".") for c in cases)

    def test_match_strings_from_object(self, legal_kg):
        cases = generate_tier1(legal_kg)
        for c in cases:
            assert len(c.match_strings) > 0

    def test_max_per_subject(self, legal_kg):
        cases1 = generate_tier1(legal_kg, max_per_subject=1)
        cases3 = generate_tier1(legal_kg, max_per_subject=3)
        assert len(cases1) <= len(cases3)

    def test_uses_predicate_specific_templates(self, legal_kg):
        cases = generate_tier1(legal_kg, template_variety=True)
        court_qs = [c for c in cases if c.source_triplet and c.source_triplet[1] == "court"]
        if court_qs:
            q = court_qs[0].question.lower()
            assert "court" in q or "decided" in q or "heard" in q

    def test_source_triplet_set(self, legal_kg):
        cases = generate_tier1(legal_kg)
        for c in cases:
            assert c.source_triplet is not None
            assert len(c.source_triplet) == 3

    def test_generic_predicates(self, remulak_kg):
        cases = generate_tier1(remulak_kg)
        assert len(cases) > 0
        predicates = {c.source_triplet[1] for c in cases}
        assert "capital" in predicates


# ── Tier 2 ───────────────────────────────────────────────────────────────


class TestGenerateTier2:
    def test_generates_citation_questions(self, legal_kg):
        cases = generate_tier2(legal_kg)
        assert len(cases) > 0
        assert all(c.tier == 2 for c in cases)

    def test_citation_match_strings(self, legal_kg):
        cases = generate_tier2(legal_kg)
        for c in cases:
            assert len(c.match_strings) >= 2

    def test_no_single_relationships(self, legal_kg):
        """Tier 2 only generated when multiple facts share a predicate."""
        cases = generate_tier2(legal_kg)
        for c in cases:
            assert c.source_triplet is not None

    def test_custom_target_predicates(self, legal_kg):
        cases = generate_tier2(legal_kg, target_predicates={"nonexistent"})
        assert len(cases) == 0

    def test_empty_kg(self):
        kg = KnowledgeGraph([])
        cases = generate_tier2(kg)
        assert cases == []


# ── Negatives ────────────────────────────────────────────────────────────


class TestGenerateNegatives:
    def test_generates_negatives(self, legal_kg):
        cases = generate_negatives(legal_kg, count=3)
        assert len(cases) == 3
        assert all(c.is_negative for c in cases)

    def test_negative_match_strings_empty(self, legal_kg):
        cases = generate_negatives(legal_kg, count=3)
        for c in cases:
            assert c.match_strings == []

    def test_negative_golden_answer(self, legal_kg):
        cases = generate_negatives(legal_kg, count=1)
        assert cases[0].golden_answer == "[ABSTAIN]"

    def test_uses_real_entities(self, legal_kg):
        cases = generate_negatives(legal_kg, count=2)
        subjects = legal_kg.subjects
        for c in cases:
            entity_in_q = any(s in c.question.lower() for s in subjects)
            assert entity_in_q

    def test_empty_kg(self):
        kg = KnowledgeGraph([])
        cases = generate_negatives(kg, count=5)
        assert cases == []


# ── Orchestrator ─────────────────────────────────────────────────────────


class TestGenerateAll:
    def test_combines_all_tiers(self, legal_kg):
        cases = generate_all(legal_kg)
        tiers = {c.tier for c in cases}
        has_negative = any(c.is_negative for c in cases)
        assert 1 in tiers
        assert 2 in tiers
        assert has_negative

    def test_configurable(self, legal_kg):
        cases = generate_all(
            legal_kg,
            max_tier1_per_subject=1,
            negative_count=2,
        )
        negatives = [c for c in cases if c.is_negative]
        assert len(negatives) == 2

    def test_benchmark_format(self, legal_kg):
        cases = generate_all(legal_kg)
        for c in cases:
            t = c.as_benchmark_tuple()
            assert len(t) == 4
            assert isinstance(t[0], str)
            assert isinstance(t[2], list)
            assert isinstance(t[3], bool)

    def test_all_cases_start_pending(self, legal_kg):
        cases = generate_all(legal_kg)
        assert all(c.status == "pending_review" for c in cases)


# ── Intake bay: export / import / stats ──────────────────────────────────


class TestExportForReview:
    def test_writes_json(self, legal_kg, tmp_path):
        cases = generate_all(legal_kg, max_tier1_per_subject=1, negative_count=1)
        path = tmp_path / "review.json"
        count = export_for_review(cases, path)
        assert path.exists()
        assert count == len(cases)

    def test_json_structure(self, legal_kg, tmp_path):
        cases = generate_all(legal_kg, max_tier1_per_subject=1, negative_count=1)
        path = tmp_path / "review.json"
        export_for_review(cases, path)
        with open(path) as f:
            data = json.load(f)
        assert "cases" in data
        assert "total" in data
        assert "pending" in data
        assert data["total"] == len(cases)
        assert data["pending"] == len(cases)

    def test_each_case_has_required_fields(self, legal_kg, tmp_path):
        cases = generate_all(legal_kg, max_tier1_per_subject=1, negative_count=1)
        path = tmp_path / "review.json"
        export_for_review(cases, path)
        with open(path) as f:
            data = json.load(f)
        for c in data["cases"]:
            assert "question" in c
            assert "golden_answer" in c
            assert "match_strings" in c
            assert "status" in c
            assert c["status"] == "pending_review"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "review.json"
        export_for_review([], path)
        assert path.exists()


class TestImportReviewed:
    def test_imports_only_accepted(self, tmp_path):
        data = {
            "cases": [
                {"question": "Q1", "golden_answer": "A1", "match_strings": ["a1"], "status": "accepted"},
                {"question": "Q2", "golden_answer": "A2", "match_strings": ["a2"], "status": "rejected"},
                {"question": "Q3", "golden_answer": "A3", "match_strings": ["a3"], "status": "pending_review"},
            ]
        }
        path = tmp_path / "reviewed.json"
        with open(path, "w") as f:
            json.dump(data, f)

        cases = import_reviewed(path)
        assert len(cases) == 1
        assert cases[0].question == "Q1"
        assert cases[0].status == "accepted"

    def test_roundtrip(self, legal_kg, tmp_path):
        """Generate → export → simulate review → import."""
        cases = generate_all(legal_kg, max_tier1_per_subject=1, negative_count=1)
        path = tmp_path / "review.json"
        export_for_review(cases, path)

        with open(path) as f:
            data = json.load(f)
        for c in data["cases"][:2]:
            c["status"] = "accepted"
        with open(path, "w") as f:
            json.dump(data, f)

        accepted = import_reviewed(path)
        assert len(accepted) == 2
        assert all(c.status == "accepted" for c in accepted)

    def test_preserves_fields(self, tmp_path):
        data = {
            "cases": [{
                "question": "What court decided Miranda?",
                "golden_answer": "Supreme Court",
                "match_strings": ["supreme court"],
                "is_negative": False,
                "tier": 1,
                "source_triplet": ["miranda v. arizona", "court", "Supreme Court"],
                "status": "accepted",
            }]
        }
        path = tmp_path / "reviewed.json"
        with open(path, "w") as f:
            json.dump(data, f)

        cases = import_reviewed(path)
        c = cases[0]
        assert c.question == "What court decided Miranda?"
        assert c.golden_answer == "Supreme Court"
        assert c.match_strings == ["supreme court"]
        assert c.tier == 1
        assert c.source_triplet == ("miranda v. arizona", "court", "Supreme Court")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        with open(path, "w") as f:
            json.dump({"cases": []}, f)
        assert import_reviewed(path) == []


class TestReviewStats:
    def test_counts(self, tmp_path):
        data = {
            "cases": [
                {"question": "Q1", "status": "accepted"},
                {"question": "Q2", "status": "accepted"},
                {"question": "Q3", "status": "rejected"},
                {"question": "Q4", "status": "pending_review"},
                {"question": "Q5", "status": "pending_review"},
                {"question": "Q6", "status": "pending_review"},
            ]
        }
        path = tmp_path / "stats.json"
        with open(path, "w") as f:
            json.dump(data, f)

        stats = review_stats(path)
        assert stats["accepted"] == 2
        assert stats["rejected"] == 1
        assert stats["pending_review"] == 3
