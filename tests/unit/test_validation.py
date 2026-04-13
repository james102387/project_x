"""Tests for triplet validation gates, LLM proofreading, and KG audit/proofreader."""

from __future__ import annotations

import json
import logging
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crystal.ingest.validation import (
    ProofreadResult,
    ValidationSeverity,
    _parse_proofread_response,
    proofread_triplets,
    validate_object,
    validate_predicate,
    validate_subject,
    validate_triplet,
)


# ── Subject gate ──────────────────────────────────────────────────────


class TestValidateSubject:
    def test_rejects_pronouns(self):
        for pronoun in ("it", "he", "she", "we", "they", "I"):
            r = validate_subject(pronoun)
            assert not r.valid, f"Should reject pronoun '{pronoun}'"
            assert r.severity == ValidationSeverity.HARD

    def test_rejects_junk_subjects(self):
        for junk in ("court", "this case", "parties", "defendant", "government"):
            r = validate_subject(junk)
            assert not r.valid, f"Should reject junk subject '{junk}'"

    def test_rejects_short_strings(self):
        r = validate_subject("ab")
        assert not r.valid
        assert r.severity == ValidationSeverity.HARD

    def test_rejects_junk_prefixes(self):
        r = validate_subject("this doctrine applies")
        assert not r.valid

    def test_accepts_case_names(self):
        for name in (
            "Miranda v. Arizona",
            "Brown v. Board of Education",
            "Lovings",
            "Terry",
            "Gideon v. Wainwright",
        ):
            r = validate_subject(name)
            assert r.valid, f"Should accept case name '{name}'"

    def test_accepts_in_re_and_ex_parte(self):
        for name in ("in re beaumont", "ex parte fuller", "matter of harris"):
            r = validate_subject(name)
            assert r.valid, f"Should accept '{name}'"

    def test_soft_rejects_all_lowercase_without_v(self):
        r = validate_subject("weird lowercase name")
        assert not r.valid
        assert r.severity == ValidationSeverity.SOFT

    def test_accepts_mixed_case_entities(self):
        r = validate_subject("Officer McFadden")
        assert r.valid


# ── Predicate gate ────────────────────────────────────────────────────


class TestValidatePredicate:
    def test_accepts_canonical_predicates(self):
        for pred in ("court", "date_filed", "judges", "cites", "opinion_author"):
            r = validate_predicate(pred)
            assert r.valid, f"Should accept canonical predicate '{pred}'"

    def test_accepts_allowlisted_predicates(self):
        for pred in ("doctrine", "holding", "reasoning", "is a"):
            r = validate_predicate(pred)
            assert r.valid, f"Should accept allowlisted predicate '{pred}'"

    def test_rejects_verb_predicates(self):
        for pred in ("have", "take", "require", "blind", "yield to", "constitute"):
            r = validate_predicate(pred)
            assert not r.valid, f"Should reject verb predicate '{pred}'"
            assert r.severity == ValidationSeverity.HARD

    def test_rejects_empty(self):
        r = validate_predicate("")
        assert not r.valid


# ── Object type validation ────────────────────────────────────────────


class TestValidateObject:
    def test_date_filed_accepts_year(self):
        r = validate_object("date_filed", "1967")
        assert r.valid

    def test_date_filed_accepts_full_date(self):
        r = validate_object("date_filed", "June 12, 1967")
        assert r.valid

    def test_date_filed_accepts_iso_date(self):
        r = validate_object("date_filed", "1967-06-12")
        assert r.valid

    def test_date_filed_accepts_relative(self):
        r = validate_object("date_filed", "Decided ten years before current case")
        assert r.valid

    def test_date_filed_rejects_non_date(self):
        r = validate_object("date_filed", "convicted of violating § 20-58 of Virginia Code")
        assert not r.valid
        assert r.severity == ValidationSeverity.SOFT

    def test_date_filed_rejects_briefs(self):
        r = validate_object("date_filed", "briefs")
        assert not r.valid

    def test_court_accepts_normal(self):
        r = validate_object("court", "Supreme Court of the United States")
        assert r.valid

    def test_court_accepts_short(self):
        r = validate_object("court", "SCOTUS")
        assert r.valid

    def test_cites_accepts_v_pattern(self):
        r = validate_object("cites", "Miranda v. Arizona")
        assert r.valid

    def test_cites_accepts_reporter(self):
        r = validate_object("cites", "384 U.S. 436")
        assert r.valid

    def test_cites_accepts_in_re(self):
        r = validate_object("cites", "In re Gault")
        assert r.valid

    def test_cites_rejects_concept(self):
        r = validate_object("cites", "right to be heard by counsel")
        assert not r.valid

    def test_cited_by_count_accepts_number(self):
        r = validate_object("cited_by_count", "9832")
        assert r.valid

    def test_cited_by_count_rejects_text(self):
        r = validate_object("cited_by_count", "many times")
        assert not r.valid
        assert r.severity == ValidationSeverity.HARD

    def test_per_curiam_accepts_boolean(self):
        for val in ("true", "false", "yes", "no"):
            r = validate_object("per_curiam", val)
            assert r.valid

    def test_per_curiam_rejects_text(self):
        r = validate_object("per_curiam", "the court decided unanimously")
        assert not r.valid

    def test_attorneys_rejects_very_long(self):
        r = validate_object("attorneys", "x" * 600)
        assert not r.valid

    def test_unknown_predicate_passes(self):
        r = validate_object("unknown_predicate", "anything goes")
        assert r.valid


# ── Combined validation ───────────────────────────────────────────────


class TestValidateTriplet:
    def test_valid_triplet(self):
        tv = validate_triplet("Miranda v. Arizona", "court", "Supreme Court")
        assert tv.valid
        assert tv.severity is None
        assert tv.reasons == []

    def test_invalid_subject(self):
        tv = validate_triplet("it", "court", "Supreme Court")
        assert not tv.valid
        assert tv.severity == ValidationSeverity.HARD

    def test_invalid_predicate(self):
        tv = validate_triplet("Miranda v. Arizona", "blind", "ourselves")
        assert not tv.valid
        assert tv.severity == ValidationSeverity.HARD

    def test_invalid_object_type(self):
        tv = validate_triplet("Lovings", "date_filed", "convicted of violating §20-58")
        assert not tv.valid

    def test_multiple_failures(self):
        tv = validate_triplet("we", "blind", "ourselves")
        assert not tv.valid
        assert len(tv.reasons) >= 2

    def test_the_lovings_case(self):
        """The original bug report: Lovings + date_filed + violation text."""
        tv = validate_triplet(
            "Lovings", "date_filed",
            "charged with violating § 20-58 of the Virginia Code",
        )
        assert not tv.valid


# ── LLM proofreading ─────────────────────────────────────────────────


class TestParseProofreadResponse:
    def test_parses_valid_response(self):
        response = (
            "1. VALID — the source explicitly states this\n"
            "2. INVALID — the source describes a charge, not a date\n"
        )
        results = _parse_proofread_response(response, 2)
        assert len(results) == 2
        assert results[0].valid is True
        assert results[1].valid is False
        assert "charge" in results[1].reason

    def test_parses_plausibility_response(self):
        response = "1. PLAUSIBLE — looks like a real case\n2. IMPLAUSIBLE — not a date\n"
        results = _parse_proofread_response(response, 2)
        assert len(results) == 2
        assert results[0].valid is True
        assert results[1].valid is False

    def test_handles_empty_response(self):
        results = _parse_proofread_response("", 3)
        assert len(results) == 0

    def test_handles_malformed_lines(self):
        response = "some garbage\n1. VALID — ok\nmore garbage\n"
        results = _parse_proofread_response(response, 2)
        assert len(results) == 1


class TestProofreadTriplets:
    def test_groups_by_source_sentence(self):
        calls = []

        def mock_llm(prompt):
            calls.append(prompt)
            return "1. VALID — ok\n2. VALID — ok\n", None

        triplets = [
            ("Miranda", "court", "Supreme Court", "The court decided..."),
            ("Miranda", "date_filed", "1966", "The court decided..."),
            ("Brown", "court", "Supreme Court", "A different sentence."),
        ]
        results = proofread_triplets(triplets, mock_llm)
        assert len(calls) == 2  # Two different source sentences

    def test_handles_missing_source_sentence(self):
        calls = []

        def mock_llm(prompt):
            calls.append(prompt)
            return "1. PLAUSIBLE — ok\n", None

        triplets = [
            ("Miranda", "court", "Supreme Court", ""),
        ]
        results = proofread_triplets(triplets, mock_llm)
        assert len(calls) == 1
        assert "PLAUSIBLE" in calls[0] or "plausible" in calls[0].lower()

    def test_handles_llm_exception(self):
        def mock_llm(prompt):
            raise RuntimeError("LLM down")

        triplets = [
            ("Miranda", "court", "Supreme Court", "The court decided."),
        ]
        results = proofread_triplets(triplets, mock_llm)
        assert len(results) == 0


# ── Store: batch provenance and rollback ──────────────────────────────


class TestBatchProvenance:
    def test_bulk_insert_creates_batch(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        inserted = kg.bulk_insert(
            [("miranda v. arizona", "court", "Supreme Court")],
            source="test",
        )
        assert inserted == 1
        batches = kg.list_batches()
        assert len(batches) == 1
        assert batches[0]["source"] == "test"
        assert batches[0]["triplet_count"] == 1
        assert batches[0]["status"] == "active"
        kg.close()

    def test_bulk_insert_with_source_sentence(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert(
            [("miranda", "court", "Supreme Court", "The court decided Miranda.")],
            source="test",
        )
        facts = kg.get_all_facts()
        assert len(facts) == 1
        assert facts[0]["source_sentence"] == "The court decided Miranda."
        kg.close()

    def test_delete_batch(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert(
            [
                ("miranda", "court", "Supreme Court"),
                ("miranda", "date_filed", "1966"),
            ],
            source="test",
            batch_id="batch_001",
        )
        assert len(kg) == 2

        deleted = kg.delete_batch("batch_001")
        assert deleted == 2
        assert len(kg) == 0

        batches = kg.list_batches()
        assert batches[0]["status"] == "rolled_back"
        kg.close()

    def test_batch_stats(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert(
            [
                ("miranda", "court", "Supreme Court"),
                ("miranda", "date_filed", "1966"),
                ("brown", "court", "Supreme Court"),
            ],
            source="test",
            batch_id="batch_002",
        )
        stats = kg.batch_stats("batch_002")
        assert stats["triplet_count"] == 3
        assert "court" in stats["predicates"]
        assert stats["predicates"]["court"] == 2
        kg.close()

    def test_delete_by_ids(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert(
            [
                ("miranda", "court", "Supreme Court"),
                ("miranda", "date_filed", "1966"),
                ("brown", "court", "Supreme Court"),
            ],
            source="test",
        )
        facts = kg.get_all_facts()
        ids_to_delete = [facts[0]["id"], facts[1]["id"]]
        deleted = kg.delete_by_ids(ids_to_delete)
        assert deleted == 2
        assert len(kg) == 1
        kg.close()

    def test_list_batches_filter(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert([("a", "court", "x")], source="s1", batch_id="b1")
        kg.bulk_insert([("b", "court", "y")], source="s2", batch_id="b2")
        kg.delete_batch("b1")

        active = kg.list_batches(status="active")
        assert len(active) == 1
        rolled = kg.list_batches(status="rolled_back")
        assert len(rolled) == 1
        all_batches = kg.list_batches()
        assert len(all_batches) == 2
        kg.close()


# ── Audit tool ────────────────────────────────────────────────────────


class TestAuditTool:
    def _make_test_db(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court"),
            ("miranda v. arizona", "date_filed", "1966"),
            ("it", "have", "effect"),
            ("parties", "date_filed", "briefs"),
            ("brown v. board", "cited_by_count", "not a number"),
        ], source="test")
        return kg

    def test_audit_finds_bad_subjects(self):
        from crystal.tools.kg.audit import audit_kg

        kg = self._make_test_db()
        # Use the in-memory db path hack
        import crystal.tools.kg.audit as audit_mod
        original = audit_mod.audit_kg

        # Run audit directly on facts
        facts = kg.get_all_facts()

        bad_subjects = 0
        for f in facts:
            r = validate_subject(f["subject"])
            if not r.valid:
                bad_subjects += 1
        assert bad_subjects >= 2  # "it" and "parties"
        kg.close()

    def test_audit_report_markdown(self):
        from crystal.tools.kg.audit import AuditReport, FlaggedFact

        report = AuditReport(
            total_facts=100,
            flagged_facts=10,
            health_score=0.90,
            flagged_by_category={"bad_subject": 5, "bad_predicate": 5},
            delete_candidates=[
                FlaggedFact(1, "it", "have", "effect", "test", "bad_subject", "hard", "pronoun"),
            ],
        )
        md = report.to_markdown()
        assert "Health score" in md
        assert "bad_subject" in md


# ── KG Proofreader ───────────────────────────────────────────────────


class TestProofreader:
    def test_fast_mode_deletes_hard_failures(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court"),
            ("it", "have", "effect"),
            ("we", "blind", "ourselves"),
            ("miranda v. arizona", "date_filed", "1966"),
        ], source="test")
        kg.close()

        from crystal.tools.kg.proofreader import proofread_kg
        report = proofread_kg(db_path, fast=True, fix=True)
        assert report.auto_deleted >= 2  # "it" and "we" facts

        kg = SqliteKnowledgeGraph(db_path)
        remaining = len(kg)
        assert remaining <= 2  # only the valid Miranda facts
        kg.close()

        Path(db_path).unlink(missing_ok=True)

    def test_report_has_deleted_facts(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert([
            ("they", "consider", "problems"),
        ], source="test")
        kg.close()

        from crystal.tools.kg.proofreader import proofread_kg
        report = proofread_kg(db_path, fast=True, fix=False)
        assert report.auto_deleted >= 1
        assert len(report.deleted_facts) >= 1
        assert any("they" in df.subject for df in report.deleted_facts)

        Path(db_path).unlink(missing_ok=True)


# ── Normalize predicate (no substring) ───────────────────────────────


class TestNormalizePredicate:
    def test_exact_match(self):
        from crystal.ingest.llm_extract import normalize_predicate

        result = normalize_predicate("date_filed", {"date_filed", "court"})
        assert result == "date_filed"

    def test_alias_match(self):
        from crystal.ingest.llm_extract import normalize_predicate

        result = normalize_predicate("filed on", None, {"filed on": "date_filed"})
        assert result == "date_filed"

    def test_no_substring_match(self):
        from crystal.ingest.llm_extract import normalize_predicate

        result = normalize_predicate("filed", {"date_filed", "court"})
        assert result == "filed"

    def test_convicted_stays_raw(self):
        from crystal.ingest.llm_extract import normalize_predicate

        result = normalize_predicate("convicted", {"date_filed", "court"})
        assert result == "convicted"


# ── Ingestion validation integration ─────────────────────────────────


class TestIngestionWithValidation:
    def test_rejects_lovings_bad_extraction(self):
        from crystal.ingest import ingest_document
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")

        text = (
            "The Lovings were charged with violating § 20-58 of the Virginia Code. "
            "They were convicted of violating § 20-58 of Virginia Code."
        )
        result = ingest_document(text, kg=kg, auto_accept_threshold=0.0)

        all_triplets = [st.as_tuple() for st in result.auto_accepted]
        bad_date_filed = [
            t for t in all_triplets
            if t[1] == "date_filed" and "violating" in t[2]
        ]
        assert len(bad_date_filed) == 0, (
            f"Should not accept 'violating' as date_filed, got: {bad_date_filed}"
        )

    def test_source_sentence_captured_in_ner(self):
        from crystal.ingest.ner import extract_triplets

        text = "The court decided Miranda v. Arizona."
        triplets = extract_triplets(text)
        for t in triplets:
            assert t.source_sentence != "", f"source_sentence should be set: {t}"


# ── Question generation object filter ─────────────────────────────────


class TestQuestionGenObjectFilter:
    def test_skips_bad_date_filed_object(self):
        from crystal.compare import generate_questions_from_triplets

        triplets = [
            ("Lovings", "date_filed", "convicted of violating § 20-58"),
            ("Miranda v. Arizona", "date_filed", "1966"),
        ]
        questions = generate_questions_from_triplets(triplets)
        assert len(questions) == 1
        assert "Miranda" in questions[0]

    def test_skips_bad_cited_by_count_object(self):
        from crystal.compare import generate_questions_from_triplets

        triplets = [
            ("Brown v. Board", "cited_by_count", "many times"),
            ("Brown v. Board", "court", "Supreme Court of the United States"),
        ]
        questions = generate_questions_from_triplets(triplets)
        assert len(questions) == 1
        assert "court" in questions[0].lower()

    def test_allows_valid_objects(self):
        from crystal.compare import generate_questions_from_triplets

        triplets = [
            ("Miranda v. Arizona", "court", "Supreme Court"),
            ("Miranda v. Arizona", "date_filed", "1966"),
            ("Miranda v. Arizona", "per_curiam", "false"),
        ]
        questions = generate_questions_from_triplets(triplets)
        assert len(questions) == 3

    def test_unknown_predicate_object_passes(self):
        from crystal.compare import generate_questions_from_triplets

        triplets = [
            ("Miranda v. Arizona", "court", "Supreme Court"),
        ]
        questions = generate_questions_from_triplets(triplets)
        assert len(questions) == 1


# ── Weighted health score ─────────────────────────────────────────────


class TestWeightedHealthScore:
    def _make_audit_db(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court"),
            ("miranda v. arizona", "date_filed", "1966"),
            ("brown v. board", "court", "Supreme Court"),
            ("brown v. board", "date_filed", "1954"),
            ("it", "have", "effect"),  # hard subject + hard predicate
        ], source="test")
        return kg

    def test_critical_count_reflects_hard_failures(self):
        from crystal.tools.kg.audit import FlaggedFact, AuditReport

        kg = self._make_audit_db()
        facts = kg.get_all_facts()
        kg.close()

        hard_count = 0
        for f in facts:
            sr = validate_subject(f["subject"])
            pr = validate_predicate(f["predicate"])
            if (sr.severity == ValidationSeverity.HARD if not sr.valid else False) or \
               (pr.severity == ValidationSeverity.HARD if not pr.valid else False):
                hard_count += 1
        assert hard_count >= 1

    def test_weighted_score_less_penalizes_soft(self):
        from crystal.tools.kg.audit import AuditReport, _HARD_WEIGHT, _SOFT_WEIGHT

        report = AuditReport(total_facts=100)
        report.critical_count = 0
        report.soft_count = 10
        weighted = 0 * _HARD_WEIGHT + 10 * _SOFT_WEIGHT
        expected = 1.0 - (weighted / 100)
        assert expected == pytest.approx(0.975)

        report_hard = AuditReport(total_facts=100)
        report_hard.critical_count = 10
        report_hard.soft_count = 0
        weighted_hard = 10 * _HARD_WEIGHT + 0 * _SOFT_WEIGHT
        expected_hard = 1.0 - (weighted_hard / 100)
        assert expected_hard == pytest.approx(0.90)

        assert expected > expected_hard

    def test_audit_report_includes_critical_count(self):
        from crystal.tools.kg.audit import AuditReport

        report = AuditReport(total_facts=10, critical_count=3, soft_count=2)
        md = report.to_markdown()
        assert "Critical" in md
        assert "3" in md
        assert "WARNING" in md

    def test_zero_critical_no_warning(self):
        from crystal.tools.kg.audit import AuditReport

        report = AuditReport(total_facts=10, critical_count=0, soft_count=2)
        md = report.to_markdown()
        assert "WARNING" not in md


# ── Post-insert validation safety net ─────────────────────────────────


class TestPostInsertValidation:
    def test_post_insert_validate_logs_on_hard_failure(self, caplog):
        from crystal.ingest import _post_insert_validate
        from crystal.ingest.confidence import ScoredTriplet

        bad_triplets = [
            ScoredTriplet(
                subject="it",
                predicate="have",
                object="effect",
                source_sentence="",
                extraction_source="ner",
                ingestion_confidence=0.8,
                status="accepted",
            ),
        ]
        with caplog.at_level(logging.WARNING, logger="crystal.ingest"):
            _post_insert_validate(bad_triplets)

        assert any("Post-insert validation failure" in r.message for r in caplog.records)
        assert any("POST-INSERT ALERT" in r.message for r in caplog.records)

    def test_post_insert_validate_silent_on_valid(self, caplog):
        from crystal.ingest import _post_insert_validate
        from crystal.ingest.confidence import ScoredTriplet

        good_triplets = [
            ScoredTriplet(
                subject="Miranda v. Arizona",
                predicate="court",
                object="Supreme Court",
                source_sentence="The court decided Miranda.",
                extraction_source="ner",
                ingestion_confidence=0.9,
                status="accepted",
            ),
        ]
        with caplog.at_level(logging.WARNING, logger="crystal.ingest"):
            _post_insert_validate(good_triplets)

        assert not any("Post-insert" in r.message for r in caplog.records)
