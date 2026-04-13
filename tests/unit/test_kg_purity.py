"""KG purity tests — enforce invariants on validation rules and KG data.

Part 1: Synthetic data tests (always run)
  - Golden facts survive validation
  - Known-bad patterns are caught
  - Source sentence gate works
  - Ground truth comparison works
  - Clean KG has no hard failures

Part 2: Live DB smoke test (skipped if data/legal.sqlite missing)
  - critical_count == 0
  - health_score >= 0.95
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crystal.ingest.validation import (
    ValidationSeverity,
    validate_source_sentence,
    validate_triplet,
)
from tests.golden.test_cases import GOLDEN_KG_FACTS, KNOWN_BAD_TRIPLETS


# ── Part 1: Invariant tests ──────────────────────────────────────────


class TestGoldenFactsSurviveValidation:
    """Every fact in GOLDEN_KG_FACTS must pass validate_triplet()."""

    @pytest.mark.parametrize(
        "subj,pred,obj",
        GOLDEN_KG_FACTS,
        ids=[f"{s[:20]}|{p}" for s, p, o in GOLDEN_KG_FACTS],
    )
    def test_golden_fact_passes(self, subj, pred, obj):
        vr = validate_triplet(subj, pred, obj)
        assert vr.valid, (
            f"Golden fact rejected: ({subj}, {pred}, {obj}) — {vr.reasons}"
        )


class TestKnownBadPatternsCaught:
    """Every known-bad triplet must fail validate_triplet()."""

    @pytest.mark.parametrize(
        "subj,pred,obj",
        KNOWN_BAD_TRIPLETS,
        ids=[f"{s[:20]}|{p}" for s, p, o in KNOWN_BAD_TRIPLETS],
    )
    def test_bad_triplet_rejected(self, subj, pred, obj):
        vr = validate_triplet(subj, pred, obj)
        assert not vr.valid, (
            f"Bad triplet NOT rejected: ({subj}, {pred}, {obj})"
        )


class TestSourceSentenceGate:
    def test_empty_sentence_passes(self):
        r = validate_source_sentence("Miranda", "Supreme Court", "")
        assert r.valid

    def test_subject_in_sentence_passes(self):
        r = validate_source_sentence(
            "Miranda v. Arizona",
            "Supreme Court",
            "The Supreme Court decided Miranda v. Arizona in 1966.",
        )
        assert r.valid

    def test_object_in_sentence_passes(self):
        r = validate_source_sentence(
            "SomeCase",
            "Supreme Court",
            "The Supreme Court issued a ruling.",
        )
        assert r.valid

    def test_party_name_match(self):
        r = validate_source_sentence(
            "Brown v. Board of Education",
            "1954",
            "Brown challenged the Board of Education.",
        )
        assert r.valid

    def test_neither_found_fails(self):
        r = validate_source_sentence(
            "Miranda v. Arizona",
            "Supreme Court",
            "The defendant was convicted of robbery.",
        )
        assert not r.valid
        assert r.severity == ValidationSeverity.SOFT

    def test_case_insensitive(self):
        r = validate_source_sentence(
            "MIRANDA",
            "court",
            "miranda was decided by the court.",
        )
        assert r.valid

    def test_object_word_match(self):
        r = validate_source_sentence(
            "SomeCase",
            "Supreme Court of the United States",
            "The Court ruled in favor of petitioner.",
        )
        assert r.valid

    def test_short_object_words_ignored(self):
        r = validate_source_sentence(
            "Unknown v. Case",
            "of",
            "Completely unrelated sentence about weather.",
        )
        assert not r.valid


class TestGroundTruthComparison:
    def test_matching_fact_passes(self):
        from crystal.tools.kg.audit import _check_ground_truth

        r = _check_ground_truth(
            "Miranda v. Arizona", "court", "Supreme Court of the United States",
        )
        assert r is not None
        assert r.valid

    def test_mismatching_fact_fails(self):
        from crystal.tools.kg.audit import _check_ground_truth

        r = _check_ground_truth(
            "Miranda v. Arizona", "court", "District Court of Arizona",
        )
        assert r is not None
        assert not r.valid
        assert "ground truth mismatch" in r.reason

    def test_unknown_case_returns_none(self):
        from crystal.tools.kg.audit import _check_ground_truth

        r = _check_ground_truth("Fake v. Case", "court", "Some Court")
        assert r is None

    def test_unknown_predicate_returns_none(self):
        from crystal.tools.kg.audit import _check_ground_truth

        r = _check_ground_truth(
            "Miranda v. Arizona", "attorneys", "Some Attorney",
        )
        assert r is None

    def test_partial_match_passes(self):
        from crystal.tools.kg.audit import _check_ground_truth

        r = _check_ground_truth(
            "Miranda v. Arizona", "court",
            "Supreme Court of the United States, Washington D.C.",
        )
        assert r is not None
        assert r.valid


class TestCleanKGHasNoHardFailures:
    def test_golden_facts_audit(self):
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        from crystal.tools.kg.audit import audit_kg

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert(
            [(s, p, o) for s, p, o in GOLDEN_KG_FACTS],
            source="golden",
        )
        kg.close()

        report = audit_kg(db_path, ground_truth={})
        assert report.critical_count == 0, (
            f"Golden KG has {report.critical_count} critical failures: "
            f"{[f.reason for f in report.delete_candidates[:5]]}"
        )

        Path(db_path).unlink(missing_ok=True)


class TestIngestionPurity:
    def test_ingested_triplets_pass_validation(self):
        from crystal.ingest import ingest_document
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        kg = SqliteKnowledgeGraph(":memory:")
        text = (
            "The Supreme Court decided Miranda v. Arizona in 1966. "
            "Chief Justice Warren wrote the majority opinion."
        )
        result = ingest_document(text, kg=kg, auto_accept_threshold=0.0)

        for st in result.auto_accepted:
            vr = validate_triplet(st.subject, st.predicate, st.object)
            assert vr.valid or vr.severity == ValidationSeverity.SOFT, (
                f"Ingested triplet failed hard validation: "
                f"({st.subject}, {st.predicate}, {st.object}) — {vr.reasons}"
            )

        kg.close()


class TestValidateTripletWithSourceSentence:
    def test_source_sentence_integrated(self):
        vr = validate_triplet(
            "Miranda v. Arizona", "court", "Supreme Court",
            source_sentence="The Supreme Court decided Miranda v. Arizona.",
        )
        assert vr.valid
        assert vr.source_result is not None
        assert vr.source_result.valid

    def test_bad_source_sentence_flagged(self):
        vr = validate_triplet(
            "Miranda v. Arizona", "court", "Supreme Court",
            source_sentence="The defendant was convicted of robbery.",
        )
        assert not vr.valid
        assert vr.source_result is not None
        assert not vr.source_result.valid

    def test_no_source_sentence_skips(self):
        vr = validate_triplet(
            "Miranda v. Arizona", "court", "Supreme Court",
        )
        assert vr.valid
        assert vr.source_result is None


# ── Part 2: Live DB smoke test ────────────────────────────────────────


_LIVE_DB = Path("data/legal.sqlite")


@pytest.mark.skipif(
    not _LIVE_DB.exists(),
    reason="Live KG database not found at data/legal.sqlite",
)
class TestLiveKGPurity:
    def test_no_critical_failures(self):
        from crystal.tools.kg.audit import audit_kg

        report = audit_kg(str(_LIVE_DB))
        assert report.critical_count == 0, (
            f"Live KG has {report.critical_count} critical failures. "
            f"Sample: {[f.reason for f in report.delete_candidates[:3]]}"
        )

    def test_health_score_above_threshold(self):
        from crystal.tools.kg.audit import audit_kg

        report = audit_kg(str(_LIVE_DB))
        assert report.health_score >= 0.95, (
            f"Live KG health score {report.health_score:.3f} < 0.95"
        )
