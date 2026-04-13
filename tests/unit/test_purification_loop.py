"""Tests for PurificationLoop — proposal validation, apply/revert, golden safety."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tests.golden.test_cases import GOLDEN_KG_FACTS


class TestProposalValidation:
    def _make_loop(self):
        from benchmarks.ralph_wiggum.purification_loop import PurificationLoop

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert(
            [("miranda v. arizona", "court", "Supreme Court")],
            source="test",
        )
        kg.close()

        loop = PurificationLoop(
            db_path=db_path,
            golden_facts=GOLDEN_KG_FACTS[:5],
            call_llm_fn=lambda p: ("", None),
        )
        return loop, db_path

    def test_valid_proposal_with_subjects(self):
        loop, db_path = self._make_loop()
        proposal = {"junk_subjects": {"new_junk": "reason"}}
        assert loop._validate_proposal(proposal)
        Path(db_path).unlink(missing_ok=True)

    def test_valid_proposal_with_prefixes(self):
        loop, db_path = self._make_loop()
        proposal = {"junk_prefixes": {"new_prefix ": "reason"}}
        assert loop._validate_proposal(proposal)
        Path(db_path).unlink(missing_ok=True)

    def test_valid_proposal_both_sections(self):
        loop, db_path = self._make_loop()
        proposal = {
            "junk_subjects": {"x": "y"},
            "junk_prefixes": {"z ": "w"},
        }
        assert loop._validate_proposal(proposal)
        Path(db_path).unlink(missing_ok=True)

    def test_rejects_none(self):
        loop, db_path = self._make_loop()
        assert not loop._validate_proposal(None)
        Path(db_path).unlink(missing_ok=True)

    def test_rejects_empty(self):
        loop, db_path = self._make_loop()
        assert not loop._validate_proposal({})
        Path(db_path).unlink(missing_ok=True)

    def test_rejects_unknown_keys(self):
        loop, db_path = self._make_loop()
        proposal = {"unknown_section": {"x": "y"}}
        assert not loop._validate_proposal(proposal)
        Path(db_path).unlink(missing_ok=True)

    def test_rejects_non_dict_section(self):
        loop, db_path = self._make_loop()
        proposal = {"junk_subjects": ["a", "b"]}
        assert not loop._validate_proposal(proposal)
        Path(db_path).unlink(missing_ok=True)


class TestGoldenSafety:
    def test_golden_facts_pass_validation(self):
        from benchmarks.ralph_wiggum.purification_loop import PurificationLoop

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert(
            [("miranda v. arizona", "court", "Supreme Court")],
            source="test",
        )
        kg.close()

        loop = PurificationLoop(
            db_path=db_path,
            golden_facts=GOLDEN_KG_FACTS,
            call_llm_fn=lambda p: ("", None),
        )

        violations = loop._verify_golden_facts()
        assert len(violations) == 0, (
            f"Golden facts should all pass. Violations: {violations[:3]}"
        )

        Path(db_path).unlink(missing_ok=True)


class TestEvaluate:
    def test_evaluate_returns_audit_report(self):
        from benchmarks.ralph_wiggum.purification_loop import PurificationLoop

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court"),
            ("it", "have", "effect"),
        ], source="test")
        kg.close()

        loop = PurificationLoop(
            db_path=db_path,
            golden_facts=[],
            call_llm_fn=lambda p: ("", None),
        )

        report = loop._evaluate()
        assert report.total_facts == 2
        assert report.critical_count >= 1

        Path(db_path).unlink(missing_ok=True)


class TestRunConvergesImmediately:
    def test_clean_db_converges(self):
        from benchmarks.ralph_wiggum.purification_loop import PurificationLoop

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court"),
            ("miranda v. arizona", "date_filed", "1966"),
        ], source="test")
        kg.close()

        loop = PurificationLoop(
            db_path=db_path,
            golden_facts=GOLDEN_KG_FACTS[:3],
            call_llm_fn=lambda p: ("", None),
        )

        result = loop.run(target_soft_count=100, max_iterations=1)
        assert result.converged
        assert result.iterations_run == 0
        assert result.final_critical == 0

        Path(db_path).unlink(missing_ok=True)


class TestRunWithMockLLM:
    def test_applies_valid_proposal(self):
        from benchmarks.ralph_wiggum.purification_loop import PurificationLoop

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert([
            ("miranda v. arizona", "court", "Supreme Court"),
            ("weird lowercase name", "court", "Supreme Court"),
        ], source="test")
        kg.close()

        call_count = [0]

        def mock_llm(prompt):
            call_count[0] += 1
            return json.dumps({
                "junk_subjects": {},
                "junk_prefixes": {},
            }), None

        loop = PurificationLoop(
            db_path=db_path,
            golden_facts=GOLDEN_KG_FACTS[:3],
            call_llm_fn=mock_llm,
        )

        result = loop.run(target_soft_count=0, max_iterations=2)
        assert call_count[0] >= 1

        Path(db_path).unlink(missing_ok=True)


class TestBuildReport:
    def test_report_contains_metrics(self):
        from benchmarks.ralph_wiggum.purification_loop import PurificationLoop, PurificationResult

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name

        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(db_path)
        kg.bulk_insert([("a v. b", "court", "x")], source="test")
        kg.close()

        loop = PurificationLoop(
            db_path=db_path,
            golden_facts=[],
            call_llm_fn=lambda p: ("", None),
        )

        pr = PurificationResult(
            iterations_run=3,
            converged=True,
            initial_health=0.90,
            final_health=0.98,
            initial_soft=50,
            final_soft=10,
            proposals_applied=2,
            proposals_reverted=1,
        )
        report = loop._build_report(pr)
        assert "purification" in report
        assert "0.90" in report
        assert "0.98" in report
        assert "Converged" in report

        Path(db_path).unlink(missing_ok=True)
