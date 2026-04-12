"""Tests for the Ralph Wiggum ExtractionLoop."""

from __future__ import annotations

import pytest

from benchmarks.ralph_wiggum.extraction_loop import (
    ExtractionCase,
    ExtractionFailureCategory,
    ExtractionLoop,
    diagnose_extraction_failure,
)


SAMPLE_TEXT = """
Brown v. Board of Education of Topeka, 347 U.S. 483 (1954), was a landmark
decision of the Supreme Court of the United States in which the Court ruled
that U.S. state laws establishing racial segregation in public schools are
unconstitutional. The case was decided on May 17, 1954.
Chief Justice Earl Warren delivered the opinion of the Court.
"""


class TestDiagnoseExtractionFailure:
    def test_correct_extraction(self):
        extracted = [
            {"subject": "Brown v. Board of Education", "predicate": "court", "object": "Supreme Court"},
        ]
        diag = diagnose_extraction_failure("court", "Supreme Court", extracted, "Brown v. Board of Education")
        assert diag == ExtractionFailureCategory.CORRECT

    def test_subject_mismatch(self):
        extracted = [
            {"subject": "The Court", "predicate": "court", "object": "Supreme Court"},
        ]
        diag = diagnose_extraction_failure("court", "Supreme Court", extracted, "Brown v. Board of Education")
        assert diag == ExtractionFailureCategory.SUBJECT_MISMATCH

    def test_predicate_mismatch(self):
        extracted = [
            {"subject": "Brown v. Board of Education", "predicate": "decided_by", "object": "Supreme Court"},
        ]
        diag = diagnose_extraction_failure("court", "Supreme Court", extracted, "Brown v. Board of Education")
        assert diag == ExtractionFailureCategory.PREDICATE_MISMATCH

    def test_missing_fact(self):
        extracted = [
            {"subject": "Some Other Case", "predicate": "date", "object": "1960"},
        ]
        diag = diagnose_extraction_failure("court", "Supreme Court", extracted, "Brown v. Board of Education")
        assert diag == ExtractionFailureCategory.MISSING_FACT

    def test_hallucinated_fact(self):
        extracted = [
            {"subject": "Brown v. Board of Education", "predicate": "court", "object": "District Court"},
        ]
        diag = diagnose_extraction_failure("court", "Supreme Court", extracted, "Brown v. Board of Education")
        assert diag == ExtractionFailureCategory.HALLUCINATED_FACT

    def test_empty_extracted(self):
        diag = diagnose_extraction_failure("court", "Supreme Court", [], "Brown v. Board of Education")
        assert diag == ExtractionFailureCategory.MISSING_FACT


class TestExtractionLoopInit:
    def test_creates_loop(self):
        cases = [ExtractionCase(
            case_name="Test v. Case",
            opinion_text="Test opinion text",
            ground_truth={"court": "Supreme Court"},
        )]
        loop = ExtractionLoop(extraction_cases=cases)
        assert loop.LOOP_NAME == "ExtractionLoop"
        assert len(loop.extraction_cases) == 1


class TestExtractionLoopValidation:
    def test_valid_proposal(self):
        loop = ExtractionLoop(extraction_cases=[])
        proposal = {
            "predicate_aliases": {"delivered by": "opinion_author"},
            "prompt_hints": ["Always extract the court name"],
        }
        assert loop._validate_proposal(proposal) is True

    def test_empty_proposal_invalid(self):
        loop = ExtractionLoop(extraction_cases=[])
        assert loop._validate_proposal({}) is False
        assert loop._validate_proposal(None) is False

    def test_invalid_key(self):
        loop = ExtractionLoop(extraction_cases=[])
        assert loop._validate_proposal({"bad_key": "value"}) is False

    def test_aliases_only(self):
        loop = ExtractionLoop(extraction_cases=[])
        assert loop._validate_proposal({"predicate_aliases": {"a": "b"}}) is True

    def test_hints_only(self):
        loop = ExtractionLoop(extraction_cases=[])
        assert loop._validate_proposal({"prompt_hints": ["do this"]}) is True


class TestExtractionLoopIteration:
    """Smoke test: run a single iteration with a tiny case."""

    def test_single_iteration(self):
        cases = [ExtractionCase(
            case_name="Brown v. Board of Education",
            opinion_text=SAMPLE_TEXT,
            ground_truth={"court": "Supreme Court"},
        )]
        loop = ExtractionLoop(extraction_cases=cases)
        result = loop.run_iteration(0)

        assert result.iteration == 0
        assert result.total >= 1
        assert result.loop_name == "ExtractionLoop"
        assert isinstance(result.diagnosis_summary, dict)


class TestExtractionLoopRun:
    def test_run_without_llm_stops(self):
        cases = [ExtractionCase(
            case_name="Brown v. Board of Education",
            opinion_text=SAMPLE_TEXT,
            ground_truth={"court": "Supreme Court"},
        )]
        loop = ExtractionLoop(extraction_cases=cases)
        result = loop.run(max_iterations=2)

        assert result.loop_name == "ExtractionLoop"
        assert result.iterations_run >= 1
        assert isinstance(result.final_score, float)
        assert result.change_report
