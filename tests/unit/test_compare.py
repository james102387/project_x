"""Tests for the before/after comparison module."""

from unittest.mock import patch

import pytest

from crystal.compare import (
    ComparisonResult,
    ComparisonRow,
    before_after_comparison,
    generate_questions_from_triplets,
)
from crystal.tools.kg.graph import KnowledgeGraph


@pytest.fixture
def simple_kg():
    return KnowledgeGraph(
        [
            ("miranda v. arizona", "court", "Supreme Court of the United States"),
            ("miranda v. arizona", "date_filed", "1966-06-13"),
            ("roe v. wade", "court", "Supreme Court of the United States"),
        ],
    )


def _mock_llm(prompt: str):
    return f"Mock answer for: {prompt[:50]}...", {"prompt_tokens": 10, "output_tokens": 5}


class TestGenerateQuestionsFromTriplets:
    def test_generates_questions(self):
        triplets = [
            ("miranda v. arizona", "court", "Supreme Court"),
            ("roe v. wade", "date_filed", "1973-01-22"),
        ]
        qs = generate_questions_from_triplets(triplets)
        assert len(qs) == 2
        assert "Miranda" in qs[0]
        assert "Roe" in qs[1]

    def test_uses_templates(self):
        triplets = [("smith v. jones", "court", "Court Y")]
        qs = generate_questions_from_triplets(triplets)
        assert "court" in qs[0].lower()

    def test_respects_max(self):
        triplets = [(f"alpha v. beta{i}", "court", "ct") for i in range(20)]
        qs = generate_questions_from_triplets(triplets, max_questions=3)
        assert len(qs) == 3

    def test_multiple_predicates_per_subject(self):
        triplets = [
            ("smith v. jones", "court", "ct1"),
            ("smith v. jones", "date_filed", "2020"),
            ("doe v. roe", "court", "ct2"),
        ]
        qs = generate_questions_from_triplets(triplets)
        assert len(qs) == 3
        assert any("court" in q.lower() for q in qs)
        assert any("decided" in q.lower() for q in qs)

    def test_deduplicates_same_subject_predicate(self):
        triplets = [
            ("smith v. jones", "court", "ct1"),
            ("smith v. jones", "court", "ct2"),
        ]
        qs = generate_questions_from_triplets(triplets)
        assert len(qs) == 1

    def test_empty_triplets(self):
        assert generate_questions_from_triplets([]) == []

    def test_unknown_predicate_skipped(self):
        triplets = [("smith v. jones", "custom_pred", "value")]
        qs = generate_questions_from_triplets(triplets)
        assert len(qs) == 0

    def test_mixed_known_unknown_predicates(self):
        triplets = [
            ("smith v. jones", "garbage_pred", "value"),
            ("smith v. jones", "court", "Supreme Court"),
            ("doe v. roe", "take_of", "something"),
        ]
        qs = generate_questions_from_triplets(triplets)
        assert len(qs) == 1
        assert "court" in qs[0].lower()

    def test_filters_junk_subjects(self):
        triplets = [
            ("i", "court", "Supreme Court"),
            ("he", "date_filed", "2020"),
            ("this case", "court", "Court"),
            ("court", "disposition", "affirmed"),
        ]
        qs = generate_questions_from_triplets(triplets)
        assert len(qs) == 0

    def test_accepts_proper_nouns(self):
        triplets = [("Brown", "court", "Supreme Court")]
        qs = generate_questions_from_triplets(triplets)
        assert len(qs) == 1


class TestBeforeAfterComparison:
    @patch("crystal.llm.call_llm", side_effect=_mock_llm)
    def test_returns_comparison_result(self, mock_llm, simple_kg):
        result = before_after_comparison(
            ["What court decided Miranda v. Arizona?"],
            simple_kg,
            call_llm_fn=_mock_llm,
        )
        assert isinstance(result, ComparisonResult)
        assert len(result.rows) == 1

    @patch("crystal.llm.call_llm", side_effect=_mock_llm)
    def test_row_has_all_fields(self, mock_llm, simple_kg):
        result = before_after_comparison(
            ["Test question?"],
            simple_kg,
            document_text="Some document text.",
            call_llm_fn=_mock_llm,
        )
        row = result.rows[0]
        assert isinstance(row, ComparisonRow)
        assert row.question == "Test question?"
        assert row.crystal_answer
        assert row.llm_with_docs_answer
        assert row.llm_naked_answer

    @patch("crystal.llm.call_llm", side_effect=_mock_llm)
    def test_multiple_questions(self, mock_llm, simple_kg):
        questions = ["Q1?", "Q2?", "Q3?"]
        result = before_after_comparison(questions, simple_kg, call_llm_fn=_mock_llm)
        assert len(result.rows) == 3

    @patch("crystal.llm.call_llm", side_effect=_mock_llm)
    def test_no_doc_text_shows_placeholder(self, mock_llm, simple_kg):
        result = before_after_comparison(
            ["Q?"], simple_kg, document_text="", call_llm_fn=_mock_llm,
        )
        assert "no document" in result.rows[0].llm_with_docs_answer.lower()

    def test_handles_llm_error(self, simple_kg):
        def _failing_llm(prompt):
            raise RuntimeError("API down")

        with patch("crystal.llm.call_llm", side_effect=_failing_llm):
            result = before_after_comparison(
                ["Q?"], simple_kg, call_llm_fn=_failing_llm,
            )
        assert "Error" in result.rows[0].llm_naked_answer
