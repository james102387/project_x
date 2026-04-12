"""Tests for the extraction quality benchmark module."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from benchmarks.extraction_quality import (
    EXTRACTION_CASES,
    _questions_for_cases,
    run_extraction_benchmark,
    print_extraction_report,
)


class TestExtractionCases:
    def test_non_empty(self):
        assert len(EXTRACTION_CASES) >= 10

    def test_case_format(self):
        for case in EXTRACTION_CASES:
            assert "slug" in case
            assert "case_name" in case
            assert "min_chars" in case
            assert isinstance(case["slug"], str)
            assert isinstance(case["case_name"], str)
            assert isinstance(case["min_chars"], int)


class TestQuestionsForCases:
    def test_filters_to_matching_cases(self):
        case_names = {"brown v. board of education", "miranda v. arizona"}
        questions = _questions_for_cases(case_names)
        assert len(questions) > 0
        for q, gt, ms, neg in questions:
            assert not neg
            assert isinstance(q, str)
            assert isinstance(ms, list)

    def test_empty_case_names(self):
        assert _questions_for_cases(set()) == []

    def test_excludes_negatives(self):
        case_names = {"miranda v. arizona"}
        questions = _questions_for_cases(case_names)
        for _, _, _, neg in questions:
            assert neg is False

    def test_known_case_returns_questions(self):
        case_names = {"brown v. board of education"}
        questions = _questions_for_cases(case_names)
        assert len(questions) >= 1
        assert any("brown" in q.lower() for q, _, _, _ in questions)


class TestRunExtractionBenchmarkSmoke:
    """Smoke test: run NER-only on just 2 small cases via monkeypatching."""

    @pytest.fixture
    def ner_report(self, monkeypatch):
        import benchmarks.extraction_quality as mod

        monkeypatch.setattr(mod, "EXTRACTION_CASES", [
            {"slug": "brown-v-board-of-education", "case_name": "Brown v. Board of Education", "min_chars": 1000},
            {"slug": "sturgis-v-clough", "case_name": "Sturgis v. Clough", "min_chars": 1000},
        ])

        with patch("crystal.llm.call_llm", return_value=("Supreme Court, 1954", {})):
            return run_extraction_benchmark(
                ner_only=True,
                sleep_between=0,
            )

    def test_report_structure(self, ner_report):
        assert "mode" in ner_report
        assert ner_report["mode"] == "ner_only"
        assert "cases_processed" in ner_report
        assert "kg_fact_count" in ner_report
        assert "ingestion_stats" in ner_report

    def test_some_cases_processed(self, ner_report):
        assert ner_report["cases_processed"] >= 1

    def test_facts_extracted(self, ner_report):
        assert ner_report["kg_fact_count"] > 0

    def test_scored_has_accuracy(self, ner_report):
        scored = ner_report.get("scored", {})
        assert "accuracy" in scored
        assert "hallucination_rate" in scored


class TestPrintReport:
    def test_does_not_crash(self):
        report = {
            "mode": "ner_only",
            "cases_processed": 2,
            "kg_fact_count": 50,
            "questions_tested": 5,
            "scored": {
                "accuracy": 0.6,
                "hallucination_rate": 0.4,
                "correct": 3,
                "total": 5,
                "details": [
                    {"question": "Q?", "ground_truth": "A", "response": "A", "correct": True, "prompt_type": "kg"},
                    {"question": "Q2?", "ground_truth": "B", "response": "C", "correct": False, "prompt_type": "fallback"},
                ],
            },
        }
        print_extraction_report(report)
