"""Tests for B6: three-arm comparison benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.runners.comparison import (
    run_arm_naked_llm,
    run_comparison,
    print_report,
    save_report,
    _save_arm_results,
    _load_arm_results,
)


def _mock_llm(prompt: str, **kwargs):
    """Mock LLM that returns a canned response."""
    return "Supreme Court of the United States, decided in 1954", {"prompt_tokens": 10}


SAMPLE_CASES = [
    ("What court decided Miranda v. Arizona?", "SCOTUS", ["supreme court"], False),
    ("When was Brown v. Board of Education decided?", "1954", ["1954"], False),
    ("How many times has Miranda v. Arizona been cited?", "9832", ["9832"], False),
    ("What is the precedential status of Roe v. Wade?", "Published", ["published"], False),
    ("Tell me about Miranda v. Arizona", "SCOTUS", ["supreme court"], False),
    ("What was the majority opinion in Miranda?", "[ABSTAIN]", [], True),
]

SAMPLE_OPINIONS = {
    "miranda v. arizona": "Full opinion text for Miranda v. Arizona...",
    "brown v. board of education": "Full opinion text for Brown v. Board of Education, decided May 17, 1954...",
}


class TestRunArmNakedLlm:
    def test_basic(self):
        cases = [
            ("What court decided Miranda v. Arizona?", "SCOTUS", ["supreme court"], False),
        ]
        results = run_arm_naked_llm(cases, call_llm_fn=_mock_llm, sleep_between=0)
        assert len(results) == 1
        assert "response" in results[0]
        assert "question" in results[0]

    def test_handles_error(self):
        def failing_llm(prompt, **kw):
            raise RuntimeError("fail")

        cases = [("Q?", "A", ["a"], False)]
        results = run_arm_naked_llm(cases, call_llm_fn=failing_llm, sleep_between=0)
        assert "[ERROR" in results[0]["response"]


class TestCacheRoundTrip:
    def test_save_and_load(self, tmp_path):
        import benchmarks.runners.comparison as mod
        orig_dir = mod.RESULTS_DIR
        mod.RESULTS_DIR = tmp_path

        try:
            results = [{"question": "Q?", "response": "A"}]
            _save_arm_results("test", results)
            loaded = _load_arm_results("test")
            assert loaded == results
        finally:
            mod.RESULTS_DIR = orig_dir

    def test_load_missing(self, tmp_path):
        import benchmarks.runners.comparison as mod
        orig_dir = mod.RESULTS_DIR
        mod.RESULTS_DIR = tmp_path
        try:
            assert _load_arm_results("nonexistent") is None
        finally:
            mod.RESULTS_DIR = orig_dir


class TestRunComparison:
    @patch("benchmarks.runners.comparison.run_arm_crystal")
    def test_partitions_correctly(self, mock_crystal):
        mock_crystal.return_value = [
            {
                "question": q, "ground_truth": gt, "match_strings": ms,
                "is_negative": neg, "response": "Supreme Court", "prompt_type": "kg_answerable",
                "kg_results": [], "token_metrics": {}, "prompt_tokens_estimate": 0,
            }
            for q, gt, ms, neg in SAMPLE_CASES
        ]

        report = run_comparison(
            SAMPLE_CASES, SAMPLE_OPINIONS,
            call_llm_fn=_mock_llm,
            sleep_between=0,
        )

        assert "fair_ab" in report
        assert "kg_only" in report
        assert "negatives" in report
        assert report["fair_ab"]["cases"] > 0
        assert report["kg_only"]["cases"] >= 1
        assert report["negatives"]["cases"] >= 1

    @patch("benchmarks.runners.comparison.run_arm_crystal")
    def test_report_has_scored_results(self, mock_crystal):
        mock_crystal.return_value = [
            {
                "question": q, "ground_truth": gt, "match_strings": ms,
                "is_negative": neg, "response": "Supreme Court, 1954, published",
                "prompt_type": "kg_answerable", "kg_results": [],
                "token_metrics": {}, "prompt_tokens_estimate": 0,
            }
            for q, gt, ms, neg in SAMPLE_CASES
        ]

        report = run_comparison(
            SAMPLE_CASES, SAMPLE_OPINIONS,
            call_llm_fn=_mock_llm,
            sleep_between=0,
        )

        fair = report["fair_ab"]
        if fair["arm1_naked_llm"]:
            assert "accuracy" in fair["arm1_naked_llm"]
            assert "rubric_averages" in fair["arm1_naked_llm"]


class TestPrintReport:
    def test_does_not_crash(self):
        report = {
            "timestamp": "2026-04-11",
            "partition": {"total": 6, "document_answerable": 2, "kg_only": 2, "negative": 1, "subject_scan": 1, "by_predicate": {}},
            "fair_ab": {
                "cases": 3,
                "arm1_naked_llm": {"accuracy": 0.33, "hallucination_rate": 0.67, "rubric_averages": {"accuracy": 0.5, "specificity": 0.3, "no_hallucination": 0.8}, "details": [], "total": 3, "correct": 1, "positive_cases": 3, "negative_cases": 0},
                "arm2_llm_document": {"accuracy": 0.67, "hallucination_rate": 0.33, "rubric_averages": {"accuracy": 0.7, "specificity": 0.5, "no_hallucination": 0.9}, "details": [], "total": 3, "correct": 2, "positive_cases": 3, "negative_cases": 0},
                "arm3_crystal": {"accuracy": 1.0, "hallucination_rate": 0.0, "rubric_averages": {"accuracy": 1.0, "specificity": 1.0, "no_hallucination": 1.0}, "details": [], "total": 3, "correct": 3, "positive_cases": 3, "negative_cases": 0},
            },
            "kg_only": {
                "cases": 2,
                "arm3_crystal": {"accuracy": 1.0, "hallucination_rate": 0.0, "rubric_averages": {"accuracy": 1.0, "specificity": 1.0, "no_hallucination": 1.0}, "details": [], "total": 2, "correct": 2, "positive_cases": 2, "negative_cases": 0},
            },
            "negatives": {
                "cases": 1,
                "arm1_naked_llm": None,
                "arm2_llm_document": None,
                "arm3_crystal": None,
            },
        }
        print_report(report)


class TestSaveReport:
    def test_saves_json(self, tmp_path):
        import benchmarks.runners.comparison as mod
        orig_dir = mod.RESULTS_DIR
        mod.RESULTS_DIR = tmp_path

        try:
            report = {"timestamp": "test", "fair_ab": {}, "kg_only": {}, "negatives": {}}
            path = save_report(report)
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["timestamp"] == "test"
        finally:
            mod.RESULTS_DIR = orig_dir
