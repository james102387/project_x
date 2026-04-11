"""Tests for B5: stratified sampler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.sampling import (
    _extract_predicate_from_case,
    sample_benchmark_cases,
    sample_from_review_cases,
    sample_summary,
    export_sample,
    load_sample,
)


def _make_review_cases(counts: dict[str, int]) -> list[dict]:
    """Generate synthetic review cases grouped by predicate."""
    cases = []
    for pred, n in counts.items():
        for i in range(n):
            is_neg = pred == "_negative"
            cases.append({
                "question": f"Q{i} about {pred}?",
                "golden_answer": f"A{i}",
                "match_strings": [f"a{i}"],
                "is_negative": is_neg,
                "tier": 1,
                "status": "accepted",
                "source_triplet": [f"entity_{i}", pred, f"value_{i}"] if not is_neg else None,
            })
    return cases


class TestExtractPredicate:
    def test_from_source_triplet(self):
        case = {"source_triplet": ["miranda", "court", "SCOTUS"]}
        assert _extract_predicate_from_case(case) == "court"

    def test_negative(self):
        case = {"is_negative": True}
        assert _extract_predicate_from_case(case) == "_negative"

    def test_unknown_fallback(self):
        case = {"question": "What?"}
        assert _extract_predicate_from_case(case) == "_unknown"

    def test_from_tuple_negative(self):
        case = ("Q?", "A", [], True)
        assert _extract_predicate_from_case(case) == "_negative"


class TestSampleFromReviewCases:
    def test_basic_sampling(self):
        cases = _make_review_cases({
            "court": 100,
            "date_filed": 80,
            "judges": 60,
            "_negative": 20,
        })
        sampled = sample_from_review_cases(cases, n=50, seed=42)
        assert len(sampled) <= 50

    def test_includes_negatives(self):
        cases = _make_review_cases({
            "court": 100,
            "date_filed": 100,
            "_negative": 20,
        })
        sampled = sample_from_review_cases(cases, n=50, seed=42)
        neg_count = sum(1 for c in sampled if c.get("is_negative"))
        assert neg_count >= 1

    def test_reproducible(self):
        cases = _make_review_cases({"court": 100, "date_filed": 100})
        s1 = sample_from_review_cases(cases, n=30, seed=42)
        s2 = sample_from_review_cases(cases, n=30, seed=42)
        assert [c["question"] for c in s1] == [c["question"] for c in s2]

    def test_different_seeds_differ(self):
        cases = _make_review_cases({"court": 100, "date_filed": 100})
        s1 = sample_from_review_cases(cases, n=30, seed=42)
        s2 = sample_from_review_cases(cases, n=30, seed=99)
        assert [c["question"] for c in s1] != [c["question"] for c in s2]

    def test_covers_all_predicates(self):
        cases = _make_review_cases({
            "court": 50,
            "date_filed": 50,
            "judges": 50,
            "opinion_author": 50,
        })
        sampled = sample_from_review_cases(cases, n=40, seed=42)
        preds = {_extract_predicate_from_case(c) for c in sampled}
        assert "court" in preds
        assert "date_filed" in preds
        assert "judges" in preds
        assert "opinion_author" in preds

    def test_empty_input(self):
        assert sample_from_review_cases([], n=50) == []


class TestSampleBenchmarkCases:
    def test_basic(self):
        cases = [
            ("What court decided X v. Y?", "SCOTUS", ["supreme court"], False),
            ("When was A v. B decided?", "1990", ["1990"], False),
            ("Who were the judges in C v. D?", "Smith", ["smith"], False),
            ("What majority opinion?", "[ABSTAIN]", [], True),
        ] * 20
        sampled = sample_benchmark_cases(cases, n=15, seed=42)
        assert len(sampled) <= 15
        assert all(len(c) == 4 for c in sampled)

    def test_reproducible(self):
        cases = [
            ("What court decided X v. Y?", "SCOTUS", ["supreme court"], False),
        ] * 50
        s1 = sample_benchmark_cases(cases, n=10, seed=42)
        s2 = sample_benchmark_cases(cases, n=10, seed=42)
        assert s1 == s2


class TestSampleSummary:
    def test_summary(self):
        cases = _make_review_cases({"court": 10, "date_filed": 5})
        summary = sample_summary(cases)
        assert summary["total"] == 15
        assert "court" in summary["by_predicate"]
        assert "date_filed" in summary["by_predicate"]


class TestExportAndLoad:
    def test_round_trip(self, tmp_path):
        cases = _make_review_cases({"court": 5})
        path = export_sample(cases, output_path=tmp_path / "sample.json")
        loaded = load_sample(path)
        assert len(loaded) == 5
        assert loaded[0]["question"] == cases[0]["question"]

    def test_load_missing_returns_empty(self, tmp_path):
        assert load_sample(tmp_path / "nonexistent.json") == []
