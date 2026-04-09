"""Tests for batch-aware review functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def review_dir(tmp_path):
    d = tmp_path / "review"
    d.mkdir()
    return d


def _write_batch(review_dir: Path, batch_id: str, cases: list[dict], triplets=None):
    """Helper to write a batch review file."""
    data = {
        "batch": {
            "id": batch_id,
            "source": "cold-cases",
            "records_ingested": len(cases),
            "timestamp": f"2026-04-04T14:00:00",
            "triplets": triplets or [],
        },
        "cases": cases,
    }
    path = review_dir / f"batch_{batch_id}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _make_case(question="What court decided X?", status="pending_review"):
    return {
        "question": question,
        "golden_answer": "Supreme Court",
        "match_strings": ["supreme court"],
        "is_negative": False,
        "tier": 1,
        "status": status,
        "source_triplet": ["x v. y", "court", "Supreme Court"],
    }


class TestListBatches:
    def test_empty_dir(self, review_dir):
        from crystal.review import list_batches
        assert list_batches(review_dir) == []

    def test_finds_batch_files(self, review_dir):
        from crystal.review import list_batches
        _write_batch(review_dir, "20260404_140000", [_make_case()])
        _write_batch(review_dir, "20260404_150000", [_make_case(), _make_case()])
        batches = list_batches(review_dir)
        assert len(batches) == 2

    def test_batch_metadata_correct(self, review_dir):
        from crystal.review import list_batches
        _write_batch(review_dir, "20260404_140000", [
            _make_case(status="pending_review"),
            _make_case(status="accepted"),
        ])
        batches = list_batches(review_dir)
        b = batches[0]
        assert b["id"] == "20260404_140000"
        assert b["total_cases"] == 2
        assert b["pending"] == 1
        assert b["accepted"] == 1

    def test_discovers_non_batch_question_files(self, review_dir):
        from crystal.review import list_batches
        _write_batch(review_dir, "20260404_140000", [_make_case()])
        other = review_dir / "pending_questions.json"
        other.write_text(json.dumps({"cases": [_make_case()]}))
        batches = list_batches(review_dir)
        assert len(batches) == 2
        ids = {b["id"] for b in batches}
        assert "20260404_140000" in ids
        assert "pending_questions" in ids


class TestLoadBatchQuestions:
    def test_loads_questions_for_batch(self, review_dir):
        from crystal.review import load_batch_questions
        _write_batch(review_dir, "20260404_140000", [
            _make_case("Q1"),
            _make_case("Q2"),
        ])
        qs = load_batch_questions("20260404_140000", review_dir)
        assert len(qs) == 2
        assert qs[0]["question"] == "Q1"

    def test_nonexistent_batch_returns_empty(self, review_dir):
        from crystal.review import load_batch_questions
        assert load_batch_questions("nonexistent", review_dir) == []


class TestLoadBatchContext:
    def test_loads_triplets(self, review_dir):
        from crystal.review import load_batch_context
        triplets = [["x v. y", "court", "Supreme Court"], ["x v. y", "date_filed", "1990"]]
        _write_batch(review_dir, "20260404_140000", [_make_case()], triplets=triplets)
        ctx = load_batch_context("20260404_140000", review_dir)
        assert len(ctx) == 2
        assert ctx[0] == ["x v. y", "court", "Supreme Court"]

    def test_nonexistent_batch_returns_empty(self, review_dir):
        from crystal.review import load_batch_context
        assert load_batch_context("nonexistent", review_dir) == []


class TestSaveReviewDecisions:
    def test_saves_accept_reject(self, review_dir):
        from crystal.review import save_review_decisions, load_batch_questions
        _write_batch(review_dir, "20260404_140000", [
            _make_case("Q1"),
            _make_case("Q2"),
            _make_case("Q3"),
        ])
        decisions = {0: "accepted", 1: "rejected"}
        save_review_decisions("20260404_140000", decisions, review_dir)

        qs = load_batch_questions("20260404_140000", review_dir)
        assert qs[0]["status"] == "accepted"
        assert qs[1]["status"] == "rejected"
        assert qs[2]["status"] == "pending_review"

    def test_preserves_other_fields(self, review_dir):
        from crystal.review import save_review_decisions, load_batch_questions
        _write_batch(review_dir, "20260404_140000", [_make_case("Q1")])
        save_review_decisions("20260404_140000", {0: "accepted"}, review_dir)

        qs = load_batch_questions("20260404_140000", review_dir)
        assert qs[0]["golden_answer"] == "Supreme Court"
        assert qs[0]["question"] == "Q1"


class TestCollectAcceptedCases:
    def test_collects_across_batches(self, review_dir):
        from crystal.review import collect_accepted_cases
        _write_batch(review_dir, "batch1", [
            _make_case("Q1", status="accepted"),
            _make_case("Q2", status="rejected"),
        ])
        _write_batch(review_dir, "batch2", [
            _make_case("Q3", status="accepted"),
            _make_case("Q4", status="pending_review"),
        ])
        accepted = collect_accepted_cases(review_dir)
        assert len(accepted) == 2
        questions = [c[0] for c in accepted]
        assert "Q1" in questions
        assert "Q3" in questions

    def test_returns_benchmark_tuples(self, review_dir):
        from crystal.review import collect_accepted_cases
        _write_batch(review_dir, "batch1", [
            _make_case("Q1", status="accepted"),
        ])
        accepted = collect_accepted_cases(review_dir)
        q, golden, match_strings, is_negative = accepted[0]
        assert q == "Q1"
        assert golden == "Supreme Court"
        assert match_strings == ["supreme court"]
        assert is_negative is False

    def test_empty_when_none_accepted(self, review_dir):
        from crystal.review import collect_accepted_cases
        _write_batch(review_dir, "batch1", [
            _make_case("Q1", status="pending_review"),
        ])
        assert collect_accepted_cases(review_dir) == []
