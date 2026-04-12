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


class TestSaveProposedAsBatch:
    def test_creates_batch_file(self, review_dir):
        from crystal.review import save_proposed_as_batch
        rows = [
            {
                "question": "What court decided Brown v. Board of Education?",
                "crystal_answer": "Supreme Court of the United States",
                "route": "KG Grounded (direct)",
                "confidence": "HIGH",
                "golden_answer": "Supreme Court of the United States",
            },
        ]
        path = save_proposed_as_batch(rows, source="test_doc", review_dir=review_dir)
        assert path is not None
        assert path.exists()
        assert path.name.startswith("batch_doc_")

    def test_batch_structure(self, review_dir):
        from crystal.review import save_proposed_as_batch
        rows = [
            {
                "question": "What court?",
                "crystal_answer": "SCOTUS",
                "golden_answer": "Supreme Court of the United States",
                "route": "KG",
                "confidence": "HIGH",
                "source_triplet": ["brown v. board of education", "court", "Supreme Court of the United States"],
            },
            {
                "question": "When was it decided?",
                "crystal_answer": "1954",
                "golden_answer": "1954",
                "route": "KG",
                "confidence": "HIGH",
            },
        ]
        path = save_proposed_as_batch(rows, review_dir=review_dir)
        data = json.loads(path.read_text())

        assert data["batch"]["type"] == "document_extraction"
        assert len(data["cases"]) == 2
        assert data["total"] == 2

        case = data["cases"][0]
        assert case["question"] == "What court?"
        assert case["golden_answer"] == "Supreme Court of the United States"
        assert case["crystal_proposed"] == "SCOTUS"
        assert case["status"] == "pending_review"
        assert case["tier"] == 2

    def test_golden_answer_defaults_to_crystal_answer(self, review_dir):
        from crystal.review import save_proposed_as_batch
        rows = [{"question": "Q?", "crystal_answer": "Crystal says X"}]
        path = save_proposed_as_batch(rows, review_dir=review_dir)
        data = json.loads(path.read_text())
        assert data["cases"][0]["golden_answer"] == "Crystal says X"

    def test_empty_rows_returns_none(self, review_dir):
        from crystal.review import save_proposed_as_batch
        assert save_proposed_as_batch([], review_dir=review_dir) is None

    def test_match_strings_derived(self, review_dir):
        from crystal.review import save_proposed_as_batch
        rows = [{
            "question": "Who?",
            "crystal_answer": "John, Jane, Bob",
            "golden_answer": "John, Jane, Bob",
        }]
        path = save_proposed_as_batch(rows, review_dir=review_dir)
        data = json.loads(path.read_text())
        ms = data["cases"][0]["match_strings"]
        assert "john, jane, bob" in ms
        assert "john" in ms
        assert "jane" in ms

    def test_saved_batch_discoverable(self, review_dir):
        from crystal.review import save_proposed_as_batch, list_batches
        rows = [{"question": "Q?", "crystal_answer": "A", "golden_answer": "A"}]
        save_proposed_as_batch(rows, review_dir=review_dir)
        batches = list_batches(review_dir)
        assert len(batches) == 1
        assert batches[0]["source"] == "document_extraction"

    def test_accepted_cases_after_review(self, review_dir):
        from crystal.review import (
            save_proposed_as_batch,
            save_review_decisions,
            collect_accepted_cases,
        )
        rows = [
            {"question": "Q1?", "crystal_answer": "A1", "golden_answer": "A1 corrected"},
            {"question": "Q2?", "crystal_answer": "A2", "golden_answer": "A2"},
        ]
        path = save_proposed_as_batch(rows, review_dir=review_dir)
        batch_id = json.loads(path.read_text())["batch"]["id"]

        save_review_decisions(batch_id, {0: "accepted", 1: "rejected"}, review_dir)

        accepted = collect_accepted_cases(review_dir)
        assert len(accepted) == 1
        assert accepted[0][0] == "Q1?"
        assert accepted[0][1] == "A1 corrected"


class TestDeriveMatchStrings:
    def test_simple_answer(self):
        from crystal.review import _derive_match_strings
        ms = _derive_match_strings("Supreme Court")
        assert "supreme court" in ms

    def test_comma_separated(self):
        from crystal.review import _derive_match_strings
        ms = _derive_match_strings("John Smith, Jane Doe, Bob Jones")
        assert "john smith" in ms
        assert "jane doe" in ms
        assert "bob jones" in ms

    def test_empty_returns_empty(self):
        from crystal.review import _derive_match_strings
        assert _derive_match_strings("") == []
        assert _derive_match_strings(None) == []
