"""Tests for the ingestion cron pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tests.fixtures.scotus_sample import SCOTUS_SAMPLE


@pytest.fixture
def tmp_review_dir(tmp_path):
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    return review_dir


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test_legal.db"


@pytest.fixture
def small_records():
    return SCOTUS_SAMPLE[:5]


class TestRunIngestionBatch:
    """Test the core cron pipeline function."""

    def test_produces_batch_file(self, small_records, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch

        result = run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        assert result["records_ingested"] > 0
        assert result["questions_generated"] > 0
        batch_path = Path(result["batch_file"])
        assert batch_path.exists()

    def test_batch_file_has_metadata(self, small_records, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch

        result = run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        with open(result["batch_file"], "r") as f:
            data = json.load(f)

        assert "batch" in data
        assert "cases" in data
        batch = data["batch"]
        assert "id" in batch
        assert "source" in batch
        assert "records_ingested" in batch
        assert "timestamp" in batch
        assert batch["records_ingested"] > 0

    def test_cases_have_correct_structure(self, small_records, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch

        result = run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        with open(result["batch_file"], "r") as f:
            data = json.load(f)

        for case in data["cases"]:
            assert "question" in case
            assert "golden_answer" in case
            assert "match_strings" in case
            assert "status" in case
            assert case["status"] == "pending_review"

    def test_batch_file_named_with_timestamp(self, small_records, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch

        result = run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        batch_path = Path(result["batch_file"])
        assert batch_path.name.startswith("batch_")
        assert batch_path.suffix == ".json"

    def test_sqlite_kg_populated(self, small_records, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        kg = SqliteKnowledgeGraph(tmp_db_path)
        assert len(kg) > 0
        kg.close()

    def test_no_duplicate_on_rerun(self, small_records, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch
        from crystal.tools.kg.store import SqliteKnowledgeGraph

        run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        kg1 = SqliteKnowledgeGraph(tmp_db_path)
        count1 = len(kg1)
        kg1.close()

        run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        kg2 = SqliteKnowledgeGraph(tmp_db_path)
        count2 = len(kg2)
        kg2.close()

        assert count2 == count1

    def test_batch_includes_source_triplets(self, small_records, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch

        result = run_ingestion_batch(
            records=small_records,
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        with open(result["batch_file"], "r") as f:
            data = json.load(f)

        assert "triplets" in data["batch"]
        assert len(data["batch"]["triplets"]) > 0
        for t in data["batch"]["triplets"]:
            assert len(t) == 3

    def test_empty_records_produces_empty_batch(self, tmp_review_dir, tmp_db_path):
        from crystal.ingest.cron import run_ingestion_batch

        result = run_ingestion_batch(
            records=[],
            review_dir=tmp_review_dir,
            db_path=tmp_db_path,
        )
        assert result["records_ingested"] == 0
        assert result["questions_generated"] == 0
