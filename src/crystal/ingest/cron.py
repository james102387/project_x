"""
Ingestion cron pipeline — single command for the full ingest cycle.

Usage:
    python -m crystal.ingest.cron --source cold-cases --batch-size 100
    python -m crystal.ingest.cron --source cold-cases --court-type F --limit 200

Pipeline:
    1. Stream records from COLD Cases (or injected records for testing)
    2. Build/update SQLite KG via SqliteKnowledgeGraph.bulk_insert()
    3. Generate questions via generate_all(kg)
    4. Export to review/batch_YYYY-MM-DD_HHMMSS.json with batch metadata
    5. Print summary
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES
from crystal.ingest.question_gen import QuestionCase, export_for_review, generate_all
from crystal.ingest.sources.cold_cases import ingest_cold_cases
from crystal.tools.kg.store import SqliteKnowledgeGraph

logger = logging.getLogger(__name__)

REVIEW_DIR = Path(__file__).parent.parent.parent.parent / "review"
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "legal.sqlite"


def run_ingestion_batch(
    *,
    records: list[dict] | None = None,
    source: str = "cold-cases",
    court_type: str | None = None,
    jurisdiction: str | None = None,
    limit: int | None = None,
    batch_size: int = 100,
    review_dir: Path | None = None,
    db_path: Path | str | None = None,
) -> dict:
    """Run the full ingestion pipeline and return a summary dict.

    If records is provided, uses those instead of streaming from HuggingFace.
    """
    review_dir = review_dir or REVIEW_DIR
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    review_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    batch_id = now.strftime("%Y%m%d_%H%M%S")

    db = SqliteKnowledgeGraph(db_path)
    all_triplet_tuples: list[tuple[str, str, str]] = []
    total_inserted = 0

    try:
        for ingest_batch in ingest_cold_cases(
            court_type=court_type,
            jurisdiction=jurisdiction,
            limit=limit,
            batch_size=batch_size,
            records=records,
        ):
            tuples = [t.as_tuple() for t in ingest_batch.triplets]
            all_triplet_tuples.extend(tuples)
            inserted = db.bulk_insert(
                tuples,
                entity_aliases=ingest_batch.entity_aliases,
                predicate_aliases=LEGAL_PREDICATE_ALIASES,
                source=source,
            )
            total_inserted += inserted

        if not all_triplet_tuples:
            logger.info("No records ingested.")
            db.close()
            return {
                "batch_id": batch_id,
                "records_ingested": 0,
                "questions_generated": 0,
                "batch_file": None,
            }

        cases = generate_all(db)
        question_dicts = [c.to_review_dict() for c in cases]

        batch_data = {
            "batch": {
                "id": batch_id,
                "source": source,
                "records_ingested": total_inserted,
                "timestamp": now.isoformat(),
                "db_path": str(db_path),
                "triplets": list(all_triplet_tuples),
            },
            "description": (
                "Auto-generated questions pending human review. "
                "Change status to 'accepted' or 'rejected'. "
                "Correct golden_answer and match_strings as needed."
            ),
            "total": len(question_dicts),
            "pending": sum(1 for c in question_dicts if c.get("status") == "pending_review"),
            "cases": question_dicts,
        }

        batch_file = review_dir / f"batch_{batch_id}.json"
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        logger.info(
            "Batch %s: %d triplets ingested, %d questions generated → %s",
            batch_id, total_inserted, len(cases), batch_file,
        )

    finally:
        db.close()

    accepted_count = _count_accepted(review_dir)
    if accepted_count > 0:
        logger.info(
            "Total accepted golden answers across all batches: %d",
            accepted_count,
        )
        if accepted_count >= 50:
            logger.info(
                "You have %d+ accepted cases — consider running the Ralph Wiggum loop: "
                "python -m benchmarks.ralph_wiggum --threshold 0.90",
                accepted_count,
            )

    return {
        "batch_id": batch_id,
        "records_ingested": total_inserted,
        "questions_generated": len(cases),
        "batch_file": str(batch_file),
    }


def _count_accepted(review_dir: Path) -> int:
    """Count total accepted cases across all batch files."""
    count = 0
    for path in review_dir.glob("batch_*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for case in data.get("cases", []):
                if case.get("status") == "accepted":
                    count += 1
        except (json.JSONDecodeError, OSError):
            continue
    return count


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Crystal ingestion cron pipeline",
    )
    parser.add_argument(
        "--source", default="cold-cases",
        help="Data source (default: cold-cases)",
    )
    parser.add_argument(
        "--court-type", default=None,
        help="COLD Cases court_type filter (e.g., 'F' for federal)",
    )
    parser.add_argument(
        "--jurisdiction", default=None,
        help="Substring filter on court_jurisdiction",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max records to process",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Records per ingestion batch (default: 100)",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Path to SQLite database (default: data/legal.sqlite)",
    )
    parser.add_argument(
        "--review-dir", default=None,
        help="Path to review output directory (default: review/)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = run_ingestion_batch(
        source=args.source,
        court_type=args.court_type,
        jurisdiction=args.jurisdiction,
        limit=args.limit,
        batch_size=args.batch_size,
        db_path=Path(args.db_path) if args.db_path else None,
        review_dir=Path(args.review_dir) if args.review_dir else None,
    )

    print(f"\n--- Ingestion Summary ---")
    print(f"Batch ID:           {result['batch_id']}")
    print(f"Records ingested:   {result['records_ingested']}")
    print(f"Questions generated: {result['questions_generated']}")
    if result.get("batch_file"):
        print(f"Batch file:         {result['batch_file']}")


if __name__ == "__main__":
    main()
