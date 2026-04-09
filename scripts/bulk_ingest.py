"""
Bulk structured-data ingestion — Phase 1a.

Streams SCOTUS cases from COLD Cases (HuggingFace), builds SQLite KG,
generates questions, and auto-accepts everything into the golden set.

No human review needed for structured API data.

Usage:
    PYTHONPATH=src:. python scripts/bulk_ingest.py --limit 500
    PYTHONPATH=src:. python scripts/bulk_ingest.py --limit 500 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES
from crystal.ingest.question_gen import generate_all
from crystal.ingest.sources.cold_cases import ingest_cold_cases
from crystal.tools.kg.store import SqliteKnowledgeGraph

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REVIEW_DIR = ROOT / "review"
DEFAULT_DB_PATH = ROOT / "data" / "legal.sqlite"


def _stream_scotus(limit: int) -> list[dict]:
    """Stream SCOTUS records from COLD Cases, filtering by court name."""
    from datasets import load_dataset

    ds = load_dataset("harvard-lil/cold-cases", split="train", streaming=True)
    records = []
    scanned = 0

    for record in ds:
        court = (record.get("court_full_name") or "").lower()
        if "supreme court of the united states" not in court:
            scanned += 1
            if scanned % 5000 == 0:
                logger.info("  scanned %d records, found %d SCOTUS so far...", scanned, len(records))
            continue

        records.append(record)
        scanned += 1

        if scanned % 5000 == 0:
            logger.info("  scanned %d records, found %d SCOTUS so far...", scanned, len(records))

        if len(records) >= limit:
            break

    return records


def run_bulk_ingest(
    *,
    limit: int = 500,
    db_path: Path = DEFAULT_DB_PATH,
    review_dir: Path = REVIEW_DIR,
    dry_run: bool = False,
) -> dict:
    """Run Phase 1a bulk ingestion with auto-accept."""
    review_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 1a: Streaming up to %d SCOTUS cases from COLD Cases...", limit)
    t0 = time.time()
    records = _stream_scotus(limit)
    stream_time = time.time() - t0
    logger.info("Streamed %d SCOTUS records in %.1fs", len(records), stream_time)

    if not records:
        logger.warning("No SCOTUS records found.")
        return {"cases_ingested": 0, "questions_generated": 0}

    logger.info("Building SQLite KG at %s...", db_path)
    t0 = time.time()
    db = SqliteKnowledgeGraph(db_path)
    total_inserted = 0

    try:
        for batch in ingest_cold_cases(records=records, batch_size=200):
            tuples = [t.as_tuple() for t in batch.triplets]
            inserted = db.bulk_insert(
                tuples,
                entity_aliases=batch.entity_aliases,
                predicate_aliases=LEGAL_PREDICATE_ALIASES,
                source="cold-cases-scotus",
            )
            total_inserted += inserted

        kg_time = time.time() - t0
        logger.info("Inserted %d triplets in %.1fs", total_inserted, kg_time)

        logger.info("Generating questions...")
        t0 = time.time()
        cases = generate_all(db, max_tier1_per_subject=6, negative_count=20)
        gen_time = time.time() - t0
        logger.info("Generated %d questions in %.1fs", len(cases), gen_time)

    finally:
        db.close()

    if dry_run:
        logger.info("[DRY RUN] Would write %d auto-accepted questions.", len(cases))
        tier_counts = {}
        for c in cases:
            tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
        for tier, count in sorted(tier_counts.items()):
            logger.info("  Tier %d: %d questions", tier, count)
        neg = sum(1 for c in cases if c.is_negative)
        logger.info("  Negatives: %d", neg)
        return {
            "cases_ingested": len(records),
            "triplets_inserted": total_inserted,
            "questions_generated": len(cases),
            "dry_run": True,
        }

    now = datetime.now(timezone.utc)
    question_dicts = []
    for c in cases:
        d = c.to_review_dict()
        d["status"] = "accepted"
        question_dicts.append(d)

    batch_data = {
        "batch": {
            "id": f"phase1a_{now.strftime('%Y%m%d_%H%M%S')}",
            "source": "cold-cases-scotus",
            "records_ingested": len(records),
            "timestamp": now.isoformat(),
            "db_path": str(db_path),
            "auto_accepted": True,
            "confidence_tier": 0,
        },
        "description": (
            "Phase 1a bulk SCOTUS ingestion — structured API data, auto-accepted. "
            "Golden answers are verbatim API field values."
        ),
        "total": len(question_dicts),
        "pending": 0,
        "cases": question_dicts,
    }

    batch_file = review_dir / f"batch_phase1a_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d auto-accepted questions to %s", len(question_dicts), batch_file)

    tier_counts = {}
    for c in cases:
        tier_counts[c.tier] = tier_counts.get(c.tier, 0) + 1
    neg = sum(1 for c in cases if c.is_negative)

    return {
        "cases_ingested": len(records),
        "triplets_inserted": total_inserted,
        "questions_generated": len(question_dicts),
        "tier_counts": tier_counts,
        "negatives": neg,
        "batch_file": str(batch_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1a: Bulk SCOTUS ingestion from COLD Cases")
    parser.add_argument("--limit", type=int, default=500, help="Max SCOTUS cases to ingest (default: 500)")
    parser.add_argument("--db-path", default=None, help="SQLite database path")
    parser.add_argument("--review-dir", default=None, help="Review output directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_bulk_ingest(
        limit=args.limit,
        db_path=Path(args.db_path) if args.db_path else DEFAULT_DB_PATH,
        review_dir=Path(args.review_dir) if args.review_dir else REVIEW_DIR,
        dry_run=args.dry_run,
    )

    print(f"\n{'='*50}")
    print(f"Phase 1a Bulk Ingestion Summary")
    print(f"{'='*50}")
    print(f"SCOTUS cases ingested:  {result['cases_ingested']}")
    print(f"Triplets inserted:      {result.get('triplets_inserted', '?')}")
    print(f"Questions generated:    {result['questions_generated']}")
    if result.get("tier_counts"):
        for tier, count in sorted(result["tier_counts"].items()):
            print(f"  Tier {tier}: {count}")
    if result.get("negatives"):
        print(f"  Negatives: {result['negatives']}")
    if result.get("batch_file"):
        print(f"Batch file:             {result['batch_file']}")
    if result.get("dry_run"):
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
