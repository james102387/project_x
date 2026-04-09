"""
Phase 1b: Citation graph ingestion from CourtListener.

For each SCOTUS case in the SQLite KG, searches CourtListener for the
matching cluster, fetches forward citations from the combined opinion,
and adds (case_name, cites, cited_case) triplets to the KG.

Auto-accepts all generated Tier 2 (relational) questions.

Requires COURTLISTENER_API_TOKEN in .env.

Usage:
    PYTHONPATH=src:. python scripts/bulk_citations.py
    PYTHONPATH=src:. python scripts/bulk_citations.py --limit 100 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import httpx

from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES, normalize_case_name
from crystal.ingest.question_gen import generate_tier2, generate_negatives
from crystal.ingest.schema import Triplet
from crystal.tools.kg.store import SqliteKnowledgeGraph

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
REVIEW_DIR = ROOT / "review"
DEFAULT_DB_PATH = ROOT / "data" / "legal.sqlite"

_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
_RATE_LIMIT_SECS = 0.6


class CitationIngester:
    def __init__(self, api_token: str):
        self.client = httpx.Client(
            headers={
                "Authorization": f"Token {api_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        self._last_request = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < _RATE_LIMIT_SECS:
            time.sleep(_RATE_LIMIT_SECS - elapsed)
        self._last_request = time.time()

    def _get(self, url: str, params: dict | None = None) -> dict | None:
        self._throttle()
        try:
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug("Request failed: %s", e)
            return None

    def search_scotus_cluster(self, case_name: str) -> int | None:
        """Search for a SCOTUS case and return its cluster ID."""
        data = self._get(f"{_BASE_URL}/search/", params={
            "q": case_name,
            "type": "o",
            "court": "scotus",
            "page_size": 5,
        })
        if not data or not data.get("results"):
            return None

        name_lower = case_name.lower().replace(" v. ", " v ").replace(".", "")
        for r in data["results"]:
            found = (r.get("caseName") or "").lower().replace(" v. ", " v ").replace(".", "")
            if name_lower in found or found in name_lower:
                return r.get("cluster_id")

        return data["results"][0].get("cluster_id")

    def get_combined_opinion_id(self, cluster_id: int) -> int | None:
        """Get the combined opinion ID for a cluster."""
        data = self._get(f"{_BASE_URL}/clusters/{cluster_id}/")
        if not data:
            return None

        for url in data.get("sub_opinions", []):
            opinion_id = url.rstrip("/").split("/")[-1]
            try:
                return int(opinion_id)
            except ValueError:
                continue
        return None

    @staticmethod
    def _extract_id(url_or_id) -> int | None:
        """Extract numeric ID from a URL string or return int directly."""
        if isinstance(url_or_id, int):
            return url_or_id
        if isinstance(url_or_id, str):
            parts = url_or_id.rstrip("/").split("/")
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
        return None

    def fetch_citations(self, opinion_id: int) -> list[int]:
        """Fetch forward citations (cases this opinion cites). Returns cited opinion IDs."""
        all_ids = []
        url = f"{_BASE_URL}/opinions-cited/"
        params = {"citing_opinion": opinion_id}

        while url:
            data = self._get(url, params=params)
            if not data:
                break
            for r in data.get("results", []):
                cited_id = self._extract_id(r.get("cited_opinion"))
                if cited_id:
                    all_ids.append(cited_id)
            url = data.get("next")
            params = None
            if not url:
                break

        return all_ids

    def resolve_opinion_name(self, opinion_id: int) -> str | None:
        """Resolve an opinion ID to a case name (opinion → cluster → case_name)."""
        data = self._get(f"{_BASE_URL}/opinions/{opinion_id}/")
        if not data:
            return None
        cluster_url = data.get("cluster")
        if not cluster_url:
            return None

        cluster_id = self._extract_id(cluster_url)
        if not cluster_id:
            return None

        cluster_data = self._get(f"{_BASE_URL}/clusters/{cluster_id}/")
        if cluster_data:
            return cluster_data.get("case_name")
        return None

    def close(self):
        self.client.close()


def run_citation_ingest(
    *,
    limit: int | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    review_dir: Path = REVIEW_DIR,
    dry_run: bool = False,
    max_citations_per_case: int = 20,
) -> dict:
    token = os.environ.get("COURTLISTENER_API_TOKEN", "")
    if not token:
        logger.error("COURTLISTENER_API_TOKEN not set")
        return {"error": "No API token"}

    review_dir.mkdir(parents=True, exist_ok=True)
    db = SqliteKnowledgeGraph(db_path)
    ingester = CitationIngester(token)

    subjects = sorted(db.subjects)
    if limit:
        subjects = subjects[:limit]

    logger.info("Phase 1b: Fetching citations for %d cases...", len(subjects))

    total_triplets = 0
    cases_with_citations = 0
    cases_searched = 0
    cases_not_found = 0
    all_new_triplets: list[tuple[str, str, str]] = []

    try:
        for i, subject in enumerate(subjects):
            if (i + 1) % 25 == 0:
                logger.info(
                    "  Progress: %d/%d cases (%d citations so far)",
                    i + 1, len(subjects), total_triplets,
                )

            cluster_id = ingester.search_scotus_cluster(subject.title())
            cases_searched += 1

            if not cluster_id:
                cases_not_found += 1
                continue

            opinion_id = ingester.get_combined_opinion_id(cluster_id)
            if not opinion_id:
                continue

            cited_ids = ingester.fetch_citations(opinion_id)
            if not cited_ids:
                continue

            cases_with_citations += 1
            case_triplets = 0

            for cited_id in cited_ids[:max_citations_per_case]:
                cited_name = ingester.resolve_opinion_name(cited_id)
                if cited_name:
                    canonical = normalize_case_name(cited_name)
                    if canonical:
                        triplet = (subject, "cites", canonical.lower())
                        all_new_triplets.append(triplet)
                        case_triplets += 1
                        total_triplets += 1

            if case_triplets > 0:
                logger.debug("  %s: %d citations", subject.title(), case_triplets)

    except KeyboardInterrupt:
        logger.info("Interrupted — saving progress (%d triplets so far)", total_triplets)
    finally:
        ingester.close()

    logger.info(
        "Citation search complete: %d cases searched, %d not found, %d with citations, %d total citation triplets",
        cases_searched, cases_not_found, cases_with_citations, total_triplets,
    )

    if dry_run:
        db.close()
        return {
            "cases_searched": cases_searched,
            "cases_not_found": cases_not_found,
            "cases_with_citations": cases_with_citations,
            "citation_triplets": total_triplets,
            "dry_run": True,
        }

    if all_new_triplets:
        inserted = db.bulk_insert(
            all_new_triplets,
            predicate_aliases=LEGAL_PREDICATE_ALIASES,
            source="courtlistener-citations",
        )
        logger.info("Inserted %d citation triplets into SQLite KG", inserted)

    cases = generate_tier2(db, target_predicates={"cites"})
    logger.info("Generated %d Tier 2 relational questions", len(cases))

    db.close()

    if cases:
        now = datetime.now(timezone.utc)
        question_dicts = []
        for c in cases:
            d = c.to_review_dict()
            d["status"] = "accepted"
            question_dicts.append(d)

        batch_data = {
            "batch": {
                "id": f"phase1b_{now.strftime('%Y%m%d_%H%M%S')}",
                "source": "courtlistener-citations",
                "records_ingested": total_triplets,
                "timestamp": now.isoformat(),
                "auto_accepted": True,
                "confidence_tier": 0,
            },
            "description": "Phase 1b citation graph — structured API data, auto-accepted.",
            "total": len(question_dicts),
            "pending": 0,
            "cases": question_dicts,
        }

        batch_file = review_dir / f"batch_phase1b_{now.strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)

        logger.info("Wrote %d auto-accepted Tier 2 questions to %s", len(question_dicts), batch_file)
    else:
        batch_file = None

    return {
        "cases_searched": cases_searched,
        "cases_not_found": cases_not_found,
        "cases_with_citations": cases_with_citations,
        "citation_triplets": total_triplets,
        "questions_generated": len(cases),
        "batch_file": str(batch_file) if batch_file else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1b: Citation graph from CourtListener")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to search (default: all)")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--review-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-citations", type=int, default=20, help="Max citations per case (default: 20)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_citation_ingest(
        limit=args.limit,
        db_path=Path(args.db_path) if args.db_path else DEFAULT_DB_PATH,
        review_dir=Path(args.review_dir) if args.review_dir else REVIEW_DIR,
        dry_run=args.dry_run,
        max_citations_per_case=args.max_citations,
    )

    print(f"\n{'='*50}")
    print(f"Phase 1b Citation Graph Summary")
    print(f"{'='*50}")
    print(f"Cases searched:         {result['cases_searched']}")
    print(f"Cases not found:        {result['cases_not_found']}")
    print(f"Cases with citations:   {result['cases_with_citations']}")
    print(f"Citation triplets:      {result['citation_triplets']}")
    if result.get("questions_generated"):
        print(f"Tier 2 questions:       {result['questions_generated']}")
    if result.get("batch_file"):
        print(f"Batch file:             {result['batch_file']}")
    if result.get("dry_run"):
        print("(dry run — no data written)")


if __name__ == "__main__":
    main()
