"""
B1: Download real opinion text from CourtListener for benchmark cases.

For each case in the benchmark set, searches CourtListener for the matching
cluster, fetches the lead opinion text (html_with_citations), strips HTML
to plain text, and caches to benchmarks/documents/{slug}.json.

Requires COURTLISTENER_API_TOKEN in .env.

Usage:
    PYTHONPATH=src:. python scripts/download_opinions.py
    PYTHONPATH=src:. python scripts/download_opinions.py --limit 10 --dry-run
    PYTHONPATH=src:. python scripts/download_opinions.py --db-path data/legal.sqlite
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

import httpx

from crystal.data.legal_ontology import normalize_case_name
from crystal.tools.kg.store import SqliteKnowledgeGraph

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DOCUMENTS_DIR = ROOT / "benchmarks" / "documents"
DEFAULT_DB_PATH = ROOT / "data" / "legal.sqlite"

_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
_RATE_LIMIT_SECS = 0.6


def _strip_html(raw: str) -> str:
    """Strip HTML tags and decode entities to produce readable plain text."""
    text = re.sub(r"<br\s*/?>", "\n", raw)
    text = re.sub(r"<p[^>]*>", "\n\n", text)
    text = re.sub(r"</p>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class OpinionDownloader:
    """Fetches opinion text from CourtListener API."""

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
        """Get the lead opinion ID for a cluster."""
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

    def fetch_opinion_text(self, opinion_id: int) -> str | None:
        """Fetch opinion text, preferring html_with_citations."""
        data = self._get(
            f"{_BASE_URL}/opinions/{opinion_id}/",
            params={"fields": "html_with_citations,plain_text,type"},
        )
        if not data:
            return None

        raw_html = data.get("html_with_citations") or ""
        if raw_html:
            return _strip_html(raw_html)

        plain = data.get("plain_text") or ""
        if plain:
            return plain.strip()

        return None

    def close(self):
        self.client.close()


def _case_slug(case_name: str) -> str:
    """Convert case name to a filesystem-safe slug."""
    slug = case_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:80]


def download_opinions(
    *,
    case_names: list[str] | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    output_dir: Path = DOCUMENTS_DIR,
    limit: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict:
    """Download opinion text for benchmark cases.

    If case_names is None, loads all subjects from the SQLite KG.
    """
    token = os.environ.get("COURTLISTENER_API_TOKEN", "")
    if not token:
        logger.error("COURTLISTENER_API_TOKEN not set")
        return {"error": "No API token"}

    if case_names is None:
        db = SqliteKnowledgeGraph(db_path)
        case_names = [s.title() for s in sorted(db.subjects)]
        db.close()

    if limit:
        case_names = case_names[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    downloader = OpinionDownloader(token)

    stats = {
        "total": len(case_names),
        "downloaded": 0,
        "skipped_existing": 0,
        "not_found": 0,
        "no_text": 0,
        "errors": [],
    }

    try:
        for i, case_name in enumerate(case_names):
            canonical = normalize_case_name(case_name)
            slug = _case_slug(canonical)

            if skip_existing and (output_dir / f"{slug}.json").exists():
                stats["skipped_existing"] += 1
                continue

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    "  [%d/%d] Downloading: %s",
                    i + 1, len(case_names), canonical,
                )

            cluster_id = downloader.search_scotus_cluster(case_name)
            if not cluster_id:
                stats["not_found"] += 1
                stats["errors"].append(f"Cluster not found: {canonical}")
                continue

            opinion_id = downloader.get_combined_opinion_id(cluster_id)
            if not opinion_id:
                stats["not_found"] += 1
                stats["errors"].append(f"Opinion not found for cluster {cluster_id}: {canonical}")
                continue

            if dry_run:
                logger.info("    [dry-run] Would download opinion %d for %s", opinion_id, canonical)
                stats["downloaded"] += 1
                continue

            text = downloader.fetch_opinion_text(opinion_id)
            if not text or len(text) < 100:
                stats["no_text"] += 1
                stats["errors"].append(f"Empty/short text for opinion {opinion_id}: {canonical}")
                continue

            doc_data = {
                "case_name": canonical,
                "slug": slug,
                "cluster_id": cluster_id,
                "opinion_id": opinion_id,
                "text": text,
                "char_count": len(text),
                "token_estimate": len(text) // 4,
            }

            out_path = output_dir / f"{slug}.json"
            out_path.write_text(json.dumps(doc_data, indent=2, ensure_ascii=False), encoding="utf-8")
            stats["downloaded"] += 1

            logger.info(
                "    ✓ %s — %d chars (~%dk tokens)",
                canonical, len(text), len(text) // 4000,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted — progress saved (%d downloaded)", stats["downloaded"])
    finally:
        downloader.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="B1: Download opinion text from CourtListener")
    parser.add_argument("--db-path", default=None, help="Path to SQLite KG (default: data/legal.sqlite)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: benchmarks/documents/)")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to download")
    parser.add_argument("--dry-run", action="store_true", help="Search but don't download text")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stats = download_opinions(
        db_path=Path(args.db_path) if args.db_path else DEFAULT_DB_PATH,
        output_dir=Path(args.output_dir) if args.output_dir else DOCUMENTS_DIR,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_existing=not args.force,
    )

    print(f"\n{'=' * 50}")
    print("B1: Opinion Text Download Summary")
    print(f"{'=' * 50}")
    print(f"Total cases:       {stats.get('total', 0)}")
    print(f"Downloaded:        {stats.get('downloaded', 0)}")
    print(f"Skipped (cached):  {stats.get('skipped_existing', 0)}")
    print(f"Not found:         {stats.get('not_found', 0)}")
    print(f"No/short text:     {stats.get('no_text', 0)}")

    if stats.get("errors"):
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats["errors"][:20]:
            print(f"  - {err}")

    if stats.get("dry_run"):
        print("\n(dry run — no files written)")


if __name__ == "__main__":
    main()
