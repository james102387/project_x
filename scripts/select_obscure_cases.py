"""
B4: Select obscure long-tail cases from the SQLite KG for benchmark fairness.

Queries the KG for cases with low cited_by_count that aren't in the existing
SCOTUS_SAMPLE fixture. Ranks by predicate coverage (more populated predicates
= more useful for benchmarking).

Usage:
    PYTHONPATH=src:. python scripts/select_obscure_cases.py
    PYTHONPATH=src:. python scripts/select_obscure_cases.py --limit 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crystal.tools.kg.store import SqliteKnowledgeGraph

ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = ROOT / "data" / "legal.sqlite"

EXISTING_FAMOUS_CASES = {
    "miranda v. arizona",
    "brown v. board of education",
    "roe v. wade",
    "marbury v. madison",
    "gideon v. wainwright",
    "mapp v. ohio",
    "plessy v. ferguson",
    "dred scott v. sandford",
    "griswold v. connecticut",
    "terry v. ohio",
    "new york times co. v. sullivan",
    "tinker v. des moines independent community school district",
    "tinker v. des moines",
    "loving v. virginia",
    "engel v. vitale",
    "texas v. johnson",
    "obergefell v. hodges",
    "citizens united v. federal election commission",
    "shelby county v. holder",
    "district of columbia v. heller",
    "korematsu v. united states",
    "united states v. nixon",
    "gibbons v. ogden",
    "mcculloch v. maryland",
    "schenck v. united states",
    "katz v. united states",
    "regents of the university of california v. bakke",
    "youngstown sheet & tube co. v. sawyer",
    "lemon v. kurtzman",
    "planned parenthood v. casey",
    "bush v. gore",
    "chevron u.s.a., inc. v. natural resources defense council, inc.",
    "dobbs v. jackson women's health organization",
    "loper bright enterprises v. raimondo",
    "gonzales v. raich",
    "furman v. georgia",
    "brandenburg v. ohio",
    "heart of atlanta motel v. united states",
    "bethel school district v. fraser",
    "hazelwood school district v. kuhlmeier",
    "roper v. simmons",
    "west virginia state board of education v. barnette",
    "lawrence v. texas",
    "mcdonald v. city of chicago",
    "grutter v. bollinger",
    "crawford v. washington",
    "riley v. california",
    "carpenter v. united states",
}


def find_obscure_cases(
    db_path: Path = DEFAULT_DB_PATH,
    max_citation_count: int = 500,
    min_predicates: int = 3,
    limit: int = 30,
) -> list[dict]:
    """Find obscure cases suitable for benchmarking.

    Returns cases sorted by citation count (ascending) with predicate coverage info.
    """
    db = SqliteKnowledgeGraph(db_path)

    candidates = []
    for subject in sorted(db.subjects):
        if subject in EXISTING_FAMOUS_CASES:
            continue

        facts = db.lookup(subject=subject)
        predicates = {f["predicate"] for f in facts}

        cite_count = None
        for f in facts:
            if f["predicate"] == "cited_by_count":
                try:
                    cite_count = int(f["object"])
                except ValueError:
                    pass
                break

        if cite_count is not None and cite_count > max_citation_count:
            continue

        if len(predicates) < min_predicates:
            continue

        doc_answerable_preds = predicates & {
            "court", "date_filed", "judges", "opinion_author", "attorneys", "cites",
        }

        fact_map = {}
        for f in facts:
            if f["predicate"] not in fact_map:
                fact_map[f["predicate"]] = f["object"]

        candidates.append({
            "subject": subject,
            "cited_by_count": cite_count,
            "predicate_count": len(predicates),
            "predicates": sorted(predicates),
            "doc_answerable_count": len(doc_answerable_preds),
            "doc_answerable_predicates": sorted(doc_answerable_preds),
            "facts": fact_map,
        })

    db.close()

    candidates.sort(key=lambda c: (c["cited_by_count"] or 0, -c["doc_answerable_count"]))
    return candidates[:limit]


def main():
    parser = argparse.ArgumentParser(description="B4: Find obscure cases for benchmark")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-citations", type=int, default=500)
    parser.add_argument("--min-predicates", type=int, default=3)
    args = parser.parse_args()

    candidates = find_obscure_cases(
        db_path=Path(args.db_path) if args.db_path else DEFAULT_DB_PATH,
        max_citation_count=args.max_citations,
        min_predicates=args.min_predicates,
        limit=args.limit,
    )

    print(f"\n{'=' * 80}")
    print(f"  B4: Obscure Case Candidates ({len(candidates)} found)")
    print(f"{'=' * 80}\n")

    for i, c in enumerate(candidates):
        print(f"  {i+1:2d}. {c['subject'].title()}")
        print(f"      Citations: {c['cited_by_count'] or '?'}")
        print(f"      Predicates ({c['predicate_count']}): {', '.join(c['predicates'])}")
        print(f"      Doc-answerable ({c['doc_answerable_count']}): {', '.join(c['doc_answerable_predicates'])}")

        facts = c["facts"]
        if "date_filed" in facts:
            print(f"      Date: {facts['date_filed']}")
        if "opinion_author" in facts:
            print(f"      Author: {facts['opinion_author']}")
        if "judges" in facts:
            judges_preview = facts["judges"][:60]
            print(f"      Judges: {judges_preview}...")
        print()


if __name__ == "__main__":
    main()
