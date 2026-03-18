"""CLI entry point: python -m crystal.ingest <document>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crystal.ingest import ingest, build_kg


def main():
    parser = argparse.ArgumentParser(
        description="Crystal KG ingestion — extract triplets from documents",
        usage="python -m crystal.ingest <document> [options]",
    )
    parser.add_argument(
        "document",
        help="Path to document (.txt for NER, .csv/.json for hand-curated)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Write extracted triplets to JSON file",
    )
    parser.add_argument(
        "--format", choices=["table", "json"], default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    path = Path(args.document)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    result = ingest(path)
    kg = build_kg(result)

    if args.format == "json":
        output = {
            "source": result.source,
            "triplets": [
                {"subject": t.subject, "predicate": t.predicate, "object": t.object}
                for t in result.triplets
            ],
            "entity_aliases": result.entity_aliases,
            "predicate_aliases": result.predicate_aliases,
            "stats": {
                "triplet_count": len(result.triplets),
                "entity_count": len(kg.entities),
                "subject_count": len(kg.subjects),
            },
        }
        text = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).write_text(text)
            print(f"Written to {args.output}")
        else:
            print(text)
    else:
        print(f"\nSource: {result.source}")
        print(f"Triplets: {len(result.triplets)}")
        print(f"Entities: {len(kg.entities)}")
        print(f"Subjects: {len(kg.subjects)}")
        if result.entity_aliases:
            print(f"Entity aliases: {len(result.entity_aliases)}")
        if result.predicate_aliases:
            print(f"Predicate aliases: {len(result.predicate_aliases)}")
        print()
        for t in result.triplets:
            print(f"  {t.subject:30s} | {t.predicate:20s} | {t.object}")
        print()


if __name__ == "__main__":
    main()
