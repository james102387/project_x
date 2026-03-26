"""CLI entry point: python -m crystal.ingest <document>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crystal.ingest import ingest, ingest_with_llm, build_kg
from crystal.ingest.loader import load_review


def _print_table(result, kg):
    """Print a human-readable table of ingestion results."""
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


def _format_result_json(result, kg):
    """Format IngestResult as a JSON-serializable dict."""
    return {
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
    parser.add_argument(
        "--llm-assist",
        action="store_true",
        help="Use LLM for sentences where NER found entities but couldn't resolve predicates (Phase 2)",
    )
    parser.add_argument(
        "--review-output",
        help="Write LLM-extracted triplets to a review JSON file (use with --llm-assist)",
    )
    parser.add_argument(
        "--load-review",
        help="Import accepted triplets from a previously reviewed JSON file",
    )
    args = parser.parse_args()

    path = Path(args.document)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if args.load_review:
        review_path = Path(args.load_review)
        if not review_path.exists():
            print(f"Error: review file not found: {review_path}", file=sys.stderr)
            sys.exit(1)
        ner_result = ingest(path)
        reviewed = load_review(review_path)
        result = ner_result.merge(reviewed)
        kg = build_kg(result)
        print(f"Merged {len(reviewed.triplets)} accepted triplets from review")
    elif args.llm_assist:
        ner_result, llm_result = ingest_with_llm(path)
        result = ner_result
        kg = build_kg(result)

        review_path = args.review_output or str(path.with_suffix(".review.json"))
        review_data = llm_result.to_review_dict()
        review_data["ner_triplet_count"] = len(ner_result.triplets)
        Path(review_path).write_text(json.dumps(review_data, indent=2))
        print(f"\nNER extracted: {len(ner_result.triplets)} triplets")
        print(f"LLM extracted: {len(llm_result.reviewable)} triplets (pending review)")
        if llm_result.skipped_sentences:
            print(f"Skipped: {len(llm_result.skipped_sentences)} sentences (LLM error)")
        print(f"Review file: {review_path}")
        print("Edit the review file → set status to 'accepted' or 'rejected'")
        print(f"Then run: python -m crystal.ingest {path} --load-review {review_path}")
    else:
        result = ingest(path)
        kg = build_kg(result)

    if args.format == "json":
        output = _format_result_json(result, kg)
        text = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).write_text(text)
            print(f"Written to {args.output}")
        else:
            print(text)
    else:
        _print_table(result, kg)


if __name__ == "__main__":
    main()
