"""
COLD Cases adapter — HuggingFace parquet streaming for bulk legal data.

Maps structured COLD Cases fields to Crystal KG triplets:
  - (case_name, court, court_full_name)
  - (case_name, date_filed, date_string)
  - (case_name, cited_by_count, count)
  - (case_name, judges, judges_string)
  - (case_name, disposition, disposition_string)
  - (case_name, nature_of_suit, nature_string)

Auto-generates entity aliases from case_name_short, case_name_full, slug, citations.

Usage:
    from crystal.ingest.sources.cold_cases import ingest_cold_cases
    for batch in ingest_cold_cases(court_type="S", limit=50):
        kg = build_kg(batch)
"""

from __future__ import annotations

from typing import Iterator

from crystal.data.legal_ontology import (
    normalize_case_name,
    generate_case_aliases,
    deduplicate_aliases,
)
from crystal.ingest.schema import IngestResult, Triplet


def _record_to_triplets(record: dict) -> tuple[list[Triplet], dict[str, str]]:
    """Convert a single COLD Cases record to triplets + entity aliases."""
    case_name = record.get("case_name", "")
    if not case_name:
        return [], {}

    canonical = normalize_case_name(case_name)
    if not canonical:
        return [], {}

    triplets: list[Triplet] = []

    court = record.get("court_full_name") or record.get("court_short_name")
    if court:
        triplets.append(Triplet(subject=canonical, predicate="court", object=court.strip()))

    date_filed = record.get("date_filed")
    if date_filed is not None:
        triplets.append(Triplet(subject=canonical, predicate="date_filed", object=str(date_filed)))

    citation_count = record.get("citation_count")
    if citation_count is not None and citation_count > 0:
        triplets.append(Triplet(
            subject=canonical, predicate="cited_by_count", object=str(citation_count),
        ))

    judges = record.get("judges")
    if judges and judges.strip():
        triplets.append(Triplet(subject=canonical, predicate="judges", object=judges.strip()))

    disposition = record.get("disposition")
    if disposition and disposition.strip():
        triplets.append(Triplet(
            subject=canonical, predicate="disposition", object=disposition.strip(),
        ))

    nature = record.get("nature_of_suit")
    if nature and nature.strip():
        triplets.append(Triplet(
            subject=canonical, predicate="nature_of_suit", object=nature.strip(),
        ))

    aliases = generate_case_aliases(
        case_name=case_name,
        case_name_short=record.get("case_name_short", ""),
        case_name_full=record.get("case_name_full", ""),
        slug=record.get("slug", ""),
        citations=record.get("citations") or [],
    )

    return triplets, aliases


def ingest_cold_cases(
    *,
    court_type: str | None = None,
    jurisdiction: str | None = None,
    limit: int | None = None,
    batch_size: int = 100,
    records: list[dict] | None = None,
) -> Iterator[IngestResult]:
    """Stream COLD Cases records and yield IngestResult batches.

    Filters:
        court_type: COLD Cases court_type code ("S" for state supreme, "F" for
                    federal appellate, etc.). Use "S" + jurisdiction filter
                    for SCOTUS.
        jurisdiction: Substring match on court_jurisdiction field.
        limit: Max number of records to process.
        records: If provided, use these dicts instead of streaming from HF.
                 Useful for testing and validation.

    Yields IngestResult batches of `batch_size` records each.
    """
    if records is not None:
        source = iter(records)
    else:
        source = _stream_from_huggingface()

    batch_triplets: list[Triplet] = []
    batch_aliases: dict[str, str] = {}
    count = 0
    batch_count = 0

    for record in source:
        if court_type and record.get("court_type") != court_type:
            continue
        if jurisdiction and jurisdiction.lower() not in (record.get("court_jurisdiction") or "").lower():
            continue

        triplets, aliases = _record_to_triplets(record)
        if not triplets:
            continue

        batch_triplets.extend(triplets)
        deduplicate_aliases(batch_aliases, aliases)
        count += 1
        batch_count += 1

        if batch_count >= batch_size:
            yield IngestResult(
                triplets=batch_triplets,
                entity_aliases=batch_aliases,
                predicate_aliases={},
                source="cold-cases",
            )
            batch_triplets = []
            batch_aliases = {}
            batch_count = 0

        if limit and count >= limit:
            break

    if batch_triplets:
        yield IngestResult(
            triplets=batch_triplets,
            entity_aliases=batch_aliases,
            predicate_aliases={},
            source="cold-cases",
        )


def _stream_from_huggingface():
    """Lazy import and stream from HuggingFace to avoid hard dependency."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for COLD Cases streaming. "
            "Install it with: pip install 'datasets>=2.14.0'"
        )
    ds = load_dataset("harvard-lil/cold-cases", split="train", streaming=True)
    yield from ds


def scotus_filter(record: dict) -> bool:
    """Check if a COLD Cases record is a SCOTUS decision."""
    court = (record.get("court_short_name") or "").lower()
    court_type = record.get("court_type", "")
    if "supreme" in court and "united states" in court:
        return True
    if court_type == "S" and "us" in (record.get("court_jurisdiction") or "").lower():
        return False  # State supreme courts, not SCOTUS
    return False
