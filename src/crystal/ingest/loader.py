"""
Hand-curated triplet loaders — CSV and JSON.

Supports bootstrapping new datasets without any NER extraction.

CSV format (header optional, auto-detected):
    subject,predicate,object
    Remulak,capital,Zelphos

JSON format:
    {
        "triplets": [
            {"subject": "Remulak", "predicate": "capital", "object": "Zelphos"}
        ],
        "entity_aliases": {"korth": "grand vizier korth"},
        "predicate_aliases": {"capital city": "capital"}
    }

    Or a flat list of [subject, predicate, object] arrays:
    [["Remulak", "capital", "Zelphos"]]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from crystal.ingest.schema import IngestResult, Triplet

CSV_HEADER_NAMES = {"subject", "predicate", "object"}


def _looks_like_header(row: list[str]) -> bool:
    """Check if a CSV row looks like a header."""
    return set(c.strip().lower() for c in row) >= CSV_HEADER_NAMES


def load_csv(path: str | Path) -> IngestResult:
    """Load triplets from a CSV file.

    Expects 3 columns: subject, predicate, object.
    If the first row matches header names, it's skipped.
    """
    path = Path(path)
    triplets: list[Triplet] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return IngestResult(source=str(path))

    start = 1 if _looks_like_header(rows[0]) else 0
    for row in rows[start:]:
        if len(row) < 3:
            continue
        subj, pred, obj = row[0].strip(), row[1].strip(), row[2].strip()
        if subj and pred and obj:
            triplets.append(Triplet(subject=subj, predicate=pred, object=obj))

    return IngestResult(triplets=triplets, source=str(path))


def load_json(path: str | Path) -> IngestResult:
    """Load triplets (and optional aliases) from a JSON file.

    Supports two formats:
    1. Object with "triplets" key (+ optional "entity_aliases", "predicate_aliases")
    2. Flat array of [subject, predicate, object] arrays
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        triplets = []
        for item in data:
            if isinstance(item, list) and len(item) >= 3:
                triplets.append(Triplet(
                    subject=str(item[0]).strip(),
                    predicate=str(item[1]).strip(),
                    object=str(item[2]).strip(),
                ))
            elif isinstance(item, dict):
                triplets.append(Triplet(
                    subject=str(item.get("subject", "")).strip(),
                    predicate=str(item.get("predicate", "")).strip(),
                    object=str(item.get("object", "")).strip(),
                ))
        return IngestResult(
            triplets=[t for t in triplets if t.subject and t.predicate and t.object],
            source=str(path),
        )

    # Object format
    triplets = []
    for item in data.get("triplets", []):
        if isinstance(item, list) and len(item) >= 3:
            triplets.append(Triplet(
                subject=str(item[0]).strip(),
                predicate=str(item[1]).strip(),
                object=str(item[2]).strip(),
            ))
        elif isinstance(item, dict):
            triplets.append(Triplet(
                subject=str(item.get("subject", "")).strip(),
                predicate=str(item.get("predicate", "")).strip(),
                object=str(item.get("object", "")).strip(),
            ))

    return IngestResult(
        triplets=[t for t in triplets if t.subject and t.predicate and t.object],
        entity_aliases=data.get("entity_aliases", {}),
        predicate_aliases=data.get("predicate_aliases", {}),
        source=str(path),
    )


def load_file(path: str | Path) -> IngestResult:
    """Auto-detect file format and load triplets."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    elif suffix == ".json":
        return load_json(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix} (expected .csv or .json)")
