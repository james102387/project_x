"""Legal KG — convenience wiring for legal data, analogous to remulak.py.

Provides both in-memory and SQLite-backed options:
  - build_legal_kg_memory(records) → in-memory KnowledgeGraph (small datasets)
  - build_legal_kg_sqlite(records, db_path) → SqliteKnowledgeGraph (large datasets)

Usage:
    from crystal.tools.kg.legal import build_legal_kg_memory
    kg = build_legal_kg_memory(scotus_records)
    kg.lookup(subject="miranda v. arizona", predicate="court")
"""

from __future__ import annotations

from pathlib import Path

from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES
from crystal.ingest import build_kg
from crystal.ingest.schema import IngestResult
from crystal.ingest.sources.cold_cases import ingest_cold_cases
from crystal.tools.kg.graph import KnowledgeGraph
from crystal.tools.kg.store import SqliteKnowledgeGraph


def build_legal_kg_memory(records: list[dict]) -> KnowledgeGraph:
    """Build an in-memory KG from COLD Cases records."""
    all_triplets = []
    all_aliases: dict[str, str] = {}
    for batch in ingest_cold_cases(records=records, batch_size=200):
        all_triplets.extend(batch.triplets)
        all_aliases.update(batch.entity_aliases)
    result = IngestResult(
        triplets=all_triplets,
        entity_aliases=all_aliases,
        predicate_aliases=LEGAL_PREDICATE_ALIASES,
    )
    return build_kg(result)


def build_legal_kg_sqlite(
    records: list[dict],
    db_path: str | Path = "legal.db",
) -> SqliteKnowledgeGraph:
    """Build a SQLite-backed KG from COLD Cases records."""
    db = SqliteKnowledgeGraph(db_path)
    for batch in ingest_cold_cases(records=records, batch_size=200):
        db.bulk_insert(
            [t.as_tuple() for t in batch.triplets],
            entity_aliases=batch.entity_aliases,
            predicate_aliases=LEGAL_PREDICATE_ALIASES,
            source="cold-cases",
        )
    return db
