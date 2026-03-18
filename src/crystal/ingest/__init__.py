"""
Crystal KG ingestion pipeline.

Entry points:
    ingest(path)           — auto-detect format, return IngestResult
    ingest_text(text)      — NER extraction from raw text
    build_kg(result)       — convert IngestResult to KnowledgeGraph

CLI:
    python -m crystal.ingest <document>
    python -m crystal.ingest data.csv
    python -m crystal.ingest facts.json
"""

from pathlib import Path

from crystal.ingest.schema import IngestResult, Triplet
from crystal.ingest.ner import ingest_text, ingest_file, extract_triplets
from crystal.ingest.loader import load_csv, load_json, load_file
from crystal.tools.kg.graph import KnowledgeGraph


def ingest(path: str | Path) -> IngestResult:
    """Auto-detect file type and ingest into an IngestResult.

    .csv / .json → hand-curated loader (exact triplets)
    .txt / other → NER extraction from text
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".json"):
        return load_file(path)
    else:
        return ingest_file(str(path))


def build_kg(result: IngestResult) -> KnowledgeGraph:
    """Convert an IngestResult into a ready-to-query KnowledgeGraph."""
    return KnowledgeGraph(
        triplets=result.as_tuples(),
        predicate_aliases=result.predicate_aliases or None,
        entity_aliases=result.entity_aliases or None,
    )
