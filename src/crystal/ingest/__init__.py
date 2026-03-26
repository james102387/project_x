"""
Crystal KG ingestion pipeline.

Entry points:
    ingest(path)                — auto-detect format, return IngestResult
    ingest_text(text)           — NER extraction from raw text
    ingest_with_llm(path)       — two-pass: NER first, LLM for gaps (D2 Phase 2)
    build_kg(result)            — convert IngestResult to KnowledgeGraph

CLI:
    python -m crystal.ingest <document>
    python -m crystal.ingest <document> --llm-assist
    python -m crystal.ingest data.csv
    python -m crystal.ingest facts.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from crystal.ingest.schema import (
    IngestResult,
    LLMExtractionResult,
    ReviewableTriplet,
    Triplet,
)
from crystal.ingest.ner import (
    extract_triplets,
    find_unresolved_sentences,
    ingest_file,
    ingest_text,
)
from crystal.ingest.loader import load_csv, load_json, load_file, load_review
from crystal.ingest.llm_extract import extract_triplets_llm
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


def ingest_with_llm(
    path: str | Path,
    call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
) -> tuple[IngestResult, LLMExtractionResult]:
    """Two-pass ingestion: NER extraction first, then LLM for unresolved sentences.

    Pass 1: Standard NER dep-tree extraction (same as ingest()).
    Pass 2: Sentences where NER found entities but no predicates are sent
             to the LLM for relationship extraction.

    Returns (ner_result, llm_result). The llm_result contains reviewable
    triplets that need human approval before being added to the KG.
    Only works on text files — CSV/JSON bypass NER entirely.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    ner_result = ingest_text(text, source=str(path))

    unresolved = find_unresolved_sentences(text)
    llm_result = extract_triplets_llm(
        unresolved,
        call_llm_fn=call_llm_fn,
    )
    llm_result.source = str(path)

    return ner_result, llm_result


def build_kg(result: IngestResult) -> KnowledgeGraph:
    """Convert an IngestResult into a ready-to-query KnowledgeGraph."""
    return KnowledgeGraph(
        triplets=result.as_tuples(),
        predicate_aliases=result.predicate_aliases or None,
        entity_aliases=result.entity_aliases or None,
    )
