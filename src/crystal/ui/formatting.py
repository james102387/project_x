"""Shared formatting helpers for the Crystal UI.

Pure functions that turn KGs, triplets, and routing metadata into Markdown
or DataFrames. No Gradio imports — safe to call from tests.
"""

from __future__ import annotations

import pandas as pd

from crystal.tools.kg.graph import KnowledgeGraph
from crystal.tools.kg.store import SqliteKnowledgeGraph

from crystal.ui.state import KG_MODES, _DEFAULT_KG_MODE, _default_kg


# ── Origin / provenance labels ───────────────────────────────────────────

_ORIGIN_DISPLAY = {
    "api_metadata": "API metadata",
    "opinion_doc": "Opinion document",
    "unknown": "Unknown",
}

_ORIGIN_FILTER_CHOICES = ["All origins", "api_metadata", "opinion_doc", "unknown"]

_KG_FACTS_HEADERS = ["Subject", "Predicate", "Object", "Origin", "Source Document"]


# ── Routing labels ───────────────────────────────────────────────────────

_ROUTE_LABELS = {
    "kg_answerable": "KG Grounded (direct)",
    "kg_augmented": "KG + LLM Reasoning",
    "pure_math": "Calculator (direct)",
    "math_answerable": "Calculator (direct)",
    "math_augmented": "Calculator + LLM Reasoning",
    "no_math": "LLM Fallback (ungrounded)",
}

_CONFIDENCE_BADGES = {
    "kg_answerable": "HIGH — answer sourced directly from verified knowledge graph",
    "kg_augmented": "MEDIUM — KG facts injected, LLM reasoning applied",
    "pure_math": "HIGH — computed mathematically",
    "math_answerable": "HIGH — computed mathematically",
    "math_augmented": "MEDIUM — math computed, LLM reasoning applied",
    "no_math": "LOW — no grounding, pure LLM generation",
}


# ── KG info / stats ──────────────────────────────────────────────────────

def _kg_info(kg: KnowledgeGraph, source: str) -> dict:
    info = {
        "source": source,
        "triplets": len(kg),
        "entities": len(kg.entities),
        "subjects": len(kg.subjects),
        "provenance": {},
    }
    if isinstance(kg, SqliteKnowledgeGraph):
        info["provenance"] = kg.provenance_counts()
    return info


def _default_kg_info() -> dict:
    return _kg_info(_default_kg, _DEFAULT_KG_MODE)


def _format_kg_stats(info: dict) -> str:
    lines = [
        f"**{info['source']}**\n",
        f"- Triplets: {info['triplets']}",
        f"- Entities: {info['entities']}",
        f"- Subjects: {info['subjects']}",
    ]
    prov = info.get("provenance", {})
    if prov:
        parts = []
        for key in ("api_metadata", "opinion_doc", "unknown"):
            count = prov.get(key, 0)
            if count > 0:
                parts.append(f"**{count}** {_ORIGIN_DISPLAY.get(key, key)}")
        if parts:
            lines.append(f"- Provenance: {', '.join(parts)}")
    return "\n".join(lines)


def _format_kg_facts(kg: KnowledgeGraph, max_facts: int = 200) -> str:
    """Format KG triplets as a readable Markdown table."""
    if not kg.triplets:
        return "No facts loaded."
    lines = ["| Subject | Predicate | Object |", "| --- | --- | --- |"]
    for s, p, o in kg.triplets[:max_facts]:
        lines.append(f"| {s} | {p} | {o} |")
    if len(kg.triplets) > max_facts:
        lines.append(f"\n*...and {len(kg.triplets) - max_facts} more*")
    return "\n".join(lines)


def _kg_facts_df(
    kg: KnowledgeGraph,
    max_facts: int = 500,
    origin_filter: str = "All origins",
    document_filter: str = "",
) -> pd.DataFrame:
    """Return KG triplets as a scrollable DataFrame with provenance columns."""
    if not kg.triplets:
        return pd.DataFrame(columns=_KG_FACTS_HEADERS)

    if isinstance(kg, SqliteKnowledgeGraph):
        data = kg.triplets_with_provenance
        if origin_filter and origin_filter != "All origins":
            data = [r for r in data if r[4] == origin_filter]
        if document_filter and document_filter.strip():
            filt = document_filter.strip().lower()
            data = [r for r in data if filt in r[5].lower()]
        rows = [
            [s, p, o, _ORIGIN_DISPLAY.get(origin, origin), src_doc]
            for s, p, o, _src, origin, src_doc in data[:max_facts]
        ]
    else:
        rows = [[s, p, o, "", ""] for s, p, o in kg.triplets[:max_facts]]
    return pd.DataFrame(rows, columns=_KG_FACTS_HEADERS)


def _get_source_document_choices(kg: KnowledgeGraph) -> list[str]:
    """Distinct source_document values for the dropdown."""
    choices = ["All documents"]
    if isinstance(kg, SqliteKnowledgeGraph):
        choices.extend(kg.source_documents())
    return choices


def _format_kg_banner(kg: KnowledgeGraph, label: str) -> str:
    return (
        f"**Active KG:** {label} · "
        f"{len(kg)} triplets · {len(kg.entities)} entities"
    )


def _format_ingest_target(kg: KnowledgeGraph, label: str) -> str:
    return (
        f"**Ingesting to: {label}** — "
        f"{len(kg)} triplets, {len(kg.entities)} entities"
    )


def _label_for_kg(kg) -> str:
    """Find the display label for a KG already in KG_MODES."""
    for label, k in KG_MODES.items():
        if k is kg:
            return label
    return "current"
