"""
Loader utilities for cached opinion documents.

Opinion text is downloaded by scripts/download_opinions.py and cached
as JSON files in benchmarks/documents/{slug}.json.

This module provides read-only access to the cached documents.
"""

from __future__ import annotations

import json
from pathlib import Path

from crystal.data.legal_ontology import normalize_case_name

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def load_opinion(slug: str) -> str | None:
    """Load cached opinion text by slug. Returns None if not cached."""
    path = DOCUMENTS_DIR / f"{slug}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("text")


def load_all_opinions() -> dict[str, str]:
    """Load all cached opinions, keyed by normalized case name (lowercased).

    Returns a dict mapping lowercase canonical case names to opinion text.
    """
    opinions: dict[str, str] = {}
    if not DOCUMENTS_DIR.exists():
        return opinions

    for path in sorted(DOCUMENTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        case_name = data.get("case_name", "")
        text = data.get("text", "")
        if case_name and text:
            key = normalize_case_name(case_name).lower()
            opinions[key] = text
    return opinions


def opinion_token_estimate(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4


def list_cached_opinions() -> list[dict]:
    """List metadata for all cached opinions (without full text)."""
    results = []
    if not DOCUMENTS_DIR.exists():
        return results

    for path in sorted(DOCUMENTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        results.append({
            "slug": data.get("slug", path.stem),
            "case_name": data.get("case_name", ""),
            "char_count": data.get("char_count", 0),
            "token_estimate": data.get("token_estimate", 0),
            "opinion_id": data.get("opinion_id"),
            "cluster_id": data.get("cluster_id"),
        })
    return results
