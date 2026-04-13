#!/usr/bin/env python3
"""Backfill source_sentence for scaffold facts in the KG.

Two strategies:
1. Synthesize metadata provenance strings for all scaffold facts from COLD Cases
   that have empty source_sentence.
2. Opportunistically match against cached opinion text in benchmarks/documents/
   for facts where the subject appears in a document we have.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "legal.sqlite"
DOCS_DIR = PROJECT_ROOT / "benchmarks" / "documents"

SCAFFOLD_SOURCE = "cold-cases-scotus"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _subject_to_slug(subject: str) -> str:
    return (
        subject.lower()
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace(" ", "-")
    )


def _extract_party_names(case_name: str) -> list[str]:
    """Extract party names from 'X v. Y' patterns."""
    for sep in [" v. ", " v ", " vs. ", " vs "]:
        if sep in case_name.lower():
            parts = re.split(r"\s+v\.?\s+", case_name, flags=re.IGNORECASE)
            return [p.strip() for p in parts if p.strip()]
    return [case_name]


def _find_sentence(text: str, subject: str, obj: str) -> str | None:
    """Find a sentence in text containing both subject (or party name) and object keywords."""
    norm_text = _normalize(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    parties = _extract_party_names(subject)
    obj_words = [w for w in obj.lower().split() if len(w) > 3][:3]

    for sent in sentences:
        norm_sent = _normalize(sent)
        subj_found = any(p.lower() in norm_sent for p in parties)
        obj_found = any(w in norm_sent for w in obj_words) if obj_words else False
        if subj_found and obj_found:
            return sent.strip()[:500]

    return None


def _load_document_text(slug: str) -> str | None:
    path = DOCS_DIR / f"{slug}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("text", "plain_text", "opinion_text", "content"):
                if key in data and data[key]:
                    return str(data[key])
            if "opinions" in data and isinstance(data["opinions"], list):
                parts = [
                    op["plain_text"]
                    for op in data["opinions"]
                    if isinstance(op, dict) and op.get("plain_text")
                ]
                if parts:
                    return "\n\n".join(parts)
    except Exception:
        pass
    return None


def backfill(db_path: Path = DB_PATH, dry_run: bool = False) -> dict:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, subject, predicate, object, source "
        "FROM triplets "
        "WHERE source = ? AND (source_sentence IS NULL OR source_sentence = '')",
        (SCAFFOLD_SOURCE,),
    ).fetchall()

    print(f"Found {len(rows)} scaffold facts with empty source_sentence")

    stats = {"total": len(rows), "metadata_synth": 0, "doc_match": 0, "skipped": 0}
    doc_cache: dict[str, str | None] = {}
    updates: list[tuple[str, int]] = []

    for row in rows:
        fid = row["id"]
        subj = row["subject"]
        pred = row["predicate"]
        obj = row["object"]

        slug = _subject_to_slug(subj)
        if slug not in doc_cache:
            doc_cache[slug] = _load_document_text(slug)

        doc_text = doc_cache[slug]
        matched_sentence = None
        if doc_text:
            matched_sentence = _find_sentence(doc_text, subj, obj)

        if matched_sentence:
            updates.append((matched_sentence, fid))
            stats["doc_match"] += 1
        else:
            synth = f"[COLD Cases metadata] {pred}: {obj[:100]}"
            updates.append((synth, fid))
            stats["metadata_synth"] += 1

    if not dry_run and updates:
        with conn:
            conn.executemany(
                "UPDATE triplets SET source_sentence = ? WHERE id = ?",
                updates,
            )
        print(f"Updated {len(updates)} rows")
    elif dry_run:
        print(f"DRY RUN — would update {len(updates)} rows")

    conn.close()

    print(f"  Document matches: {stats['doc_match']}")
    print(f"  Metadata synth:   {stats['metadata_synth']}")
    return stats


def main():
    dry_run = "--dry-run" in sys.argv
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        sys.exit(1)
    backfill(DB_PATH, dry_run=dry_run)


if __name__ == "__main__":
    main()
