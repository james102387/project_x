"""
SQLite-backed Knowledge Graph — same lookup()/traverse() API as the in-memory KG.

Designed for datasets that exceed comfortable in-memory size (100K+ triplets).
The in-memory KnowledgeGraph stays for Remulak, tests, and small datasets.

Usage:
    from crystal.tools.kg.store import SqliteKnowledgeGraph
    kg = SqliteKnowledgeGraph("legal.db")
    kg.bulk_insert(triplets, entity_aliases)
    results = kg.lookup(subject="Miranda v. Arizona", predicate="court")
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypeAlias

from crystal.tools.kg.fuzzy import fuzzy_match

Triplet: TypeAlias = tuple[str, str, str]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS triplets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(subject, predicate, object)
);
CREATE INDEX IF NOT EXISTS idx_forward ON triplets(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_reverse ON triplets(predicate, object);
CREATE INDEX IF NOT EXISTS idx_subject ON triplets(subject);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alias_canonical ON entity_aliases(canonical);

CREATE TABLE IF NOT EXISTS predicate_aliases (
    alias TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT PRIMARY KEY,
    last_sync TEXT,
    last_id TEXT,
    item_count INTEGER
);
"""


class SqliteKnowledgeGraph:
    """SQLite-backed KG with the same query interface as the in-memory KG."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        fuzzy_threshold: float = 80.0,
    ) -> None:
        self.db_path = str(db_path)
        self.fuzzy_threshold = fuzzy_threshold
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Bulk loading ─────────────────────────────────────────────────────

    def bulk_insert(
        self,
        triplets: list[Triplet],
        entity_aliases: dict[str, str] | None = None,
        predicate_aliases: dict[str, str] | None = None,
        source: str = "",
    ) -> int:
        """Insert triplets and aliases in a single transaction. Returns count inserted."""
        inserted = 0
        with self._conn:
            for subj, pred, obj in triplets:
                try:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO triplets (subject, predicate, object, source) "
                        "VALUES (?, ?, ?, ?)",
                        (subj.lower(), pred.lower(), obj, source),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass

            for alias, canonical in (entity_aliases or {}).items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO entity_aliases (alias, canonical) VALUES (?, ?)",
                    (alias.lower(), canonical.lower()),
                )

            for alias, canonical in (predicate_aliases or {}).items():
                self._conn.execute(
                    "INSERT OR REPLACE INTO predicate_aliases (alias, canonical) VALUES (?, ?)",
                    (alias.lower(), canonical.lower()),
                )

        return inserted

    # ── Entity/predicate resolution ──────────────────────────────────────

    @property
    def entities(self) -> set[str]:
        """All known entity strings (subjects and objects)."""
        rows = self._conn.execute(
            "SELECT DISTINCT subject FROM triplets "
            "UNION SELECT DISTINCT lower(object) FROM triplets"
        ).fetchall()
        return {r[0] for r in rows}

    @property
    def subjects(self) -> set[str]:
        """All known subject strings."""
        rows = self._conn.execute(
            "SELECT DISTINCT subject FROM triplets"
        ).fetchall()
        return {r[0] for r in rows}

    def _resolve_entity(self, text: str) -> tuple[str, str]:
        """Resolve a surface form to a canonical entity via 3-tier cascade."""
        low = text.lower()

        row = self._conn.execute(
            "SELECT 1 FROM triplets WHERE subject = ? LIMIT 1", (low,)
        ).fetchone()
        if row:
            return low, "exact"

        row = self._conn.execute(
            "SELECT canonical FROM entity_aliases WHERE alias = ?", (low,)
        ).fetchone()
        if row:
            return row[0], "alias"

        result = fuzzy_match(low, self.entities, self.fuzzy_threshold)
        if result is not None:
            return result[0], "fuzzy"

        return low, "none"

    def _resolve_predicate(self, predicate: str) -> str:
        """Map an alias to its canonical predicate, or return as-is."""
        low = predicate.lower()
        row = self._conn.execute(
            "SELECT canonical FROM predicate_aliases WHERE alias = ?", (low,)
        ).fetchone()
        return row[0] if row else low

    def _resolve_predicate_cascade(
        self, predicate: str, subject: str | None = None,
    ) -> tuple[str, str]:
        """Resolve a predicate via 3-tier cascade: exact → alias → fuzzy."""
        low = predicate.lower()

        if subject:
            row = self._conn.execute(
                "SELECT 1 FROM triplets WHERE subject = ? AND predicate = ? LIMIT 1",
                (subject.lower(), low),
            ).fetchone()
            if row:
                return low, "exact"
        else:
            row = self._conn.execute(
                "SELECT 1 FROM triplets WHERE predicate = ? LIMIT 1", (low,)
            ).fetchone()
            if row:
                return low, "exact"

        alias_row = self._conn.execute(
            "SELECT canonical FROM predicate_aliases WHERE alias = ?", (low,)
        ).fetchone()
        if alias_row:
            return alias_row[0], "alias"

        if subject:
            pred_rows = self._conn.execute(
                "SELECT DISTINCT predicate FROM triplets WHERE subject = ?",
                (subject.lower(),),
            ).fetchall()
        else:
            pred_rows = self._conn.execute(
                "SELECT DISTINCT predicate FROM triplets"
            ).fetchall()

        candidates = [r[0] for r in pred_rows]
        if candidates:
            result = fuzzy_match(low, candidates, self.fuzzy_threshold)
            if result is not None:
                return result[0], "fuzzy"

        return low, "none"

    def has_entity(self, text: str) -> bool:
        low = text.lower()
        row = self._conn.execute(
            "SELECT 1 FROM triplets WHERE subject = ? LIMIT 1", (low,)
        ).fetchone()
        if row:
            return True
        row = self._conn.execute(
            "SELECT 1 FROM entity_aliases WHERE alias = ?", (low,)
        ).fetchone()
        return row is not None

    # ── Core lookup ──────────────────────────────────────────────────────

    def lookup(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[dict]:
        """Query the KG. Same interface as the in-memory KnowledgeGraph."""
        if predicate:
            predicate = self._resolve_predicate(predicate)

        if subject and predicate:
            rows = self._conn.execute(
                "SELECT subject, predicate, object FROM triplets "
                "WHERE subject = ? AND predicate = ?",
                (subject.lower(), predicate.lower()),
            ).fetchall()
        elif predicate and obj:
            rows = self._conn.execute(
                "SELECT subject, predicate, object FROM triplets "
                "WHERE predicate = ? AND lower(object) = ?",
                (predicate.lower(), obj.lower()),
            ).fetchall()
        elif subject:
            rows = self._conn.execute(
                "SELECT subject, predicate, object FROM triplets WHERE subject = ?",
                (subject.lower(),),
            ).fetchall()
        else:
            return []

        return [
            {"subject": r["subject"], "predicate": r["predicate"], "object": r["object"]}
            for r in rows
        ]

    # ── Multi-hop traversal ──────────────────────────────────────────────

    def traverse(self, entity: str, max_depth: int = 2) -> list[dict]:
        """BFS traversal, same semantics as in-memory KG."""
        visited: set[str] = set()
        result: list[dict] = []
        frontier: list[tuple[str, int]] = [(entity.lower(), 0)]

        subject_set = self.subjects

        while frontier:
            current, depth = frontier.pop(0)
            if current in visited:
                continue
            visited.add(current)

            facts = self.lookup(subject=current)
            for fact in facts:
                if fact not in result:
                    result.append(fact)
                obj_lower = fact["object"].lower()
                if (
                    depth + 1 <= max_depth
                    and obj_lower not in visited
                    and obj_lower in subject_set
                ):
                    frontier.append((obj_lower, depth + 1))

        return result

    # ── Stats ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM triplets").fetchone()
        return row[0]

    def __repr__(self) -> str:
        return f"SqliteKnowledgeGraph({len(self)} triplets, db='{self.db_path}')"

    @property
    def triplets(self) -> list[Triplet]:
        """All triplets as (subject, predicate, object) tuples."""
        rows = self._conn.execute(
            "SELECT subject, predicate, object FROM triplets"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]
