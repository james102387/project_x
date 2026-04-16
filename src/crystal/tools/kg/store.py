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

import copy
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

from crystal.tools.kg.fuzzy import fuzzy_match

Triplet: TypeAlias = tuple[str, str, str]


ORIGIN_API_METADATA = "api_metadata"
ORIGIN_OPINION_DOC = "opinion_doc"
ORIGIN_UNKNOWN = "unknown"

VALID_ORIGINS = {
    ORIGIN_API_METADATA,
    ORIGIN_OPINION_DOC,
    ORIGIN_UNKNOWN,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS triplets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT,
    source_sentence TEXT DEFAULT '',
    batch_id TEXT,
    origin TEXT DEFAULT 'unknown',
    source_document TEXT DEFAULT '',
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

CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id TEXT PRIMARY KEY,
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    triplet_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
);
"""

_MIGRATIONS = [
    ("col_source_sentence", "ALTER TABLE triplets ADD COLUMN source_sentence TEXT DEFAULT ''"),
    ("col_batch_id", "ALTER TABLE triplets ADD COLUMN batch_id TEXT"),
    ("tbl_ingestion_batches", """
        CREATE TABLE IF NOT EXISTS ingestion_batches (
            batch_id TEXT PRIMARY KEY,
            source TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            triplet_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        )
    """),
    ("col_origin", "ALTER TABLE triplets ADD COLUMN origin TEXT DEFAULT 'unknown'"),
    ("col_source_document", "ALTER TABLE triplets ADD COLUMN source_document TEXT DEFAULT ''"),
]


def _generate_batch_id(source: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(f"{source}{ts}".encode()).hexdigest()[:8]
    return f"batch_{ts}_{h}"


class SqliteKnowledgeGraph:
    """SQLite-backed KG with the same query interface as the in-memory KG."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        fuzzy_threshold: float = 80.0,
    ) -> None:
        self.db_path = str(db_path)
        self.fuzzy_threshold = fuzzy_threshold
        self._conn = self._open_connection(self.db_path)
        self._init_schema()
        self._run_migrations()

    @staticmethod
    def _open_connection(db_path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def __deepcopy__(self, memo):
        """Return a new instance pointing at the same DB file.

        sqlite3.Connection objects can't be pickled/deepcopied.
        Gradio's gr.State calls deepcopy on the initial value, so we
        re-open the connection instead.
        """
        return SqliteKnowledgeGraph(self.db_path, self.fuzzy_threshold)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_conn"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._conn = self._open_connection(self.db_path)

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _run_migrations(self) -> None:
        """Add new columns/tables to existing databases."""
        existing_cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(triplets)").fetchall()
        }
        _COL_SKIP = {
            "col_source_sentence": "source_sentence",
            "col_batch_id": "batch_id",
            "col_origin": "origin",
            "col_source_document": "source_document",
        }
        for name, sql in _MIGRATIONS:
            col = _COL_SKIP.get(name)
            if col and col in existing_cols:
                continue
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError:
                pass
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_batch ON triplets(batch_id)",
            "CREATE INDEX IF NOT EXISTS idx_origin ON triplets(origin)",
            "CREATE INDEX IF NOT EXISTS idx_source_document ON triplets(source_document)",
        ]:
            try:
                self._conn.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Bulk loading ─────────────────────────────────────────────────────

    def bulk_insert(
        self,
        triplets: list[Triplet | tuple[str, str, str, str]],
        entity_aliases: dict[str, str] | None = None,
        predicate_aliases: dict[str, str] | None = None,
        source: str = "",
        batch_id: str | None = None,
        origin: str = ORIGIN_UNKNOWN,
        source_document: str = "",
    ) -> int:
        """Insert triplets and aliases in a single transaction.

        Triplets can be 3-tuples (subject, predicate, object),
        4-tuples (subject, predicate, object, source_sentence), or
        5-tuples (subject, predicate, object, source_sentence, per_row_origin).

        Args:
            origin: Default origin for all rows (overridden by per-row 5th element).
            source_document: The specific document or payload that produced these facts.

        Returns count inserted.
        """
        if batch_id is None:
            batch_id = _generate_batch_id(source)

        inserted = 0
        with self._conn:
            for t in triplets:
                if len(t) >= 5:
                    subj, pred, obj, src_sent, row_origin = (
                        t[0], t[1], t[2], t[3], t[4],
                    )
                elif len(t) >= 4:
                    subj, pred, obj, src_sent = t[0], t[1], t[2], t[3]
                    row_origin = origin
                else:
                    subj, pred, obj = t[0], t[1], t[2]
                    src_sent = ""
                    row_origin = origin
                cursor = self._conn.execute(
                    "INSERT OR IGNORE INTO triplets "
                    "(subject, predicate, object, source, source_sentence, "
                    "batch_id, origin, source_document) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (subj.lower(), pred.lower(), obj, source, src_sent,
                     batch_id, row_origin, source_document),
                )
                inserted += cursor.rowcount if cursor.rowcount > 0 else 0

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

            if inserted > 0:
                self._conn.execute(
                    "INSERT OR REPLACE INTO ingestion_batches "
                    "(batch_id, source, triplet_count, status) VALUES (?, ?, ?, 'active')",
                    (batch_id, source, inserted),
                )

        return inserted

    # ── Batch management ─────────────────────────────────────────────────

    def delete_batch(self, batch_id: str) -> int:
        """Delete all triplets from a batch and mark it rolled back. Returns count deleted."""
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM triplets WHERE batch_id = ?", (batch_id,),
            )
            deleted = cursor.rowcount
            self._conn.execute(
                "UPDATE ingestion_batches SET status = 'rolled_back', triplet_count = 0 "
                "WHERE batch_id = ?",
                (batch_id,),
            )
        return deleted

    def list_batches(self, status: str | None = None) -> list[dict]:
        """List ingestion batches, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT batch_id, source, created_at, triplet_count, status "
                "FROM ingestion_batches WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT batch_id, source, created_at, triplet_count, status "
                "FROM ingestion_batches ORDER BY created_at DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def batch_stats(self, batch_id: str) -> dict:
        """Get statistics for a specific batch."""
        row = self._conn.execute(
            "SELECT batch_id, source, created_at, triplet_count, status "
            "FROM ingestion_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if not row:
            return {}
        stats = dict(row)
        pred_counts = self._conn.execute(
            "SELECT predicate, COUNT(*) as cnt FROM triplets "
            "WHERE batch_id = ? GROUP BY predicate ORDER BY cnt DESC",
            (batch_id,),
        ).fetchall()
        stats["predicates"] = {r["predicate"]: r["cnt"] for r in pred_counts}
        return stats

    def delete_by_ids(self, ids: list[int]) -> int:
        """Delete specific triplets by their row IDs. Returns count deleted."""
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        with self._conn:
            cursor = self._conn.execute(
                f"DELETE FROM triplets WHERE id IN ({placeholders})", ids,
            )
        return cursor.rowcount

    def get_all_facts(self) -> list[dict]:
        """Return all triplets with full metadata for auditing."""
        rows = self._conn.execute(
            "SELECT id, subject, predicate, object, source, source_sentence, "
            "batch_id, origin, source_document "
            "FROM triplets"
        ).fetchall()
        return [dict(r) for r in rows]

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
        """Query the KG. Same interface as the in-memory KnowledgeGraph.

        Returns dicts with subject, predicate, object, source, origin,
        and source_document.
        """
        _COLS = "subject, predicate, object, source, origin, source_document"

        if predicate:
            predicate = self._resolve_predicate(predicate)

        if subject and predicate:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM triplets "
                "WHERE subject = ? AND predicate = ?",
                (subject.lower(), predicate.lower()),
            ).fetchall()
        elif predicate and obj:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM triplets "
                "WHERE predicate = ? AND lower(object) = ?",
                (predicate.lower(), obj.lower()),
            ).fetchall()
        elif subject:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM triplets WHERE subject = ?",
                (subject.lower(),),
            ).fetchall()
        else:
            return []

        return [
            {"subject": r["subject"], "predicate": r["predicate"],
             "object": r["object"], "source": r["source"] or "",
             "origin": r["origin"] or ORIGIN_UNKNOWN,
             "source_document": r["source_document"] or ""}
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

    @property
    def triplets_with_source(self) -> list[tuple[str, str, str, str]]:
        """All triplets as (subject, predicate, object, source) tuples."""
        rows = self._conn.execute(
            "SELECT subject, predicate, object, source FROM triplets"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3] or "") for r in rows]

    @property
    def triplets_with_provenance(
        self,
    ) -> list[tuple[str, str, str, str, str, str]]:
        """All triplets as (subject, predicate, object, source, origin, source_document)."""
        rows = self._conn.execute(
            "SELECT subject, predicate, object, source, origin, source_document "
            "FROM triplets"
        ).fetchall()
        return [
            (r[0], r[1], r[2], r[3] or "", r[4] or ORIGIN_UNKNOWN, r[5] or "")
            for r in rows
        ]

    def provenance_counts(self) -> dict[str, int]:
        """Count triplets by origin type."""
        rows = self._conn.execute(
            "SELECT COALESCE(origin, 'unknown') AS o, COUNT(*) AS cnt "
            "FROM triplets GROUP BY o ORDER BY cnt DESC"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def source_documents(self) -> list[str]:
        """Distinct source_document values, excluding empty."""
        rows = self._conn.execute(
            "SELECT DISTINCT source_document FROM triplets "
            "WHERE source_document != '' ORDER BY source_document"
        ).fetchall()
        return [r[0] for r in rows]

    def lookup_by_origin(self, origin: str) -> list[dict]:
        """Return all triplets with a given origin."""
        rows = self._conn.execute(
            "SELECT id, subject, predicate, object, source, source_sentence, "
            "batch_id, origin, source_document "
            "FROM triplets WHERE origin = ?",
            (origin,),
        ).fetchall()
        return [dict(r) for r in rows]

    def lookup_by_document(self, source_document: str) -> list[dict]:
        """Return all triplets from a specific source document."""
        rows = self._conn.execute(
            "SELECT id, subject, predicate, object, source, source_sentence, "
            "batch_id, origin, source_document "
            "FROM triplets WHERE source_document = ?",
            (source_document,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Backfill provenance ───────────────────────────────────────────────

    def backfill_provenance(self) -> dict[str, int]:
        """Classify existing rows that have origin='unknown' based on heuristics.

        Returns counts of rows updated per origin category.
        """
        counts: dict[str, int] = {}

        with self._conn:
            cur = self._conn.execute(
                "UPDATE triplets SET origin = 'api_metadata' "
                "WHERE origin IN ('unknown', '') AND ("
                "  source LIKE '%cold-cases%' OR source LIKE '%courtlistener%' "
                "  OR source LIKE '%scaffold%' OR source LIKE '%cron%'"
                ")"
            )
            counts["api_metadata"] = cur.rowcount

            cur = self._conn.execute(
                "UPDATE triplets SET origin = 'opinion_doc' "
                "WHERE origin IN ('unknown', '') AND ("
                "  source = 'pasted_text' OR source LIKE '%.txt'"
                "  OR source LIKE '%.csv' OR source LIKE '%.json'"
                ") AND source NOT LIKE '%cold-cases%' "
                "AND source NOT LIKE '%courtlistener%' "
                "AND source NOT LIKE '%scaffold%' "
                "AND source NOT LIKE '%cron%'"
            )
            counts["opinion_doc"] = cur.rowcount

            cur = self._conn.execute(
                "UPDATE triplets SET source_document = source "
                "WHERE source_document IN ('', NULL) AND source != ''"
            )
            counts["source_document_backfilled"] = cur.rowcount

        return counts
