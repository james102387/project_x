"""KG Proofreader — thorough, LLM-powered deep clean of the knowledge graph.

Two-pass system:
  Pass 1 (fast gates): deterministic regex/rule validators, zero LLM cost.
  Pass 2 (LLM proofreading): semantic verification against source sentences.

Tiered trust policy:
  - auto_delete: hard gate failures (no LLM needed)
  - auto_delete: soft gate failures that the LLM also rejects (tiered)
  - human_review: LLM-only rejections where gates passed
  - clean: passes both gates and LLM

CLI: python -m crystal.tools.kg.proofread --db data/legal.sqlite [--fix] [--fast] [--sample N]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from crystal.ingest.validation import (
    ValidationSeverity,
    proofread_triplets,
    validate_object,
    validate_predicate,
    validate_subject,
)

logger = logging.getLogger(__name__)


@dataclass
class DeletedFact:
    id: int
    subject: str
    predicate: str
    object: str
    reason: str


@dataclass
class ReviewFact:
    id: int
    subject: str
    predicate: str
    object: str
    source_sentence: str
    llm_reason: str


@dataclass
class ProofreaderReport:
    total_scanned: int = 0
    auto_deleted: int = 0
    human_review_count: int = 0
    pass2_validated: int = 0
    pass2_skipped: int = 0
    health_score_before: float = 1.0
    health_score_after: float = 1.0
    deleted_facts: list[DeletedFact] = field(default_factory=list)
    review_facts: list[ReviewFact] = field(default_factory=list)
    human_review_batch_path: str = ""

    def to_markdown(self) -> str:
        lines = [
            "# KG Proofreader Report",
            "",
            f"- **Total scanned:** {self.total_scanned}",
            f"- **Auto-deleted:** {self.auto_deleted}",
            f"- **Human review:** {self.human_review_count}",
            f"- **LLM validated:** {self.pass2_validated}",
            f"- **LLM skipped:** {self.pass2_skipped}",
            f"- **Health score:** {self.health_score_before:.2f} -> {self.health_score_after:.2f}",
        ]

        if self.human_review_batch_path:
            lines.append(f"- **Review batch:** {self.human_review_batch_path}")

        if self.deleted_facts:
            lines.extend(["", "## Deleted facts (sample)", ""])
            for df in self.deleted_facts[:30]:
                lines.append(
                    f"- [{df.id}] ({df.subject}, {df.predicate}, "
                    f"{df.object[:50]}...) — {df.reason}"
                )

        if self.review_facts:
            lines.extend(["", "## Flagged for human review (sample)", ""])
            for rf in self.review_facts[:20]:
                lines.append(
                    f"- [{rf.id}] ({rf.subject}, {rf.predicate}, "
                    f"{rf.object[:50]}...) — {rf.llm_reason}"
                )

        return "\n".join(lines)


def proofread_kg(
    db_path: str | Path,
    *,
    call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
    fast: bool = False,
    sample_n: int | None = None,
    fix: bool = False,
) -> ProofreaderReport:
    """Run the two-pass proofreader on the KG.

    Args:
        db_path: Path to the SQLite KG database.
        call_llm_fn: LLM caller for Pass 2. If None and not fast, skips LLM.
        fast: Skip LLM pass entirely (fast gates only).
        sample_n: If set, LLM-check only this many random facts (not all).
        fix: Actually delete flagged facts (default: report only).
    """
    from crystal.tools.kg.store import SqliteKnowledgeGraph

    kg = SqliteKnowledgeGraph(db_path)
    all_facts = kg.get_all_facts()
    report = ProofreaderReport(total_scanned=len(all_facts))

    # ── Pass 1: Fast gates ────────────────────────────────────────────
    hard_fail_ids: list[int] = []
    hard_fail_facts: list[DeletedFact] = []
    soft_fail_facts: list[dict] = []
    passed_facts: list[dict] = []

    for fact in all_facts:
        fid = fact["id"]
        subj = fact["subject"]
        pred = fact["predicate"]
        obj = fact["object"]

        sr = validate_subject(subj)
        pr = validate_predicate(pred)
        or_ = validate_object(pred, obj)

        worst_severity = None
        reasons = []
        for r in (sr, pr, or_):
            if not r.valid:
                reasons.append(r.reason)
                if r.severity == ValidationSeverity.HARD:
                    worst_severity = ValidationSeverity.HARD
                elif worst_severity is None:
                    worst_severity = ValidationSeverity.SOFT

        if worst_severity == ValidationSeverity.HARD:
            hard_fail_ids.append(fid)
            hard_fail_facts.append(DeletedFact(
                fid, subj, pred, obj, "; ".join(reasons),
            ))
        elif worst_severity == ValidationSeverity.SOFT:
            fact["_soft_reasons"] = reasons
            soft_fail_facts.append(fact)
        else:
            passed_facts.append(fact)

    report.auto_deleted = len(hard_fail_ids)
    report.deleted_facts.extend(hard_fail_facts)
    report.health_score_before = 1.0 - (
        (len(hard_fail_ids) + len(soft_fail_facts)) / max(1, len(all_facts))
    )

    # ── Pass 2: LLM proofreading ─────────────────────────────────────
    if not fast and call_llm_fn is not None:
        llm_candidates = soft_fail_facts + passed_facts

        if sample_n is not None and sample_n < len(llm_candidates):
            llm_candidates = random.sample(llm_candidates, sample_n)
            report.pass2_skipped = len(soft_fail_facts) + len(passed_facts) - sample_n
        else:
            report.pass2_skipped = 0

        if llm_candidates:
            triplet_data = [
                (f["subject"], f["predicate"], f["object"], f.get("source_sentence", ""))
                for f in llm_candidates
            ]
            llm_results = proofread_triplets(triplet_data, call_llm_fn)

            for i, fact in enumerate(llm_candidates):
                fid = fact["id"]
                pr = llm_results.get(i)
                is_soft = "_soft_reasons" in fact

                if pr is None:
                    report.pass2_skipped += 1
                    continue

                if pr.valid:
                    report.pass2_validated += 1
                else:
                    if is_soft:
                        reasons = fact["_soft_reasons"]
                        hard_fail_ids.append(fid)
                        report.auto_deleted += 1
                        report.deleted_facts.append(DeletedFact(
                            fid, fact["subject"], fact["predicate"],
                            fact["object"],
                            f"tiered: gate={'; '.join(reasons)}, llm={pr.reason}",
                        ))
                    else:
                        report.human_review_count += 1
                        report.review_facts.append(ReviewFact(
                            fid, fact["subject"], fact["predicate"],
                            fact["object"],
                            fact.get("source_sentence", ""),
                            pr.reason,
                        ))
    else:
        report.pass2_skipped = len(soft_fail_facts) + len(passed_facts)

    # ── Fix: delete flagged facts ─────────────────────────────────────
    if fix and hard_fail_ids:
        unique_ids = list(dict.fromkeys(hard_fail_ids))
        deleted = kg.delete_by_ids(unique_ids)
        logger.info("Deleted %d facts from KG", deleted)

    remaining = len(all_facts) - len(set(hard_fail_ids)) if fix else len(all_facts)
    report.health_score_after = 1.0 - (
        report.human_review_count / max(1, remaining)
    )

    # ── Save human review batch ───────────────────────────────────────
    if report.review_facts:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        batch_path = f"review/proofread_{ts}.json"
        batch_data = {
            "batch": {
                "id": f"proofread_{ts}",
                "source": str(db_path),
                "type": "kg_proofreader",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "total": len(report.review_facts),
            "cases": [
                {
                    "question": f"Is ({rf.subject}, {rf.predicate}, {rf.object[:80]}) valid?",
                    "golden_answer": "",
                    "fact_id": rf.id,
                    "subject": rf.subject,
                    "predicate": rf.predicate,
                    "object": rf.object,
                    "source_sentence": rf.source_sentence,
                    "llm_reason": rf.llm_reason,
                    "status": "pending_review",
                }
                for rf in report.review_facts
            ],
        }
        try:
            Path(batch_path).parent.mkdir(parents=True, exist_ok=True)
            Path(batch_path).write_text(json.dumps(batch_data, indent=2))
            report.human_review_batch_path = batch_path
        except Exception as e:
            logger.warning("Failed to save review batch: %s", e)

    kg.close()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KG Proofreader — deep clean the knowledge graph")
    parser.add_argument("--db", default="data/legal.sqlite", help="Path to SQLite KG database")
    parser.add_argument("--fix", action="store_true", help="Actually delete flagged facts")
    parser.add_argument("--fast", action="store_true", help="Skip LLM pass (fast gates only)")
    parser.add_argument("--sample", type=int, default=None, help="LLM-check N random facts only")
    parser.add_argument("--json", action="store_true", help="Output report as JSON")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    call_llm_fn = None
    if not args.fast:
        try:
            from crystal.llm import call_llm
            call_llm_fn = call_llm
        except Exception:
            logger.warning("Could not load LLM; running in fast mode")

    report = proofread_kg(
        args.db,
        call_llm_fn=call_llm_fn,
        fast=args.fast,
        sample_n=args.sample,
        fix=args.fix,
    )

    if args.json:
        print(json.dumps({
            "total_scanned": report.total_scanned,
            "auto_deleted": report.auto_deleted,
            "human_review_count": report.human_review_count,
            "pass2_validated": report.pass2_validated,
            "pass2_skipped": report.pass2_skipped,
            "health_score_before": report.health_score_before,
            "health_score_after": report.health_score_after,
        }, indent=2))
    else:
        print(report.to_markdown())


if __name__ == "__main__":
    main()
