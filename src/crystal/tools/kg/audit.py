"""KG audit tool — fast, deterministic health check with zero LLM cost.

Runs the same validation gates as the ingestion pipeline against
every fact in the KG. Produces an AuditReport with flagged facts,
health score, and delete/review candidate lists.

CLI: python -m crystal.tools.kg.audit --db data/legal.sqlite [--fix]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from crystal.ingest.validation import (
    ValidationResult,
    ValidationSeverity,
    validate_object,
    validate_predicate,
    validate_source_sentence,
    validate_subject,
)

logger = logging.getLogger(__name__)


@dataclass
class FlaggedFact:
    id: int
    subject: str
    predicate: str
    object: str
    source: str
    category: str
    severity: str
    reason: str


_HARD_WEIGHT = 1.0
_SOFT_WEIGHT = 0.25


@dataclass
class AuditReport:
    total_facts: int = 0
    flagged_facts: int = 0
    critical_count: int = 0
    soft_count: int = 0
    flagged_by_category: dict[str, int] = field(default_factory=dict)
    health_score: float = 1.0
    delete_candidates: list[FlaggedFact] = field(default_factory=list)
    review_candidates: list[FlaggedFact] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# KG Audit Report",
            "",
            f"- **Total facts:** {self.total_facts}",
            f"- **Flagged:** {self.flagged_facts} ({self.flagged_facts / max(1, self.total_facts) * 100:.1f}%)",
            f"- **Critical (hard failures):** {self.critical_count}",
            f"- **Soft (cosmetic/borderline):** {self.soft_count}",
            f"- **Health score:** {self.health_score:.3f}",
            f"- **Delete candidates:** {len(self.delete_candidates)}",
            f"- **Review candidates:** {len(self.review_candidates)}",
            "",
        ]
        if self.critical_count > 0:
            lines.append(
                "**WARNING:** Critical failures detected. These facts could cause "
                "Crystal to give confidently wrong answers via kg_answerable."
            )
            lines.append("")

        lines.append("## Flagged by category")
        lines.append("")
        for cat, count in sorted(self.flagged_by_category.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}:** {count}")

        if self.delete_candidates:
            lines.extend(["", "## Delete candidates (sample)", ""])
            for f in self.delete_candidates[:20]:
                lines.append(f"- [{f.id}] ({f.subject}, {f.predicate}, {f.object[:60]}...) — {f.reason}")

        if self.review_candidates:
            lines.extend(["", "## Review candidates (sample)", ""])
            for f in self.review_candidates[:20]:
                lines.append(f"- [{f.id}] ({f.subject}, {f.predicate}, {f.object[:60]}...) — {f.reason}")

        return "\n".join(lines)


GROUND_TRUTH: dict[str, dict[str, str]] = {
    "miranda v. arizona": {
        "court": "Supreme Court of the United States",
        "date_filed": "1966-06-13",
        "opinion_author": "Warren",
    },
    "brown v. board of education": {
        "court": "Supreme Court of the United States",
        "date_filed": "1954-05-17",
        "opinion_author": "Warren",
    },
    "loving v. virginia": {
        "court": "Supreme Court of the United States",
        "date_filed": "1967-06-12",
        "opinion_author": "Warren",
    },
    "roe v. wade": {
        "court": "Supreme Court of the United States",
        "date_filed": "1973-01-22",
        "opinion_author": "Blackmun",
    },
    "marbury v. madison": {
        "court": "Supreme Court of the United States",
        "date_filed": "1803-02-24",
        "opinion_author": "Marshall",
    },
    "gideon v. wainwright": {
        "court": "Supreme Court of the United States",
        "date_filed": "1963-03-18",
        "opinion_author": "Black",
    },
    "terry v. ohio": {
        "court": "Supreme Court of the United States",
        "date_filed": "1968-06-10",
    },
    "dred scott v. sandford": {
        "court": "Supreme Court of the United States",
        "date_filed": "1857-03-06",
        "opinion_author": "Taney",
    },
}


def _check_ground_truth(subject: str, predicate: str, obj: str) -> ValidationResult | None:
    """Check a fact against known ground truth. Returns None if no truth available."""
    gt_case = GROUND_TRUTH.get(subject.lower())
    if gt_case is None:
        return None
    gt_value = gt_case.get(predicate.lower())
    if gt_value is None:
        return None
    if gt_value.lower() == obj.strip().lower():
        return ValidationResult(True)
    if gt_value.lower() in obj.strip().lower():
        return ValidationResult(True)
    return ValidationResult(
        False, ValidationSeverity.SOFT,
        f"ground truth mismatch: expected '{gt_value}', got '{obj[:80]}'",
    )


def audit_kg(
    db_path: str | Path,
    *,
    ground_truth: dict[str, dict[str, str]] | None = None,
) -> AuditReport:
    """Run a full audit on the KG at db_path.

    Args:
        db_path: Path to the SQLite KG database.
        ground_truth: Optional override for ground truth data.
            Defaults to the built-in GROUND_TRUTH dict.
    """
    from crystal.tools.kg.store import SqliteKnowledgeGraph

    gt = ground_truth if ground_truth is not None else GROUND_TRUTH

    kg = SqliteKnowledgeGraph(db_path)
    facts = kg.get_all_facts()
    kg.close()

    report = AuditReport(total_facts=len(facts))
    seen_keys: dict[tuple[str, str], list[int]] = {}
    fact_worst_severity: dict[int, ValidationSeverity] = {}

    def _flag(fid: int, subj: str, pred: str, obj: str, source: str,
              cat: str, sev: ValidationSeverity, reason: str) -> None:
        report.flagged_by_category[cat] = report.flagged_by_category.get(cat, 0) + 1
        ff = FlaggedFact(fid, subj, pred, obj, source, cat, sev.value, reason)
        if sev == ValidationSeverity.HARD:
            report.delete_candidates.append(ff)
        else:
            report.review_candidates.append(ff)
        cur = fact_worst_severity.get(fid)
        if cur is None or sev == ValidationSeverity.HARD:
            fact_worst_severity[fid] = sev

    for fact in facts:
        fid = fact["id"]
        subj = fact["subject"]
        pred = fact["predicate"]
        obj = fact["object"]
        source = fact.get("source", "")
        source_sentence = fact.get("source_sentence", "")
        flagged = False

        sr = validate_subject(subj)
        if not sr.valid:
            _flag(fid, subj, pred, obj, source, "bad_subject",
                  sr.severity or ValidationSeverity.HARD, sr.reason)
            flagged = True

        pr = validate_predicate(pred)
        if not pr.valid:
            _flag(fid, subj, pred, obj, source, "bad_predicate",
                  pr.severity or ValidationSeverity.HARD, pr.reason)
            flagged = True

        or_ = validate_object(pred, obj)
        if not or_.valid:
            _flag(fid, subj, pred, obj, source, "bad_object",
                  or_.severity or ValidationSeverity.SOFT, or_.reason)
            flagged = True

        if source_sentence:
            ss_r = validate_source_sentence(subj, obj, source_sentence)
            if not ss_r.valid:
                _flag(fid, subj, pred, obj, source, "source_mismatch",
                      ss_r.severity or ValidationSeverity.SOFT, ss_r.reason)
                flagged = True

        if gt:
            gt_r = _check_ground_truth(subj, pred, obj)
            if gt_r is not None and not gt_r.valid:
                _flag(fid, subj, pred, obj, source, "ground_truth_mismatch",
                      gt_r.severity or ValidationSeverity.SOFT, gt_r.reason)
                flagged = True

        key = (subj.lower(), pred.lower())
        seen_keys.setdefault(key, []).append(fid)

        if flagged:
            report.flagged_facts += 1

    for key, ids in seen_keys.items():
        if len(ids) > 5:
            cat = "high_duplication"
            report.flagged_by_category[cat] = report.flagged_by_category.get(cat, 0) + 1

    hard_count = sum(1 for s in fact_worst_severity.values() if s == ValidationSeverity.HARD)
    soft_count = sum(1 for s in fact_worst_severity.values() if s == ValidationSeverity.SOFT)
    report.critical_count = hard_count
    report.soft_count = soft_count

    weighted_penalty = hard_count * _HARD_WEIGHT + soft_count * _SOFT_WEIGHT
    report.health_score = max(0.0, 1.0 - (weighted_penalty / max(1, report.total_facts)))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit the Crystal KG for data quality issues")
    parser.add_argument("--db", default="data/legal.sqlite", help="Path to SQLite KG database")
    parser.add_argument("--fix", action="store_true", help="Delete hard-failure facts")
    parser.add_argument("--json", action="store_true", help="Output report as JSON")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    report = audit_kg(args.db)

    if args.json:
        print(json.dumps({
            "total_facts": report.total_facts,
            "flagged_facts": report.flagged_facts,
            "critical_count": report.critical_count,
            "soft_count": report.soft_count,
            "health_score": report.health_score,
            "flagged_by_category": report.flagged_by_category,
            "delete_candidates": len(report.delete_candidates),
            "review_candidates": len(report.review_candidates),
        }, indent=2))
    else:
        print(report.to_markdown())

    if args.fix and report.delete_candidates:
        from crystal.tools.kg.store import SqliteKnowledgeGraph
        kg = SqliteKnowledgeGraph(args.db)
        ids_to_delete = [f.id for f in report.delete_candidates]
        deleted = kg.delete_by_ids(ids_to_delete)
        kg.close()
        print(f"\nDeleted {deleted} facts.")


if __name__ == "__main__":
    main()
