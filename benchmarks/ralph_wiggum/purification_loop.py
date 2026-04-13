"""PurificationLoop — iteratively tightens KG validation rules.

Unlike other Ralph Wiggum loops that evaluate question-answering accuracy,
this loop evaluates data quality: it runs audit_kg(), analyzes soft failures,
and proposes rule changes to validation.py. The golden facts safety constraint
prevents over-tightening.

Does NOT extend BaseLoop (different evaluation paradigm). Follows the same
propose → validate → apply → verify → keep/revert pattern.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from benchmarks.ralph_wiggum.base import (
    call_ralph_llm,
    git_cmd,
    parse_llm_proposal,
)

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent.parent / "src"
_VALIDATION_FILE = _SRC_ROOT / "crystal" / "ingest" / "validation.py"


@dataclass
class PurificationResult:
    converged: bool = False
    iterations_run: int = 0
    initial_health: float = 0.0
    final_health: float = 0.0
    initial_soft: int = 0
    final_soft: int = 0
    initial_critical: int = 0
    final_critical: int = 0
    golden_violations: int = 0
    proposals_applied: int = 0
    proposals_reverted: int = 0
    change_report: str = ""


_PROPOSAL_PROMPT = """\
You are improving validation rules for a legal knowledge graph.

The audit found {soft_count} soft failures and {critical_count} critical failures
in {total_facts} total facts. Your goal: tighten rules so that more bad facts
are caught, without rejecting legitimate legal data.

Current validation rules are in `src/crystal/ingest/validation.py`.

Here are the current junk subjects (frozenset `_JUNK_SUBJECTS`):
{junk_subjects}

Here are the current junk prefixes (tuple `_JUNK_PREFIXES`):
{junk_prefixes}

Here are sample soft failures from the audit:
{sample_failures}

Propose ADDITIONS to improve the rules. You may add:
- `junk_subjects`: new strings to add to `_JUNK_SUBJECTS`
- `junk_prefixes`: new prefix strings to add to `_JUNK_PREFIXES`

RULES:
- Only propose additions, never deletions
- Only add strings you are confident are junk (not case names)
- Be conservative: it's worse to reject a valid case name than to miss junk
- Do not add anything that looks like a real legal case name

Respond with ONLY a JSON object:
```json
{{
  "junk_subjects": {{"new_junk_1": "reason", "new_junk_2": "reason"}},
  "junk_prefixes": {{"new_prefix ": "reason"}}
}}
```

If nothing should be added, return empty dicts."""


def _read_current_rules() -> tuple[str, str]:
    """Read current _JUNK_SUBJECTS and _JUNK_PREFIXES from validation.py."""
    content = _VALIDATION_FILE.read_text(encoding="utf-8")

    subjects_match = re.search(
        r"_JUNK_SUBJECTS\s*=\s*frozenset\(\{(.*?)\}\)",
        content, re.DOTALL,
    )
    subjects_str = subjects_match.group(1).strip() if subjects_match else "(not found)"

    prefixes_match = re.search(
        r'_JUNK_PREFIXES\s*=\s*\((.*?)\)',
        content, re.DOTALL,
    )
    prefixes_str = prefixes_match.group(1).strip() if prefixes_match else "(not found)"

    return subjects_str, prefixes_str


def _insert_junk_subjects(new_subjects: list[str]) -> int:
    """Add new entries to _JUNK_SUBJECTS in validation.py."""
    content = _VALIDATION_FILE.read_text(encoding="utf-8")
    inserted = 0

    for subj in new_subjects:
        subj_clean = subj.strip().lower()
        if not subj_clean:
            continue
        if f'"{subj_clean}"' in content or f"'{subj_clean}'" in content:
            continue
        target = '\n})\n\n_JUNK_PREFIXES'
        insert_line = f'    "{subj_clean}",\n'
        idx = content.find(target)
        if idx >= 0:
            content = content[:idx] + "\n" + insert_line + content[idx:]
            inserted += 1

    if inserted:
        _VALIDATION_FILE.write_text(content, encoding="utf-8")
    return inserted


def _insert_junk_prefixes(new_prefixes: list[str]) -> int:
    """Add new entries to _JUNK_PREFIXES in validation.py."""
    content = _VALIDATION_FILE.read_text(encoding="utf-8")
    inserted = 0

    for prefix in new_prefixes:
        prefix_clean = prefix.strip().lower()
        if not prefix_clean:
            continue
        if f'"{prefix_clean}"' in content or f"'{prefix_clean}'" in content:
            continue
        pattern = re.compile(r'(_JUNK_PREFIXES\s*=\s*\(.*?)\)', re.DOTALL)
        match = pattern.search(content)
        if match:
            new_entry = f', "{prefix_clean}"'
            content = content[:match.end(1)] + new_entry + content[match.end(1):]
            inserted += 1

    if inserted:
        _VALIDATION_FILE.write_text(content, encoding="utf-8")
    return inserted


def _remove_junk_subjects(subjects: list[str]) -> None:
    """Remove entries from _JUNK_SUBJECTS (for revert)."""
    content = _VALIDATION_FILE.read_text(encoding="utf-8")
    for subj in subjects:
        subj_clean = subj.strip().lower()
        pattern = re.compile(rf'\s*"{re.escape(subj_clean)}",?\s*\n', re.MULTILINE)
        content = pattern.sub("\n", content)
    _VALIDATION_FILE.write_text(content, encoding="utf-8")


def _remove_junk_prefixes(prefixes: list[str]) -> None:
    """Remove entries from _JUNK_PREFIXES (for revert)."""
    content = _VALIDATION_FILE.read_text(encoding="utf-8")
    for prefix in prefixes:
        prefix_clean = prefix.strip().lower()
        content = content.replace(f', "{prefix_clean}"', "")
        content = content.replace(f'"{prefix_clean}", ', "")
    _VALIDATION_FILE.write_text(content, encoding="utf-8")


class PurificationLoop:
    """Iteratively tightens KG validation rules based on audit results."""

    LOOP_NAME = "purification"
    TARGET_FILES = [str(_VALIDATION_FILE)]

    def __init__(
        self,
        db_path: str | Path,
        *,
        golden_facts: list[tuple[str, str, str]] | None = None,
        call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
        use_git: bool = False,
    ) -> None:
        self.db_path = str(db_path)
        self.call_llm_fn = call_llm_fn or call_ralph_llm
        self.use_git = use_git

        if golden_facts is None:
            from tests.golden.test_cases import GOLDEN_KG_FACTS
            self.golden_facts = GOLDEN_KG_FACTS
        else:
            self.golden_facts = golden_facts

    def _evaluate(self):
        from crystal.tools.kg.audit import audit_kg
        return audit_kg(self.db_path)

    def _verify_golden_facts(self) -> list[tuple[str, str, str, str]]:
        """Returns list of (subject, predicate, object, reason) for golden facts
        that fail validation. Empty list = all golden facts safe."""
        from crystal.ingest.validation import validate_triplet

        violations = []
        for subj, pred, obj in self.golden_facts:
            vr = validate_triplet(subj, pred, obj)
            if not vr.valid:
                reasons = "; ".join(vr.reasons)
                violations.append((subj, pred, obj, reasons))
        return violations

    def _build_proposal_prompt(self, report) -> str:
        subjects_str, prefixes_str = _read_current_rules()

        sample = report.review_candidates[:30]
        sample_text = "\n".join(
            f"- ({f.subject}, {f.predicate}, {f.object[:60]}) — {f.reason}"
            for f in sample
        )

        return _PROPOSAL_PROMPT.format(
            soft_count=report.soft_count,
            critical_count=report.critical_count,
            total_facts=report.total_facts,
            junk_subjects=subjects_str,
            junk_prefixes=prefixes_str,
            sample_failures=sample_text or "(none)",
        )

    def _validate_proposal(self, proposal: dict | None) -> bool:
        if not proposal or not isinstance(proposal, dict):
            return False

        allowed = {"junk_subjects", "junk_prefixes"}
        if not any(k in allowed for k in proposal):
            return False

        for key in proposal:
            if key not in allowed:
                return False
            section = proposal[key]
            if not isinstance(section, dict):
                return False

        return True

    def _apply_proposal(self, proposal: dict) -> dict[str, int]:
        counts: dict[str, int] = {}

        js = proposal.get("junk_subjects", {})
        if js:
            n = _insert_junk_subjects(list(js.keys()))
            counts["junk_subjects"] = n

        jp = proposal.get("junk_prefixes", {})
        if jp:
            n = _insert_junk_prefixes(list(jp.keys()))
            counts["junk_prefixes"] = n

        return counts

    def _revert_proposal(self, proposal: dict) -> None:
        js = proposal.get("junk_subjects", {})
        if js:
            _remove_junk_subjects(list(js.keys()))

        jp = proposal.get("junk_prefixes", {})
        if jp:
            _remove_junk_prefixes(list(jp.keys()))

    def run(
        self,
        target_soft_count: int = 50,
        max_iterations: int = 5,
    ) -> PurificationResult:
        result = PurificationResult()

        baseline = self._evaluate()
        result.initial_health = baseline.health_score
        result.initial_soft = baseline.soft_count
        result.initial_critical = baseline.critical_count

        logger.info(
            "Purification baseline: health=%.3f, critical=%d, soft=%d, total=%d",
            baseline.health_score, baseline.critical_count, baseline.soft_count,
            baseline.total_facts,
        )

        golden_violations = self._verify_golden_facts()
        if golden_violations:
            logger.error(
                "Golden facts already failing validation (%d violations) — "
                "cannot run purification loop safely.",
                len(golden_violations),
            )
            for s, p, o, reason in golden_violations:
                logger.error("  (%s, %s, %s) — %s", s, p, o, reason)
            result.golden_violations = len(golden_violations)
            return result

        if baseline.critical_count == 0 and baseline.soft_count <= target_soft_count:
            logger.info("Already converged (soft=%d <= target=%d)",
                        baseline.soft_count, target_soft_count)
            result.converged = True
            result.final_health = baseline.health_score
            result.final_soft = baseline.soft_count
            result.final_critical = baseline.critical_count
            return result

        current_report = baseline
        consecutive_no_progress = 0

        for i in range(max_iterations):
            result.iterations_run = i + 1
            logger.info("Purification iteration %d/%d", i + 1, max_iterations)

            prompt = self._build_proposal_prompt(current_report)
            try:
                raw_response, _ = self.call_llm_fn(prompt)
            except Exception:
                logger.exception("LLM call failed")
                break

            proposal = parse_llm_proposal(raw_response)
            if not self._validate_proposal(proposal):
                logger.warning("Invalid proposal, stopping")
                break

            if self.use_git:
                git_cmd("add", str(_VALIDATION_FILE))

            counts = self._apply_proposal(proposal)
            total_changes = sum(counts.values())
            if total_changes == 0:
                logger.info("Proposal had no new additions, stopping")
                break

            logger.info("Applied %d changes: %s", total_changes, counts)

            golden_check = self._verify_golden_facts()
            if golden_check:
                logger.warning(
                    "Proposal would reject %d golden facts — reverting",
                    len(golden_check),
                )
                self._revert_proposal(proposal)
                result.proposals_reverted += 1
                result.golden_violations += len(golden_check)
                consecutive_no_progress += 1
                if consecutive_no_progress >= 2:
                    logger.info("No progress for 2 iterations, stopping")
                    break
                continue

            new_report = self._evaluate()
            logger.info(
                "After proposal: health=%.3f→%.3f, soft=%d→%d",
                current_report.health_score, new_report.health_score,
                current_report.soft_count, new_report.soft_count,
            )

            if new_report.health_score >= current_report.health_score:
                result.proposals_applied += 1
                current_report = new_report
                consecutive_no_progress = 0

                if self.use_git:
                    git_cmd("add", str(_VALIDATION_FILE))
                    git_cmd(
                        "commit", "-m",
                        f"purification: +{total_changes} rules "
                        f"(health {new_report.health_score:.3f})",
                    )

                if new_report.critical_count == 0 and new_report.soft_count <= target_soft_count:
                    logger.info("Converged!")
                    result.converged = True
                    break
            else:
                logger.info("Health score did not improve — reverting")
                self._revert_proposal(proposal)
                result.proposals_reverted += 1
                consecutive_no_progress += 1

                if self.use_git:
                    git_cmd("checkout", "--", str(_VALIDATION_FILE))

                if consecutive_no_progress >= 3:
                    logger.info("No progress for 3 iterations, stopping")
                    break

        result.final_health = current_report.health_score
        result.final_soft = current_report.soft_count
        result.final_critical = current_report.critical_count
        result.change_report = self._build_report(result)
        return result

    def _build_report(self, result: PurificationResult) -> str:
        lines = [
            f"## {self.LOOP_NAME} loop",
            "",
            f"- **Iterations:** {result.iterations_run}",
            f"- **Converged:** {result.converged}",
            f"- **Health:** {result.initial_health:.3f} → {result.final_health:.3f}",
            f"- **Critical:** {result.initial_critical} → {result.final_critical}",
            f"- **Soft:** {result.initial_soft} → {result.final_soft}",
            f"- **Proposals applied:** {result.proposals_applied}",
            f"- **Proposals reverted:** {result.proposals_reverted}",
            f"- **Golden violations:** {result.golden_violations}",
        ]
        return "\n".join(lines)
