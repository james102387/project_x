"""ExtractionLoop — targets ingestion quality (subject/predicate/fact accuracy).

Evaluates by running document extraction on opinions with known CourtListener
metadata, comparing extracted triplets against ground truth for those cases.

Mutation targets:
  - LEGAL_EXTRACTION_PROMPT (instruction wording, examples)
  - normalize_predicate() alias additions in legal_ontology.py
  - INGEST_AUTO_ACCEPT threshold in confidence.py

Handles: subject_mismatch, predicate_mismatch, missing_fact, hallucinated_fact
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from benchmarks.ralph_wiggum.base import (
    BaseLoop,
    IterationResult,
    LoopResult,
    build_change_report,
    call_ralph_llm,
    insert_dict_entries,
    parse_llm_proposal,
    remove_dict_entries,
)

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent.parent / "src"
_EXTRACT_FILE = _SRC_ROOT / "crystal" / "ingest" / "llm_extract.py"
_ALIAS_FILE = _SRC_ROOT / "crystal" / "data" / "legal_ontology.py"
_CONFIDENCE_FILE = _SRC_ROOT / "crystal" / "ingest" / "confidence.py"


class ExtractionFailureCategory:
    SUBJECT_MISMATCH = "subject_mismatch"
    PREDICATE_MISMATCH = "predicate_mismatch"
    MISSING_FACT = "missing_fact"
    HALLUCINATED_FACT = "hallucinated_fact"
    CORRECT = "correct"


@dataclass
class ExtractionCase:
    """A case for extraction evaluation with known ground truth."""
    case_name: str
    opinion_text: str
    ground_truth: dict[str, str]


_PROMPT = """\
You are optimizing a legal document extraction system's prompt and normalization.
The system extracts (subject, predicate, object) triplets from court opinions.

Below are extraction failures — facts that should have been extracted but weren't,
or facts extracted incorrectly.

FAILURES:
{failures}

CURRENT LEGAL_EXTRACTION_PROMPT (excerpt):
{prompt_excerpt}

CURRENT LEGAL_PREDICATE_ALIASES:
{aliases}

Propose changes to improve extraction quality:

1. predicate_aliases: add surface forms → canonical predicates to help
   normalize_predicate() match more extracted predicates to the ontology.
2. prompt_hints: suggest specific instruction lines to add to the extraction
   prompt (e.g., "Always extract the court name as-is from the document header").

Rules:
- Only propose additions (never deletions)
- predicate_aliases keys = surface forms found in documents, values = ontology predicates
- prompt_hints = list of instruction strings to append to the extraction prompt
- Focus on the most common failure patterns

Respond with ONLY a JSON object:
```json
{{
  "predicate_aliases": {{"surface_form": "canonical_predicate", ...}},
  "prompt_hints": ["instruction 1", "instruction 2"]
}}
```
Include only sections with changes. If no useful changes: {{}}
"""


def _load_ground_truth_for_case(case_name: str) -> dict[str, str]:
    """Load known CourtListener metadata for a case as {predicate: value}."""
    try:
        from crystal.tools.kg.legal import load_legal_kg
        kg = load_legal_kg()
    except Exception:
        return {}

    from crystal.data.legal_ontology import normalize_case_name
    key = normalize_case_name(case_name).lower()
    results = kg.lookup(subject=key)
    if not results:
        for s in [case_name, case_name.lower()]:
            results = kg.lookup(subject=s)
            if results:
                break

    facts = {}
    for r in results:
        pred = r.get("predicate", "")
        obj = r.get("object", "")
        if pred and obj:
            facts[pred] = obj
    return facts


def diagnose_extraction_failure(
    expected_pred: str,
    expected_value: str,
    extracted_triplets: list[dict],
    case_name: str,
) -> str:
    """Classify an extraction failure."""
    case_lower = case_name.lower()
    pred_lower = expected_pred.lower()
    val_lower = expected_value.lower()

    for t in extracted_triplets:
        t_subj = t.get("subject", "").lower()
        t_pred = t.get("predicate", "").lower()
        t_obj = t.get("object", "").lower()

        if t_pred == pred_lower and val_lower in t_obj:
            if case_lower not in t_subj and t_subj not in case_lower:
                return ExtractionFailureCategory.SUBJECT_MISMATCH
            return ExtractionFailureCategory.CORRECT

        if case_lower in t_subj or t_subj in case_lower:
            if val_lower in t_obj:
                return ExtractionFailureCategory.PREDICATE_MISMATCH

    for t in extracted_triplets:
        t_subj = t.get("subject", "").lower()
        t_pred = t.get("predicate", "").lower()
        if case_lower in t_subj and t_pred == pred_lower:
            return ExtractionFailureCategory.HALLUCINATED_FACT

    return ExtractionFailureCategory.MISSING_FACT


class ExtractionLoop(BaseLoop):
    """Ralph Wiggum loop targeting document extraction quality."""

    LOOP_NAME = "ExtractionLoop"
    FAILURE_CATEGORIES = frozenset({
        ExtractionFailureCategory.SUBJECT_MISMATCH,
        ExtractionFailureCategory.PREDICATE_MISMATCH,
        ExtractionFailureCategory.MISSING_FACT,
        ExtractionFailureCategory.HALLUCINATED_FACT,
    })
    TARGET_FILES = [_EXTRACT_FILE, _ALIAS_FILE, _CONFIDENCE_FILE]

    def __init__(
        self,
        extraction_cases: list[ExtractionCase],
        *,
        call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
        extraction_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
        use_git: bool = False,
        on_iteration=None,
    ) -> None:
        self.extraction_cases = extraction_cases
        self.extraction_llm_fn = extraction_llm_fn
        self._call_llm_fn = call_llm_fn
        self.use_git = use_git
        self._on_iteration = on_iteration
        self.results_log: list[dict] = []

    @property
    def call_llm_fn(self):
        return self._call_llm_fn

    def run_iteration(self, iteration: int) -> IterationResult:
        """Run extraction on all cases, compare against ground truth."""
        from crystal.ingest import ingest_document
        from crystal.data.legal_ontology import LEGAL_PREDICATES

        all_failures = []
        total_facts = 0
        correct_facts = 0

        for case in self.extraction_cases:
            result = ingest_document(
                case.opinion_text,
                call_llm_fn=self.extraction_llm_fn,
                auto_accept_threshold=0.0,
                domain="legal",
            )

            extracted = [
                {"subject": st.subject, "predicate": st.predicate, "object": st.object}
                for st in result.auto_accepted + result.pending_review
            ]

            for pred, expected_val in case.ground_truth.items():
                total_facts += 1
                diag = diagnose_extraction_failure(
                    pred, expected_val, extracted, case.case_name,
                )
                if diag == ExtractionFailureCategory.CORRECT:
                    correct_facts += 1
                else:
                    all_failures.append({
                        "question": f"Extract {pred} for {case.case_name}",
                        "golden_answer": f"{pred}: {expected_val}",
                        "match_strings": [expected_val.lower()[:50]],
                        "is_negative": False,
                        "response": json.dumps(extracted[:5], default=str)[:200],
                        "case_name": case.case_name,
                        "expected_predicate": pred,
                        "expected_value": expected_val,
                        "diagnosis": diag,
                        "n_extracted": len(extracted),
                    })

        score = correct_facts / total_facts if total_facts > 0 else 0.0

        diagnosis_summary: dict[str, int] = {}
        for f in all_failures:
            cat = f.get("diagnosis", "unknown")
            diagnosis_summary[cat] = diagnosis_summary.get(cat, 0) + 1

        return IterationResult(
            iteration=iteration,
            score=score,
            total=total_facts,
            correct=correct_facts,
            loop_name=self.LOOP_NAME,
            failures=all_failures,
            diagnosis_summary=diagnosis_summary,
        )

    def _my_failures(self, failures: list[dict]) -> list[dict]:
        return [f for f in failures if f.get("diagnosis") in self.FAILURE_CATEGORIES]

    def _build_proposal_prompt(self, failures: list[dict]) -> str:
        from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES
        from crystal.ingest.llm_extract import LEGAL_EXTRACTION_PROMPT

        failure_text = ""
        for f in failures[:15]:
            failure_text += (
                f"  Case: {f.get('case_name', 'unknown')}\n"
                f"  Expected: {f.get('expected_predicate')}: {f.get('expected_value')}\n"
                f"  Diagnosis: {f.get('diagnosis')}\n"
                f"  Extracted triplets (sample): {f.get('response', '')[:150]}\n"
                f"  ---\n"
            )

        prompt_excerpt = LEGAL_EXTRACTION_PROMPT[:500] + "..."

        return _PROMPT.format(
            failures=failure_text,
            prompt_excerpt=prompt_excerpt,
            aliases=json.dumps(dict(LEGAL_PREDICATE_ALIASES), indent=2),
        )

    def _validate_proposal(self, proposal: dict) -> bool:
        allowed = {"predicate_aliases", "prompt_hints"}
        if not proposal or not isinstance(proposal, dict):
            return False
        if not any(k in allowed for k in proposal):
            return False
        for key in proposal:
            if key not in allowed:
                return False
            if key == "predicate_aliases":
                section = proposal[key]
                if not isinstance(section, dict):
                    return False
                for k, v in section.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        return False
            elif key == "prompt_hints":
                if not isinstance(proposal[key], list):
                    return False
                for hint in proposal[key]:
                    if not isinstance(hint, str):
                        return False
        return True

    def _apply_proposal(self, proposal: dict) -> dict[str, int]:
        counts: dict[str, int] = {}

        if "predicate_aliases" in proposal and proposal["predicate_aliases"]:
            counts["predicate_aliases"] = insert_dict_entries(
                _ALIAS_FILE, "LEGAL_PREDICATE_ALIASES",
                proposal["predicate_aliases"],
            )

        if "prompt_hints" in proposal and proposal["prompt_hints"]:
            content = _EXTRACT_FILE.read_text(encoding="utf-8")
            hints = proposal["prompt_hints"]
            added = 0
            for hint in hints:
                if hint not in content:
                    marker = "Return ONLY the JSON array."
                    if marker in content:
                        content = content.replace(
                            marker,
                            f"- {hint}\n\n{marker}",
                        )
                        added += 1
            if added:
                _EXTRACT_FILE.write_text(content, encoding="utf-8")
            counts["prompt_hints"] = added

        return counts

    def _revert_proposal(self, proposal: dict) -> None:
        if "predicate_aliases" in proposal:
            remove_dict_entries(
                _ALIAS_FILE, "LEGAL_PREDICATE_ALIASES",
                proposal["predicate_aliases"],
            )

    def run(
        self,
        threshold: float = 0.70,
        max_iterations: int = 10,
    ) -> LoopResult:
        history: list[IterationResult] = []
        best_score = 0.0
        best_iteration = 0
        consecutive_discards = 0

        for i in range(max_iterations):
            result = self.run_iteration(i)
            history.append(result)

            if result.score > best_score:
                best_score = result.score
                best_iteration = i
                consecutive_discards = 0
            else:
                consecutive_discards += 1

            logger.info(
                "%s iter %d: %.1f%% (%d/%d facts, %d failures)",
                self.LOOP_NAME, i, result.score * 100,
                result.correct, result.total, len(result.failures),
            )

            self.results_log.append({
                "iteration": i, "score": result.score,
                "correct": result.correct, "total": result.total,
                "failures": len(result.failures),
                "diagnosis": result.diagnosis_summary,
            })

            if self._on_iteration:
                self._on_iteration(result)

            if result.score >= threshold:
                logger.info("%s: converged at %.1f%%", self.LOOP_NAME, result.score * 100)
                report = build_change_report(history, self.LOOP_NAME)
                return LoopResult(
                    converged=True, final_score=result.score,
                    iterations_run=i + 1, best_score=best_score,
                    best_iteration=best_iteration, loop_name=self.LOOP_NAME,
                    history=history, change_report=report,
                )

            if not self._call_llm_fn:
                if i > 0 and result.score == history[i - 1].score:
                    logger.info("%s: score unchanged, no LLM. Stopping.", self.LOOP_NAME)
                    break
                continue

            my_failures = self._my_failures(result.failures)
            if not my_failures:
                logger.info("%s: no failures in my categories. Stopping.", self.LOOP_NAME)
                break

            if consecutive_discards >= 3:
                logger.info("%s: 3 consecutive discards. Stopping.", self.LOOP_NAME)
                break

            proposal = self._propose_changes(result.failures)
            if not proposal:
                logger.info("%s: no proposal. Stopping.", self.LOOP_NAME)
                break

            counts = self._apply_proposal(proposal)
            total_applied = sum(counts.values())
            if total_applied == 0:
                logger.info("%s: proposal had no new entries.", self.LOOP_NAME)
                continue

            result.proposed_changes = proposal
            desc = ", ".join(f"{k}: +{v}" for k, v in counts.items())
            logger.info("%s: applied %s", self.LOOP_NAME, desc)

        final = history[-1] if history else IterationResult(0, 0.0, 0, 0)
        report = build_change_report(history, self.LOOP_NAME)
        return LoopResult(
            converged=final.score >= threshold, final_score=final.score,
            iterations_run=len(history), best_score=best_score,
            best_iteration=best_iteration, loop_name=self.LOOP_NAME,
            history=history, change_report=report,
        )

    def _propose_changes(self, failures: list[dict]) -> dict | None:
        if not self._call_llm_fn or not failures:
            return None

        my_failures = self._my_failures(failures)
        if not my_failures:
            return None

        prompt = self._build_proposal_prompt(my_failures)

        try:
            response_text, _ = self._call_llm_fn(prompt)
            proposal = parse_llm_proposal(response_text)
            if proposal and self._validate_proposal(proposal):
                return proposal
            logger.info("%s: LLM proposal invalid or empty", self.LOOP_NAME)
        except Exception as e:
            logger.warning("%s: LLM proposal failed: %s", self.LOOP_NAME, e)

        return None
