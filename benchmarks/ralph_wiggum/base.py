"""BaseLoop — shared evaluation, diagnosis, scoring, and reporting.

Every specialized loop inherits from BaseLoop and overrides:
  - FAILURE_CATEGORIES: which diagnosis categories this loop handles
  - TARGET_FILES: which files this loop is allowed to modify
  - _build_proposal_prompt(): LLM prompt scoped to this loop's domain
  - _validate_proposal(): structural checks for this loop's proposal format
  - _apply_proposal(): write the proposal to the target files
  - _revert_proposal(): undo the proposal (non-git path)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

import spacy

from benchmarks.scoring.fitness import binary_correct, fitness_score

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent.parent / "src"

RALPH_MODEL = os.environ.get("RALPH_MODEL", "claude-sonnet-4-20250514")


# ── Failure diagnosis ────────────────────────────────────────────────


class FailureCategory:
    ENTITY_MISMATCH = "entity_mismatch"
    PREDICATE_MISMATCH = "predicate_mismatch"
    ROUTING_ERROR = "routing_error"
    FRAMING_ERROR = "framing_error"
    MATH_FALSE_POSITIVE = "math_false_positive"
    NO_DETECTION = "no_detection"
    CORRECT = "correct"


def diagnose_failure(
    question: str,
    golden_answer: str,
    match_strings: list[str],
    is_negative: bool,
    pipeline_result: dict,
    detection: dict | None,
) -> str:
    """Classify a failure into a component-level category."""
    response = pipeline_result.get("final_response", "")
    prompt_type = pipeline_result.get("prompt_type", "")
    correct = binary_correct(response, match_strings, is_negative)

    if correct:
        return FailureCategory.CORRECT

    if is_negative:
        if prompt_type in ("kg_answerable", "kg_augmented"):
            return FailureCategory.ROUTING_ERROR
        return FailureCategory.FRAMING_ERROR

    if prompt_type in ("pure_math", "math_answerable", "math_augmented"):
        if detection and detection.get("tool") != "calculator":
            return FailureCategory.MATH_FALSE_POSITIVE

    if detection is None or not pipeline_result.get("kg_entities_found"):
        return FailureCategory.NO_DETECTION

    kg_results = pipeline_result.get("kg_results", [])
    if kg_results:
        flat_facts = []
        for entry in kg_results:
            flat_facts.extend(entry.get("results", []))

        response_lower = response.lower()
        golden_lower = golden_answer.lower()
        for fact in flat_facts:
            if fact["object"].lower() in golden_lower:
                if fact["object"].lower() not in response_lower:
                    return FailureCategory.FRAMING_ERROR
                break
        else:
            if flat_facts:
                entity_in_golden = any(
                    fact["subject"].lower() in golden_lower for fact in flat_facts
                )
                if not entity_in_golden:
                    return FailureCategory.ENTITY_MISMATCH
                return FailureCategory.PREDICATE_MISMATCH

    if prompt_type == "no_math" and detection is not None:
        return FailureCategory.ROUTING_ERROR

    return FailureCategory.ENTITY_MISMATCH


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class IterationResult:
    iteration: int
    score: float
    total: int
    correct: int
    loop_name: str = ""
    failures: list[dict] = field(default_factory=list)
    proposed_changes: dict = field(default_factory=dict)
    diagnosis_summary: dict = field(default_factory=dict)


@dataclass
class LoopResult:
    converged: bool
    final_score: float
    iterations_run: int
    best_score: float
    best_iteration: int
    loop_name: str = ""
    history: list[IterationResult] = field(default_factory=list)
    change_report: str = ""


# ── LLM interface ────────────────────────────────────────────────────


def call_ralph_llm(prompt: str) -> tuple[str, dict | None]:
    """Call Anthropic Claude for code-modification proposals."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export it."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=RALPH_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return text, usage


def parse_llm_proposal(raw: str) -> dict | None:
    """Parse an LLM response into a structured proposal dict."""
    if not raw:
        return None

    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    text = json_match.group(1) if json_match else raw

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ── File mutation helpers ────────────────────────────────────────────


def insert_dict_entries(
    filepath: Path,
    dict_name: str,
    entries: dict[str, str],
) -> int:
    """Insert new key-value pairs into a Python dict literal in a file."""
    content = filepath.read_text(encoding="utf-8")
    inserted = 0

    pattern = re.compile(
        rf"^({dict_name}\b.*?=\s*\{{)(.*?)(\n\}})",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        logger.warning("Could not find dict %s in %s", dict_name, filepath)
        return 0

    dict_body = match.group(2)
    new_lines = []
    for key, value in entries.items():
        if f'"{key}"' in dict_body or f"'{key}'" in dict_body:
            continue
        new_lines.append(f'    "{key}": "{value}",')
        inserted += 1

    if not new_lines:
        return 0

    insert_text = "\n" + "\n".join(new_lines)
    new_content = (
        content[:match.end(2)]
        + insert_text
        + content[match.end(2):]
    )
    filepath.write_text(new_content, encoding="utf-8")
    return inserted


def remove_dict_entries(
    filepath: Path,
    dict_name: str,
    entries: dict[str, str],
) -> None:
    """Remove specific entries from a Python dict literal in a file."""
    content = filepath.read_text(encoding="utf-8")
    for key, value in entries.items():
        line_pattern = re.compile(
            rf'^\s*"{re.escape(key)}":\s*"{re.escape(value)}",?\s*\n',
            re.MULTILINE,
        )
        content = line_pattern.sub("", content)
    filepath.write_text(content, encoding="utf-8")


# ── Git operations ───────────────────────────────────────────────────


def git_cmd(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd or Path(__file__).parent.parent.parent,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# ── Pipeline runner ──────────────────────────────────────────────────


def run_full_pipeline(question: str, kg, *, mock_llm_fn=None) -> dict:
    """Run a question through the full Crystal LangGraph pipeline."""
    from crystal.graph import build_crystal_graph
    from crystal.state import make_initial_state

    app = build_crystal_graph()
    state = make_initial_state(question, kg=kg)

    if mock_llm_fn:
        with patch(
            "crystal.nodes.llm_nodes.crystal.llm.call_llm",
            side_effect=mock_llm_fn,
        ):
            return app.invoke(state)
    return app.invoke(state)


def default_mock_llm(prompt: str):
    return "[LLM fallback — no specific answer]", {
        "prompt_tokens": 10, "output_tokens": 5,
    }


# ── Change report ────────────────────────────────────────────────────


def build_change_report(history: list[IterationResult], loop_name: str = "") -> str:
    title = f"# {loop_name} — Change Report\n" if loop_name else "# Ralph Wiggum — Change Report\n"
    lines = [title]

    if not history:
        lines.append("No iterations were run.\n")
        return "\n".join(lines)

    lines.append("## Summary")
    lines.append(f"- Iterations: {len(history)}")
    lines.append(f"- Initial score: {history[0].score:.1%}")
    lines.append(f"- Final score: {history[-1].score:.1%}")
    best = max(history, key=lambda h: h.score)
    lines.append(f"- Best score: {best.score:.1%} (iteration {best.iteration})")
    lines.append("")

    for result in history:
        lines.append(f"## Iteration {result.iteration}")
        lines.append(f"Score: {result.score:.1%} ({result.correct}/{result.total})")

        if result.diagnosis_summary:
            lines.append("\n### Failure Diagnosis")
            for category, count in sorted(result.diagnosis_summary.items()):
                if count > 0:
                    lines.append(f"  - {category}: {count}")

        if result.proposed_changes:
            lines.append("\n### Proposed Changes")
            lines.append(f"```json\n{json.dumps(result.proposed_changes, indent=2)}\n```")

        remaining = [f for f in result.failures if not f.get("correct", False)]
        if remaining:
            lines.append(f"\n### Remaining Failures ({len(remaining)})")
            for f in remaining[:5]:
                lines.append(f"  - [{f.get('diagnosis', '?')}] {f['question'][:80]}")
            if len(remaining) > 5:
                lines.append(f"  - ... and {len(remaining) - 5} more")

        lines.append("")

    return "\n".join(lines)


# ── Base loop ────────────────────────────────────────────────────────


class BaseLoop(ABC):
    """Shared evaluation and loop mechanics. Subclasses own one mutation domain."""

    LOOP_NAME: str = "BaseLoop"
    FAILURE_CATEGORIES: frozenset[str] = frozenset()
    TARGET_FILES: list[Path] = []

    def __init__(
        self,
        kg,
        cases: list[tuple[str, str, list[str], bool]],
        nlp=None,
        on_iteration: Callable[[IterationResult], None] | None = None,
        call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
        use_git: bool = False,
        use_full_pipeline: bool = True,
        mock_llm_fn=None,
    ) -> None:
        self.kg = kg
        self.cases = cases
        self.nlp = nlp or spacy.load("en_core_web_sm")
        self.on_iteration = on_iteration
        self.call_llm_fn = call_llm_fn
        self.use_git = use_git
        self.use_full_pipeline = use_full_pipeline
        self.mock_llm_fn = mock_llm_fn or default_mock_llm
        self.results_log: list[dict] = []

    def _run_single_case(
        self,
        question: str,
        match_strings: list[str],
        is_negative: bool,
    ) -> tuple[str, bool, dict | None, dict | None]:
        if self.use_full_pipeline:
            pipeline_result = run_full_pipeline(
                question, self.kg, mock_llm_fn=self.mock_llm_fn,
            )
            response = pipeline_result.get("final_response", "")
            detection = None
            kg_detections = pipeline_result.get("kg_detections", [])
            if kg_detections:
                detection = kg_detections[0]
        else:
            from crystal.detectors.kg import detect_kg_query
            from crystal.nodes.compiler import _format_kg_results

            pipeline_result = None
            doc = self.nlp(question)
            detection = detect_kg_query(doc, self.kg)

            if detection is None:
                response = "No KG match found." if not is_negative else "No match found in the knowledge graph."
            else:
                response = _format_kg_results([{
                    "tool": "kg", "success": True,
                    "results": detection["results"],
                }])

        correct = binary_correct(response, match_strings, is_negative)
        return response, correct, detection, pipeline_result

    def _analyze_failures(self, results: list[dict]) -> list[dict]:
        failures = []
        for r in results:
            if r["correct"]:
                continue
            failure = {
                "question": r["question"],
                "golden_answer": r["golden_answer"],
                "response": r["response"][:200],
                "match_strings": r["match_strings"],
                "is_negative": r["is_negative"],
            }
            detection = r.get("detection")
            pipeline_result = r.get("pipeline_result") or {}

            diagnosis = diagnose_failure(
                r["question"], r["golden_answer"], r["match_strings"],
                r["is_negative"], pipeline_result, detection,
            )
            failure["diagnosis"] = diagnosis

            if detection:
                failure["detected_entity"] = detection.get("entity", "")
                failure["lookup_type"] = detection.get("lookup_type", "")
                failure["predicate_phrase"] = detection.get("predicate_phrase", "")
                failure["n_results"] = len(detection.get("results", []))
                failure["match_tier"] = detection.get("match_tier", "")
                failure["match_score"] = detection.get("match_score", 0.0)
            else:
                failure["detection"] = None

            failure["prompt_type"] = pipeline_result.get("prompt_type", "")
            failures.append(failure)
        return failures

    def _my_failures(self, failures: list[dict]) -> list[dict]:
        """Filter to only the failure categories this loop handles."""
        if not self.FAILURE_CATEGORIES:
            return failures
        return [f for f in failures if f.get("diagnosis") in self.FAILURE_CATEGORIES]

    def run_iteration(self, iteration: int) -> IterationResult:
        results: list[dict] = []

        for question, golden_answer, match_strings, is_negative in self.cases:
            response, correct, detection, pipeline_result = self._run_single_case(
                question, match_strings, is_negative,
            )
            results.append({
                "question": question,
                "golden_answer": golden_answer,
                "response": response,
                "match_strings": match_strings,
                "is_negative": is_negative,
                "correct": correct,
                "detection": detection,
                "pipeline_result": pipeline_result,
            })

        correct_count = sum(1 for r in results if r["correct"])
        total = len(results)
        score = correct_count / total if total > 0 else 0.0
        failures = self._analyze_failures(results)

        diagnosis_summary: dict[str, int] = {}
        for f in failures:
            cat = f.get("diagnosis", "unknown")
            diagnosis_summary[cat] = diagnosis_summary.get(cat, 0) + 1

        return IterationResult(
            iteration=iteration,
            score=score,
            total=total,
            correct=correct_count,
            loop_name=self.LOOP_NAME,
            failures=failures,
            diagnosis_summary=diagnosis_summary,
        )

    @abstractmethod
    def _build_proposal_prompt(self, failures: list[dict]) -> str:
        """Build an LLM prompt scoped to this loop's mutation domain."""

    @abstractmethod
    def _validate_proposal(self, proposal: dict) -> bool:
        """Structural validation for this loop's proposal format."""

    @abstractmethod
    def _apply_proposal(self, proposal: dict) -> dict[str, int]:
        """Write the proposal to target files. Returns {target: count}."""

    @abstractmethod
    def _revert_proposal(self, proposal: dict) -> None:
        """Undo the proposal (non-git revert path)."""

    def _propose_changes(self, failures: list[dict]) -> dict | None:
        if not self.call_llm_fn or not failures:
            return None

        my_failures = self._my_failures(failures)
        if not my_failures:
            logger.info("%s: no failures in my categories, skipping", self.LOOP_NAME)
            return None

        prompt = self._build_proposal_prompt(my_failures)

        try:
            response_text, _ = self.call_llm_fn(prompt)
            proposal = parse_llm_proposal(response_text)
            if proposal and self._validate_proposal(proposal):
                return proposal
            logger.info("%s: LLM proposal invalid or empty", self.LOOP_NAME)
        except Exception as e:
            logger.warning("%s: LLM proposal failed: %s", self.LOOP_NAME, e)

        return None

    def run(
        self,
        threshold: float = 0.90,
        max_iterations: int = 20,
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
                "%s iter %d: %.1f%% (%d/%d, %d failures)",
                self.LOOP_NAME, i, result.score * 100,
                result.correct, result.total, len(result.failures),
            )

            self.results_log.append({
                "iteration": i, "score": result.score,
                "correct": result.correct, "total": result.total,
                "failures": len(result.failures),
                "diagnosis": result.diagnosis_summary,
            })

            if self.on_iteration:
                self.on_iteration(result)

            if result.score >= threshold:
                logger.info("%s: converged at %.1f%%", self.LOOP_NAME, result.score * 100)
                report = build_change_report(history, self.LOOP_NAME)
                return LoopResult(
                    converged=True, final_score=result.score,
                    iterations_run=i + 1, best_score=best_score,
                    best_iteration=best_iteration, loop_name=self.LOOP_NAME,
                    history=history, change_report=report,
                )

            if not self.call_llm_fn:
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

            if self.use_git:
                files_to_stage = [str(f) for f in self.TARGET_FILES if f.exists()]
                if files_to_stage:
                    git_cmd("add", *files_to_stage)
                    git_cmd("commit", "-m",
                            f"ralph/{self.LOOP_NAME}: {desc} (iter {i}, score {result.score:.3f})")

            next_result = self.run_iteration(i + 1)
            if next_result.score > result.score:
                logger.info("%s: improved %.1f%% → %.1f%%",
                            self.LOOP_NAME, result.score * 100, next_result.score * 100)
            else:
                logger.info("%s: no improvement, reverting", self.LOOP_NAME)
                if self.use_git:
                    git_cmd("reset", "--hard", "HEAD~1")
                else:
                    self._revert_proposal(proposal)

        final = history[-1] if history else IterationResult(0, 0.0, 0, 0)
        report = build_change_report(history, self.LOOP_NAME)
        return LoopResult(
            converged=final.score >= threshold, final_score=final.score,
            iterations_run=len(history), best_score=best_score,
            best_iteration=best_iteration, loop_name=self.LOOP_NAME,
            history=history, change_report=report,
        )
