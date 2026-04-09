"""
Ralph Wiggum self-testing loop — iterative optimization of Crystal's detection.

Runs generated questions through the Crystal pipeline, scores results,
and proposes modifications to the detector's pattern tables until a
fitness threshold is met.

Constrained action space (what the loop can modify):
  - QUESTION_PREDICATE_MAP entries in detectors/kg.py
  - LEGAL_PREDICATE_ALIASES entries in data/legal_ontology.py

What it cannot modify:
  - Core algorithms (find_entity_spans, extract_predicate_phrase)
  - Compiler logic
  - Graph data structures
  - The fitness scoring function (binary_correct, fitness_score)

Autoresearch pattern (adapted from karpathy/autoresearch):
  - Changes on a ralph/* git branch
  - Each iteration: LLM proposes dict additions, commit, evaluate
  - If score improved: keep commit, advance branch
  - If worse: git reset --hard to previous good commit
  - No human approval required

Usage:
    from benchmarks.ralph_wiggum import RalphWiggumLoop
    loop = RalphWiggumLoop(kg=legal_kg, cases=generated_cases)
    result = loop.run(threshold=0.90, max_iterations=10)

CLI:
    python -m benchmarks.ralph_wiggum --threshold 0.90
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

import spacy

from benchmarks.scoring.fitness import binary_correct, fitness_score
from crystal.detectors.kg import detect_kg_query
from crystal.nodes.compiler import _format_kg_results

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent / "src"
_PREDICATE_MAP_FILE = _SRC_ROOT / "crystal" / "detectors" / "kg.py"
_ALIAS_FILE = _SRC_ROOT / "crystal" / "data" / "legal_ontology.py"

RALPH_MODEL = os.environ.get("RALPH_MODEL", "claude-sonnet-4-20250514")


def call_ralph_llm(prompt: str) -> tuple[str, dict | None]:
    """Call Anthropic Claude for code-modification proposals.

    Separated from crystal.llm (Gemini Flash) because code modification
    requires a stronger model. Set RALPH_MODEL and ANTHROPIC_API_KEY in .env.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or export it. "
            "Ralph Wiggum needs an Anthropic API key for code proposals."
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


_PROPOSAL_PROMPT = """\
You are optimizing a question-answering system's detection layer. The system maps \
natural language questions to knowledge graph lookups.

Below are questions that the system answered INCORRECTLY. For each failure, you see:
- The question asked
- The golden answer (correct answer)
- The predicate phrase the detector extracted (or None if it couldn't extract one)
- What the detector resolved it to (or None if it failed)
- Whether the question is a negative case (expects abstention)

FAILURES:
{failures}

CURRENT QUESTION_PREDICATE_MAP (maps extracted phrases → KG predicates):
{predicate_map}

CURRENT LEGAL_PREDICATE_ALIASES (maps surface forms → canonical predicates):
{aliases}

Your task: propose NEW entries (additions only, never deletions) to fix as many \
failures as possible.

Rules:
1. Only propose string→string mappings
2. Keys should be the extracted predicate phrase (or a likely phrase variant)
3. Values must be an existing KG predicate: court, date_filed, judges, disposition, \
cited_by_count, nature_of_suit, cites
4. Do NOT propose entries that already exist
5. For negative cases (is_negative=True), do NOT propose mappings — abstention is correct

Respond with ONLY a JSON object in this format:
```json
{{
  "predicate_map": {{"phrase": "kg_predicate", ...}},
  "predicate_aliases": {{"surface_form": "canonical_predicate", ...}}
}}
```

Include only sections that have new entries. If no useful additions can be made, \
respond with an empty JSON object: {{}}
"""


@dataclass
class IterationResult:
    """Result of a single loop iteration."""
    iteration: int
    score: float
    total: int
    correct: int
    failures: list[dict] = field(default_factory=list)
    proposed_changes: dict = field(default_factory=dict)


@dataclass
class LoopResult:
    """Final result of the Ralph Wiggum loop."""
    converged: bool
    final_score: float
    iterations_run: int
    best_score: float
    best_iteration: int
    history: list[IterationResult] = field(default_factory=list)


# ── Proposal validation and application ──────────────────────────────────


def _validate_proposal(proposal: dict | None) -> bool:
    """Validate an LLM-generated proposal. Additions only, string→string."""
    if not proposal or not isinstance(proposal, dict):
        return False

    allowed_sections = {"predicate_map", "predicate_aliases"}
    if not any(k in allowed_sections for k in proposal):
        return False

    for key in proposal:
        if key not in allowed_sections:
            return False
        section = proposal[key]
        if not isinstance(section, dict):
            return False
        for k, v in section.items():
            if not isinstance(k, str) or not isinstance(v, str):
                return False
            if not k.strip() or not v.strip():
                return False

    return True


def _parse_llm_proposal(raw: str) -> dict | None:
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


def _apply_proposal(
    proposal: dict,
    *,
    predicate_map_file: Path | None = None,
    alias_file: Path | None = None,
) -> dict[str, int]:
    """Apply validated proposal by inserting entries into target files.

    Returns counts of entries actually inserted per section.
    """
    pm_file = predicate_map_file or _PREDICATE_MAP_FILE
    al_file = alias_file or _ALIAS_FILE
    counts: dict[str, int] = {}

    if "predicate_map" in proposal and proposal["predicate_map"]:
        inserted = _insert_dict_entries(
            pm_file,
            "QUESTION_PREDICATE_MAP",
            proposal["predicate_map"],
        )
        counts["predicate_map"] = inserted

    if "predicate_aliases" in proposal and proposal["predicate_aliases"]:
        inserted = _insert_dict_entries(
            al_file,
            "LEGAL_PREDICATE_ALIASES",
            proposal["predicate_aliases"],
        )
        counts["predicate_aliases"] = inserted

    return counts


def _insert_dict_entries(
    filepath: Path,
    dict_name: str,
    entries: dict[str, str],
) -> int:
    """Insert new key-value pairs into a Python dict literal in a file.

    Finds the closing brace of the named dict and inserts entries before it.
    Skips entries whose keys already exist in the file.
    """
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


# ── Git operations ───────────────────────────────────────────────────────


def _git_cmd(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd or Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_commit(message: str) -> str:
    """Stage target files and commit. Returns short hash."""
    _git_cmd("add", str(_PREDICATE_MAP_FILE), str(_ALIAS_FILE))
    _git_cmd("commit", "-m", message)
    return _git_cmd("rev-parse", "--short", "HEAD")


def _git_reset_hard(ref: str) -> None:
    """Hard reset to a given ref."""
    _git_cmd("reset", "--hard", ref)


# ── Core loop ────────────────────────────────────────────────────────────


class RalphWiggumLoop:
    """The self-testing loop that optimizes Crystal's detection patterns."""

    def __init__(
        self,
        kg,
        cases: list[tuple[str, str, list[str], bool]],
        nlp=None,
        on_iteration: Callable[[IterationResult], None] | None = None,
        call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
        use_git: bool = False,
    ) -> None:
        self.kg = kg
        self.cases = cases
        self.nlp = nlp or spacy.load("en_core_web_sm")
        self.on_iteration = on_iteration
        self.call_llm_fn = call_llm_fn
        self.use_git = use_git
        self.results_log: list[dict] = []

    def _run_single_case(
        self,
        question: str,
        match_strings: list[str],
        is_negative: bool,
    ) -> tuple[str, bool, dict | None]:
        """Run a single question through the detector and format a response."""
        doc = self.nlp(question)
        detection = detect_kg_query(doc, self.kg)

        if detection is None:
            if is_negative:
                response = "No match found in the knowledge graph."
            else:
                response = "No KG match found."
        else:
            response = _format_kg_results([{
                "tool": "kg",
                "success": True,
                "results": detection["results"],
            }])

        correct = binary_correct(response, match_strings, is_negative)
        return response, correct, detection

    def _analyze_failures(
        self,
        results: list[dict],
    ) -> list[dict]:
        """Identify and categorize failures for diagnostic output."""
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
            if detection:
                failure["detected_entity"] = detection.get("entity", "")
                failure["lookup_type"] = detection.get("lookup_type", "")
                failure["predicate_phrase"] = detection.get("predicate_phrase", "")
                failure["n_results"] = len(detection.get("results", []))
            else:
                failure["detection"] = None
            failures.append(failure)
        return failures

    def run_iteration(self, iteration: int) -> IterationResult:
        """Run all cases and return scored results."""
        results: list[dict] = []

        for question, golden_answer, match_strings, is_negative in self.cases:
            response, correct, detection = self._run_single_case(
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
            })

        correct_count = sum(1 for r in results if r["correct"])
        total = len(results)
        score = correct_count / total if total > 0 else 0.0
        failures = self._analyze_failures(results)

        return IterationResult(
            iteration=iteration,
            score=score,
            total=total,
            correct=correct_count,
            failures=failures,
        )

    def _propose_changes(self, failures: list[dict]) -> dict | None:
        """Ask the LLM to propose dict additions based on failures."""
        if not self.call_llm_fn:
            return None

        if not failures:
            return None

        from crystal.detectors.kg import QUESTION_PREDICATE_MAP
        from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES

        failure_text = ""
        for f in failures[:15]:
            failure_text += (
                f"  Question: {f['question']}\n"
                f"  Golden answer: {f['golden_answer']}\n"
                f"  Predicate phrase: {f.get('predicate_phrase', 'None')}\n"
                f"  Resolved to: {f.get('lookup_type', 'None')}\n"
                f"  Is negative: {f['is_negative']}\n"
                f"  ---\n"
            )

        prompt = _PROPOSAL_PROMPT.format(
            failures=failure_text,
            predicate_map=json.dumps(QUESTION_PREDICATE_MAP, indent=2),
            aliases=json.dumps(
                {k: v for k, v in LEGAL_PREDICATE_ALIASES.items()},
                indent=2,
            ),
        )

        try:
            response_text, _ = self.call_llm_fn(prompt)
            proposal = _parse_llm_proposal(response_text)
            if proposal and _validate_proposal(proposal):
                return proposal
            logger.info("LLM proposal invalid or empty")
        except Exception as e:
            logger.warning("LLM proposal failed: %s", e)

        return None

    def run(
        self,
        threshold: float = 0.90,
        max_iterations: int = 20,
    ) -> LoopResult:
        """Run the autonomous self-improvement loop.

        With call_llm_fn set: proposes and applies changes, uses git for
        keep/discard (if use_git=True).

        Without call_llm_fn: evaluation-only mode (breaks after 1 iteration
        since score won't change).
        """
        history: list[IterationResult] = []
        best_score = 0.0
        best_iteration = 0
        consecutive_discards = 0
        baseline_ref = None

        if self.use_git:
            baseline_ref = _git_cmd("rev-parse", "--short", "HEAD")

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
                "Iteration %d: %.1f%% (%d/%d correct, %d failures)",
                i, result.score * 100, result.correct, result.total,
                len(result.failures),
            )

            self.results_log.append({
                "iteration": i,
                "score": result.score,
                "correct": result.correct,
                "total": result.total,
                "failures": len(result.failures),
            })

            if self.on_iteration:
                self.on_iteration(result)

            if result.score >= threshold:
                logger.info("Converged at %.1f%% (threshold %.1f%%)", result.score * 100, threshold * 100)
                return LoopResult(
                    converged=True,
                    final_score=result.score,
                    iterations_run=i + 1,
                    best_score=best_score,
                    best_iteration=best_iteration,
                    history=history,
                )

            if not self.call_llm_fn:
                if i > 0 and result.score == history[i - 1].score:
                    logger.info("Score unchanged, no LLM available for modifications. Stopping.")
                    break
                continue

            if consecutive_discards >= 3:
                logger.info("3 consecutive discards — exhausted useful mutations. Stopping.")
                break

            proposal = self._propose_changes(result.failures)
            if not proposal:
                logger.info("No proposal from LLM, stopping.")
                break

            pre_change_ref = None
            if self.use_git:
                pre_change_ref = _git_cmd("rev-parse", "--short", "HEAD")

            counts = _apply_proposal(proposal)
            total_applied = sum(counts.values())
            if total_applied == 0:
                logger.info("Proposal had no new entries to apply.")
                continue

            result.proposed_changes = proposal
            desc = ", ".join(f"{k}: +{v}" for k, v in counts.items())
            logger.info("Applied: %s", desc)

            if self.use_git:
                commit_hash = _git_commit(
                    f"ralph: {desc} (iter {i}, score {result.score:.3f})"
                )
                self.results_log[-1]["commit"] = commit_hash
                self.results_log[-1]["status"] = "pending"

            next_result = self.run_iteration(i + 1)
            if next_result.score > result.score:
                logger.info(
                    "Score improved: %.1f%% → %.1f%% — keeping changes",
                    result.score * 100, next_result.score * 100,
                )
                if self.use_git:
                    self.results_log[-1]["status"] = "keep"
            else:
                logger.info(
                    "Score did not improve: %.1f%% → %.1f%% — reverting",
                    result.score * 100, next_result.score * 100,
                )
                if self.use_git and pre_change_ref:
                    _git_reset_hard(pre_change_ref)
                    self.results_log[-1]["status"] = "discard"
                else:
                    _revert_proposal(proposal)

        final = history[-1] if history else IterationResult(0, 0.0, 0, 0)
        return LoopResult(
            converged=final.score >= threshold,
            final_score=final.score,
            iterations_run=len(history),
            best_score=best_score,
            best_iteration=best_iteration,
            history=history,
        )


def _revert_proposal(proposal: dict) -> None:
    """Remove entries that were just added (non-git revert path)."""
    if "predicate_map" in proposal:
        _remove_dict_entries(
            _PREDICATE_MAP_FILE,
            "QUESTION_PREDICATE_MAP",
            proposal["predicate_map"],
        )
    if "predicate_aliases" in proposal:
        _remove_dict_entries(
            _ALIAS_FILE,
            "LEGAL_PREDICATE_ALIASES",
            proposal["predicate_aliases"],
        )


def _remove_dict_entries(
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


# ── CLI ──────────────────────────────────────────────────────────────────


def _write_results_tsv(results: list[dict], path: Path) -> None:
    """Write results log to TSV file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("iteration\tscore\tcorrect\ttotal\tfailures\tcommit\tstatus\n")
        for r in results:
            f.write(
                f"{r['iteration']}\t{r['score']:.4f}\t{r['correct']}\t"
                f"{r['total']}\t{r['failures']}\t"
                f"{r.get('commit', '')}\t{r.get('status', '')}\n"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ralph Wiggum self-improvement loop",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Fitness score threshold for convergence (default: 0.90)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=20,
        help="Maximum iterations (default: 20)",
    )
    parser.add_argument(
        "--review-dir", default=None,
        help="Path to review directory with accepted cases",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="Path to SQLite KG database",
    )
    parser.add_argument(
        "--use-git", action="store_true",
        help="Use git branching (create ralph/* branch, commit/revert)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluation only, no LLM proposals",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from crystal.review import collect_accepted_cases, REVIEW_DIR
    from crystal.tools.kg.store import SqliteKnowledgeGraph

    review_dir = Path(args.review_dir) if args.review_dir else REVIEW_DIR
    cases = collect_accepted_cases(review_dir)

    if not cases:
        from benchmarks.ground_truth.legal import LEGAL_BENCHMARK_CASES
        logger.warning(
            "No accepted cases found in %s. Falling back to LEGAL_BENCHMARK_CASES (%d cases).",
            review_dir, len(LEGAL_BENCHMARK_CASES),
        )
        cases = LEGAL_BENCHMARK_CASES

    logger.info("Loaded %d test cases", len(cases))

    if args.db_path:
        kg = SqliteKnowledgeGraph(args.db_path)
    else:
        from crystal.tools.kg.legal import build_legal_kg_memory
        from tests.fixtures.scotus_sample import SCOTUS_SAMPLE
        logger.info("No --db-path provided, building in-memory KG from SCOTUS sample fixtures")
        kg = build_legal_kg_memory(SCOTUS_SAMPLE)

    call_llm_fn = None
    if not args.dry_run:
        try:
            call_ralph_llm("ping")
            call_llm_fn = call_ralph_llm
            logger.info(
                "Anthropic API available (model: %s) — will propose code changes",
                RALPH_MODEL,
            )
        except Exception as e:
            logger.warning("Anthropic API not available (%s) — running in evaluation-only mode", e)

    if args.use_git:
        from datetime import datetime
        branch_name = f"ralph/{datetime.now().strftime('%Y-%m-%d')}"
        current = _git_cmd("branch", "--show-current")
        if not current.startswith("ralph/"):
            _git_cmd("checkout", "-b", branch_name)
            logger.info("Created branch: %s", branch_name)

    loop = RalphWiggumLoop(
        kg=kg,
        cases=cases,
        call_llm_fn=call_llm_fn,
        use_git=args.use_git,
    )
    result = loop.run(
        threshold=args.threshold,
        max_iterations=args.max_iterations,
    )

    results_path = Path(__file__).parent / "ralph_results.tsv"
    _write_results_tsv(loop.results_log, results_path)

    print(f"\n--- Ralph Wiggum Summary ---")
    print(f"Converged:       {result.converged}")
    print(f"Final score:     {result.final_score:.1%}")
    print(f"Best score:      {result.best_score:.1%} (iteration {result.best_iteration})")
    print(f"Iterations run:  {result.iterations_run}")
    print(f"Results log:     {results_path}")

    if hasattr(kg, "close"):
        kg.close()


if __name__ == "__main__":
    main()
