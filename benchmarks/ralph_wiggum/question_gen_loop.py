"""QuestionGenLoop — mutates QUESTION_GEN_PROMPT in compare.py.

Evaluates question generation quality by checking whether Crystal can
correctly answer generated questions from the source triplets (self-
consistency). Proposes prompt improvements when questions are unanswerable,
too vague, or produce garbage golden answers.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

from benchmarks.ralph_wiggum.base import (
    BaseLoop,
    FailureCategory,
    IterationResult,
    parse_llm_proposal,
)
from benchmarks.scoring.fitness import binary_correct

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent.parent / "src"
_COMPARE_FILE = _SRC_ROOT / "crystal" / "compare.py"

QUESTION_QUALITY = "question_quality"
NO_QUESTIONS_GENERATED = "no_questions_generated"


class QuestionGenLoop(BaseLoop):
    """Ralph Wiggum loop that improves QUESTION_GEN_PROMPT.

    Instead of evaluating Crystal's QA accuracy on fixed cases,
    this loop:
      1. Generates questions from sample triplets using the current prompt
      2. Runs each through Crystal
      3. Checks if Crystal's answer matches the generated golden answer
      4. Proposes prompt changes for failures
    """

    LOOP_NAME = "QuestionGenLoop"
    FAILURE_CATEGORIES = frozenset({QUESTION_QUALITY, NO_QUESTIONS_GENERATED})
    TARGET_FILES = [_COMPARE_FILE]

    def __init__(
        self,
        kg,
        cases: list[tuple[str, str, list[str], bool]],
        *,
        sample_triplets: list[tuple[str, str, str]] | None = None,
        nlp=None,
        call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
        use_git: bool = False,
        use_full_pipeline: bool = True,
        mock_llm_fn=None,
    ) -> None:
        super().__init__(
            kg=kg,
            cases=cases,
            nlp=nlp,
            call_llm_fn=call_llm_fn,
            use_git=use_git,
            use_full_pipeline=use_full_pipeline,
            mock_llm_fn=mock_llm_fn,
        )
        self._sample_triplets = sample_triplets or self._extract_triplets_from_kg()
        self._original_prompt: str | None = None

    def _extract_triplets_from_kg(self) -> list[tuple[str, str, str]]:
        """Pull a sample of triplets from the KG for question generation."""
        triplets = []
        for subj in sorted(self.kg.subjects)[:20]:
            for fact in self.kg.lookup(subject=subj)[:3]:
                triplets.append((subj, fact["predicate"], fact["object"]))
        return triplets

    def run_iteration(self, iteration: int = 0) -> IterationResult:
        """Generate questions and evaluate self-consistency."""
        from crystal.compare import generate_questions_llm, generate_questions_from_triplets
        from crystal.graph import build_crystal_graph
        from crystal.state import make_initial_state

        if self.call_llm_fn and self._sample_triplets:
            generated = generate_questions_llm(
                self._sample_triplets,
                self.call_llm_fn,
                max_questions=15,
                max_per_subject=3,
            )
        else:
            questions = generate_questions_from_triplets(
                self._sample_triplets, max_questions=15,
            )
            generated = [
                {"question": q, "golden_answer": "", "source_triplet": []}
                for q in questions
            ]

        if not generated:
            logger.warning(
                "QuestionGenLoop generated 0 questions from %d triplets — "
                "treating as failure (score=0.0) so the loop can mutate the "
                "prompt instead of reporting vacuous success.",
                len(self._sample_triplets),
            )
            return IterationResult(
                iteration=iteration,
                score=0.0,
                total=1,
                correct=0,
                loop_name=self.LOOP_NAME,
                failures=[{
                    "question": "(no questions generated)",
                    "golden_answer": "",
                    "crystal_response": "",
                    "response": "",
                    "source_triplet": [],
                    "diagnosis": NO_QUESTIONS_GENERATED,
                }],
                diagnosis_summary={NO_QUESTIONS_GENERATED: 1},
            )

        graph = build_crystal_graph()
        correct_count = 0
        failures: list[dict] = []
        total = len(generated)

        for item in generated:
            q = item["question"]
            golden = item["golden_answer"]
            match_strings = [golden.lower()] if golden else []

            try:
                state = make_initial_state(q, kg=self.kg)
                result = graph.invoke(state)
                response = result.get("final_response", "")
            except Exception:
                response = ""

            correct = binary_correct(response, match_strings, False) if match_strings else False
            if correct:
                correct_count += 1
            else:
                failures.append({
                    "question": q,
                    "golden_answer": golden,
                    "crystal_response": response[:200],
                    "response": response[:200],
                    "source_triplet": item.get("source_triplet", []),
                    "diagnosis": QUESTION_QUALITY,
                })

        score = correct_count / total if total else 0.0
        diagnosis_summary = {
            QUESTION_QUALITY: len(failures),
            FailureCategory.CORRECT: correct_count,
        }

        return IterationResult(
            iteration=iteration,
            score=score,
            total=total,
            correct=correct_count,
            loop_name=self.LOOP_NAME,
            failures=failures,
            diagnosis_summary=diagnosis_summary,
        )

    def _build_proposal_prompt(self, failures: list[dict]) -> str:
        current_prompt = _read_question_gen_prompt()

        failure_text = ""
        for f in failures[:10]:
            failure_text += (
                f"  Question: {f['question']}\n"
                f"  Expected golden answer: {f['golden_answer']}\n"
                f"  Crystal's response: {f['crystal_response']}\n"
                f"  Source triplet: {f.get('source_triplet', [])}\n"
                f"  ---\n"
            )

        return f"""\
You are improving a question generation prompt for a legal knowledge graph QA system.

The current prompt generates questions from KG triplets, but some generated questions \
cannot be correctly answered by the QA system, or produce poor golden answers.

CURRENT QUESTION_GEN_PROMPT:
\"\"\"
{current_prompt}
\"\"\"

FAILURES (questions where Crystal's answer didn't match the golden answer):
{failure_text}

Propose an improved QUESTION_GEN_PROMPT that:
1. Generates more answerable questions (Crystal can find the answer in its KG)
2. Produces golden answers that match what the KG actually stores
3. Avoids overly abstract or vague questions
4. Generates questions about specific facts, not general knowledge

Respond with a JSON object:
```json
{{
  "question_gen_prompt": "the full improved prompt text..."
}}
```
"""

    def _validate_proposal(self, proposal: dict) -> bool:
        if not proposal or not isinstance(proposal, dict):
            return False
        prompt = proposal.get("question_gen_prompt")
        if not isinstance(prompt, str) or len(prompt) < 50:
            return False
        if "{subject}" not in prompt or "{facts}" not in prompt:
            return False
        return True

    def _apply_proposal(self, proposal: dict) -> dict[str, int]:
        self._original_prompt = _read_question_gen_prompt()
        new_prompt = proposal["question_gen_prompt"]
        _write_question_gen_prompt(new_prompt)
        return {"QUESTION_GEN_PROMPT": 1}

    def _revert_proposal(self, proposal: dict) -> None:
        if self._original_prompt is not None:
            _write_question_gen_prompt(self._original_prompt)
            self._original_prompt = None


def _read_question_gen_prompt() -> str:
    """Read the current QUESTION_GEN_PROMPT from compare.py."""
    content = _COMPARE_FILE.read_text(encoding="utf-8")
    match = re.search(
        r'QUESTION_GEN_PROMPT\s*=\s*"""(.*?)"""',
        content,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def _write_question_gen_prompt(new_prompt: str) -> None:
    """Replace QUESTION_GEN_PROMPT in compare.py."""
    content = _COMPARE_FILE.read_text(encoding="utf-8")
    new_content = re.sub(
        r'(QUESTION_GEN_PROMPT\s*=\s*""").*?(""")',
        rf'\1\n{new_prompt}\n\2',
        content,
        flags=re.DOTALL,
    )
    _COMPARE_FILE.write_text(new_content, encoding="utf-8")
