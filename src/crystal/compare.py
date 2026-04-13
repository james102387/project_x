"""Before/after comparison — demonstrates answer quality improvement post-ingestion.

Runs the same questions through:
  1. Crystal with the current (post-ingestion) KG
  2. Naked LLM with the document text in context
  3. Naked LLM without any document context

Returns a structured comparison table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from crystal.graph import build_crystal_graph
from crystal.ingest.validation import validate_object, validate_subject
from crystal.state import make_initial_state

logger = logging.getLogger(__name__)

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_crystal_graph()
    return _graph


@dataclass
class ComparisonRow:
    question: str
    crystal_answer: str
    crystal_route: str
    llm_with_docs_answer: str
    llm_naked_answer: str


@dataclass
class ComparisonResult:
    rows: list[ComparisonRow] = field(default_factory=list)
    document_text: str = ""


def before_after_comparison(
    questions: list[str],
    kg,
    document_text: str = "",
    call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
) -> ComparisonResult:
    """Run questions through Crystal + KG, LLM+docs, and naked LLM.

    Args:
        questions: Questions to compare.
        kg: The post-ingestion KG to use with Crystal.
        document_text: Original document text to provide to LLM as context.
        call_llm_fn: LLM caller function.
    """
    if call_llm_fn is None:
        from crystal.llm import call_llm
        call_llm_fn = call_llm

    graph = _get_graph()
    result = ComparisonResult(document_text=document_text[:500])

    for q in questions:
        crystal_answer = ""
        crystal_route = ""
        try:
            state = make_initial_state(q, kg=kg)
            final = graph.invoke(state)
            crystal_answer = final.get("final_response", "")
            crystal_route = final.get("prompt_type", "unknown")
        except Exception as e:
            crystal_answer = f"Error: {e}"
            crystal_route = "error"

        llm_with_docs = ""
        if document_text:
            try:
                context_prompt = (
                    f"Based on the following document excerpt, answer this question:\n\n"
                    f"--- DOCUMENT ---\n{document_text[:4000]}\n--- END ---\n\n"
                    f"Question: {q}"
                )
                llm_with_docs, _ = call_llm_fn(context_prompt)
            except Exception as e:
                llm_with_docs = f"Error: {e}"
        else:
            llm_with_docs = "(no document text provided)"

        llm_naked = ""
        try:
            llm_naked, _ = call_llm_fn(q)
        except Exception as e:
            llm_naked = f"Error: {e}"

        result.rows.append(ComparisonRow(
            question=q,
            crystal_answer=crystal_answer,
            crystal_route=crystal_route,
            llm_with_docs_answer=llm_with_docs,
            llm_naked_answer=llm_naked,
        ))

    return result


_PRED_TEMPLATES = {
    "court": "What court decided {subject}?",
    "date_filed": "When was {subject} decided?",
    "judges": "Who were the judges in {subject}?",
    "opinion_author": "Who wrote the opinion in {subject}?",
    "cites": "What cases does {subject} cite?",
    "attorneys": "Who were the attorneys in {subject}?",
    "disposition": "What was the outcome of {subject}?",
    "cited_by_count": "What is the citation count for {subject}?",
    "precedential_status": "What is the precedential status of {subject}?",
    "per_curiam": "Was {subject} a per curiam decision?",
}


_JUNK_SUBJECTS = {
    "i", "he", "she", "it", "we", "they", "this", "that", "court",
    "courts", "state", "states", "law", "case", "cases", "the court",
    "this case", "the state", "defendant", "plaintiff", "petitioner",
    "respondent", "appellant", "appellee", "parties", "conclusions",
    "provisions", "progress", "burden", "equity", "presentations",
    "opinion", "dissent", "concurrence", "judgment", "order",
}

_JUNK_PREFIXES = ("this ", "the ", "said ", "such ", "that ")


def _is_plausible_case_name(subject: str) -> bool:
    """Check if a subject looks like a legal case name, not NER noise.

    Delegates to the shared validation module for the core checks,
    then applies additional question-generation-specific heuristics.
    """
    s = subject.strip().lower()
    if len(s) < 3 or len(s) > 120:
        return False

    vr = validate_subject(subject)
    if not vr.valid:
        return False

    if " v. " in s or " v " in s:
        parts = s.replace(" v. ", " v ").split(" v ", 1)
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ""
        if left in _JUNK_SUBJECTS or right in _JUNK_SUBJECTS:
            return False
        if len(left) < 2 or len(right) < 2:
            return False
        return True

    words = subject.strip().split()
    if len(words) > 6:
        return False
    if not any(w[0].isupper() for w in words if w):
        return False
    if s.startswith("justice ") or s.startswith("chief justice "):
        return False
    return True


def generate_questions_from_triplets(triplets: list[tuple[str, str, str]], max_questions: int = 5) -> list[str]:
    """Generate questions from extracted triplets using known predicate templates.

    Only generates questions for predicates we have templates for and subjects
    that look like real case names or legal entities.
    Allows multiple questions per subject (one per predicate).
    """
    questions = []
    seen = set()

    for subj, pred, obj in triplets:
        if len(questions) >= max_questions:
            break

        template = _PRED_TEMPLATES.get(pred.lower())
        if not template:
            continue

        if not _is_plausible_case_name(subj):
            continue

        if not validate_object(pred, obj).valid:
            continue

        key = (subj.lower(), pred.lower())
        if key in seen:
            continue
        seen.add(key)

        questions.append(template.format(subject=subj.title()))

    return questions
