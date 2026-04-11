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


def generate_questions_from_triplets(triplets: list[tuple[str, str, str]], max_questions: int = 5) -> list[str]:
    """Generate simple questions from extracted triplets for comparison."""
    questions = []
    seen_subjects = set()

    _PRED_TEMPLATES = {
        "court": "What court decided {subject}?",
        "date_filed": "When was {subject} decided?",
        "judges": "Who were the judges in {subject}?",
        "opinion_author": "Who wrote the opinion in {subject}?",
        "cites": "What cases does {subject} cite?",
        "attorneys": "Who were the attorneys in {subject}?",
        "disposition": "What was the outcome of {subject}?",
    }

    for subj, pred, obj in triplets:
        if len(questions) >= max_questions:
            break
        if subj in seen_subjects:
            continue

        template = _PRED_TEMPLATES.get(pred.lower())
        if template:
            questions.append(template.format(subject=subj.title()))
            seen_subjects.add(subj)
        elif pred and subj:
            questions.append(f"What is the {pred} of {subj.title()}?")
            seen_subjects.add(subj)

    return questions
