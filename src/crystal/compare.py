"""Before/after comparison — demonstrates answer quality improvement post-ingestion.

Runs the same questions through:
  1. Crystal with the current (post-ingestion) KG
  2. Naked LLM with the document text in context
  3. Naked LLM without any document context

Returns a structured comparison table.

Also provides LLM-based question generation from extracted triplets.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from crystal.graph import build_crystal_graph
from crystal.ingest.question_gen import canonical_template
from crystal.ingest.validation import (
    _JUNK_SUBJECTS as VALIDATION_JUNK_SUBJECTS,
    validate_object,
    validate_subject,
)
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


def _is_plausible_case_name(subject: str) -> bool:
    """Check if a subject looks like a legal case name, not NER noise.

    Delegates to the shared validation module for the core checks,
    then applies additional question-generation-specific heuristics.
    Uses the canonical junk-subject set from `crystal.ingest.validation`.
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
        if left in VALIDATION_JUNK_SUBJECTS or right in VALIDATION_JUNK_SUBJECTS:
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
    """Generate questions from extracted triplets using canonical predicate templates.

    Only generates questions for predicates with a canonical template
    (see `crystal.ingest.question_gen.PREDICATE_QUESTION_FORMS`) and for
    subjects that look like real case names or legal entities.
    Allows multiple questions per subject (one per predicate).
    """
    questions = []
    seen = set()

    for subj, pred, obj in triplets:
        if len(questions) >= max_questions:
            break

        template = canonical_template(pred)
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


# ── LLM-based question generation ─────────────────────────────────────

# Stored as a module-level constant so the QuestionGenLoop can mutate it.
QUESTION_GEN_PROMPT = """
\
You are generating test questions from knowledge graph facts about legal cases.

Given the facts below about a legal entity, generate {max_questions} natural-language \
questions that can be answered directly from these facts. For each question, provide \
the golden answer (the correct answer derived from the facts).

Guidelines:
- Vary question style: "What did the court hold?", "Who wrote the opinion?", \
"When was the case decided?" — not just "What is the X of Y?"
- For holdings and doctrines, ask about legal principles, not the predicate name. \
Example: "What constitutional right did Gideon v. Wainwright establish?" \
instead of "What is the holding of Gideon v. Wainwright?"
- For reasoning, ask about rationale: "Why did the court rule that way?" or \
"What was the basis for the decision?"
- Each question must be answerable from the provided facts alone.
- Golden answers should be concise and factual, drawn directly from the object values.
- If there are fewer meaningful facts than {max_questions}, generate fewer questions.

Entity: {subject}

Facts:
{facts}

Respond with a JSON array of objects, each with "question" and "golden_answer" keys. \
Example: [{{"question": "What did the court hold in X v. Y?", "golden_answer": "The court held that..."}}]

JSON:
"""


def generate_questions_llm(
    triplets: list[tuple[str, str, str]],
    call_llm_fn: Callable[[str], tuple[str, dict | None]],
    *,
    max_questions: int = 10,
    max_per_subject: int = 3,
) -> list[dict]:
    """Generate questions from triplets using the LLM for natural phrasing.

    Groups triplets by subject, sends each group to the LLM with
    QUESTION_GEN_PROMPT, and parses out questions + golden answers.

    Falls back to template generation for any subject where the LLM
    call fails or returns unparseable output.

    Returns list of dicts with: question, golden_answer, source_triplet.
    """
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for subj, pred, obj in triplets:
        if not _is_plausible_case_name(subj):
            continue
        if not validate_object(pred, obj).valid:
            continue
        grouped[subj].append((subj, pred, obj))

    results: list[dict] = []

    for subj, facts in grouped.items():
        if len(results) >= max_questions:
            break

        facts_text = "\n".join(
            f"- {pred}: {obj}" for _, pred, obj in facts
        )
        prompt = QUESTION_GEN_PROMPT.format(
            subject=subj.title(),
            facts=facts_text,
            max_questions=min(max_per_subject, max_questions - len(results)),
        )

        try:
            response_text, _ = call_llm_fn(prompt)
            parsed = _parse_question_response(response_text)
        except Exception:
            logger.warning("LLM question gen failed for %s, using templates", subj)
            parsed = []

        if not parsed:
            for _, pred, obj in facts[:max_per_subject]:
                template = canonical_template(pred)
                if template:
                    parsed.append({
                        "question": template.format(subject=subj.title()),
                        "golden_answer": obj,
                    })

        for item in parsed[:max_per_subject]:
            if len(results) >= max_questions:
                break
            src = facts[0] if facts else ("", "", "")
            for _, pred, obj in facts:
                if obj.lower() in item.get("golden_answer", "").lower():
                    src = (subj, pred, obj)
                    break
            results.append({
                "question": item["question"],
                "golden_answer": item["golden_answer"],
                "source_triplet": list(src),
            })

    return results


def _parse_question_response(text: str) -> list[dict]:
    """Parse LLM response into list of {question, golden_answer} dicts."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    valid = []
    for item in items:
        if (isinstance(item, dict)
                and item.get("question")
                and item.get("golden_answer")):
            valid.append({
                "question": str(item["question"]).strip(),
                "golden_answer": str(item["golden_answer"]).strip(),
            })
    return valid
