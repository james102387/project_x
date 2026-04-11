"""Prompt classification and the main compiler node."""

from __future__ import annotations

from crystal.metrics import estimate_metrics
from crystal.detectors.math import (
    ADDITION_VERBS,
    ADDITION_CONJUNCTIONS,
    ADDITION_NOUNS,
    ADDITION_SYMBOLS,
    ALL_SEMANTIC_VERBS,
)
from crystal.nodes.compiler.kg import _format_kg_results, _build_kg_augmented_prompt
from crystal.nodes.compiler.math import _format_result, _build_simplified_prompt
from crystal.nodes.planner import CONFIDENCE_HIGH


QUESTION_FILLER = {
    "what", "what's", "whats", "how", "much", "is", "are", "does",
    "do", "can", "you", "please", "find", "give", "me", "the",
    "calculate", "compute", "?", "equals", "equal", "'s",
}

REASONING_SIGNALS = {
    "should", "wise", "wisely", "better", "recommend", "advice", "enough",
    "why", "explain", "reason", "because",
    "compare", "which", "prefer",
    "will", "would", "could", "future", "next",
    "total", "combined", "sum", "together", "overall", "aggregate",
}


def _has_reasoning_signals(doc) -> bool:
    """Check if the spaCy doc contains tokens that demand LLM reasoning."""
    for token in doc:
        if token.lemma_.lower() in REASONING_SIGNALS:
            return True
    return False


def _classify_prompt_type(
    raw_prompt: str,
    doc,
    tool_results: list[dict],
    *,
    grounding_confidence: float = 1.0,
) -> str:
    """Determine how to handle a prompt with successful tool results.

    pure_math:        every token is a number, math keyword, or question filler.
    math_answerable:  narrative framing around math, but no reasoning signals.
    math_augmented:   narrative math with reasoning signals requiring LLM.
    kg_answerable:    KG lookup returned results — return facts directly.
    kg_augmented:     KG facts + reasoning signals — inject facts for LLM.
    no_math:          no successful tool results.

    Medium-confidence KG grounding forces kg_augmented so the LLM can
    cross-check the facts instead of returning them blindly.
    """
    if not tool_results or not any(r.get("success") for r in tool_results):
        return "no_math"

    kg_results = [r for r in tool_results if r.get("tool") == "kg" and r.get("success")]
    if kg_results:
        has_subject_scan = any(
            r.get("lookup_type") == "subject_scan" for r in kg_results
        )
        if grounding_confidence < CONFIDENCE_HIGH:
            return "kg_augmented"
        if _has_reasoning_signals(doc) or has_subject_scan:
            return "kg_augmented"
        return "kg_answerable"

    math_keywords = ADDITION_VERBS | ADDITION_CONJUNCTIONS | ADDITION_NOUNS | ADDITION_SYMBOLS
    all_filler = QUESTION_FILLER | math_keywords

    has_content_words = False
    for token in doc:
        if token.pos_ in ("PUNCT", "SPACE", "NUM", "SYM"):
            continue
        if token.text.lower() in all_filler or token.lemma_.lower() in all_filler:
            continue
        if token.pos_ == "CCONJ" or (
            token.pos_ in ("ADP", "PART") and token.text.lower() in ("to", "of")
        ):
            continue
        has_content_words = True
        break

    if not has_content_words:
        return "pure_math"

    if _has_reasoning_signals(doc):
        return "math_augmented"

    return "math_answerable"


def prompt_compiler_node(state: dict) -> dict:
    """Classify the prompt and build the appropriate output."""
    raw = state["raw_prompt"]
    doc = state["spacy_doc"]
    tool_results = state.get("tool_results", [])
    grounding_confidence = state.get("grounding_confidence", 1.0)

    prompt_type = _classify_prompt_type(
        raw, doc, tool_results, grounding_confidence=grounding_confidence,
    )

    if prompt_type == "kg_answerable":
        display = _format_kg_results(tool_results)
        metrics = estimate_metrics(raw, "", prompt_type)
        return {
            "prompt_type": prompt_type,
            "compiled_prompt": "",
            "final_response": display,
            "token_metrics": metrics.to_dict(),
        }

    if prompt_type == "kg_augmented":
        compiled = _build_kg_augmented_prompt(
            raw, tool_results, grounding_confidence=grounding_confidence,
        )
        metrics = estimate_metrics(raw, compiled, prompt_type)
        return {
            "prompt_type": "kg_augmented",
            "compiled_prompt": compiled,
            "final_response": "",
            "token_metrics": metrics.to_dict(),
        }

    if prompt_type in ("pure_math", "math_answerable"):
        results = [r for r in tool_results if r.get("success")]
        if results:
            display = _format_result(results[0]["result"])
            metrics = estimate_metrics(raw, "", prompt_type)
            return {
                "prompt_type": prompt_type,
                "compiled_prompt": "",
                "final_response": display,
                "token_metrics": metrics.to_dict(),
            }

    elif prompt_type == "math_augmented":
        compiled = _build_simplified_prompt(raw, tool_results)
        metrics = estimate_metrics(raw, compiled, prompt_type)
        return {
            "prompt_type": "math_augmented",
            "compiled_prompt": compiled,
            "final_response": "",
            "token_metrics": metrics.to_dict(),
        }

    metrics = estimate_metrics(raw, raw, "no_math")
    return {
        "prompt_type": "no_math",
        "compiled_prompt": raw,
        "final_response": "",
        "token_metrics": metrics.to_dict(),
    }
