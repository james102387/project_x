"""
Prompt compiler node — the core of Crystal.

Classifies the prompt into one of these types:
  pure_math:        "5 + 3"              → return result directly, skip LLM
  math_answerable:  "John has 10..."     → return result directly, skip LLM
  math_augmented:   "...is she wise?"    → inject result, simplify prompt for LLM
  kg_answerable:    "What is the capital of Remulak?" → return KG facts, skip LLM
  kg_augmented:     "Why is Remulak's government a technocratic council?" → inject KG facts, LLM reasons
  no_math:          (shouldn't reach here — plan_builder catches it)
"""

import numpy as np

from crystal.metrics import estimate_metrics
from crystal.detectors.math import (
    ADDITION_VERBS,
    ADDITION_CONJUNCTIONS,
    ADDITION_NOUNS,
    ADDITION_SYMBOLS,
    ALL_SEMANTIC_VERBS,
)


QUESTION_FILLER = {
    "what", "what's", "whats", "how", "much", "is", "are", "does",
    "do", "can", "you", "please", "find", "give", "me", "the",
    "calculate", "compute", "?", "equals", "equal", "'s",
}

REASONING_SIGNALS = {
    # advisory
    "should", "wise", "wisely", "better", "recommend", "advice", "enough",
    # explanatory
    "why", "explain", "reason", "because",
    # comparative
    "compare", "which", "prefer",
    # predictive
    "will", "would", "could", "future", "next",
}


def _has_reasoning_signals(doc) -> bool:
    """Check if the spaCy doc contains tokens that demand LLM reasoning."""
    for token in doc:
        if token.lemma_.lower() in REASONING_SIGNALS:
            return True
    return False


def _classify_prompt_type(raw_prompt: str, doc, tool_results: list[dict]) -> str:
    """
    Determine how to handle a prompt with successful tool results.

    pure_math:        every token is a number, math keyword, or question filler.
    math_answerable:  narrative framing around math, but no reasoning signals.
    math_augmented:   narrative math with reasoning signals requiring LLM.
    kg_answerable:    KG lookup returned results — return facts directly.
    no_math:          no successful tool results.
    """
    if not tool_results or not any(r.get("success") for r in tool_results):
        return "no_math"

    kg_results = [r for r in tool_results if r.get("tool") == "kg" and r.get("success")]
    if kg_results:
        if _has_reasoning_signals(doc):
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


def _format_result(result_val) -> str:
    """Format a numeric result for direct display."""
    if isinstance(result_val, (np.integer, int)):
        return str(int(result_val))
    if isinstance(result_val, float) and result_val == int(result_val):
        return str(int(result_val))
    return str(result_val)


def _format_kg_results(tool_results: list[dict]) -> str:
    """Format KG lookup results for direct display."""
    lines = []
    for r in tool_results:
        if not r.get("success") or r.get("tool") != "kg":
            continue
        for fact in r["results"]:
            lines.append(f"{fact['subject']} — {fact['predicate']}: {fact['object']}")
    return "\n".join(lines)


def _build_kg_augmented_prompt(raw_prompt: str, tool_results: list[dict]) -> str:
    """Build a prompt with grounded KG facts injected for LLM reasoning."""
    facts = []
    for r in tool_results:
        if not r.get("success") or r.get("tool") != "kg":
            continue
        for fact in r["results"]:
            facts.append(f"  {fact['subject']} — {fact['predicate']}: {fact['object']}")

    return (
        f"The user asked: '{raw_prompt}'\n\n"
        f"The following facts have been verified from the knowledge graph:\n"
        f"{chr(10).join(facts)}\n\n"
        f"Answer the user's question using these grounded facts. "
        f"Do not invent information beyond what is provided."
    )


def _build_simplified_prompt(raw_prompt: str, tool_results: list[dict]) -> str:
    """Build a simplified prompt with computed results injected."""
    evaluations = []
    for r in tool_results:
        if not r.get("success"):
            continue

        if r["operation"] == "semantic_math":
            steps = r.get("steps", [])
            step_parts = []
            for s in steps:
                if s["op"] == "state":
                    step_parts.append(f"start with {s['value']}")
                elif s["op"] == "add":
                    step_parts.append(f"+ {s['value']} ({s['verb']})")
                elif s["op"] == "subtract":
                    step_parts.append(f"- {s['value']} ({s['verb']})")
            evaluations.append(f"{', '.join(step_parts)} = {r['result']}")
        else:
            args_str = " + ".join(str(a) for a in r["args"])
            evaluations.append(f"{args_str} = {r['result']}")

    return (
        f"The user asked: '{raw_prompt}'\n\n"
        f"The following has been computed with verified precision:\n"
        f"  {'; '.join(evaluations)}\n\n"
        f"Answer the user's question using the computed result. "
        f"Do not recalculate."
    )


def prompt_compiler_node(state: dict) -> dict:
    """Classify the prompt and build the appropriate output."""
    raw = state["raw_prompt"]
    doc = state["spacy_doc"]
    tool_results = state.get("tool_results", [])

    prompt_type = _classify_prompt_type(raw, doc, tool_results)

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
        compiled = _build_kg_augmented_prompt(raw, tool_results)
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
