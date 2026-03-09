"""LLM nodes — handle the final LLM call for augmented and fallback paths."""

import crystal.llm


def direct_return_node(state: dict) -> dict:
    """Pure math / math answerable — result already in final_response."""
    return {}


def llm_augmented_node(state: dict) -> dict:
    """Send the compiled (simplified) prompt to the LLM."""
    response, usage = crystal.llm.call_llm(state["compiled_prompt"])
    metrics = dict(state.get("token_metrics") or {})
    if usage:
        metrics["actual_prompt_tokens"] = usage["prompt_tokens"]
        metrics["actual_output_tokens"] = usage["output_tokens"]
    return {
        "llm_response": response,
        "final_response": response,
        "token_metrics": metrics,
    }


def llm_fallback_node(state: dict) -> dict:
    """Send the raw prompt directly to the LLM — no tool augmentation."""
    response, usage = crystal.llm.call_llm(state["raw_prompt"])
    metrics = dict(state.get("token_metrics") or {})
    if usage:
        metrics["actual_prompt_tokens"] = usage["prompt_tokens"]
        metrics["actual_output_tokens"] = usage["output_tokens"]
    return {
        "prompt_type": "no_math",
        "llm_response": response,
        "final_response": response,
        "token_metrics": metrics,
    }
