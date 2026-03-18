"""LLM nodes — handle the final LLM call for augmented and fallback paths."""

import crystal.llm


def _update_metrics_from_usage(metrics: dict, usage: dict | None) -> dict:
    """Merge API usage data into the metrics dict."""
    if not usage:
        return metrics
    metrics["actual_prompt_tokens"] = usage.get("prompt_tokens")
    metrics["actual_output_tokens"] = usage.get("output_tokens")
    metrics["actual_reasoning_tokens"] = usage.get("reasoning_tokens")
    metrics["actual_total_tokens"] = usage.get("total_tokens")
    metrics["actual_cached_tokens"] = usage.get("cached_tokens")
    return metrics


def direct_return_node(state: dict) -> dict:
    """Pure math / math answerable — result already in final_response."""
    return {}


def llm_augmented_node(state: dict) -> dict:
    """Send the compiled (simplified) prompt to the LLM."""
    response, usage = crystal.llm.call_llm(state["compiled_prompt"])
    metrics = dict(state.get("token_metrics") or {})
    _update_metrics_from_usage(metrics, usage)
    return {
        "llm_response": response,
        "final_response": response,
        "token_metrics": metrics,
    }


def llm_fallback_node(state: dict) -> dict:
    """Send the raw prompt directly to the LLM — no tool augmentation."""
    response, usage = crystal.llm.call_llm(state["raw_prompt"])
    metrics = dict(state.get("token_metrics") or {})
    _update_metrics_from_usage(metrics, usage)
    return {
        "prompt_type": "no_math",
        "llm_response": response,
        "final_response": response,
        "token_metrics": metrics,
    }
