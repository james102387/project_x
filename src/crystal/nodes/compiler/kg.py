"""KG-specific prompt compilation — formatting and augmentation.

Prompt framing is qualified by grounding confidence:
  HIGH  (>= 0.9): "verified from the knowledge graph"
  MEDIUM (0.7–0.9): "possibly relevant context" — LLM uses own judgment
"""

from __future__ import annotations

from crystal.nodes.planner import CONFIDENCE_HIGH


def _format_kg_results(tool_results: list[dict]) -> str:
    """Format KG lookup results for direct display."""
    lines = []
    for r in tool_results:
        if not r.get("success") or r.get("tool") != "kg":
            continue
        for fact in r["results"]:
            lines.append(f"{fact['subject']} — {fact['predicate']}: {fact['object']}")
    return "\n".join(lines)


def _build_kg_augmented_prompt(
    raw_prompt: str,
    tool_results: list[dict],
    *,
    grounding_confidence: float = 1.0,
) -> str:
    """Build a prompt with grounded KG facts injected for LLM reasoning.

    Framing adapts to confidence level to avoid anchoring the LLM on
    uncertain facts — the "never worse than LLM" contract.
    """
    facts = []
    for r in tool_results:
        if not r.get("success") or r.get("tool") != "kg":
            continue
        for fact in r["results"]:
            facts.append(f"  {fact['subject']} — {fact['predicate']}: {fact['object']}")

    facts_block = chr(10).join(facts)

    if grounding_confidence >= CONFIDENCE_HIGH:
        return (
            f"The user asked: '{raw_prompt}'\n\n"
            f"The following facts have been verified from the knowledge graph:\n"
            f"{facts_block}\n\n"
            f"Answer the user's question using these grounded facts. "
            f"Do not invent information beyond what is provided."
        )

    return (
        f"The user asked: '{raw_prompt}'\n\n"
        f"The following facts were found in the knowledge graph and may be "
        f"relevant, but the match confidence is moderate. Use your own "
        f"judgment:\n"
        f"{facts_block}\n\n"
        f"Answer the user's question. If these facts conflict with what "
        f"you know, explain the discrepancy rather than blindly accepting "
        f"the grounding. Do not invent information."
    )
