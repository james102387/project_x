"""KG-specific prompt compilation — formatting and augmentation."""

from __future__ import annotations


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
