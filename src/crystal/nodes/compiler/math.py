"""Math-specific prompt compilation — formatting and simplification."""

from __future__ import annotations

import numpy as np


def _format_result(result_val) -> str:
    """Format a numeric result for direct display."""
    if isinstance(result_val, (np.integer, int)):
        return str(int(result_val))
    if isinstance(result_val, float) and result_val == int(result_val):
        return str(int(result_val))
    return str(result_val)


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
