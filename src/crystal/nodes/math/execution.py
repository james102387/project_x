"""Calculator execution node — runs math operations from preprocessed payloads."""

import numpy as np


def calculator_node(state: dict) -> dict:
    """Execute calculator operations. Supports explicit add and semantic_math."""
    preprocessed = state.get("preprocessed", [])
    tool_results = []

    for item in preprocessed:
        if item["tool"] != "calculator" or not item.get("ready"):
            tool_results.append({
                "tool": item["tool"],
                "success": False,
                "error": item.get("error", "Not ready for execution"),
            })
            continue

        if item["operation"] == "add":
            result = np.sum(item["args"])
            tool_results.append({
                "tool": "calculator",
                "operation": "add",
                "args": item["args"],
                "result": result,
                "success": True,
            })
        elif item["operation"] == "semantic_math":
            tool_results.append({
                "tool": "calculator",
                "operation": "semantic_math",
                "args": item["args"],
                "steps": item.get("steps", []),
                "result": item["result"],
                "success": True,
            })

    return {"tool_results": tool_results}
