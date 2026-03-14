"""KG execution node — runs KG lookups from preprocessed payloads."""


def kg_node(state: dict) -> dict:
    """Execute KG lookups. Results are already resolved at detection time."""
    preprocessed = state.get("preprocessed", [])
    tool_results = list(state.get("tool_results", []))
    kg_results = list(state.get("kg_results", []))

    for item in preprocessed:
        if item["tool"] != "kg" or not item.get("ready"):
            if item["tool"] == "kg":
                tool_results.append({
                    "tool": "kg",
                    "success": False,
                    "error": item.get("error", "Not ready for execution"),
                })
            continue

        result_entry = {
            "tool": "kg",
            "operation": "lookup",
            "entity": item["entity"],
            "results": item["results"],
            "success": True,
        }
        tool_results.append(result_entry)
        kg_results.append(result_entry)

    return {"tool_results": tool_results, "kg_results": kg_results}
