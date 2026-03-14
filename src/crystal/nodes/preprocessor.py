"""Preprocessor node — transforms plan items into tool-ready payloads."""


def preprocessor_node(state: dict) -> dict:
    """Validate and format arguments for each tool in the plan."""
    plan = state.get("plan", [])
    preprocessed = []

    for plan_item in plan:
        if plan_item["tool"] == "calculator":
            args = plan_item["args"]
            if all(isinstance(a, (int, float)) for a in args):
                entry = {
                    "tool": "calculator",
                    "operation": plan_item["operation"],
                    "args": args,
                    "ready": True,
                }
                if plan_item["operation"] == "semantic_math":
                    entry["steps"] = plan_item.get("steps", [])
                    entry["result"] = plan_item.get("result")
                preprocessed.append(entry)
            else:
                preprocessed.append({
                    "tool": "calculator",
                    "operation": plan_item["operation"],
                    "args": args,
                    "ready": False,
                    "error": "Non-numeric arguments detected",
                })

        elif plan_item["tool"] == "kg":
            results = plan_item.get("results", [])
            preprocessed.append({
                "tool": "kg",
                "operation": "lookup",
                "entity": plan_item["entity"],
                "results": results,
                "ready": bool(results),
                **({"error": "No KG results found"} if not results else {}),
            })

    return {"preprocessed": preprocessed}
