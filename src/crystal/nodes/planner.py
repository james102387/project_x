"""Plan builder node — converts tool detections into an execution plan."""


def plan_builder_node(state: dict) -> dict:
    """
    Build a compiler plan from tool detections.
    MVP: one detection → one plan item. Empty detections → empty plan.
    """
    detections = state.get("tool_detections", [])

    if not detections:
        return {"plan": [], "fallback_to_llm": True}

    plan = []
    for detection in detections:
        if detection["tool"] == "kg":
            plan.append({
                "tool": "kg",
                "operation": "lookup",
                "entity": detection["entity"],
                "results": detection["results"],
            })
        else:
            entry = {
                "tool": detection["tool"],
                "operation": detection["operation"],
                "args": detection["raw_args"],
            }
            if detection["operation"] == "semantic_math":
                entry["steps"] = detection.get("steps", [])
                entry["result"] = detection.get("result")
            plan.append(entry)

    return {"plan": plan, "fallback_to_llm": False}
