"""KG detection node — runs entity matching against loaded knowledge graphs."""

from crystal.detectors.kg import detect_kg_query
from crystal.tools.kg import remulak_kg


def kg_detection_node(state: dict) -> dict:
    """
    Scan the prompt for known KG entities with question structure.

    Runs after math detection. Always fires regardless of whether math
    detections exist — the planner merges both detection types.

    Uses state["kg"] if provided, otherwise falls back to remulak_kg.
    """
    detections = list(state.get("tool_detections", []))

    doc = state["spacy_doc"]
    kg = state.get("kg") or remulak_kg
    detection = detect_kg_query(doc, kg)

    kg_detections = []
    kg_entities_found = []

    if detection is not None:
        detections.append(detection)
        kg_detections.append(detection)
        kg_entities_found = [
            s["entity"] for s in detection.get("entity_spans", [])
        ]

    return {
        "tool_detections": detections,
        "kg_detections": kg_detections,
        "kg_entities_found": kg_entities_found,
    }
