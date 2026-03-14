"""KG detection node — runs entity matching against loaded knowledge graphs."""

from crystal.detectors.kg import detect_kg_query
from crystal.tools.kg import remulak_kg


def kg_detection_node(state: dict) -> dict:
    """
    Scan the prompt for known KG entities with question structure.

    Runs after math detection. Only fires if no math detection was
    made (math takes priority for math queries).
    """
    detections = list(state.get("tool_detections", []))

    if detections:
        return {"tool_detections": detections}

    doc = state["spacy_doc"]
    detection = detect_kg_query(doc, remulak_kg)

    if detection is not None:
        detections.append(detection)

    return {"tool_detections": detections}
