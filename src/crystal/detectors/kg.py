"""
Knowledge Graph detector — matches prompts against KG entities.

Two-stage detection:
1. Entity recognition: scan tokens for spans matching known KG entities
2. Question structure: confirm the prompt is actually asking about the entity
   (WH-word, interrogative structure, or imperative like "tell me about")

Predicate extraction: attempts to pull a predicate phrase from the question
to do a targeted lookup rather than dumping all facts.
"""

from __future__ import annotations

from crystal.tools.kg.graph import KnowledgeGraph


QUESTION_WORDS = {"what", "who", "where", "when", "which", "how", "whose", "whom"}
REQUEST_VERBS = {"tell", "describe", "explain", "show", "list", "give", "name"}

# Words to strip when extracting predicate phrases from questions
NOISE_WORDS = {
    "what", "who", "where", "when", "which", "how", "whose", "whom",
    "is", "are", "was", "were", "the", "a", "an",
    "do", "does", "did", "tell", "me", "about", "describe", "show",
    "?", "'s", "s",
}


def find_entity_spans(doc, kg: KnowledgeGraph) -> list[dict]:
    """Find token spans in the doc that match known KG entities.

    Tries longest match first (multi-word entities like "Grand Vizier Korth"
    before single-word entities like "Korth").
    """
    text_lower = doc.text.lower()
    matches = []
    matched_ranges: list[tuple[int, int]] = []

    sorted_entities = sorted(kg.entities, key=len, reverse=True)

    for entity in sorted_entities:
        start = 0
        while True:
            idx = text_lower.find(entity, start)
            if idx == -1:
                break

            end = idx + len(entity)

            if _overlaps(idx, end, matched_ranges):
                start = end
                continue

            if idx > 0 and text_lower[idx - 1].isalnum():
                start = end
                continue
            if end < len(text_lower) and text_lower[end].isalnum():
                start = end
                continue

            matched_ranges.append((idx, end))
            matches.append({
                "entity": entity,
                "char_start": idx,
                "char_end": end,
            })
            start = end

    return matches


def _overlaps(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    for rs, re in ranges:
        if start < re and end > rs:
            return True
    return False


def has_question_structure(doc) -> bool:
    """Check whether the doc has an interrogative or request structure."""
    for token in doc:
        if token.lemma_.lower() in QUESTION_WORDS:
            return True
        if token.pos_ == "VERB" and token.lemma_.lower() in REQUEST_VERBS:
            return True
    if doc.text.rstrip().endswith("?"):
        return True
    return False


def extract_predicate_phrase(doc, entity_spans: list[dict]) -> str | None:
    """Extract a candidate predicate phrase from the question.

    Strategy: collect non-noise tokens that aren't part of an entity.
    Preserves prepositions like "of" that may be internal to a predicate
    (e.g., "head of state") but strips trailing "of" that connects to an entity.

    E.g., "What is the capital of Remulak?" → "capital"
          "Who is the head of state of Remulak?" → "head of state"
          "How old is Grand Vizier Korth?" → "old"
    """
    entity_ranges = [(s["char_start"], s["char_end"]) for s in entity_spans]

    tokens = []
    for token in doc:
        if token.pos_ in ("PUNCT", "SPACE"):
            continue

        token_start = token.idx
        token_end = token.idx + len(token.text)
        in_entity = any(
            token_start >= es and token_end <= ee
            for es, ee in entity_ranges
        )
        if in_entity:
            continue

        word = token.text.lower().rstrip("?.,!")
        if word in NOISE_WORDS:
            continue

        tokens.append(word)

    # Strip trailing prepositions ("of", "in") that connected to the entity
    while tokens and tokens[-1] in ("of", "in", "on", "for", "from", "by"):
        tokens.pop()

    phrase = " ".join(tokens).strip()
    return phrase if phrase else None


# Common question-word → predicate mappings for natural phrasing
QUESTION_PREDICATE_MAP = {
    "old": "age",
    "big": "diameter",
    "large": "diameter",
    "many people": "population",
    "many moons": "number of moons",
    "many continents": "number of continents",
    "many languages": "number of languages",
}


def detect_kg_query(
    doc,
    kg: KnowledgeGraph,
) -> dict | None:
    """Run KG detection against a spaCy doc.

    Returns a detection dict if the prompt matches a KG entity AND has
    question structure, otherwise None.
    """
    entity_spans = find_entity_spans(doc, kg)
    if not entity_spans:
        return None

    if not has_question_structure(doc):
        return None

    primary = entity_spans[0]
    entity_text = primary["entity"]

    predicate_phrase = extract_predicate_phrase(doc, entity_spans)

    if predicate_phrase:
        resolved = QUESTION_PREDICATE_MAP.get(predicate_phrase, predicate_phrase)
        results = kg.lookup(subject=entity_text, predicate=resolved)

        if not results and resolved != predicate_phrase:
            results = kg.lookup(subject=entity_text, predicate=predicate_phrase)

    if not predicate_phrase or not results:
        results = kg.lookup(subject=entity_text)

    if not results:
        return None

    return {
        "tool": "kg",
        "operation": "lookup",
        "entity": entity_text,
        "entity_spans": entity_spans,
        "predicate_phrase": predicate_phrase,
        "results": results,
        "matched_pattern": "entity_question",
    }
