"""
Knowledge Graph detector — matches prompts against KG entities.

Two-stage detection:
1. Entity recognition: scan tokens for spans matching known KG entities
   (exact substring → alias → fuzzy cascade)
2. Question structure: confirm the prompt is actually asking about the entity
   (WH-word, interrogative structure, or imperative like "tell me about")

Predicate extraction: attempts to pull a predicate phrase from the question
to do a targeted lookup rather than dumping all facts.
"""

from __future__ import annotations

from crystal.tools.kg.graph import KnowledgeGraph


QUESTION_WORDS = {"what", "who", "where", "when", "which", "how", "whose", "whom", "why"}
REQUEST_VERBS = {"tell", "describe", "explain", "show", "list", "give", "name"}

NOISE_WORDS = {
    "what", "who", "where", "when", "which", "how", "whose", "whom", "why",
    "is", "are", "was", "were", "the", "a", "an",
    "do", "does", "did", "tell", "me", "about", "describe", "show",
    "?", "'s", "s",
}


def find_entity_spans(doc, kg: KnowledgeGraph) -> list[dict]:
    """Find token spans in the doc that match known KG entities.

    Uses a 3-tier cascade: exact substring → alias → fuzzy.
    Tries longest match first (multi-word entities before single-word).
    """
    text_lower = doc.text.lower()
    matches = []
    matched_ranges: list[tuple[int, int]] = []

    # Tier 1: exact substring scan against entity index
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
                "match_tier": "exact",
                "match_score": 1.0,
                "original_text": text_lower[idx:end],
            })
            start = end

    if matches:
        return matches

    # Tier 2 & 3: alias / fuzzy — extract noun phrases from spaCy doc and
    # try the KG's entity resolution cascade on each.
    candidate_spans = _extract_candidate_spans(doc)

    for span_text, char_start, char_end in candidate_spans:
        resolved, tier = kg._resolve_entity(span_text)
        if tier == "none":
            continue
        if _overlaps(char_start, char_end, matched_ranges):
            continue

        # Reject fuzzy matches where length differs too much (derived forms
        # like "Remulakian" → "remulak" rather than genuine typos)
        if tier == "fuzzy":
            len_ratio = len(span_text) / max(len(resolved), 1)
            if len_ratio > 1.3 or len_ratio < 0.7:
                continue

        score = 1.0 if tier in ("exact", "alias") else 0.0
        if tier == "fuzzy":
            from crystal.tools.kg.fuzzy import fuzzy_match
            result = fuzzy_match(span_text, kg.entities, kg.fuzzy_threshold)
            score = result[1] if result else 0.0

        matched_ranges.append((char_start, char_end))
        matches.append({
            "entity": resolved,
            "char_start": char_start,
            "char_end": char_end,
            "match_tier": tier,
            "match_score": score,
            "original_text": span_text,
        })

    return matches


def _extract_candidate_spans(doc) -> list[tuple[str, int, int]]:
    """Extract candidate entity spans from spaCy noun chunks and named entities.

    Returns (text_lower, char_start, char_end) tuples, longest first.
    """
    candidates: list[tuple[str, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for ent in doc.ents:
        key = (ent.start_char, ent.end_char)
        if key not in seen:
            seen.add(key)
            candidates.append((ent.text.lower(), ent.start_char, ent.end_char))

    for chunk in doc.noun_chunks:
        key = (chunk.start_char, chunk.end_char)
        if key not in seen:
            seen.add(key)
            candidates.append((chunk.text.lower(), chunk.start_char, chunk.end_char))

    # Also try individual tokens that look like proper nouns
    for token in doc:
        if token.pos_ in ("PROPN", "NOUN") and len(token.text) > 2:
            key = (token.idx, token.idx + len(token.text))
            if key not in seen:
                seen.add(key)
                candidates.append((token.text.lower(), token.idx, token.idx + len(token.text)))

    candidates.sort(key=lambda c: c[2] - c[1], reverse=True)
    return candidates


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

    while tokens and tokens[-1] in ("of", "in", "on", "for", "from", "by"):
        tokens.pop()

    phrase = " ".join(tokens).strip()
    return phrase if phrase else None


QUESTION_PREDICATE_MAP = {
    "old": "age",
    "big": "diameter",
    "large": "diameter",
    "many people": "population",
    "many moons": "number of moons",
    "many continents": "number of continents",
    "many languages": "number of languages",
    "born": "birthplace",
    "known": "known for",
    "famous": "known for",
    "long last": "duration",
    "long": "duration",
    # legal
    "filed": "date_filed",
    "decided": "date_filed",
    "ruled": "disposition",
}


def detect_kg_query(
    doc,
    kg: KnowledgeGraph,
    *,
    multi_hop: bool = False,
    max_depth: int = 2,
) -> dict | None:
    """Run KG detection against a spaCy doc.

    Returns a detection dict if the prompt matches a KG entity AND has
    question structure, otherwise None.

    When multi_hop=True, uses kg.traverse() to collect facts reachable
    within max_depth hops from the primary entity.
    """
    entity_spans = find_entity_spans(doc, kg)
    if not entity_spans:
        return None

    if not has_question_structure(doc):
        return None

    # Prefer entities that are KG subjects (have forward-lookup facts).
    subject_spans = [s for s in entity_spans if kg.lookup(subject=s["entity"])]
    primary = subject_spans[0] if subject_spans else entity_spans[0]
    entity_text = primary["entity"]

    entity_match_tier = primary.get("match_tier", "exact")
    entity_match_score = primary.get("match_score", 1.0)
    original_text = primary.get("original_text", entity_text)

    predicate_phrase = extract_predicate_phrase(doc, entity_spans)
    lookup_type = "subject_scan"
    predicate_match_tier = "none"
    results = []

    # All subject entities to query — for multi-entity questions like
    # "total population of X and Y", we need facts from every entity.
    all_subjects = [s["entity"] for s in subject_spans] if subject_spans else [entity_text]

    if predicate_phrase:
        # Strip noise words that leak in from multi-entity phrasing
        clean_predicate = " ".join(
            w for w in predicate_phrase.split()
            if w not in ("and", "or", "both", "total", "combined", "sum", "together")
        ).strip() or predicate_phrase

        for subj in all_subjects:
            resolved = QUESTION_PREDICATE_MAP.get(clean_predicate, clean_predicate)
            hits = kg.lookup(subject=subj, predicate=resolved)

            if not hits and resolved != clean_predicate:
                hits = kg.lookup(subject=subj, predicate=clean_predicate)

            if not hits:
                fuzzy_pred, pred_tier = kg._resolve_predicate_cascade(
                    clean_predicate, subject=subj,
                )
                if pred_tier != "none":
                    hits = kg.lookup(subject=subj, predicate=fuzzy_pred)

            if hits:
                lookup_type = "targeted"
                predicate_match_tier = "exact"
                for h in hits:
                    if h not in results:
                        results.append(h)

    if not results:
        for subj in all_subjects:
            if multi_hop:
                hits = kg.traverse(subj, max_depth=max_depth)
                lookup_type = "multi_hop"
            else:
                hits = kg.lookup(subject=subj)
                lookup_type = "subject_scan"
            for h in hits:
                if h not in results:
                    results.append(h)

    if not results:
        return None

    return {
        "tool": "kg",
        "operation": "lookup",
        "entity": entity_text,
        "entity_spans": entity_spans,
        "predicate_phrase": predicate_phrase,
        "lookup_type": lookup_type,
        "results": results,
        "matched_pattern": "entity_question",
        "match_tier": entity_match_tier,
        "match_score": entity_match_score,
        "original_text": original_text,
        "predicate_match_tier": predicate_match_tier,
    }
