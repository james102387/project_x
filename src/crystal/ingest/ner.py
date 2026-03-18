"""
NER-based triplet extraction (D2 Phase 1).

Uses spaCy dependency trees to extract (subject, predicate, object) triplets
from natural language text. Operates sentence-by-sentence.

Supported patterns (verified empirically against en_core_web_sm):
  1. Copular:      "X is/are Y"       → nsubj + attr/acomp
  2. Possessive:   "X has/have Y"     → nsubj + dobj
  3. Active:       "X verbs Y"        → nsubj + dobj
  4. Passive:      "X was Vd by Y"    → nsubjpass + agent + pobj
  5. Prepositional: "X verb PREP Y"   → nsubj + prep + pobj

Entities are extracted from noun chunks and named entities.
Predicates are derived from the root verb (+ preposition for pattern 5).
"""

from __future__ import annotations

import spacy
from spacy.tokens import Doc, Span, Token

from crystal.ingest.schema import IngestResult, Triplet

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _span_text(span_or_token: Span | Token) -> str:
    """Clean text from a span or token, stripping leading determiners."""
    if isinstance(span_or_token, Token):
        return span_or_token.text.strip()
    text = span_or_token.text.strip()
    # Strip leading "the/a/an" for cleaner entity names
    for det in ("the ", "a ", "an "):
        if text.lower().startswith(det):
            stripped = text[len(det):].strip()
            if stripped:
                return stripped
    return text


def _get_subject_span(token: Token, include_prep: bool = True) -> str:
    """Get the full subject span including compound modifiers.

    When include_prep is True, includes "of X" prepositional phrases
    (e.g., "Festival of Vohn"). Set to False when the copular pattern
    handles the prep separately.
    """
    span_tokens = [token]
    for t in sorted(token.children, key=lambda t: t.i):
        if t.dep_ in ("compound", "flat", "amod"):
            span_tokens.append(t)
        elif t.dep_ == "punct" and t.text in ("-", "/"):
            span_tokens.append(t)
        elif t.dep_ == "prep" and include_prep:
            for sub_t in sorted(t.subtree, key=lambda x: x.i):
                span_tokens.append(sub_t)
        elif t.dep_ == "det":
            continue
    if not span_tokens:
        return token.text
    return _join_tokens(sorted(span_tokens, key=lambda t: t.i))


def _join_tokens(tokens: list[Token]) -> str:
    """Join tokens, collapsing spaces around hyphens and punctuation."""
    if not tokens:
        return ""
    parts = []
    for i, t in enumerate(tokens):
        if t.pos_ == "PUNCT" and t.text in ("-", "/"):
            if parts and parts[-1].endswith(" "):
                parts[-1] = parts[-1].rstrip()
            parts.append(t.text)
            continue
        if parts and not parts[-1].endswith("-") and not parts[-1].endswith("/"):
            parts.append(" ")
        parts.append(t.text)
    return "".join(parts).strip()


def _get_object_span(token: Token) -> str:
    """Get the full object span from a token's subtree, including compounds."""
    subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
    obj_tokens = []
    for t in subtree_tokens:
        if t.dep_ in ("relcl", "acl", "advcl"):
            continue
        if t.dep_ == "det":
            continue
        obj_tokens.append(t)
    if not obj_tokens:
        return token.text
    return _join_tokens(sorted(obj_tokens, key=lambda t: t.i))


def _extract_from_sentence(sent: Span) -> list[Triplet]:
    """Extract triplets from a single sentence using dep tree patterns."""
    triplets: list[Triplet] = []
    root = None
    for token in sent:
        if token.dep_ == "ROOT":
            root = token
            break
    if root is None:
        return triplets

    # Find subject
    subj_token = None
    subj_dep = None
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj_token = child
            subj_dep = child.dep_
            break

    if subj_token is None:
        return triplets

    is_copular = root.lemma_ == "be" and subj_dep == "nsubj"
    subject = _get_subject_span(subj_token, include_prep=not is_copular)
    if not subject:
        return triplets

    # Pattern 1: Copular — "X is/are Y"
    if is_copular:
        # Check subject for "of" prep → flip pattern
        # "The capital of Remulak is Zelphos" → (Remulak, capital, Zelphos)
        # "Zelphos is the capital of Remulak" → (Remulak, capital, Zelphos)
        subj_of_pobj = None
        for prep in subj_token.children:
            if prep.dep_ == "prep" and prep.text.lower() == "of":
                for pobj in prep.children:
                    if pobj.dep_ == "pobj":
                        subj_of_pobj = _get_object_span(pobj)
                        break

        for child in root.children:
            if child.dep_ in ("attr", "acomp"):
                obj_text = _get_object_span(child)
                # Check attr for "of" prep (handles "Zelphos is the capital of Remulak")
                attr_of_pobj = None
                for prep in child.children:
                    if prep.dep_ == "prep" and prep.text.lower() == "of":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj":
                                attr_of_pobj = _get_object_span(pobj)
                                break

                if subj_of_pobj:
                    # "The capital of Remulak is Zelphos"
                    triplets.append(Triplet(
                        subject=subj_of_pobj,
                        predicate=subject.lower(),
                        object=obj_text,
                    ))
                elif attr_of_pobj:
                    # "Zelphos is the capital of Remulak"
                    predicate = _span_text(child)
                    # Strip the "of X" from the predicate
                    for prep in child.children:
                        if prep.dep_ == "prep":
                            break
                    pred_tokens = [t for t in child.subtree
                                   if t.i < prep.i and t.dep_ != "det"]
                    pred_text = " ".join(t.text for t in sorted(pred_tokens, key=lambda t: t.i)).strip()
                    triplets.append(Triplet(
                        subject=attr_of_pobj,
                        predicate=pred_text.lower() if pred_text else predicate.lower(),
                        object=subject,
                    ))
                else:
                    # Simple copular: "Remulak is a planet"
                    predicate = "is a" if child.dep_ == "attr" else "is"
                    triplets.append(Triplet(
                        subject=subject,
                        predicate=predicate,
                        object=obj_text,
                    ))
            elif child.dep_ == "prep":
                # "X is PREP Y" — e.g. "X is in Y"
                prep_text = child.text.lower()
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        obj = _get_object_span(pobj)
                        triplets.append(Triplet(
                            subject=subject,
                            predicate=prep_text,
                            object=obj,
                        ))
        return triplets

    # Pattern 4: Passive — "X was Vd by Y"
    if subj_dep == "nsubjpass":
        predicate = root.text.lower()
        for child in root.children:
            if child.dep_ == "agent":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        agent = _get_object_span(pobj)
                        triplets.append(Triplet(
                            subject=subject,
                            predicate=f"{predicate} by",
                            object=agent,
                        ))
            elif child.dep_ == "prep":
                prep_text = child.text.lower()
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        obj = _get_object_span(pobj)
                        triplets.append(Triplet(
                            subject=subject,
                            predicate=f"{predicate} {prep_text}",
                            object=obj,
                        ))
        return triplets

    # Pattern 2/3: Active — "X verbs Y" / "X has Y"
    predicate = root.lemma_
    for child in root.children:
        if child.dep_ == "dobj":
            obj = _get_object_span(child)
            triplets.append(Triplet(
                subject=subject,
                predicate=predicate,
                object=obj,
            ))
        elif child.dep_ == "prep":
            # Pattern 5: "X verb PREP Y"
            prep_text = child.text.lower()
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    obj = _get_object_span(pobj)
                    triplets.append(Triplet(
                        subject=subject,
                        predicate=f"{predicate} {prep_text}",
                        object=obj,
                    ))

    return triplets


def extract_triplets(text: str, nlp=None) -> list[Triplet]:
    """Extract triplets from a block of text, sentence by sentence."""
    if nlp is None:
        nlp = _get_nlp()
    doc = nlp(text)
    triplets: list[Triplet] = []
    for sent in doc.sents:
        triplets.extend(_extract_from_sentence(sent))
    return triplets


def ingest_text(text: str, source: str = "", nlp=None) -> IngestResult:
    """Run NER extraction on a text block and return an IngestResult."""
    triplets = extract_triplets(text, nlp=nlp)
    return IngestResult(
        triplets=triplets,
        source=source,
    )


def ingest_file(path: str, nlp=None) -> IngestResult:
    """Read a text file and extract triplets via NER."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return ingest_text(text, source=path, nlp=nlp)
