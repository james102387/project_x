"""
LLM-assisted triplet extraction (D2 Phase 2).

For sentences where NER found entities but dep-tree patterns couldn't
resolve predicates, uses the LLM to extract (subject, predicate, object)
relationships. Designed as an offline batch job with human-reviewable output.

Domain-specific prompts (e.g. LEGAL_EXTRACTION_PROMPT) steer the LLM
toward the ontology's canonical predicates for higher-confidence extraction.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from crystal.ingest.schema import (
    LLMExtractionResult,
    ReviewableTriplet,
)

EXTRACTION_PROMPT = """\
Extract factual relationships from the sentences below as (subject, predicate, object) triplets.

For each sentence, potential entities have been identified. Extract concrete relationships \
between these or any other entities present.

Rules:
- Only extract factual claims, not opinions or speculation.
- Use the most specific entity name from the sentence.
- Predicates should be concise verb phrases (1-3 words, e.g. "borders", "exports", "founded by").
- Assign a confidence level to each extraction:
  "high" = clear, unambiguous relationship stated explicitly
  "medium" = likely relationship, somewhat implicit
  "low" = inferred or uncertain

Respond with ONLY a JSON array. Each element:
{{"subject": "...", "predicate": "...", "object": "...", "confidence": "high|medium|low", "sentence_index": N}}

Sentences:
{sentences}

JSON:"""


LEGAL_EXTRACTION_PROMPT = """\
Extract factual relationships from the legal text below as (subject, predicate, object) triplets.

PREFERRED PREDICATES (use these when possible):
- court: which court decided the case
- date_filed: when the case was filed or decided
- judges: justices or judges involved
- opinion_author: who wrote the majority opinion
- cites: cases cited by this opinion (use "Party v. Party" format)
- attorneys: lawyers who argued the case
- per_curiam: whether the opinion is unsigned (true/false)
- precedential_status: publication status (e.g. "Published", "Precedential")
- holding: the court's legal conclusion
- doctrine: legal principles applied
- reasoning: key reasoning or rationale

CASE NAME FORMAT: Always use "Party v. Party" format (e.g. "Miranda v. Arizona", not "Miranda").

Rules:
- Only extract factual claims stated in the text, not speculation.
- For case citations, extract as: {{"subject": "This Case v. Name", "predicate": "cites", "object": "Cited Case v. Name"}}
- For judges/justices: use full names when available.
- Assign confidence:
  "high" = explicitly stated fact (dates, named judges, direct citations)
  "medium" = clearly implied but requires interpretation
  "low" = inferred or ambiguous

Respond with ONLY a JSON array. Each element:
{{"subject": "...", "predicate": "...", "object": "...", "confidence": "high|medium|low", "sentence_index": N}}

Text excerpts:
{sentences}

JSON:"""


def _get_prompt(domain: str) -> str:
    """Return the extraction prompt for a given domain."""
    if domain == "legal":
        return LEGAL_EXTRACTION_PROMPT
    return EXTRACTION_PROMPT


def normalize_predicate(
    raw_predicate: str,
    ontology_predicates: set[str] | None = None,
    predicate_aliases: dict[str, str] | None = None,
) -> str:
    """Normalize an extracted predicate against the ontology.

    Tries exact match, then alias lookup. No substring fallback — loose
    matching caused predicates like "convicted" to map to "date_filed".
    Returns the canonical predicate if matched, or the raw predicate unchanged.
    """
    if not raw_predicate:
        return raw_predicate

    low = raw_predicate.lower().strip()

    if ontology_predicates and low in ontology_predicates:
        return low

    if predicate_aliases and low in predicate_aliases:
        return predicate_aliases[low]

    return low


def _format_sentences(sentences: list[tuple[str, list[str]]]) -> str:
    """Format sentence+entity pairs for the LLM prompt."""
    lines = []
    for i, (sent, entities) in enumerate(sentences):
        lines.append(f'{i + 1}. "{sent}" (entities: {", ".join(entities)})')
    return "\n".join(lines)


def _parse_llm_response(
    response: str,
    sentences: list[tuple[str, list[str]]],
) -> list[ReviewableTriplet]:
    """Parse JSON triplets from LLM response, with robust fallback."""
    text = response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Find first JSON array in response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    reviewable: list[ReviewableTriplet] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not (subj and pred and obj):
            continue

        confidence = str(item.get("confidence", "medium")).lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"

        idx = item.get("sentence_index")
        if isinstance(idx, int) and 1 <= idx <= len(sentences):
            source_sentence = sentences[idx - 1][0]
        else:
            source_sentence = ""

        reviewable.append(ReviewableTriplet(
            subject=subj,
            predicate=pred,
            object=obj,
            source_sentence=source_sentence,
            confidence=confidence,
        ))

    return reviewable


def extract_triplets_llm(
    sentences: list[tuple[str, list[str]]],
    call_llm_fn: Callable[[str], tuple[str, dict | None]] | None = None,
    batch_size: int = 20,
    domain: str = "general",
) -> LLMExtractionResult:
    """Use LLM to extract triplets from sentences that NER couldn't handle.

    Args:
        sentences: List of (sentence_text, entity_names) from find_unresolved_sentences().
        call_llm_fn: LLM caller matching call_llm(prompt) -> (text, usage).
                      Defaults to crystal.llm.call_llm if not provided.
        batch_size: Max sentences per LLM call to stay within context limits.
        domain: "legal" for legal-tuned prompt, "general" for generic.

    Returns:
        LLMExtractionResult with reviewable triplets (all pending_review).
    """
    if call_llm_fn is None:
        from crystal.llm import call_llm
        call_llm_fn = call_llm

    if not sentences:
        return LLMExtractionResult()

    prompt_template = _get_prompt(domain)
    all_reviewable: list[ReviewableTriplet] = []
    skipped: list[str] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        formatted = _format_sentences(batch)
        prompt = prompt_template.format(sentences=formatted)

        try:
            response_text, _usage = call_llm_fn(prompt)
            parsed = _parse_llm_response(response_text, batch)
            if parsed:
                all_reviewable.extend(parsed)
            else:
                skipped.extend(sent for sent, _ in batch)
        except Exception:
            skipped.extend(sent for sent, _ in batch)

    return LLMExtractionResult(
        reviewable=all_reviewable,
        skipped_sentences=skipped,
    )
