"""
LLM-assisted triplet extraction (D2 Phase 2).

For sentences where NER found entities but dep-tree patterns couldn't
resolve predicates, uses the LLM to extract (subject, predicate, object)
relationships. Designed as an offline batch job with human-reviewable output.
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
) -> LLMExtractionResult:
    """Use LLM to extract triplets from sentences that NER couldn't handle.

    Args:
        sentences: List of (sentence_text, entity_names) from find_unresolved_sentences().
        call_llm_fn: LLM caller matching call_llm(prompt) -> (text, usage).
                      Defaults to crystal.llm.call_llm if not provided.
        batch_size: Max sentences per LLM call to stay within context limits.

    Returns:
        LLMExtractionResult with reviewable triplets (all pending_review).
    """
    if call_llm_fn is None:
        from crystal.llm import call_llm
        call_llm_fn = call_llm

    if not sentences:
        return LLMExtractionResult()

    all_reviewable: list[ReviewableTriplet] = []
    skipped: list[str] = []

    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        formatted = _format_sentences(batch)
        prompt = EXTRACTION_PROMPT.format(sentences=formatted)

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
