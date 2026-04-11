"""
B3: Document-context baseline runner — LLM + real opinion text.

Simulates a lawyer pasting a case opinion into ChatGPT and asking
a question about it. This is the fair comparison arm that addresses
Concern 1 (naked LLM is a strawman).

Same interface as baseline.py: takes cases, returns scored result dicts.
Only runs on document-answerable questions (determined by B2 answerability).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from crystal.data.legal_ontology import normalize_case_name

_MAX_DOC_CHARS = 100_000

_CITATION_CASE_PATTERN = re.compile(r"\b\d+\s+U\.S\.\s+\d+\b")


def _extract_case_name(question: str) -> str | None:
    """Extract the case name from a benchmark question.

    Handles both named cases ("Miranda v. Arizona") and citation
    formats ("384 U.S. 436"). Uses the " v. " separator as an anchor.
    """
    cite_match = _CITATION_CASE_PATTERN.search(question)
    if cite_match:
        return cite_match.group(0)

    v_idx = question.find(" v. ")
    if v_idx == -1:
        v_idx = question.find(" v ")
    if v_idx == -1:
        return None

    before = question[:v_idx]
    after = question[v_idx:]

    words_before = before.split()
    start_idx = len(words_before) - 1
    while start_idx > 0:
        w = words_before[start_idx - 1]
        if w[0].isupper() or w in ("of", "the", "&", "and", "for"):
            start_idx -= 1
        elif w.rstrip(".,") and w.rstrip(".,")[0].isupper():
            start_idx -= 1
        else:
            break
    plaintiff_words = words_before[start_idx:]
    while plaintiff_words and plaintiff_words[0].lower() in ("of", "the", "for", "is", "in", "was"):
        plaintiff_words = plaintiff_words[1:]
    plaintiff = " ".join(plaintiff_words)

    v_sep = " v. " if " v. " in question else " v "
    rest = question[v_idx + len(v_sep):]
    rest = rest.rstrip("?").strip()
    end_words = rest.split()
    end_idx = 0
    for w in end_words:
        stripped = w.rstrip("?.,")
        if stripped and (stripped[0].isupper() or stripped in ("of", "the", "&", "and", "for")):
            end_idx += 1
        else:
            break
    if end_idx == 0:
        end_idx = min(len(end_words), 3)
    defendant = " ".join(end_words[:end_idx])
    defendant = defendant.rstrip("?.,")

    if plaintiff and defendant:
        return f"{plaintiff} v. {defendant}"
    return None


def _build_document_prompt(opinion_text: str, question: str) -> str:
    """Build the prompt that mirrors a lawyer pasting a case into ChatGPT."""
    doc = opinion_text
    if len(doc) > _MAX_DOC_CHARS:
        doc = doc[:_MAX_DOC_CHARS] + "\n\n[Document truncated at 100,000 characters]"

    return (
        "Here is the full text of a court opinion:\n\n"
        "---\n"
        f"{doc}\n"
        "---\n\n"
        "Based on this document, answer the following question. "
        "Be concise and specific.\n\n"
        f"{question}"
    )


def _unpack_case(case: tuple) -> tuple[str, str, list[str], bool]:
    if len(case) == 4:
        return case[0], case[1], case[2], case[3]
    return case[0], case[1], case[2], False


def _resolve_document(question: str, opinions: dict[str, str]) -> tuple[str, str]:
    """Extract case name from question and look up its opinion text.

    Returns (prompt, prompt_source) where prompt_source is
    "document" or "no_document".
    """
    case_name = _extract_case_name(question)
    doc_text = ""
    if case_name:
        normalized = normalize_case_name(case_name).lower()
        doc_text = opinions.get(normalized, "")

    if not doc_text:
        return question, "no_document"
    return _build_document_prompt(doc_text, question), "document"


def run_document_baseline(
    cases: list[tuple],
    opinions: dict[str, str],
    *,
    call_llm_fn=None,
    sleep_between: float = 4.0,
) -> list[dict]:
    """Arm 2: LLM + real opinion text for each question.

    Args:
        cases: Benchmark case tuples (question, golden, match_strings, is_negative).
        opinions: Dict mapping normalized lowercase case names to opinion text.
        call_llm_fn: Optional injectable LLM function (for testing).
        sleep_between: Delay between LLM calls.

    Returns list of result dicts compatible with score_batch_rubric().
    """
    if call_llm_fn is None:
        from crystal.llm import call_llm
        call_llm_fn = call_llm

    results = []
    for i, case in enumerate(cases):
        question, ground_truth, match_strings, is_negative = _unpack_case(case)
        print(f"  [{i+1}/{len(cases)}] {question}")

        prompt, prompt_source = _resolve_document(question, opinions)

        try:
            response, usage = call_llm_fn(prompt)
        except Exception as e:
            response = f"[ERROR: {e}]"
            usage = None

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "is_negative": is_negative,
            "response": response,
            "usage": usage,
            "prompt_source": prompt_source,
            "prompt_chars": len(prompt),
            "prompt_tokens_estimate": len(prompt) // 4,
        })

        if sleep_between > 0:
            time.sleep(sleep_between)

    return results


def run_document_baseline_cached(
    cases: list[tuple],
    opinions: dict[str, str],
    *,
    arm_name: str = "arm2_doc",
    model_tag: str = "",
    call_llm_fn=None,
    sleep_between: float = 4.0,
) -> list[dict]:
    """Arm 2 with per-question caching. Same interface as run_document_baseline."""
    from benchmarks.cache import get_cached, set_cached

    if call_llm_fn is None:
        from crystal.llm import call_llm
        call_llm_fn = call_llm

    results = []
    cache_hits = 0

    for i, case in enumerate(cases):
        question, ground_truth, match_strings, is_negative = _unpack_case(case)

        cached = get_cached(question, arm_name, model_tag)
        if cached is not None:
            cache_hits += 1
            print(f"  [Arm2 {i+1}/{len(cases)}] (cached) {question[:60]}")
            results.append(cached)
            continue

        print(f"  [Arm2 {i+1}/{len(cases)}] {question[:60]}")
        prompt, prompt_source = _resolve_document(question, opinions)

        try:
            response, usage = call_llm_fn(prompt)
        except Exception as e:
            response = f"[ERROR: {e}]"
            usage = None

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "match_strings": match_strings,
            "is_negative": is_negative,
            "response": response,
            "usage": usage,
            "prompt_source": prompt_source,
            "prompt_chars": len(prompt),
            "prompt_tokens_estimate": len(prompt) // 4,
        }
        results.append(result)
        set_cached(question, arm_name, model_tag, result)

        if sleep_between > 0:
            time.sleep(sleep_between)

    if cache_hits:
        print(f"  → {cache_hits}/{len(cases)} from cache")
    return results
