"""
B2: Document-answerability audit.

Classifies benchmark questions as document-answerable, KG-only, or negative
based on the predicate they target. This determines which questions get a
fair three-arm comparison vs. being shown as Crystal-only advantages.

Predicate classification is structural, not text-search:
- Document-answerable predicates appear in real opinion text (court name,
  date, judges, author, citations, attorneys).
- KG-only predicates exist only in CourtListener index metadata and never
  appear in the opinion document itself (citation count, precedential
  status, per curiam flag).
"""

from __future__ import annotations

import re

DOCUMENT_ANSWERABLE_PREDICATES: set[str] = {
    "court",
    "date_filed",
    "judges",
    "opinion_author",
    "cites",
    "attorneys",
}

KG_ONLY_PREDICATES: set[str] = {
    "cited_by_count",
    "precedential_status",
    "per_curiam",
}

_PREDICATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("cited_by_count", re.compile(
        r"how many times|citation count|times? cited|how often cited",
        re.IGNORECASE,
    )),
    ("precedential_status", re.compile(
        r"precedential status|published opinion|publication status",
        re.IGNORECASE,
    )),
    ("opinion_author", re.compile(
        r"who wrote|who authored|written by|authored by|wrote the opinion",
        re.IGNORECASE,
    )),
    ("attorneys", re.compile(
        r"attorney|lawyer|counsel|who represented|who argued|attorneys",
        re.IGNORECASE,
    )),
    ("judges", re.compile(
        r"(?:who were the |what )judge|justice|who decided|who heard|who ruled"
        r"|list the judges|judges in",
        re.IGNORECASE,
    )),
    ("court", re.compile(
        r"what court|which court|court (?:decided|heard|ruled)",
        re.IGNORECASE,
    )),
    ("date_filed", re.compile(
        r"when was|what date|when did|date (?:filed|decided)|was filed|was decided",
        re.IGNORECASE,
    )),
    ("cites", re.compile(
        r"what (?:cases? )?(?:does|did) .+ cite|cites what|citations? of",
        re.IGNORECASE,
    )),
]


def infer_predicate(question: str) -> str | None:
    """Infer the target predicate from a benchmark question.

    Returns the canonical predicate name, or None if the question doesn't
    clearly target a single predicate (e.g., subject-scan questions like
    "Tell me about X").
    """
    q = question.strip()
    for predicate, pattern in _PREDICATE_PATTERNS:
        if pattern.search(q):
            return predicate
    return None


def classify_question(
    question: str,
    match_strings: list[str],
    is_negative: bool = False,
) -> str:
    """Classify a benchmark case as document_answerable, kg_only, or negative.

    Returns one of: "document_answerable", "kg_only", "negative", "subject_scan".
    """
    if is_negative:
        return "negative"

    predicate = infer_predicate(question)

    if predicate is None:
        return "subject_scan"

    if predicate in DOCUMENT_ANSWERABLE_PREDICATES:
        return "document_answerable"
    if predicate in KG_ONLY_PREDICATES:
        return "kg_only"

    return "document_answerable"


def partition_cases(
    cases: list[tuple[str, str, list[str], bool]],
) -> tuple[list, list, list, list]:
    """Split benchmark cases into (doc_answerable, kg_only, negatives, subject_scan).

    Each returned list contains the original tuples.
    """
    doc_answerable = []
    kg_only = []
    negatives = []
    subject_scan = []

    for case in cases:
        question, _golden, match_strings, is_negative = case
        category = classify_question(question, match_strings, is_negative)

        if category == "document_answerable":
            doc_answerable.append(case)
        elif category == "kg_only":
            kg_only.append(case)
        elif category == "negative":
            negatives.append(case)
        else:
            subject_scan.append(case)

    return doc_answerable, kg_only, negatives, subject_scan


def partition_summary(cases: list[tuple[str, str, list[str], bool]]) -> dict:
    """Return a summary dict of partition counts and predicate distribution."""
    doc, kg, neg, scan = partition_cases(cases)
    predicate_counts: dict[str, int] = {}
    for case in cases:
        q, _, ms, is_neg = case
        pred = infer_predicate(q) or ("negative" if is_neg else "subject_scan")
        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

    return {
        "total": len(cases),
        "document_answerable": len(doc),
        "kg_only": len(kg),
        "negative": len(neg),
        "subject_scan": len(scan),
        "by_predicate": predicate_counts,
    }
