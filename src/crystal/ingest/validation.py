"""Triplet validation gates — prevent garbage from entering the KG.

Four fast, deterministic gates (no LLM cost):
  1. Subject gate: rejects pronouns, common nouns, short strings
  2. Predicate gate: rejects non-canonical predicates
  3. Object type gate: validates object format against predicate expectations
  4. Source sentence gate: verifies subject/object appear in tagged source text

Plus an LLM proofreading function for semantic validation against source text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class ValidationSeverity(Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class ValidationResult:
    valid: bool
    severity: ValidationSeverity | None = None
    reason: str = ""


# ── Subject gate ──────────────────────────────────────────────────────

_JUNK_SUBJECTS = frozenset({
    "i", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "this", "that", "these", "those", "who", "whom", "which", "what",
    "itself", "himself", "herself", "themselves", "ourselves",
    "one", "ones", "some", "any", "all", "none", "each", "every",
    "court", "courts", "state", "states", "law", "case", "cases",
    "the court", "this case", "the state", "the case",
    "defendant", "plaintiff", "petitioner", "respondent",
    "appellant", "appellee", "parties", "party",
    "conclusions", "provisions", "progress", "burden", "equity",
    "presentations", "opinion", "dissent", "concurrence",
    "judgment", "order", "government", "officer", "officers",
    "constitution", "statute", "statute of limitations",
    "evidence", "testimony", "witness", "witnesses",
    "majority", "minority", "court below",
    "personal interest", "stake", "scheme", "notion",
    "briefs", "motion", "petition", "mandamus",
})

_JUNK_PREFIXES = ("this ", "the ", "said ", "such ", "that ", "some of ", "some ")

_PRONOUN_PATTERN = re.compile(
    r"^(i|me|my|mine|he|him|his|she|her|hers|it|its|we|us|our|ours|"
    r"they|them|their|theirs|this|that|these|those|who|whom|which|what)$",
    re.IGNORECASE,
)


def validate_subject(subject: str) -> ValidationResult:
    """Check if a subject is a plausible legal entity (case name, person, org)."""
    s = subject.strip().lower()

    if len(s) < 3:
        return ValidationResult(False, ValidationSeverity.HARD, f"subject too short: '{s}'")

    if len(s) > 200:
        return ValidationResult(False, ValidationSeverity.HARD, "subject too long (>200 chars)")

    if s in _JUNK_SUBJECTS:
        return ValidationResult(False, ValidationSeverity.HARD, f"junk subject: '{s}'")

    if _PRONOUN_PATTERN.match(s):
        return ValidationResult(False, ValidationSeverity.HARD, f"pronoun subject: '{s}'")

    if any(s.startswith(p) and " v" not in s for p in _JUNK_PREFIXES):
        return ValidationResult(False, ValidationSeverity.HARD, f"junk prefix: '{s}'")

    if s.startswith("in re ") or s.startswith("ex parte ") or s.startswith("matter of "):
        return ValidationResult(True)

    words = subject.strip().split()
    if all(w[0].islower() for w in words if w):
        if " v. " not in subject and " v " not in subject:
            return ValidationResult(
                False, ValidationSeverity.SOFT,
                f"all-lowercase subject without 'v.': '{s}'",
            )

    return ValidationResult(True)


# ── Predicate gate ────────────────────────────────────────────────────

_ALLOWED_PREDICATES = frozenset({
    "cites", "cited_by_count", "court", "date_filed", "judges",
    "disposition", "nature_of_suit", "opinion_author", "per_curiam",
    "attorneys", "precedential_status",
    "doctrine", "holding", "reasoning", "is a",
})


def validate_predicate(predicate: str) -> ValidationResult:
    """Check if a predicate is a known canonical or allowlisted predicate."""
    p = predicate.strip().lower()

    if not p:
        return ValidationResult(False, ValidationSeverity.HARD, "empty predicate")

    if p in _ALLOWED_PREDICATES:
        return ValidationResult(True)

    return ValidationResult(
        False, ValidationSeverity.HARD,
        f"non-canonical predicate: '{p}'",
    )


# ── Object type validation ────────────────────────────────────────────

_DATE_YEAR_PATTERN = re.compile(r"\b\d{4}\b")
_DATE_MONTH_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.IGNORECASE,
)
_DATE_RELATIVE_PATTERN = re.compile(
    r"\b(?:before|after|earlier|prior|later|year|years|ago|"
    r"during|following|preceding|since|until)\b",
    re.IGNORECASE,
)
_DATE_FORMAT_PATTERN = re.compile(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}")

_COURT_TERMS = re.compile(
    r"\b(?:court|supreme|circuit|district|appeals|appellate|"
    r"tribunal|chancery|magistrate|bankruptcy|"
    r"scotus|u\.s\.|united states)\b",
    re.IGNORECASE,
)

_CITATION_V_PATTERN = re.compile(r"\bv\.?\s", re.IGNORECASE)
_CITATION_REPORTER = re.compile(r"\d+\s+\S+\.?\s+\d+")

_BOOLEAN_VALUES = frozenset({
    "yes", "no", "true", "false", "1", "0",
})


def _validate_date_filed(obj: str) -> ValidationResult:
    o = obj.strip()
    if _DATE_YEAR_PATTERN.search(o):
        return ValidationResult(True)
    if _DATE_MONTH_PATTERN.search(o):
        return ValidationResult(True)
    if _DATE_RELATIVE_PATTERN.search(o):
        return ValidationResult(True)
    if _DATE_FORMAT_PATTERN.search(o):
        return ValidationResult(True)
    return ValidationResult(
        False, ValidationSeverity.SOFT,
        f"date_filed object has no date-like pattern: '{o[:80]}'",
    )


def _validate_court(obj: str) -> ValidationResult:
    if len(obj) > 200:
        return ValidationResult(
            False, ValidationSeverity.SOFT,
            "court object too long (>200 chars)",
        )
    if _COURT_TERMS.search(obj):
        return ValidationResult(True)
    if len(obj) < 50:
        return ValidationResult(True)
    return ValidationResult(
        False, ValidationSeverity.SOFT,
        f"court object has no court-like terms: '{obj[:80]}'",
    )


def _validate_cites(obj: str) -> ValidationResult:
    if _CITATION_V_PATTERN.search(obj):
        return ValidationResult(True)
    if _CITATION_REPORTER.search(obj):
        return ValidationResult(True)
    if "in re " in obj.lower() or "ex parte " in obj.lower() or "matter of " in obj.lower():
        return ValidationResult(True)
    return ValidationResult(
        False, ValidationSeverity.SOFT,
        f"cites object has no citation pattern: '{obj[:80]}'",
    )


def _validate_cited_by_count(obj: str) -> ValidationResult:
    o = obj.strip()
    if o.isdigit():
        return ValidationResult(True)
    try:
        int(o.replace(",", ""))
        return ValidationResult(True)
    except ValueError:
        pass
    return ValidationResult(
        False, ValidationSeverity.HARD,
        f"cited_by_count object is non-numeric: '{o[:80]}'",
    )


def _validate_per_curiam(obj: str) -> ValidationResult:
    if obj.strip().lower() in _BOOLEAN_VALUES:
        return ValidationResult(True)
    return ValidationResult(
        False, ValidationSeverity.HARD,
        f"per_curiam object is non-boolean: '{obj[:80]}'",
    )


def _validate_short_text(obj: str, predicate: str) -> ValidationResult:
    if len(obj) > 500:
        return ValidationResult(
            False, ValidationSeverity.SOFT,
            f"{predicate} object too long ({len(obj)} chars)",
        )
    return ValidationResult(True)


def _validate_prose_text(obj: str, predicate: str) -> ValidationResult:
    if len(obj) > 2000:
        return ValidationResult(
            False, ValidationSeverity.SOFT,
            f"{predicate} object extremely long ({len(obj)} chars)",
        )
    return ValidationResult(True)


PREDICATE_OBJECT_VALIDATORS: dict[str, Callable[[str], ValidationResult]] = {
    "date_filed": _validate_date_filed,
    "court": _validate_court,
    "cites": _validate_cites,
    "cited_by_count": _validate_cited_by_count,
    "per_curiam": _validate_per_curiam,
    "opinion_author": lambda o: _validate_short_text(o, "opinion_author"),
    "judges": lambda o: _validate_short_text(o, "judges"),
    "attorneys": lambda o: _validate_short_text(o, "attorneys"),
    "precedential_status": lambda o: _validate_short_text(o, "precedential_status"),
    "disposition": lambda o: _validate_short_text(o, "disposition"),
    "nature_of_suit": lambda o: _validate_short_text(o, "nature_of_suit"),
    "doctrine": lambda o: _validate_prose_text(o, "doctrine"),
    "holding": lambda o: _validate_prose_text(o, "holding"),
    "reasoning": lambda o: _validate_prose_text(o, "reasoning"),
}


def validate_object(predicate: str, obj: str) -> ValidationResult:
    """Validate that an object makes sense for the given predicate."""
    validator = PREDICATE_OBJECT_VALIDATORS.get(predicate.lower())
    if validator is None:
        return ValidationResult(True)
    return validator(obj)


# ── Source sentence verification ──────────────────────────────────────


def _normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy containment checks."""
    return " ".join(text.lower().split())


def validate_source_sentence(
    subject: str, obj: str, source_sentence: str,
) -> ValidationResult:
    """Check that subject and object appear in the tagged source sentence.

    Skips validation (returns valid) when source_sentence is empty — legacy
    data may not have provenance. When present, checks case-insensitive
    containment. For "X v. Y" subjects, checks both party names independently.
    """
    if not source_sentence or not source_sentence.strip():
        return ValidationResult(True)

    ss = _normalize_for_match(source_sentence)
    subj_low = _normalize_for_match(subject)

    subj_found = False
    if subj_low in ss:
        subj_found = True
    elif " v. " in subj_low or " v " in subj_low:
        parts = subj_low.replace(" v. ", " v ").split(" v ", 1)
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ""
        if left and left in ss:
            subj_found = True
        elif right and right in ss:
            subj_found = True

    obj_low = _normalize_for_match(obj)
    obj_found = False
    if obj_low in ss:
        obj_found = True
    else:
        obj_words = [w for w in obj_low.split() if len(w) > 3]
        if obj_words and any(w in ss for w in obj_words):
            obj_found = True

    if not subj_found and not obj_found:
        return ValidationResult(
            False, ValidationSeverity.SOFT,
            f"neither subject nor object found in source sentence "
            f"(subj='{subject[:40]}', obj='{obj[:40]}', sent='{source_sentence[:60]}')",
        )

    return ValidationResult(True)


# ── Combined validation ───────────────────────────────────────────────

@dataclass
class TripletValidation:
    valid: bool
    subject_result: ValidationResult
    predicate_result: ValidationResult
    object_result: ValidationResult
    source_result: ValidationResult | None = None

    @property
    def severity(self) -> ValidationSeverity | None:
        """Return the worst severity among failed checks."""
        if self.valid:
            return None
        results = [self.subject_result, self.predicate_result, self.object_result]
        if self.source_result is not None:
            results.append(self.source_result)
        for r in results:
            if not r.valid and r.severity == ValidationSeverity.HARD:
                return ValidationSeverity.HARD
        return ValidationSeverity.SOFT

    @property
    def reasons(self) -> list[str]:
        results = [self.subject_result, self.predicate_result, self.object_result]
        if self.source_result is not None:
            results.append(self.source_result)
        return [r.reason for r in results if not r.valid and r.reason]


def validate_triplet(
    subject: str, predicate: str, obj: str, source_sentence: str = "",
) -> TripletValidation:
    """Run all validation gates on a triplet.

    When source_sentence is provided, also runs the source-sentence
    verification gate.
    """
    sr = validate_subject(subject)
    pr = validate_predicate(predicate)
    or_ = validate_object(predicate, obj)
    ss_r = None
    if source_sentence:
        ss_r = validate_source_sentence(subject, obj, source_sentence)
    all_valid = sr.valid and pr.valid and or_.valid
    if ss_r is not None and not ss_r.valid:
        all_valid = False
    return TripletValidation(
        valid=all_valid,
        subject_result=sr,
        predicate_result=pr,
        object_result=or_,
        source_result=ss_r,
    )


# ── LLM proofreading ─────────────────────────────────────────────────

_PROOFREAD_PROMPT = """\
You are verifying extracted facts against their source text.
For each fact, determine if it is accurately supported by the source text.

Source text:
"{source_sentence}"

Facts to verify:
{facts}

For EACH fact, respond with exactly one line:
<number>. VALID or INVALID — <brief reason>

Example:
1. VALID — the source text explicitly states this relationship
2. INVALID — the source describes a criminal charge, not a filing date"""

_PLAUSIBILITY_PROMPT = """\
You are checking the plausibility of legal knowledge graph facts.
For each fact, determine if it is a plausible legal fact.

Facts to verify:
{facts}

Consider:
- Does the subject look like a real case name, legal entity, or person?
- Does the object make sense for the predicate type?
- For "date_filed": object should be a date or time reference
- For "court": object should be a court name
- For "cites": object should be a case name or citation
- For "judges"/"opinion_author": object should be a person's name

For EACH fact, respond with exactly one line:
<number>. PLAUSIBLE or IMPLAUSIBLE — <brief reason>"""


@dataclass
class ProofreadResult:
    """Result of LLM proofreading for a single triplet."""
    index: int
    valid: bool
    reason: str = ""


def _parse_proofread_response(response: str, count: int) -> list[ProofreadResult]:
    """Parse LLM proofreading response into structured results."""
    results = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"(\d+)\.\s*(VALID|INVALID|PLAUSIBLE|IMPLAUSIBLE)\s*[—\-–:]\s*(.*)",
            line, re.IGNORECASE,
        )
        if match:
            idx = int(match.group(1)) - 1
            verdict = match.group(2).upper()
            reason = match.group(3).strip()
            is_valid = verdict in ("VALID", "PLAUSIBLE")
            if 0 <= idx < count:
                results.append(ProofreadResult(index=idx, valid=is_valid, reason=reason))
    return results


def proofread_triplets(
    triplets: list[tuple[str, str, str, str]],
    call_llm_fn: Callable[[str], tuple[str, dict | None]],
    batch_size: int = 15,
) -> dict[int, ProofreadResult]:
    """LLM-proofread triplets against their source sentences.

    Args:
        triplets: List of (subject, predicate, object, source_sentence).
        call_llm_fn: LLM caller function.
        batch_size: Max triplets per LLM call.

    Returns:
        Dict mapping original index to ProofreadResult.
    """
    results: dict[int, ProofreadResult] = {}

    with_source = [(i, s, p, o, ss) for i, (s, p, o, ss) in enumerate(triplets) if ss.strip()]
    without_source = [(i, s, p, o, ss) for i, (s, p, o, ss) in enumerate(triplets) if not ss.strip()]

    by_sentence: dict[str, list[tuple[int, str, str, str]]] = {}
    for i, s, p, o, ss in with_source:
        by_sentence.setdefault(ss, []).append((i, s, p, o))

    for sentence, facts in by_sentence.items():
        for batch_start in range(0, len(facts), batch_size):
            batch = facts[batch_start:batch_start + batch_size]
            facts_text = "\n".join(
                f"{j + 1}. ({s}, {p}, {o})" for j, (_, s, p, o) in enumerate(batch)
            )
            prompt = _PROOFREAD_PROMPT.format(
                source_sentence=sentence[:2000],
                facts=facts_text,
            )
            try:
                response, _ = call_llm_fn(prompt)
                parsed = _parse_proofread_response(response, len(batch))
                for pr in parsed:
                    orig_idx = batch[pr.index][0]
                    results[orig_idx] = ProofreadResult(
                        index=orig_idx, valid=pr.valid, reason=pr.reason,
                    )
            except Exception:
                pass

    for batch_start in range(0, len(without_source), batch_size):
        batch = without_source[batch_start:batch_start + batch_size]
        facts_text = "\n".join(
            f"{j + 1}. ({s}, {p}, {o})" for j, (_, s, p, o, _) in enumerate(batch)
        )
        prompt = _PLAUSIBILITY_PROMPT.format(facts=facts_text)
        try:
            response, _ = call_llm_fn(prompt)
            parsed = _parse_proofread_response(response, len(batch))
            for pr in parsed:
                orig_idx = batch[pr.index][0]
                results[orig_idx] = ProofreadResult(
                    index=orig_idx, valid=pr.valid, reason=pr.reason,
                )
        except Exception:
            pass

    return results
