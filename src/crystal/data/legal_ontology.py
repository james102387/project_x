"""
Legal ontology for Crystal KG — citations-first design.

Phase 0: 4 canonical predicates for structured COLD Cases fields.
Entity naming: canonical subject = normalized case_name.
Aliases auto-generated from case_name_short, case_name_full, slug, citations.

Designed for SCOTUS subset (~35K decisions) with expansion path to full corpus.
"""

from __future__ import annotations

import re


# ── Canonical predicates ─────────────────────────────────────────────────
# Phase 0: structured fields only. LLM-extracted predicates (holdings,
# doctrines) deferred to L4/L5.

LEGAL_PREDICATES: list[str] = [
    "cites",
    "cited_by_count",
    "court",
    "date_filed",
    "document_slug",
    "judges",
    "disposition",
    "nature_of_suit",
    "opinion_author",
    "per_curiam",
    "attorneys",
    "precedential_status",
]


# ── Predicate aliases ────────────────────────────────────────────────────
# Surface forms → canonical predicates for KG resolution.

LEGAL_PREDICATE_ALIASES: dict[str, str] = {
    # citation
    "references": "cites",
    "relies on": "cites",
    "cites to": "cites",
    "cited": "cites",
    "citing": "cites",
    "citation count": "cited_by_count",
    "times cited": "cited_by_count",
    "number of citations": "cited_by_count",
    "how many times cited": "cited_by_count",
    "how often cited": "cited_by_count",
    "many times has been cited": "cited_by_count",
    "many times cited": "cited_by_count",
    "cited": "cited_by_count",
    # court — predicate phrases extracted by kg.py's extract_predicate_phrase
    "decided by": "court",
    "heard by": "court",
    "adjudicated by": "court",
    "which court": "court",
    "what court": "court",
    "tribunal": "court",
    "court decided": "court",
    "court heard": "court",
    "court ruled": "court",
    # date
    "filed on": "date_filed",
    "date decided": "date_filed",
    "when filed": "date_filed",
    "when decided": "date_filed",
    "decided on": "date_filed",
    "year": "date_filed",
    "date": "date_filed",
    "filed": "date_filed",
    # judges
    "judge": "judges",
    "justice": "judges",
    "justices": "judges",
    "presiding judge": "judges",
    "decided by judge": "judges",
    "who decided": "judges",
    "who judged": "judges",
    # disposition
    "outcome": "disposition",
    "ruling": "disposition",
    "result": "disposition",
    "how resolved": "disposition",
    "decision": "disposition",
    # nature of suit
    "type of case": "nature_of_suit",
    "case type": "nature_of_suit",
    "subject matter": "nature_of_suit",
    "kind of case": "nature_of_suit",
    # opinion author
    "author": "opinion_author",
    "written by": "opinion_author",
    "who wrote": "opinion_author",
    "authored by": "opinion_author",
    "opinion written by": "opinion_author",
    "majority author": "opinion_author",
    # per curiam
    "per curiam opinion": "per_curiam",
    "unsigned opinion": "per_curiam",
    # attorneys
    "attorney": "attorneys",
    "lawyer": "attorneys",
    "lawyers": "attorneys",
    "counsel": "attorneys",
    "represented by": "attorneys",
    "who represented": "attorneys",
    "who argued": "attorneys",
    # precedential status
    "published": "precedential_status",
    "publication status": "precedential_status",
    "precedential": "precedential_status",
}


# ── Court jurisdiction mapping ───────────────────────────────────────────
# COLD Cases jurisdiction codes → human-readable names.

COURT_TYPE_MAP: dict[str, str] = {
    "F": "Federal Appellate",
    "FD": "Federal District",
    "FB": "Federal Bankruptcy",
    "FS": "Federal Special",
    "S": "State Supreme",
    "SA": "State Appellate",
    "ST": "State Trial",
    "SS": "State Special",
    "SAG": "State Attorney General",
    "C": "Committee",
    "T": "Tribal",
    "I": "International",
}


# ── Citation parsing ─────────────────────────────────────────────────────

_CITE_PATTERN = re.compile(
    r"(\d+)\s+"
    r"(U\.S\.|S\.\s*Ct\.|L\.\s*Ed\.\s*(?:2d)?|F\.\s*(?:[23]d|Supp\.?\s*(?:[23]d)?)"
    r"|A\.\s*(?:[23]d)?|N\.E\.\s*(?:[23]d)?|N\.W\.\s*(?:[23]d)?|S\.E\.\s*(?:[23]d)?"
    r"|S\.W\.\s*(?:[23]d)?|So\.\s*(?:[23]d)?|P\.\s*(?:[23]d)?)"
    r"\s+(\d+)"
)


def parse_citation(cite_str: str) -> str | None:
    """Extract a normalized volume/reporter/page citation.

    "384 U.S. 436" → "384 U.S. 436"
    "garbage text" → None
    """
    if not cite_str:
        return None
    match = _CITE_PATTERN.search(cite_str)
    if match:
        volume, reporter, page = match.groups()
        reporter_clean = re.sub(r"\s+", " ", reporter.strip())
        return f"{volume} {reporter_clean} {page}"
    return None


# ── Case name normalization ──────────────────────────────────────────────

_V_PATTERN = re.compile(r"\s+v\.?\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_STRIP_PARENS = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_case_name(name: str) -> str:
    """Normalize a case name to a canonical form.

    - Standardize "v." / "vs." / "vs" / "v" to "v."
    - Collapse whitespace
    - Strip trailing parenthetical (e.g., "(1966)")
    - Title case
    """
    if not name:
        return ""
    text = _STRIP_PARENS.sub("", name)
    text = re.sub(r"\s+vs\.?\s+", " v. ", text, flags=re.IGNORECASE)
    text = _V_PATTERN.sub(" v. ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    parts = text.split(" v. ", 1)
    if len(parts) == 2:
        return f"{parts[0].strip().title()} v. {parts[1].strip().title()}"
    return text.title()


_FIRST_PARTY_BLOCKLIST: frozenset[str] = frozenset({
    # US states & territories — high collision risk as first-party names
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas",
    "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "puerto rico", "guam",
    # Federal/generic government entities
    "united states", "state", "city", "county", "commonwealth",
    "district of columbia", "people", "government",
    # Generic institutional prefixes that aren't disambiguating
    "board", "commission", "committee", "department", "office",
    "regents", "trustees",
})


def _is_safe_first_party(name: str) -> bool:
    """Check whether a first-party name is specific enough to be an alias."""
    lower = name.lower()
    if lower in _FIRST_PARTY_BLOCKLIST:
        return False
    if len(lower) <= 3:
        return False
    words = lower.split()
    if len(words) == 1 and len(lower) <= 5:
        return False
    return True


def generate_case_aliases(
    case_name: str,
    case_name_short: str = "",
    case_name_full: str = "",
    slug: str = "",
    citations: list[str] | None = None,
) -> dict[str, str]:
    """Generate entity alias mappings for a case.

    All aliases point to the normalized case_name as canonical entity.
    """
    canonical = normalize_case_name(case_name)
    if not canonical:
        return {}

    aliases: dict[str, str] = {}
    canonical_lower = canonical.lower()

    for variant in [case_name_short, case_name_full]:
        if variant:
            normalized = normalize_case_name(variant)
            key = normalized.lower()
            if key and key != canonical_lower:
                aliases[key] = canonical_lower

    if slug:
        slug_clean = slug.replace("-", " ").strip().lower()
        if slug_clean and slug_clean != canonical_lower:
            aliases[slug_clean] = canonical_lower

    for cite in (citations or []):
        parsed = parse_citation(cite)
        if parsed:
            aliases[parsed.lower()] = canonical_lower
        cite_clean = cite.strip().lower()
        if cite_clean and cite_clean != canonical_lower:
            aliases[cite_clean] = canonical_lower

    if slug:
        parts = slug.split("-v-", 1)
        if len(parts) == 2:
            short_v = f"{parts[0].replace('-', ' ')} v. {parts[1].replace('-', ' ')}".strip().lower()
            if short_v != canonical_lower and short_v not in aliases:
                aliases[short_v] = canonical_lower

    if " v. " in canonical:
        first_party = canonical.split(" v. ", 1)[0].strip()
        if _is_safe_first_party(first_party):
            aliases[first_party.lower()] = canonical_lower

    return aliases


def deduplicate_aliases(
    accumulated: dict[str, str],
    new_aliases: dict[str, str],
) -> dict[str, str]:
    """Merge new_aliases into accumulated, dropping any that would collide.

    A collision occurs when the same alias key maps to a different canonical
    entity. Silently dropping collisions is safer than letting .update()
    silently overwrite — we lose a convenience alias but never a wrong one.

    Returns the updated accumulated dict (mutated in place for efficiency).
    """
    for key, canonical in new_aliases.items():
        existing = accumulated.get(key)
        if existing is None:
            accumulated[key] = canonical
        elif existing != canonical:
            del accumulated[key]
    return accumulated
