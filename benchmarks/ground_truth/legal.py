"""
Ground-truth benchmark questions for the legal knowledge graph (Phase 0).

Same format as ground_truth.py: (question, ground_truth_answer, match_strings, is_negative)

Questions are designed so:
1. A naked LLM *could* answer these (real cases, public record) — unlike Remulak
2. Crystal + KG should answer them exactly and faster from the knowledge graph
3. Negative cases verify abstention when the KG has no data for that predicate

The value proposition for legal data is precision and sourcing, not impossibility.
Crystal provides grounded, verifiable answers with explicit KG provenance.
"""

LEGAL_BENCHMARK_CASES: list[tuple[str, str, list[str], bool]] = [
    # --- Targeted: court predicate ---
    (
        "What court decided Miranda v. Arizona?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    (
        "What court heard Gideon v. Wainwright?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    # --- Targeted: date_filed predicate ---
    (
        "When was Brown v. Board of Education decided?",
        "1954-05-17",
        ["1954"],
        False,
    ),
    (
        "When was Obergefell v. Hodges filed?",
        "2015-06-26",
        ["2015"],
        False,
    ),
    # --- Targeted: judges predicate ---
    (
        "Who were the judges in Roe v. Wade?",
        "Burger, Douglas, Brennan, Stewart, White, Marshall, Blackmun, Powell, Rehnquist",
        ["burger", "blackmun"],
        False,
    ),
    (
        "Who were the judges in Marbury v. Madison?",
        "Marshall, Paterson, Chase, Washington",
        ["marshall"],
        False,
    ),
    # --- Targeted: disposition predicate ---
    (
        "What was the ruling in Bush v. Gore?",
        "Reversed and remanded",
        ["reversed"],
        False,
    ),
    (
        "What was the disposition of Miranda v. Arizona?",
        "Reversed and remanded",
        ["reversed", "remanded"],
        False,
    ),
    # --- Targeted: cited_by_count predicate ---
    (
        "How many times has Miranda v. Arizona been cited?",
        "9832",
        ["9832"],
        False,
    ),
    # --- Targeted: nature_of_suit predicate ---
    (
        "What type of case was Brown v. Board of Education?",
        "Civil Rights",
        ["civil rights"],
        False,
    ),
    # --- Alias resolution ---
    (
        "When was Chevron v. NRDC decided?",
        "1984-06-25",
        ["1984"],
        False,
    ),
    # --- Adversarial negatives (no KG predicate exists) ---
    (
        "What was the majority opinion in Miranda v. Arizona?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "Who wrote the dissent in Brown v. Board of Education?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What statute did Roe v. Wade overturn?",
        "[ABSTAIN]",
        [],
        True,
    ),
]


# ── Known gaps (detection enhancements needed) ───────────────────────────
# These require detector changes before they can pass.
# Tracked here so the Ralph Wiggum loop can eventually target them.

LEGAL_KNOWN_GAPS: list[tuple[str, str, list[str], str]] = [
    (
        "What court decided 384 U.S. 436?",
        "Supreme Court of the United States (Miranda v. Arizona)",
        ["supreme court"],
        "Citation-format entity spans ('384 U.S. 436') split by spaCy into "
        "separate tokens. Needs regex-based citation span detection in kg.py.",
    ),
    (
        "Who decided Citizens United v. Federal Election Commission?",
        "Roberts, Stevens, Scalia, Kennedy, Thomas, Ginsburg, Breyer, Alito, Sotomayor",
        ["roberts"],
        "Ambiguous predicate 'decided' — maps to date_filed but question expects "
        "judges. Needs context-aware predicate resolution (WH-word = 'who' → judges).",
    ),
]

