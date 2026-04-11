"""
Ground-truth benchmark questions for the legal knowledge graph.

Same format as ground_truth.py: (question, ground_truth_answer, match_strings, is_negative)

Questions are designed so:
1. A naked LLM *could* answer these (real cases, public record) — unlike Remulak
2. Crystal + KG should answer them exactly and faster from the knowledge graph
3. Negative cases verify abstention when the KG has no data for that predicate

The value proposition for legal data is precision and sourcing, not impossibility.
Crystal provides grounded, verifiable answers with explicit KG provenance.

Coverage: court, date_filed, judges, cited_by_count, opinion_author,
precedential_status, attorneys, per_curiam, cites, plus alias resolution,
WH-word variation, citation-format entities, and adversarial negatives.
"""

LEGAL_BENCHMARK_CASES: list[tuple[str, str, list[str], bool]] = [
    # ── court predicate ──────────────────────────────────────────────────
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
    (
        "Which court decided Marbury v. Madison?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    (
        "What court heard Plessy v. Ferguson?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    # ── date_filed predicate ─────────────────────────────────────────────
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
    (
        "When was Roe v. Wade decided?",
        "1973-01-22",
        ["1973"],
        False,
    ),
    (
        "What date was Loving v. Virginia filed?",
        "1967-06-12",
        ["1967"],
        False,
    ),
    (
        "When was Chevron v. NRDC decided?",
        "1984-06-25",
        ["1984"],
        False,
    ),
    # ── judges predicate ─────────────────────────────────────────────────
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
    (
        "Who decided Citizens United v. Federal Election Commission?",
        "Roberts, Stevens, Scalia, Kennedy, Thomas, Ginsburg, Breyer, Alito, Sotomayor",
        ["roberts"],
        False,
    ),
    (
        "What judges heard Mapp v. Ohio?",
        "Warren, Black, Frankfurter, Douglas, Clark, Harlan, Brennan, Whittaker, Stewart",
        ["warren", "clark"],
        False,
    ),
    # ── cited_by_count predicate ─────────────────────────────────────────
    (
        "How many times has Miranda v. Arizona been cited?",
        "9832",
        ["9832"],
        False,
    ),
    (
        "What is the citation count for Brown v. Board of Education?",
        "12450",
        ["12450"],
        False,
    ),
    # ── opinion_author predicate ─────────────────────────────────────────
    (
        "Who wrote the opinion in Dred Scott v. Sandford?",
        "Taney",
        ["taney"],
        False,
    ),
    (
        "Who authored the opinion in Mcculloch v. Maryland?",
        "Marshall",
        ["marshall"],
        False,
    ),
    # ── precedential_status predicate ────────────────────────────────────
    (
        "What is the precedential status of Miranda v. Arizona?",
        "Published",
        ["published"],
        False,
    ),
    (
        "Is Brown v. Board of Education a published opinion?",
        "Published",
        ["published"],
        False,
    ),
    (
        "What is the precedential status of Roe v. Wade?",
        "Published",
        ["published"],
        False,
    ),
    # ── attorneys predicate ──────────────────────────────────────────────
    (
        "Who were the attorneys in Gideon v. Wainwright?",
        "Abe Fortas argued the cause for petitioner",
        ["fortas"],
        False,
    ),
    # ── WH-word variation (who decided → judges vs date_filed) ───────────
    (
        "Who decided Loving v. Virginia?",
        "Warren, Black, Douglas, Clark, Harlan, Brennan, Stewart, White, Fortas",
        ["warren"],
        False,
    ),
    (
        "When was Loving v. Virginia decided?",
        "1967-06-12",
        ["1967"],
        False,
    ),
    (
        "Who decided Mapp v. Ohio?",
        "Warren, Black, Frankfurter, Douglas, Clark, Harlan, Brennan, Whittaker, Stewart",
        ["warren", "clark"],
        False,
    ),
    (
        "When was Mapp v. Ohio decided?",
        "1961-06-19",
        ["1961"],
        False,
    ),
    # ── Citation-format entity detection ─────────────────────────────────
    (
        "What court decided 384 U.S. 436?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    (
        "When was 347 U.S. 483 decided?",
        "1954-05-17",
        ["1954"],
        False,
    ),
    # ── Alias resolution ─────────────────────────────────────────────────
    (
        "When was Chevron v. NRDC decided?",
        "1984-06-25",
        ["1984"],
        False,
    ),
    (
        "What court heard Terry v. Ohio?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    # ── Cross-predicate (same entity, different predicates) ──────────────
    (
        "What court decided Roe v. Wade?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    (
        "Who were the judges in Brown v. Board of Education?",
        "Warren, Black, Reed, Frankfurter, Douglas, Jackson, Burton, Clark, Minton",
        ["warren"],
        False,
    ),
    (
        "What is the precedential status of Marbury v. Madison?",
        "Published",
        ["published"],
        False,
    ),
    # ── Request verb variations ────────────────────────────────────────────
    (
        "Tell me the court that decided Miranda v. Arizona",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    (
        "List the judges in Gideon v. Wainwright",
        "Warren, Black, Douglas, Clark, Harlan, Brennan, Stewart, White, Goldberg",
        ["warren"],
        False,
    ),
    (
        "Tell me when Brown v. Board of Education was filed",
        "1954-05-17",
        ["1954"],
        False,
    ),
    # ── Subject scan (unknown predicate → all facts) ─────────────────────
    (
        "Tell me about Miranda v. Arizona",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    (
        "What do we know about Roe v. Wade?",
        "1973",
        ["1973"],
        False,
    ),
    (
        "Tell me about Marbury v. Madison",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    # ── More cross-predicate coverage ────────────────────────────────────
    (
        "What is the precedential status of Loving v. Virginia?",
        "Published",
        ["published"],
        False,
    ),
    (
        "What court decided Mapp v. Ohio?",
        "Supreme Court of the United States",
        ["supreme court"],
        False,
    ),
    # ── Adversarial negatives (no KG predicate exists) ───────────────────
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
    (
        "What was the constitutional basis for Marbury v. Madison?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What lower court did Miranda v. Arizona come from?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What was the vote count in Roe v. Wade?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What doctrine did Chevron v. NRDC establish?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "Who filed the amicus brief in Citizens United?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What was the oral argument date for Brown v. Board of Education?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What law school did Chief Justice Marshall attend?",
        "[ABSTAIN]",
        [],
        True,
    ),
]


# ── Known gaps (detection enhancements needed) ───────────────────────────
# Resolved gaps are kept here as comments for history.
# Active gaps require detector changes before they can pass.

# RESOLVED 2026-04-09: citation span regex pre-scan in kg.py detector
# RESOLVED 2026-04-09: WH-word-aware predicate override ("who decided" → judges)

LEGAL_KNOWN_GAPS: list[tuple[str, str, list[str], str]] = []

