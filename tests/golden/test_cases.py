"""
Golden test cases — hand-crafted ground truth.

Every case has a known expected outcome. No LLM calls needed.
Format: (prompt, expected_type, expected_result_or_None)

Prompt types:
  pure_math       — bare arithmetic, skip LLM entirely
  math_answerable — narrative math where Crystal has the full answer, skip LLM
  math_augmented  — math embedded in a question requiring LLM reasoning
  kg_answerable   — KG lookup returned facts, skip LLM
  kg_augmented    — KG facts + reasoning signals, inject facts for LLM
  no_match        — no computable math detected
"""

PURE_MATH_CASES = [
    ("add 5 and 3", "pure_math", 8),
    ("can you add 12 to 8", "pure_math", 20),
    ("please add 100 and 250", "pure_math", 350),
    ("add 1 and 2 and 3", "pure_math", 6),
    ("5 plus 3", "pure_math", 8),
    ("what's 5 plus 3", "pure_math", 8),
    ("what is 10 plus 20", "pure_math", 30),
    ("100 plus 200 plus 50", "pure_math", 350),
    ("the sum of 5 and 3", "pure_math", 8),
    ("what's the sum of 10 and 20", "pure_math", 30),
    ("find the total of 7 and 8", "pure_math", 15),
    ("5 + 3", "pure_math", 8),
    ("what's 5 + 3", "pure_math", 8),
    ("10 + 20 + 30", "pure_math", 60),
]

MATH_ANSWERABLE_CASES = [
    ("If I have 5 apples and add 3 more, how many do I have?", "math_answerable", 8),
    ("John has 10 apples and buys 5 more", "math_answerable", 15),
    ("She earned 500 dollars and then earned 300 more", "math_answerable", 800),
    ("I found 3 coins and collected 7 more", "math_answerable", 10),
    ("He started with 100 and won 50", "math_answerable", 150),
    ("I have 20 dollars and spent 8", "math_answerable", 12),
    ("She had 15 cookies and gave away 6", "math_answerable", 9),
    ("He owned 50 shares and sold 20", "math_answerable", 30),
    ("I had 100 dollars, earned 50, and spent 30", "math_answerable", 120),
    ("She started with 10, found 5, and lost 3", "math_answerable", 12),
    ("Adam has 10 chairs, sells 6, and then makes 7 more", "math_answerable", 11),
]

MATH_AUGMENTED_CASES = [
    ("She earned 500 and spent 300, is she managing her money wisely?", "math_augmented", 200),
    ("He had 100 shares and sold 60, should he buy more?", "math_augmented", 40),
    ("I started with 50 and lost 30, explain what happened", "math_augmented", 20),
]

KG_ANSWERABLE_CASES = [
    # --- Exact predicate (forward lookup) ---
    ("What is the capital of Remulak?", "kg_answerable", "Remulak — capital: Zelphos"),
    ("Who is the leader of Remulak?", "kg_answerable", "Remulak — leader: Grand Vizier Korth"),
    ("What is the population of Draveth?", "kg_answerable", "Draveth — population: 1.8 billion"),
    ("What is the population of Sulari?", "kg_answerable", "Sulari — population: 1.4 billion"),
    ("What is the capital of Khotane?", "kg_answerable", "Khotane — capital: Tessavri"),
    ("What is the climate of Yelvri?", "kg_answerable", "Yelvri — climate: polar tundra"),
    ("What is the population of Remulak?", "kg_answerable", "Remulak — population: 4.3 billion"),
    # --- Alias predicates ---
    ("What is the capital city of Remulak?", "kg_answerable", "Remulak — capital: Zelphos"),
    ("Who is the head of state of Remulak?", "kg_answerable", "Remulak — leader: Grand Vizier Korth"),
    ("What is the currency of Remulak?", "kg_answerable", "Remulak — currency: the Vreth"),
    ("What is the weather of Sulari?", "kg_answerable", "Sulari — climate: arid desert highlands"),
    ("What is the government of Remulak?", "kg_answerable", "Remulak — government type: technocratic council"),
    # --- Multi-word entity ---
    ("How old is Grand Vizier Korth?", "kg_answerable", "Grand Vizier Korth — age: 142 standard years"),
    ("Where was Grand Vizier Korth born?", "kg_answerable", "Grand Vizier Korth — birthplace: Zelphos"),
    # --- Request verbs ---
    ("Tell me about the climate of Draveth", "kg_answerable", "Draveth — climate: temperate with long dry seasons"),
    ("Describe the major landmark of Khotane", "kg_answerable", "Khotane — major landmark: The Hanging Forests of Druen"),
    # --- Reverse lookup phrasing ---
    ("What is Draveth known for?", "kg_answerable", "Draveth — known for: government and military academies"),
    ("What is Sulari known for?", "kg_answerable", "Sulari — known for: mining and heavy industry"),
    # --- Technology & military ---
    ("What is the technology level of Remulak?", "kg_answerable", "Remulak — technology level: post-fusion, pre-singularity"),
    ("What is the most popular sport on Remulak?", "kg_answerable", "Remulak — most popular sport: skyracing"),
]

KG_AUGMENTED_CASES = [
    ("Why does Remulak have a technocratic council?", "kg_augmented",
     "technocratic council"),
    ("Should Grand Vizier Korth retire given his age?", "kg_augmented",
     "142 standard years"),
    ("Explain why Draveth is important to Remulak", "kg_augmented",
     "government and military academies"),
]

NEGATIVE_CASES = [
    ("add me to the list", "no_match", None),
    ("the sum of all fears", "no_match", None),
    ("what's your plus side", "no_match", None),
    ("I lost my keys", "no_match", None),
    ("She earned a reputation", "no_match", None),
    ("He gave a speech", "no_match", None),
    ("They found common ground", "no_match", None),
    ("I spent time thinking", "no_match", None),
    ("He sold the idea to his boss", "no_match", None),
    ("She won the argument", "no_match", None),
    ("I have a question", "no_match", None),
    ("hello how are you", "no_match", None),
    ("what's the weather like", "no_match", None),
    ("tell me about the history of mathematics", "no_match", None),
    # KG negatives — entities not in the knowledge graph
    ("What is the capital of France?", "no_match", None),
    ("Who is the president of the United States?", "no_match", None),
    ("What is the population of Tokyo?", "no_match", None),
]

# Adversarial KG negatives — entities exist but the requested predicate doesn't.
# These are used by the benchmark rubric (D1) to verify abstention behavior.
# The pipeline detects the entity and does a subject scan (predicate not found),
# so these classify as kg_augmented — inject facts as context for LLM reasoning.
# Zelphos is object-only (not a subject), so it gets no match.
KG_ADVERSARIAL_NEGATIVES = [
    ("What is the GDP of Remulak?", "kg_augmented", None),
    ("Who was the leader before Grand Vizier Korth?", "kg_augmented", None),
    ("What is the second largest city on Remulak?", "kg_augmented", None),
    ("How many oceans does Remulak have?", "kg_augmented", None),
    ("What is the population of Zelphos?", "no_match", None),
]

# ── Fuzzy matching cases (D5) ──────────────────────────────────────────────
# Format: (prompt, expected_type, expected_result_or_None)
# These test the 3-tier entity/predicate resolution cascade.

KG_FUZZY_ENTITY_ALIAS_CASES = [
    # Entity alias: "Korth" → "Grand Vizier Korth"
    ("How old is Korth?", "kg_answerable", "Grand Vizier Korth — age: 142 standard years"),
    # Entity alias: "Quorum" → "The Quorum of Twelve"
    ("What is the term length of the Quorum?", "kg_answerable", "The Quorum of Twelve — term length: 25 standard years"),
    # Entity alias: "Fracture War" → "The Sulari Fracture War"
    ("How long did the Fracture War last?", "kg_answerable", "The Sulari Fracture War — duration: 12 standard years"),
]

KG_FUZZY_STRING_CASES = [
    # Typo: "Remulack" fuzzy → "remulak"
    ("What is the capital of Remulack?", "kg_answerable", "Remulak — capital: Zelphos"),
    # Typo: "Draevth" fuzzy → "draveth"
    ("What is the population of Draevth?", "kg_answerable", "Draveth — population: 1.8 billion"),
]


# Back-compat alias: tests that import MATH_IN_CONTEXT_CASES get the answerable set
MATH_IN_CONTEXT_CASES = MATH_ANSWERABLE_CASES

ALL_CASES = (
    PURE_MATH_CASES + MATH_ANSWERABLE_CASES + MATH_AUGMENTED_CASES
    + KG_ANSWERABLE_CASES + KG_AUGMENTED_CASES + NEGATIVE_CASES
    + KG_ADVERSARIAL_NEGATIVES
    + KG_FUZZY_ENTITY_ALIAS_CASES + KG_FUZZY_STRING_CASES
)


# ── Golden KG facts — known-correct triplets for purity testing ──────
#
# Format: (subject, predicate, object)
# These must always pass validate_triplet(). If a validation rule change
# causes any of these to fail, the change is reverted.
GOLDEN_KG_FACTS: list[tuple[str, str, str]] = [
    # Miranda v. Arizona
    ("Miranda v. Arizona", "court", "Supreme Court of the United States"),
    ("Miranda v. Arizona", "date_filed", "1966-06-13"),
    ("Miranda v. Arizona", "opinion_author", "Warren"),
    ("Miranda v. Arizona", "cited_by_count", "9832"),
    ("Miranda v. Arizona", "precedential_status", "Published"),
    # Brown v. Board of Education
    ("Brown v. Board of Education", "court", "Supreme Court of the United States"),
    ("Brown v. Board of Education", "date_filed", "1954-05-17"),
    ("Brown v. Board of Education", "opinion_author", "Warren"),
    # Loving v. Virginia
    ("Loving v. Virginia", "court", "Supreme Court of the United States"),
    ("Loving v. Virginia", "date_filed", "1967-06-12"),
    ("Loving v. Virginia", "opinion_author", "Warren"),
    # Roe v. Wade
    ("Roe v. Wade", "court", "Supreme Court of the United States"),
    ("Roe v. Wade", "date_filed", "1973-01-22"),
    ("Roe v. Wade", "opinion_author", "Blackmun"),
    # Marbury v. Madison
    ("Marbury v. Madison", "court", "Supreme Court of the United States"),
    ("Marbury v. Madison", "date_filed", "1803-02-24"),
    ("Marbury v. Madison", "opinion_author", "Marshall"),
    ("Marbury v. Madison", "per_curiam", "false"),
    # Gideon v. Wainwright
    ("Gideon v. Wainwright", "court", "Supreme Court of the United States"),
    ("Gideon v. Wainwright", "opinion_author", "Black"),
    # Terry v. Ohio
    ("Terry v. Ohio", "court", "Supreme Court of the United States"),
    ("Terry v. Ohio", "date_filed", "1968-06-10"),
    # Dred Scott v. Sandford
    ("Dred Scott v. Sandford", "court", "Supreme Court of the United States"),
    ("Dred Scott v. Sandford", "opinion_author", "Taney"),
    # Edge cases: legitimate legal formats
    ("In re Gault", "court", "Supreme Court of the United States"),
    ("Ex parte Milligan", "court", "Supreme Court of the United States"),
]


# Known-bad triplets — these must always FAIL validate_triplet().
KNOWN_BAD_TRIPLETS: list[tuple[str, str, str]] = [
    ("it", "have", "effect"),
    ("we", "blind", "ourselves"),
    ("they", "consider", "problems"),
    ("Lovings", "date_filed", "convicted of violating § 20-58 of the Virginia Code"),
    ("parties", "date_filed", "briefs"),
    ("defendant", "yield to", "prosecution"),
    ("court", "require", "evidence"),
    ("Brown v. Board", "cited_by_count", "many times"),
    ("this case", "constitute", "landmark"),
]
