"""
Ground-truth benchmark questions for the Remulak knowledge graph.

Each entry: (question, ground_truth_answer, match_strings, is_negative)
  - question: natural language prompt
  - ground_truth_answer: the correct factual answer
  - match_strings: list of substrings that MUST appear in a correct response
    (case-insensitive). A response is "correct" if ALL match strings are present.
  - is_negative: True if the system should abstain (no KG data exists).
    Defaults to False for backward compatibility — scoring checks tuple length.

These questions are designed so:
  1. A naked LLM cannot answer them (Remulak is fictional)
  2. Crystal + KG should answer them exactly from the knowledge graph
  3. Negative cases verify the system abstains rather than fabricates
"""

BENCHMARK_CASES: list[tuple[str, str, list[str], bool]] = [
    # --- Geography ---
    (
        "What is the capital of Remulak?",
        "Zelphos",
        ["zelphos"],
        False,
    ),
    (
        "What star system is Remulak in?",
        "Veldra-7",
        ["veldra-7"],
        False,
    ),
    (
        "How many moons does Remulak have?",
        "3",
        ["3"],
        False,
    ),
    (
        "What is the largest continent on Remulak?",
        "Draveth",
        ["draveth"],
        False,
    ),
    (
        "What is the diameter of Remulak?",
        "14,200 km",
        ["14,200"],
        False,
    ),
    # --- Continents ---
    (
        "What is the population of Draveth?",
        "1.8 billion",
        ["1.8 billion"],
        False,
    ),
    (
        "What is the capital of Sulari?",
        "Mevrath",
        ["mevrath"],
        False,
    ),
    (
        "What is Khotane known for?",
        "agriculture and bioengineering",
        ["agriculture", "bioengineering"],
        False,
    ),
    (
        "What is the climate of Yelvri?",
        "polar tundra",
        ["polar tundra"],
        False,
    ),
    # --- Government ---
    (
        "Who is the leader of Remulak?",
        "Grand Vizier Korth",
        ["grand vizier korth"],
        False,
    ),
    (
        "What is the governing body of Remulak?",
        "The Quorum of Twelve",
        ["quorum of twelve"],
        False,
    ),
    (
        "How old is Grand Vizier Korth?",
        "142 standard years",
        ["142"],
        False,
    ),
    (
        "Where was Grand Vizier Korth born?",
        "Zelphos",
        ["zelphos"],
        False,
    ),
    # --- Demographics ---
    (
        "What is the population of Remulak?",
        "4.3 billion",
        ["4.3 billion"],
        False,
    ),
    (
        "What is the official language of Remulak?",
        "Veldrasi",
        ["veldrasi"],
        False,
    ),
    # --- Economy ---
    (
        "What is the currency of Remulak?",
        "the Vreth",
        ["vreth"],
        False,
    ),
    (
        "What is the primary trade partner of Remulak?",
        "the Confederacy of Talshek",
        ["talshek"],
        False,
    ),
    # --- Military & Technology ---
    (
        "What is the technology level of Remulak?",
        "post-fusion, pre-singularity",
        ["post-fusion"],
        False,
    ),
    (
        "What is the most popular sport on Remulak?",
        "skyracing",
        ["skyracing"],
        False,
    ),
    # --- Culture & History ---
    (
        "What is the major holiday on Remulak?",
        "The Festival of Vohn",
        ["festival of vohn"],
        False,
    ),
    # --- Adversarial negatives (no KG triplet exists) ---
    (
        "What is the GDP of Remulak?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "Who was the leader before Grand Vizier Korth?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What is the second largest city on Remulak?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "How many oceans does Remulak have?",
        "[ABSTAIN]",
        [],
        True,
    ),
    (
        "What is the population of Zelphos?",
        "[ABSTAIN]",
        [],
        True,
    ),
]
