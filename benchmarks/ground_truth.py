"""
Ground-truth benchmark questions for the Remulak knowledge graph.

Each entry: (question, ground_truth_answer, match_strings)
  - question: natural language prompt
  - ground_truth_answer: the correct factual answer
  - match_strings: list of substrings that MUST appear in a correct response
    (case-insensitive). A response is "correct" if ALL match strings are present.

These questions are designed so:
  1. A naked LLM cannot answer them (Remulak is fictional)
  2. Crystal + KG should answer them exactly from the knowledge graph
"""

BENCHMARK_CASES: list[tuple[str, str, list[str]]] = [
    # --- Geography ---
    (
        "What is the capital of Remulak?",
        "Zelphos",
        ["zelphos"],
    ),
    (
        "What star system is Remulak in?",
        "Veldra-7",
        ["veldra-7"],
    ),
    (
        "How many moons does Remulak have?",
        "3",
        ["3"],
    ),
    (
        "What is the largest continent on Remulak?",
        "Draveth",
        ["draveth"],
    ),
    (
        "What is the diameter of Remulak?",
        "14,200 km",
        ["14,200"],
    ),
    # --- Continents ---
    (
        "What is the population of Draveth?",
        "1.8 billion",
        ["1.8 billion"],
    ),
    (
        "What is the capital of Sulari?",
        "Mevrath",
        ["mevrath"],
    ),
    (
        "What is Khotane known for?",
        "agriculture and bioengineering",
        ["agriculture", "bioengineering"],
    ),
    (
        "What is the climate of Yelvri?",
        "polar tundra",
        ["polar tundra"],
    ),
    # --- Government ---
    (
        "Who is the leader of Remulak?",
        "Grand Vizier Korth",
        ["grand vizier korth"],
    ),
    (
        "What is the governing body of Remulak?",
        "The Quorum of Twelve",
        ["quorum of twelve"],
    ),
    (
        "How old is Grand Vizier Korth?",
        "142 standard years",
        ["142"],
    ),
    (
        "Where was Grand Vizier Korth born?",
        "Zelphos",
        ["zelphos"],
    ),
    # --- Demographics ---
    (
        "What is the population of Remulak?",
        "4.3 billion",
        ["4.3 billion"],
    ),
    (
        "What is the official language of Remulak?",
        "Veldrasi",
        ["veldrasi"],
    ),
    # --- Economy ---
    (
        "What is the currency of Remulak?",
        "the Vreth",
        ["vreth"],
    ),
    (
        "What is the primary trade partner of Remulak?",
        "the Confederacy of Talshek",
        ["talshek"],
    ),
    # --- Military & Technology ---
    (
        "What is the technology level of Remulak?",
        "post-fusion, pre-singularity",
        ["post-fusion"],
    ),
    (
        "What is the most popular sport on Remulak?",
        "skyracing",
        ["skyracing"],
    ),
    # --- Culture & History ---
    (
        "What is the major holiday on Remulak?",
        "The Festival of Vohn",
        ["festival of vohn"],
    ),
]
