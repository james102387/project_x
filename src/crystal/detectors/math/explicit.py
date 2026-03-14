"""
Explicit math pattern matchers.

Each matcher takes a spaCy Doc and returns a list of numbers if the pattern
matches, or None if it doesn't. These detect unambiguous, explicit math
expressions like "5 + 3", "add 5 and 3", "the sum of 10 and 20".
"""

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------
ADDITION_VERBS = {"add"}
ADDITION_CONJUNCTIONS = {"plus"}
ADDITION_NOUNS = {"sum", "total"}
ADDITION_SYMBOLS = {"+"}


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def extract_numbers(tokens) -> list[int | float]:
    """Extract numeric values from spaCy tokens with POS=NUM."""
    numbers = []
    for t in tokens:
        try:
            numbers.append(float(t.text) if "." in t.text else int(t.text))
        except ValueError:
            pass
    return numbers


# ---------------------------------------------------------------------------
# Pattern: Verb-led — "add 5 and 3", "can you add 12 to 8"
# ---------------------------------------------------------------------------
def match_verb_pattern(doc) -> list[int | float] | None:
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in ADDITION_VERBS:
            num_tokens = [child for child in token.subtree if child.pos_ == "NUM"]
            numbers = extract_numbers(num_tokens)
            if len(numbers) >= 2:
                return numbers
    return None


# ---------------------------------------------------------------------------
# Pattern: Conjunction — "5 plus 3", "what is 10 plus 20"
# ---------------------------------------------------------------------------
def match_conjunction_pattern(doc) -> list[int | float] | None:
    for token in doc:
        if token.pos_ == "CCONJ" and token.lemma_.lower() in ADDITION_CONJUNCTIONS:
            head = token.head
            if head.pos_ == "NUM":
                num_tokens = [t for t in doc if t.pos_ == "NUM"]
                numbers = extract_numbers(num_tokens)
                if len(numbers) >= 2:
                    return numbers
    return None


# ---------------------------------------------------------------------------
# Pattern: Noun-led — "the sum of 5 and 3", "find the total of 7 and 8"
# ---------------------------------------------------------------------------
def match_noun_pattern(doc) -> list[int | float] | None:
    for token in doc:
        if token.pos_ == "NOUN" and token.lemma_.lower() in ADDITION_NOUNS:
            for child in token.children:
                if child.dep_ == "prep" and child.lemma_.lower() == "of":
                    num_tokens = [t for t in child.subtree if t.pos_ == "NUM"]
                    numbers = extract_numbers(num_tokens)
                    if len(numbers) >= 2:
                        return numbers
    return None


# ---------------------------------------------------------------------------
# Pattern: Symbol — "5 + 3", "10 + 20 + 30"
# ---------------------------------------------------------------------------
def match_symbol_pattern(doc) -> list[int | float] | None:
    plus_tokens = [t for t in doc if t.text == "+"]
    if not plus_tokens:
        return None
    num_tokens = [t for t in doc if t.pos_ == "NUM" and t.text != "+"]
    numbers = extract_numbers(num_tokens)
    if len(numbers) >= 2:
        return numbers
    return None


# ---------------------------------------------------------------------------
# Registry — order matters, first match wins
# ---------------------------------------------------------------------------
EXPLICIT_PATTERNS = [
    ("verb", match_verb_pattern),
    ("conjunction", match_conjunction_pattern),
    ("noun", match_noun_pattern),
    ("symbol", match_symbol_pattern),
]
