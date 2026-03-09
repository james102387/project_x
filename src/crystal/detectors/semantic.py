"""
Semantic verb detector for implied math in word problems.

Classifies verbs into acquire/lose/state categories and extracts
multi-step arithmetic operations from natural language.

Example: "John has 10 apples and buys 5 more"
    → [{"op": "state", "value": 10, "verb": "has"},
       {"op": "add",   "value": 5,  "verb": "buys"}]
    → result: 15
"""

from .calculator import extract_numbers

# ---------------------------------------------------------------------------
# Verb-semantic classification
# High-confidence, low-ambiguity verbs only.
# ---------------------------------------------------------------------------
ACQUIRE_VERBS = {"buy", "earn", "receive", "gain", "find", "collect", "win", "pick", "add", "make"}
LOSE_VERBS = {"sell", "spend", "give", "lose", "donate", "drop", "pay"}
STATE_VERBS = {"have", "own", "hold", "start", "begin", "carry"}

# Directional modifiers that reinforce acquire/lose semantics
ACQUIRE_MODIFIERS = {"more", "extra", "additional", "another"}
LOSE_MODIFIERS = {"less", "fewer", "away"}

# Combined set (useful for prompt type classification)
ALL_SEMANTIC_VERBS = ACQUIRE_VERBS | LOSE_VERBS | STATE_VERBS


# ---------------------------------------------------------------------------
# Semantic verb pattern matcher
# ---------------------------------------------------------------------------
def match_semantic_verb_pattern(doc) -> list[dict] | None:
    """
    Detect implied math from verb semantics in word problems.

    Walks each VERB token, classifies it as state/acquire/lose, and
    extracts NUM tokens from its direct children (not full subtree).
    Skips conjoined verb children to avoid cross-clause number capture.

    Returns a list of operation steps, or None if < 2 verb-number pairs found.
    """
    steps = []
    seen_num_indices = set()

    for token in doc:
        if token.pos_ != "VERB":
            continue

        lemma = token.lemma_.lower()

        op = None
        if lemma in STATE_VERBS:
            op = "state"
        elif lemma in ACQUIRE_VERBS:
            op = "add"
        elif lemma in LOSE_VERBS:
            op = "subtract"
        else:
            continue

        # Collect NUM tokens from direct children + one level deep.
        # IMPORTANT: skip children that are conjoined verbs (dep="conj" and
        # pos="VERB") — those will be processed as their own verb in the loop.
        num_tokens = []
        for child in token.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                continue
            if child.pos_ == "NUM" and child.i not in seen_num_indices:
                num_tokens.append(child)
            for grandchild in child.children:
                if grandchild.dep_ == "conj" and grandchild.pos_ == "VERB":
                    continue
                if grandchild.pos_ == "NUM" and grandchild.i not in seen_num_indices:
                    num_tokens.append(grandchild)

        if not num_tokens:
            continue

        numbers = extract_numbers(num_tokens)
        if not numbers:
            continue

        # Check for directional modifiers
        child_texts = set()
        for child in token.children:
            child_texts.add(child.text.lower())
            for grandchild in child.children:
                child_texts.add(grandchild.text.lower())

        has_acquire_mod = bool(child_texts & ACQUIRE_MODIFIERS)
        has_lose_mod = bool(child_texts & LOSE_MODIFIERS)

        if has_lose_mod and op == "state":
            op = "subtract"
        elif has_acquire_mod and op == "state":
            op = "add"

        # Mark NUM tokens as claimed
        for nt in num_tokens:
            seen_num_indices.add(nt.i)

        for num in numbers:
            steps.append({"op": op, "value": num, "verb": token.text})

    if len(steps) < 2:
        return None
    return steps


# ---------------------------------------------------------------------------
# Step evaluator
# ---------------------------------------------------------------------------
def evaluate_semantic_steps(steps: list[dict]) -> dict | None:
    """
    Evaluate a sequence of semantic verb steps into a final result.

    Rules:
        - First "state" verb sets the initial value
        - If no "state" verb, first "add" verb sets the initial value
        - Subsequent "add" steps add to the running total
        - "subtract" steps subtract from the running total
        - Additional "state" verbs treated as add
    """
    if not steps:
        return None

    result = 0
    start_idx = 0

    # Look for a state verb first
    for i, step in enumerate(steps):
        if step["op"] == "state":
            result = step["value"]
            start_idx = i + 1
            break
    else:
        first = steps[0]
        result = first["value"]
        start_idx = 1

    for step in steps[start_idx:]:
        if step["op"] == "add":
            result += step["value"]
        elif step["op"] == "subtract":
            result -= step["value"]
        elif step["op"] == "state":
            result += step["value"]

    args = [s["value"] for s in steps]

    return {
        "operation": "semantic_math",
        "args": args,
        "steps": steps,
        "result": result,
    }
