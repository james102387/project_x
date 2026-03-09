"""
Golden test cases — hand-crafted ground truth.

Every case has a known expected outcome. No LLM calls needed.
Format: (prompt, expected_type, expected_result_or_None)

Prompt types:
  pure_math       — bare arithmetic, skip LLM entirely
  math_answerable — narrative math where Crystal has the full answer, skip LLM
  math_augmented  — math embedded in a question requiring LLM reasoning
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
]

# Back-compat alias: tests that import MATH_IN_CONTEXT_CASES get the answerable set
MATH_IN_CONTEXT_CASES = MATH_ANSWERABLE_CASES

ALL_CASES = PURE_MATH_CASES + MATH_ANSWERABLE_CASES + MATH_AUGMENTED_CASES + NEGATIVE_CASES
