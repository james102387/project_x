"""
Holdout golden benchmark — second batch of opinion questions for validation.

This file is populated AFTER the Ralph Wiggum loop has been optimized on
opinion_golden.py. If extraction improvements from batch one generalize to
batch two, the system is not overfit.

Use a DIFFERENT set of cases than opinion_golden.py.
Suggested: 4th Amendment cases (Mapp v. Ohio, Terry v. Ohio, Katz v. United
States) or Commerce Clause cases.

IMPORTANT: Hand-authored by a human. See opinion_golden.py for workflow.

Format: (question, golden_answer, match_strings, is_negative)
"""

OPINION_HOLDOUT_CASES: list[tuple[str, str, list[str], bool]] = []
