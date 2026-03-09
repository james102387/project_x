# Session Journal

Ephemeral scratchpad for the current working session.
Archive useful findings into `DEVLOG.md` before clearing.

---

## How to use this file

**At session start:** The LLM reads this + `DEVLOG.md` for context.

**During session:** Write observations and reasoning here as you work.

**At session end:** Move anything worth keeping into `DEVLOG.md`.
Then clear the "Current Session" section for next time.

## Current Session

- Added three savings views to metrics: token_savings_pct, savings_pct (legacy isolated N+N²), marginal_savings_pct
- marginal_cost() function: N + 2BN + N² where B = base context (default 2000 tokens)
- Key finding: for math_augmented, token savings is ~-314% (3-4x more tokens), marginal is ~-318%, but isolated N+N² was -1535% (wildly overstated)
- Token and marginal views agree closely — the quadratic term is negligible when the prompt is small relative to base context
- 95/95 tests passing, 42/42 golden cases passing
