# Architecture Decisions

## Core Principle
Crystal cannot return a less accurate answer than the LLM would alone.
This means: strict pattern matching, no fuzzy routing, and the LLM is always the fallback.

## Router as Optimizer, Not Gatekeeper
The router only intercepts when it has high confidence it can improve on the LLM.
A false positive (routing a non-math query to the calculator) is worse than a false
negative (letting a math query fall through to the LLM). The LLM handles false
negatives gracefully; the calculator does not handle false positives at all.

## Why spaCy Over Regex
Regex catches explicit math patterns but fails on ambiguity: "add me to the list"
triggers on "add". spaCy's dependency parse lets us check that "add" governs NUM
tokens, eliminating false positives structurally rather than through brittle exclusion lists.

## Prompt Compiler, Not Answer Generator
Crystal does not replace the LLM. It evaluates deterministic expressions and injects
verified results into the prompt. The LLM still owns the conversation. Three paths:
1. **Pure math** — return result directly, skip LLM entirely
2. **Math in context** — simplify the prompt with pre-computed results, send to LLM
3. **No match** — pass through to LLM untouched

## Explicit Patterns Before Semantic
Explicit math ("5 + 3", "add 5 and 3") is high-confidence, low-ambiguity.
Semantic verb detection ("John buys 5 more") is medium-confidence.
Explicit always fires first; semantic only runs if explicit found nothing.

## Semantic Verb Scope
Limited to ~15 high-confidence verbs to avoid brittleness:
- **Acquire:** buy, earn, receive, gain, find, collect, win, pick, add
- **Lose:** sell, spend, give, lose, donate, drop, pay
- **State:** have, own, hold, start, begin, carry

Ambiguous verbs (make, get, take, break) are deliberately excluded.

## Plan Schema
The compiler plan is a list by design, even though the MVP only produces single-item
plans. This avoids refactoring when multi-step execution is added later.

## LangGraph Node Design
Each pipeline stage is an independent node with a single responsibility.
The CrystalState TypedDict flows through all nodes. Nodes never call each
other directly — they communicate exclusively through state.
