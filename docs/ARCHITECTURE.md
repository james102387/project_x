# Architecture Decisions

## Core Principle
Crystal cannot return a less accurate answer than the LLM would alone.
This means: strict pattern matching, no fuzzy routing, and the LLM is always the fallback.

## Router as Optimizer, Not Gatekeeper
The router only intercepts when it has high confidence it can improve on the LLM.
A false positive (routing to a tool that can't handle the query) is worse than a
false negative (letting a query fall through to the LLM). The LLM handles false
negatives gracefully; deterministic tools do not handle false positives at all.

## Prompt Compiler, Not Answer Generator
Crystal does not replace the LLM. It evaluates deterministic expressions and injects
verified results into the prompt. The LLM still owns the conversation. Three paths:
1. **Tool-answerable** — tool returns a complete answer, skip LLM entirely
2. **Tool-augmented** — inject grounded results into a simplified prompt for the LLM
3. **No match** — pass through to LLM untouched

Each tool defines its own "answerable" vs "augmented" boundary. The compiler
classifies based on what the tool detected and whether the query requires
reasoning beyond the tool's output.

## Detector → Tool → Compiler Pipeline
Every capability follows the same three-stage pattern:

1. **Detector** — analyzes raw text, emits structured detections into state
2. **Tool/Execution node** — runs the deterministic computation
3. **Compiler** — classifies the result and either returns it or augments the LLM prompt

Adding a new capability means implementing these three pieces and wiring them
into the graph. The planner merges detections from all active detectors into
a unified plan.

## LangGraph Node Design
Each pipeline stage is an independent node with a single responsibility.
The CrystalState TypedDict flows through all nodes. Nodes never call each
other directly — they communicate exclusively through state.

## Plan Schema
The compiler plan is a list by design, even though the MVP only produces single-item
plans. This avoids refactoring when multi-step execution is added later.

## Data Separate From Tools
Pure data (triplet corpora, lookup tables, vocabularies) lives in `data/`.
Tool logic (execution engines, algorithms) lives in `tools/`. This prevents
coupling between a tool's interface and a specific dataset, and makes it
trivial to swap or add datasets without touching tool code.

---

## Tool-Specific Notes

### Calculator (Math Detection)

**Why spaCy over regex:** Regex catches explicit math patterns but fails on
ambiguity: "add me to the list" triggers on "add". spaCy's dependency parse
lets us check that "add" governs NUM tokens, eliminating false positives
structurally rather than through brittle exclusion lists.

**Explicit before semantic:** Explicit math ("5 + 3") is high-confidence.
Semantic verb detection ("John buys 5 more") is medium-confidence. Explicit
always fires first; semantic only runs if explicit found nothing.

**Semantic verb scope:** Limited to ~15 high-confidence verbs to avoid
brittleness. Ambiguous verbs (make, get, take, break) are deliberately excluded.

### Knowledge Graph

**Generic engine, pluggable data:** The KG tool (`tools/kg/graph.py`) is a
dataset-agnostic hash-table engine. It accepts `(subject, predicate, object)`
triplets and builds forward/reverse indexes for O(1) lookup.

**Dataset wiring:** Synthetic datasets live in `data/` as pure data modules.
Instance modules (e.g. `tools/kg/remulak.py`) instantiate the engine with a
specific dataset. Adding a new KG corpus = write a data module +
instantiate `KnowledgeGraph(triplets)` in a new file under `tools/kg/`.

**Two-stage detection:** Entity recognition (hash lookup against the KG's
entity index) followed by question-structure analysis with predicate extraction.
