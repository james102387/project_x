# Crystal — Neuro-Symbolic Prompt Compiler

A prompt compiler for LLMs that deterministically evaluates structured expressions 
(math, knowledge graph lookups) and injects verified results into prompts before 
the LLM sees them.

## Core Principle

**Crystal cannot return a less accurate answer than the LLM would alone.**  
If no pattern matches, the prompt passes through to the LLM untouched.

## Architecture

```
Input → spaCy Parser → Tool Detectors → Plan Builder → Preprocessor → Tool Execution → Prompt Compiler
    → (pure math)       → Direct Return (no LLM)
    → (math in context) → Simplified Prompt → LLM → Response
    → (no match)        → Raw Prompt → LLM → Response
```

Each stage is a LangGraph node. The state object flows through all nodes 
and accumulates information at each step.

## Current Tools

- **Calculator** — Explicit math (`5 + 3`, `add 5 and 3`, `the sum of 10 and 20`)
- **Calculator (Semantic)** — Implied math via verb classification (`John has 10 apples and buys 5 more`)
- **Knowledge Graph** — Generic hash-table KG engine over `(subject, predicate, object)` triplets with forward and reverse indexes

## Synthetic Datasets

- **Remulak** — 75 triplets describing a fictional alien planet (geography, government, economy, culture). No LLM can answer from training data, proving KG value.

## Planned Tools

- **KG Crystallization** — Automated knowledge promotion and maintenance

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set your LLM API key:
```bash
export GOOGLE_API_KEY="your-key-here"
```

## Running Tests

```bash
# Golden tests only (no LLM, no API key needed)
pytest tests/ -m "not llm"

# Full test suite (requires API key)
pytest tests/
```

## Project Structure

```
crystal/
├── .cursorrules            # Minimal agent rules (session workflow + hard constraints)
├── src/crystal/
│   ├── state.py            # CrystalState schema
│   ├── graph.py            # LangGraph wiring
│   ├── llm.py              # LLM client wrapper
│   ├── metrics.py          # Token counting and savings metrics
│   ├── data/
│   │   └── remulak.py      # Remulak synthetic triplet dataset (pure data)
│   ├── detectors/
│   │   ├── math/
│   │   │   ├── explicit.py # Explicit math pattern matchers
│   │   │   └── semantic.py # Verb-semantic word problem detection
│   │   └── kg.py           # KG entity + question structure detector
│   ├── tools/
│   │   ├── calculator.py   # NumPy calculator execution
│   │   └── kg/
│   │       ├── graph.py    # Generic KG hash-table engine
│   │       └── remulak.py  # Remulak convenience instance
│   └── nodes/
│       ├── parser.py         # spaCy parser node
│       ├── math_detection.py # Math detection node
│       ├── kg_detection.py   # KG detection node
│       ├── kg.py             # KG execution node
│       ├── planner.py        # Plan builder node
│       ├── preprocessor.py   # Preprocessor node
│       ├── compiler.py       # Prompt compiler node
│       └── llm_nodes.py      # LLM augmented + fallback nodes
├── tests/
│   ├── conftest.py         # Shared fixtures (spaCy nlp, LLM cache)
│   ├── golden/
│   │   └── test_cases.py   # Hand-crafted golden test cases
│   ├── fixtures/
│   │   └── llm_cache.json  # Cached LLM responses for offline testing
│   ├── unit/
│   │   ├── detectors/      # Calculator, semantic, KG detector tests
│   │   ├── tools/          # KG tool (lookup, aliases, entity index) tests
│   │   └── nodes/          # Compiler, metrics tests
│   └── integration/
│       ├── test_pipeline.py  # Full local pipeline (no LLM)
│       └── test_llm.py       # Full LangGraph pipeline (requires API key)
├── scripts/
│   └── run.py              # Interactive runner (golden tests, single prompts, parse trees)
├── docs/
│   ├── ARCHITECTURE.md     # Design decisions and rationale
│   ├── PATTERNS.md         # Documented spaCy patterns with dep trees
│   ├── DEVLOG.md           # Persistent development journal (LLM reads this at session start)
│   └── SESSION.md          # Ephemeral session scratchpad (cleared between sessions)
├── requirements.txt
└── pyproject.toml
```

## LLM Workflow

When starting a new coding session with an LLM:

1. Point it at `docs/DEVLOG.md` for full project history and known issues
2. Point it at `docs/SESSION.md` for the current session scratchpad
3. The LLM writes observations and reasoning to `SESSION.md` as it works
4. At session end, archive findings from `SESSION.md` into `DEVLOG.md`
