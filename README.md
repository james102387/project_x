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

## Planned Tools

- **Knowledge Graph** — Static hash table lookup with synonym resolution
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
│   ├── detectors/
│   │   ├── calculator.py   # Explicit math pattern matchers
│   │   └── semantic.py     # Verb-semantic word problem detection
│   ├── tools/
│   │   └── calculator.py   # NumPy calculator execution
│   └── nodes/
│       ├── parser.py       # spaCy parser node
│       ├── detector.py     # Calculator detector node
│       ├── planner.py      # Plan builder node
│       ├── preprocessor.py # Preprocessor node
│       ├── compiler.py     # Prompt compiler node
│       └── llm_nodes.py    # LLM augmented + fallback nodes
├── tests/
│   ├── golden/
│   │   └── test_cases.py   # Hand-crafted golden test cases
│   ├── test_detectors.py   # Unit tests for pattern matchers
│   ├── test_semantic.py    # Unit tests for semantic verb detection
│   ├── test_compiler.py    # Unit tests for prompt classification + compilation
│   ├── test_pipeline.py    # Integration tests (full local pipeline)
│   └── conftest.py         # Shared fixtures (spaCy nlp, sample docs)
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
