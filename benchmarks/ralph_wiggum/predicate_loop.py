"""PredicateLoop — mutates QUESTION_PREDICATE_MAP and LEGAL_PREDICATE_ALIASES.

Handles: predicate_mismatch failures.
Target files: detectors/kg.py, data/legal_ontology.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from benchmarks.ralph_wiggum.base import (
    BaseLoop,
    FailureCategory,
    insert_dict_entries,
    remove_dict_entries,
)

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent.parent / "src"
_PREDICATE_MAP_FILE = _SRC_ROOT / "crystal" / "detectors" / "kg.py"
_ALIAS_FILE = _SRC_ROOT / "crystal" / "data" / "legal_ontology.py"

_PROMPT = """\
You are optimizing a question-answering system's predicate detection layer.
The system maps natural language phrases to knowledge graph predicates.

Below are questions where the system found the RIGHT entity but the WRONG predicate.

FAILURES:
{failures}

CURRENT QUESTION_PREDICATE_MAP (extracted phrases → KG predicates):
{predicate_map}

CURRENT LEGAL_PREDICATE_ALIASES (surface forms → canonical predicates):
{aliases}

Propose ADDITIONS to fix as many failures as possible.

Rules:
1. Only propose additions (never deletions)
2. predicate_map keys = extracted predicate phrases, values = KG predicates
3. predicate_aliases keys = surface forms, values = canonical predicates
4. For negative cases (is_negative=True), do NOT propose mappings

Respond with ONLY a JSON object:
```json
{{
  "predicate_map": {{"phrase": "kg_predicate", ...}},
  "predicate_aliases": {{"surface_form": "canonical_predicate", ...}}
}}
```
Include only sections with changes. If no useful changes: {{}}
"""


class PredicateLoop(BaseLoop):
    LOOP_NAME = "PredicateLoop"
    FAILURE_CATEGORIES = frozenset({FailureCategory.PREDICATE_MISMATCH})
    TARGET_FILES = [_PREDICATE_MAP_FILE, _ALIAS_FILE]

    def _build_proposal_prompt(self, failures: list[dict]) -> str:
        from crystal.detectors.kg import QUESTION_PREDICATE_MAP
        from crystal.data.legal_ontology import LEGAL_PREDICATE_ALIASES

        failure_text = ""
        for f in failures[:15]:
            failure_text += (
                f"  Question: {f['question']}\n"
                f"  Golden answer: {f['golden_answer']}\n"
                f"  Predicate phrase: {f.get('predicate_phrase', 'None')}\n"
                f"  Detected entity: {f.get('detected_entity', 'None')}\n"
                f"  ---\n"
            )

        return _PROMPT.format(
            failures=failure_text,
            predicate_map=json.dumps(QUESTION_PREDICATE_MAP, indent=2),
            aliases=json.dumps(dict(LEGAL_PREDICATE_ALIASES), indent=2),
        )

    def _validate_proposal(self, proposal: dict) -> bool:
        allowed = {"predicate_map", "predicate_aliases"}
        if not proposal or not isinstance(proposal, dict):
            return False
        if not any(k in allowed for k in proposal):
            return False
        for key in proposal:
            if key not in allowed:
                return False
            section = proposal[key]
            if not isinstance(section, dict):
                return False
            for k, v in section.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    return False
                if not k.strip() or not v.strip():
                    return False
        return True

    def _apply_proposal(self, proposal: dict) -> dict[str, int]:
        counts: dict[str, int] = {}
        if "predicate_map" in proposal and proposal["predicate_map"]:
            counts["predicate_map"] = insert_dict_entries(
                _PREDICATE_MAP_FILE, "QUESTION_PREDICATE_MAP",
                proposal["predicate_map"],
            )
        if "predicate_aliases" in proposal and proposal["predicate_aliases"]:
            counts["predicate_aliases"] = insert_dict_entries(
                _ALIAS_FILE, "LEGAL_PREDICATE_ALIASES",
                proposal["predicate_aliases"],
            )
        return counts

    def _revert_proposal(self, proposal: dict) -> None:
        if "predicate_map" in proposal:
            remove_dict_entries(
                _PREDICATE_MAP_FILE, "QUESTION_PREDICATE_MAP",
                proposal["predicate_map"],
            )
        if "predicate_aliases" in proposal:
            remove_dict_entries(
                _ALIAS_FILE, "LEGAL_PREDICATE_ALIASES",
                proposal["predicate_aliases"],
            )
