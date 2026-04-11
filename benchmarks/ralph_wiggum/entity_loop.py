"""EntityLoop — mutates entity alias tables.

Handles: entity_mismatch failures.
Target files: data/legal_ontology.py (entity alias additions only)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from benchmarks.ralph_wiggum.base import (
    BaseLoop,
    FailureCategory,
)

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent.parent / "src"
_ALIAS_FILE = _SRC_ROOT / "crystal" / "data" / "legal_ontology.py"

_PROMPT = """\
You are optimizing a question-answering system's entity resolution layer.
The system resolves entity names in questions to knowledge graph subjects.

Below are questions where the system matched the WRONG entity or no entity at all.

FAILURES:
{failures}

Your task: propose entity alias ADDITIONS so these questions resolve correctly.
Entity aliases map alternative names to canonical entity names in the KG.

Rules:
1. Only propose additions (never deletions or renames)
2. Keys = alternative name (as it appears in questions), Values = canonical KG entity name
3. For negative cases (is_negative=True), do NOT propose aliases
4. Be conservative — only propose aliases you're confident about

Respond with ONLY a JSON object:
```json
{{
  "entity_aliases": {{"alt_name": "canonical_entity", ...}}
}}
```
If no useful changes: {{}}
"""


class EntityLoop(BaseLoop):
    LOOP_NAME = "EntityLoop"
    FAILURE_CATEGORIES = frozenset({FailureCategory.ENTITY_MISMATCH})
    TARGET_FILES = [_ALIAS_FILE]

    def _build_proposal_prompt(self, failures: list[dict]) -> str:
        failure_text = ""
        for f in failures[:15]:
            failure_text += (
                f"  Question: {f['question']}\n"
                f"  Golden answer: {f['golden_answer']}\n"
                f"  Detected entity: {f.get('detected_entity', 'None')}\n"
                f"  Match tier: {f.get('match_tier', 'None')}\n"
                f"  Match score: {f.get('match_score', 'None')}\n"
                f"  ---\n"
            )

        return _PROMPT.format(failures=failure_text)

    def _validate_proposal(self, proposal: dict) -> bool:
        if not proposal or not isinstance(proposal, dict):
            return False
        if "entity_aliases" not in proposal:
            return False
        for key in proposal:
            if key != "entity_aliases":
                return False
        section = proposal["entity_aliases"]
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
        aliases = proposal.get("entity_aliases", {})
        if aliases:
            counts["entity_aliases"] = 0
            for alias_key, canonical in aliases.items():
                logger.info(
                    "Entity alias proposed: %s → %s (requires manual addition to KG)",
                    alias_key, canonical,
                )
                counts["entity_aliases"] += 1
        return counts

    def _revert_proposal(self, proposal: dict) -> None:
        pass
