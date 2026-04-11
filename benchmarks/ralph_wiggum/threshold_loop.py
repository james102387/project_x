"""ThresholdLoop — mutates CONFIDENCE_LOW numeric threshold.

Handles: routing_error failures.
Target files: nodes/planner.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from benchmarks.ralph_wiggum.base import (
    BaseLoop,
    FailureCategory,
)

logger = logging.getLogger(__name__)

_SRC_ROOT = Path(__file__).parent.parent.parent / "src"
_PLANNER_FILE = _SRC_ROOT / "crystal" / "nodes" / "planner.py"

MIN_THRESHOLD = 0.5
MAX_THRESHOLD = 0.85

_PROMPT = """\
You are tuning a confidence threshold for a question-answering system's routing.
The system uses CONFIDENCE_LOW to decide whether to trust a KG match or fall back to the LLM.

Current CONFIDENCE_LOW = {current_threshold}
Allowed range: [{min_threshold}, {max_threshold}]

Below are routing failures — questions where the system either:
- Routed TO the KG when it should have fallen back to the LLM, or
- Routed AWAY from the KG when it should have used the KG

FAILURES:
{failures}

Rules:
1. Only propose a threshold change if 3+ routing failures suggest it would help
2. Lowering the threshold means MORE questions use KG (riskier but more coverage)
3. Raising the threshold means FEWER questions use KG (safer but less coverage)
4. Stay within [{min_threshold}, {max_threshold}]

Respond with ONLY a JSON object:
```json
{{
  "confidence_threshold": 0.XX
}}
```
If no change needed: {{}}
"""


class ThresholdLoop(BaseLoop):
    LOOP_NAME = "ThresholdLoop"
    FAILURE_CATEGORIES = frozenset({FailureCategory.ROUTING_ERROR})
    TARGET_FILES = [_PLANNER_FILE]

    def _get_current_threshold(self) -> float:
        content = _PLANNER_FILE.read_text(encoding="utf-8")
        match = re.search(r"^CONFIDENCE_LOW\s*=\s*([\d.]+)", content, re.MULTILINE)
        if match:
            return float(match.group(1))
        return 0.7

    def _build_proposal_prompt(self, failures: list[dict]) -> str:
        failure_text = ""
        for f in failures[:15]:
            failure_text += (
                f"  Question: {f['question']}\n"
                f"  Golden answer: {f['golden_answer']}\n"
                f"  Prompt type: {f.get('prompt_type', 'None')}\n"
                f"  Detected entity: {f.get('detected_entity', 'None')}\n"
                f"  Match tier: {f.get('match_tier', 'None')}\n"
                f"  Match score: {f.get('match_score', 'None')}\n"
                f"  ---\n"
            )

        return _PROMPT.format(
            current_threshold=self._get_current_threshold(),
            min_threshold=MIN_THRESHOLD,
            max_threshold=MAX_THRESHOLD,
            failures=failure_text,
        )

    def _validate_proposal(self, proposal: dict) -> bool:
        if not proposal or not isinstance(proposal, dict):
            return False
        if "confidence_threshold" not in proposal:
            return False
        for key in proposal:
            if key != "confidence_threshold":
                return False
        val = proposal["confidence_threshold"]
        if val is None:
            return False
        if not isinstance(val, (int, float)):
            return False
        if val < MIN_THRESHOLD or val > MAX_THRESHOLD:
            return False
        return True

    def _apply_proposal(self, proposal: dict) -> dict[str, int]:
        new_val = proposal.get("confidence_threshold")
        if new_val is None:
            return {}
        self._prev_threshold = self._get_current_threshold()
        if _update_threshold(new_val):
            return {"confidence_threshold": 1}
        return {}

    def _revert_proposal(self, proposal: dict) -> None:
        prev = getattr(self, "_prev_threshold", None)
        if prev is not None:
            _update_threshold(prev)


def _update_threshold(new_value: float) -> bool:
    """Update CONFIDENCE_LOW in the planner file."""
    content = _PLANNER_FILE.read_text(encoding="utf-8")
    pattern = re.compile(r"^(CONFIDENCE_LOW\s*=\s*)[\d.]+", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        logger.warning("Could not find CONFIDENCE_LOW in %s", _PLANNER_FILE)
        return False
    new_content = (
        content[:match.start()]
        + f"{match.group(1)}{new_value}"
        + content[match.end():]
    )
    _PLANNER_FILE.write_text(new_content, encoding="utf-8")
    return True
